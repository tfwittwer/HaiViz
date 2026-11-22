//! LAS extra-bytes fields: the descriptors from the Extra Bytes VLR, the values
//! they describe, and the colour ramps of the Extra-bytes colour mode.
//!
//! The `las` crate hands every point its extra bytes as an opaque
//! `Point::extra_bytes`, but does not parse the descriptors that say what is in
//! them, so that is done here. Descriptor layout (LAS 1.4 R15, table 24), 192
//! bytes: `reserved[2] data_type options name[32] unused[4] no_data[3] min[3]
//! max[3] scale[3] offset[3] description[32]`; only element 0 of each
//! three-element array is used.
//!
//! copc_compare's `c2c` output is the case this was written for: i32 fields
//! `c2c_dist` (always), `c2c_dx`/`c2c_dy`/`c2c_dz` and `c2c_plane`, in metres
//! after scaling. A point with no reference neighbour within max-dist holds 5 ×
//! max-dist (the "cap") in every one of them. Max-dist itself is not stored,
//! but whenever any point is beyond it the cap is `c2c_dist`'s descriptor max.

use las::Header;
use log::warn;

const DESCRIPTOR_SIZE: usize = 192;
const USER_ID: &str = "LASF_Spec";
const RECORD_ID: u16 = 4;

const OPT_NO_DATA: u8 = 1 << 0;
const OPT_MIN: u8 = 1 << 1;
const OPT_MAX: u8 = 1 << 2;
const OPT_SCALE: u8 = 1 << 3;
const OPT_OFFSET: u8 = 1 << 4;

const OFF_DATA_TYPE: usize = 2;
const OFF_OPTIONS: usize = 3;
const OFF_NAME: usize = 4;
const OFF_NO_DATA: usize = 40;
const OFF_MIN: usize = 64;
const OFF_MAX: usize = 88;
const OFF_SCALE: usize = 112;
const OFF_OFFSET: usize = 136;

/// The fields copc_compare c2c writes. Every one of them holds the cap for a
/// point that found no neighbour.
const C2C_FIELDS: [&str; 5] = ["c2c_dist", "c2c_dx", "c2c_dy", "c2c_dz", "c2c_plane"];
const C2C_DIST: &str = "c2c_dist";

/// Points at the c2c cap ("beyond max-dist"), whatever the range.
pub const BEYOND_COLOR: [f32; 3] = [0.42, 0.42, 0.42];
/// Points whose file has no such field, or whose value is the field's no-data
/// value or not a number.
pub const NO_VALUE_COLOR: [f32; 3] = [0.2, 0.2, 0.2];

/// One field of the extra bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraField {
    pub name: String,
    /// Byte offset within `Point::extra_bytes`, and byte count.
    pub start: usize,
    pub len: usize,
    /// LAS data type: 0 undocumented, 1..=10 scalars, 11..=30 deprecated
    /// two- and three-element arrays.
    pub data_type: u8,
    /// 1 and 0 when the descriptor does not set them.
    pub scale: f64,
    pub offset: f64,
    /// Descriptor statistics and no-data value, as stored (before scale).
    pub min_raw: Option<f64>,
    pub max_raw: Option<f64>,
    pub no_data_raw: Option<f64>,
}

impl ExtraField {
    /// A single number, which is what can be coloured by and shown scaled.
    pub fn is_numeric(&self) -> bool {
        (1..=10).contains(&self.data_type)
    }

    pub fn is_c2c(&self) -> bool {
        C2C_FIELDS.contains(&self.name.as_str())
    }

    /// The stored number in a point's extra bytes, before scale and offset.
    /// `None` for non-numeric fields and for a record too short to hold it.
    pub fn raw(&self, extra: &[u8]) -> Option<f64> {
        let b = extra.get(self.start..self.start + self.len)?;
        decode(self.data_type, b)
    }

    fn scaled(&self, raw: f64) -> f64 {
        raw * self.scale + self.offset
    }

    pub fn min(&self) -> Option<f64> {
        self.min_raw.map(|r| self.scaled(r))
    }

    pub fn max(&self) -> Option<f64> {
        self.max_raw.map(|r| self.scaled(r))
    }

    /// Decimals that show every step of the scale, e.g. 4 for 0.0001.
    pub fn decimals(&self) -> usize {
        if (9..=10).contains(&self.data_type) && self.scale == 1.0 {
            // An unscaled float has no step to take the precision from.
            return 6;
        }
        decimals_of(self.scale).max(decimals_of(self.offset))
    }

    /// The scaled value, or `None` when the record does not hold one: a
    /// non-numeric field, the no-data value, or not a number.
    pub fn value(&self, extra: &[u8]) -> Option<f64> {
        let raw = self.raw(extra)?;
        if Some(raw) == self.no_data_raw {
            return None;
        }
        Some(self.scaled(raw)).filter(|v| v.is_finite())
    }

    /// Lowest and highest value over `extra` records (the no-data value and
    /// non-numbers left out). For fields whose descriptor has no statistics.
    pub fn value_range<'a>(&self, extra: impl Iterator<Item = &'a [u8]>, cap_raw: Option<f64>) -> Option<(f64, f64)> {
        let mut r: Option<(f64, f64)> = None;
        for e in extra {
            let Some(raw) = self.raw(e) else { continue };
            if Some(raw) == cap_raw && self.is_c2c() {
                continue;
            }
            let Some(v) = self.value(e) else { continue };
            r = Some(r.map_or((v, v), |(lo, hi)| (lo.min(v), hi.max(v))));
        }
        r
    }
}

/// The smallest number of decimals at which `x` is exact, up to 9.
fn decimals_of(x: f64) -> usize {
    (0..=9)
        .find(|&d| {
            let m = x * 10f64.powi(d as i32);
            (m - m.round()).abs() <= 1e-6 * m.abs().max(1.0)
        })
        .unwrap_or(9)
}

/// A scalar of LAS data type `t` from its little-endian bytes.
fn decode(t: u8, b: &[u8]) -> Option<f64> {
    Some(match (t, b.len()) {
        (1, 1) => f64::from(b[0]),
        (2, 1) => f64::from(b[0] as i8),
        (3, 2) => f64::from(u16::from_le_bytes([b[0], b[1]])),
        (4, 2) => f64::from(i16::from_le_bytes([b[0], b[1]])),
        (5, 4) => f64::from(u32::from_le_bytes(b.try_into().ok()?)),
        (6, 4) => f64::from(i32::from_le_bytes(b.try_into().ok()?)),
        (7, 8) => u64::from_le_bytes(b.try_into().ok()?) as f64,
        (8, 8) => i64::from_le_bytes(b.try_into().ok()?) as f64,
        (9, 4) => f64::from(f32::from_le_bytes(b.try_into().ok()?)),
        (10, 8) => f64::from_le_bytes(b.try_into().ok()?),
        _ => return None,
    })
}

/// An "anytype" statistic or no-data value: u64 for the unsigned types, i64
/// for the signed ones, f64 for the floats.
fn decode_any(t: u8, b: &[u8]) -> Option<f64> {
    let b: [u8; 8] = b.try_into().ok()?;
    Some(match t {
        1 | 3 | 5 | 7 => u64::from_le_bytes(b) as f64,
        2 | 4 | 6 | 8 => i64::from_le_bytes(b) as f64,
        9 | 10 => f64::from_le_bytes(b),
        _ => return None,
    })
}

/// Bytes per point a descriptor covers.
fn descriptor_len(d: &[u8]) -> Option<usize> {
    const BASE: [usize; 10] = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8];
    let t = d[OFF_DATA_TYPE];
    Some(match t {
        0 => usize::from(d[OFF_OPTIONS]),
        1..=10 => BASE[usize::from(t - 1)],
        11..=20 => 2 * BASE[usize::from(t - 11)],
        21..=30 => 3 * BASE[usize::from(t - 21)],
        _ => return None,
    })
}

fn las_str(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_string()
}

/// Every extra-bytes field of one file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtraFields {
    pub fields: Vec<ExtraField>,
    /// The c2c "beyond max-dist" value, raw, when the file is c2c output and
    /// has points beyond max-dist. See [`c2c_cap`].
    pub c2c_cap_raw: Option<f64>,
}

impl ExtraFields {
    pub fn from_header(h: &Header) -> Self {
        let n_extra = usize::from(h.point_format().extra_bytes);
        let vlr = h.vlrs().iter().chain(h.evlrs())
            .find(|v| v.user_id == USER_ID && v.record_id == RECORD_ID);
        Self::parse(vlr.map(|v| v.data.as_slice()), n_extra)
    }

    /// The fields of an Extra Bytes VLR payload for records with `n_extra`
    /// extra bytes. Bytes no descriptor covers become one field named
    /// `(undescribed)`. A malformed VLR is logged and as much of it used as
    /// lines up with the records.
    pub fn parse(payload: Option<&[u8]>, n_extra: usize) -> Self {
        let payload = payload.unwrap_or_default();
        if !payload.len().is_multiple_of(DESCRIPTOR_SIZE) {
            warn!("Extra Bytes VLR is {} bytes, not a multiple of {DESCRIPTOR_SIZE}", payload.len());
        }
        let mut fields = Vec::new();
        let mut start = 0;
        for d in payload.chunks_exact(DESCRIPTOR_SIZE) {
            let name = las_str(&d[OFF_NAME..OFF_NAME + 32]);
            let Some(len) = descriptor_len(d) else {
                warn!("Extra bytes field '{name}' has unknown data type {}; ignoring it and those after it", d[OFF_DATA_TYPE]);
                break;
            };
            if start + len > n_extra {
                warn!("Extra bytes field '{name}' ends at byte {}, but the records have {n_extra}; ignoring it and those after it", start + len);
                break;
            }
            let (t, opt) = (d[OFF_DATA_TYPE], d[OFF_OPTIONS]);
            // Options mean something else for an undocumented field (its byte
            // count), and statistics only make sense for a scalar.
            let scalar = (1..=10).contains(&t);
            let set = |bit: u8| scalar && opt & bit != 0;
            let f64_at = |at: usize| f64::from_le_bytes(d[at..at + 8].try_into().unwrap());
            let any_at = |at: usize| decode_any(t, &d[at..at + 8]);
            let scale = if set(OPT_SCALE) { f64_at(OFF_SCALE) } else { 1.0 };
            fields.push(ExtraField {
                name,
                start,
                len,
                data_type: t,
                // A zero scale would make every value 0.
                scale: if scale != 0.0 && scale.is_finite() { scale } else { 1.0 },
                offset: if set(OPT_OFFSET) { f64_at(OFF_OFFSET) } else { 0.0 },
                min_raw: if set(OPT_MIN) { any_at(OFF_MIN) } else { None },
                max_raw: if set(OPT_MAX) { any_at(OFF_MAX) } else { None },
                no_data_raw: if set(OPT_NO_DATA) { any_at(OFF_NO_DATA) } else { None },
            });
            start += len;
        }
        if start < n_extra {
            fields.push(ExtraField {
                name: "(undescribed)".into(),
                start,
                len: n_extra - start,
                data_type: 0,
                scale: 1.0,
                offset: 0.0,
                min_raw: None,
                max_raw: None,
                no_data_raw: None,
            });
        }
        let c2c_cap_raw = c2c_cap(&fields);
        Self { fields, c2c_cap_raw }
    }

    pub fn get(&self, name: &str) -> Option<&ExtraField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Whether `raw`, read from field `f`, is the c2c "beyond max-dist" cap.
    ///
    /// Exact per field: a point that found a neighbour has |dist|, |dx|, |dy|,
    /// |dz| and |plane| all ≤ max-dist (the plane passes through the
    /// neighbours' centroid, which is itself within max-dist), and the cap is
    /// 5 × max-dist.
    pub fn is_beyond(&self, f: &ExtraField, raw: f64) -> bool {
        f.is_c2c() && Some(raw) == self.c2c_cap_raw
    }

    /// The cap in metres.
    pub fn c2c_cap(&self) -> Option<f64> {
        let f = self.get(C2C_DIST)?;
        self.c2c_cap_raw.map(|r| f.scaled(r))
    }

    /// One Point Info line per field: name and value.
    pub fn describe(&self, extra: &[u8]) -> Vec<(String, String)> {
        self.fields.iter().map(|f| {
            let text = match f.raw(extra) {
                Some(raw) if self.is_beyond(f, raw) => "beyond max-dist".to_string(),
                Some(raw) if Some(raw) == f.no_data_raw => "no data".to_string(),
                Some(raw) => {
                    let unit = if f.is_c2c() { " m" } else { "" };
                    format!("{:.*}{unit}", f.decimals(), f.scaled(raw))
                }
                None => match extra.get(f.start..f.start + f.len) {
                    Some(b) => b.iter().map(|x| format!("{x:02x}")).collect(),
                    None => "—".to_string(),
                },
            };
            (f.name.clone(), text)
        }).collect()
    }
}

/// The raw c2c cap, if this is c2c output with points beyond max-dist.
///
/// Taken from `c2c_dist`'s descriptor max, which is the cap whenever any point
/// is beyond max-dist. When none is, that max is just the largest distance, and
/// treating it as the cap would grey out the farthest point and make the
/// default range a fifth of what it should be. The other c2c fields tell the
/// two apart: beyond points put the cap into every one of them, while real
/// components and plane distances stay within max-dist, below the cap. So if a
/// file has any of them and one of their maxima differs, nothing is at the cap.
/// A `c2c_dist`-only file has no such check and the max is taken as the cap.
fn c2c_cap(fields: &[ExtraField]) -> Option<f64> {
    let dist = fields.iter().find(|f| f.name == C2C_DIST)?;
    let cap = dist.max_raw?;
    let others_agree = fields.iter()
        .filter(|f| f.is_c2c() && f.name != C2C_DIST)
        .all(|f| f.max_raw == Some(cap));
    (others_agree && cap > 0.0).then_some(cap)
}

/// What the colour controls know about one field name across every loaded
/// file.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSummary {
    pub name: String,
    /// Scaled descriptor statistics, widened over the files.
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Finest step over the files.
    pub decimals: usize,
    /// c2c cap in metres, largest over the files.
    pub c2c_cap: Option<f64>,
}

impl FieldSummary {
    pub fn is_c2c(&self) -> bool {
        C2C_FIELDS.contains(&self.name.as_str())
    }
}

/// The numeric fields of every file, by name, in order of first appearance.
/// Statistics are only merged where every file that has the field has them,
/// so one file without them does not make the range look known.
pub fn summarise<'a>(sets: impl Iterator<Item = &'a ExtraFields>) -> Vec<FieldSummary> {
    let mut out: Vec<FieldSummary> = Vec::new();
    for set in sets {
        let cap = set.c2c_cap();
        for f in set.fields.iter().filter(|f| f.is_numeric()) {
            let c2c_cap = if f.is_c2c() { cap } else { None };
            match out.iter_mut().find(|s| s.name == f.name) {
                Some(s) => {
                    s.min = s.min.zip(f.min()).map(|(a, b)| a.min(b));
                    s.max = s.max.zip(f.max()).map(|(a, b)| a.max(b));
                    s.decimals = s.decimals.max(f.decimals());
                    s.c2c_cap = match (s.c2c_cap, c2c_cap) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    };
                }
                None => out.push(FieldSummary {
                    name: f.name.clone(),
                    min: f.min(),
                    max: f.max(),
                    decimals: f.decimals(),
                    c2c_cap,
                }),
            }
        }
    }
    out
}

/// The field to colour by when the operator has not picked one.
pub fn default_field(fields: &[FieldSummary]) -> Option<&FieldSummary> {
    fields.iter().find(|f| f.name == C2C_DIST).or(fields.first())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ramp {
    /// Low to high, for magnitudes.
    Sequential,
    /// Neutral at 0, for signed values.
    Diverging,
}

impl Ramp {
    pub fn label(self) -> &'static str {
        match self {
            Ramp::Sequential => "Sequential",
            Ramp::Diverging => "Diverging",
        }
    }
}

/// The colour-by-field selection, the same for every file. `lo`/`hi` are in
/// the field's scaled units.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraColor {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
    pub ramp: Ramp,
}

impl ExtraColor {
    /// The starting selection for `f`. The range comes from the header, never
    /// from the points: COPC nodes stream in as the view moves, so a range
    /// gathered from them would keep shifting the colours. `loaded` is the
    /// fallback for a field whose header has no statistics; it is taken once,
    /// here, and then stays put.
    pub fn default_for(f: &FieldSummary, loaded: Option<(f64, f64)>) -> Self {
        let signed = f.is_c2c() && f.name != C2C_DIST;
        let (lo, hi) = match (f.c2c_cap, signed) {
            (Some(cap), false) => (0.0, cap / 5.0),
            (Some(cap), true) => (-cap / 5.0, cap / 5.0),
            (None, _) => {
                let (lo, hi) = f.min.zip(f.max).or(loaded).unwrap_or((0.0, 1.0));
                if signed {
                    let m = lo.abs().max(hi.abs());
                    (-m, m)
                } else if f.name == C2C_DIST {
                    (0.0, hi)
                } else {
                    (lo, hi)
                }
            }
        };
        let (lo, hi) = if hi > lo { (lo, hi) } else { (lo - 0.5, lo + 0.5) };
        let ramp = if signed { Ramp::Diverging } else { Ramp::Sequential };
        Self { name: f.name.clone(), lo, hi, ramp }
    }
}

/// An [`ExtraColor`] resolved against one file's descriptors: what the
/// per-point colouring needs, with no lookups left.
#[derive(Clone, Copy, Debug)]
pub struct ExtraRamp {
    start: usize,
    len: usize,
    data_type: u8,
    scale: f64,
    offset: f64,
    no_data_raw: Option<f64>,
    /// Raw cap, for c2c fields only.
    cap_raw: Option<f64>,
    lo: f64,
    hi: f64,
    ramp: Ramp,
}

impl ExtraRamp {
    /// `None` when there is no selection or this file has no numeric field of
    /// that name; its points are then drawn in [`NO_VALUE_COLOR`].
    pub fn resolve(sel: Option<&ExtraColor>, fields: &ExtraFields) -> Option<Self> {
        let sel = sel?;
        let f = fields.get(&sel.name).filter(|f| f.is_numeric())?;
        Some(Self {
            start: f.start,
            len: f.len,
            data_type: f.data_type,
            scale: f.scale,
            offset: f.offset,
            no_data_raw: f.no_data_raw,
            cap_raw: if f.is_c2c() { fields.c2c_cap_raw } else { None },
            lo: sel.lo,
            hi: sel.hi,
            ramp: sel.ramp,
        })
    }

    pub fn color(&self, extra: &[u8]) -> [f32; 3] {
        let Some(raw) = extra.get(self.start..self.start + self.len).and_then(|b| decode(self.data_type, b)) else {
            return NO_VALUE_COLOR;
        };
        if Some(raw) == self.cap_raw {
            return BEYOND_COLOR;
        }
        if Some(raw) == self.no_data_raw {
            return NO_VALUE_COLOR;
        }
        let v = raw * self.scale + self.offset;
        if !v.is_finite() {
            return NO_VALUE_COLOR;
        }
        let t = match self.ramp {
            Ramp::Sequential => unit(v, self.lo, self.hi),
            Ramp::Diverging => diverging_t(v, self.lo, self.hi),
        };
        ramp_color(self.ramp, t)
    }
}

/// The colour at `t` in `[0, 1]` along a ramp.
pub fn ramp_color(ramp: Ramp, t: f32) -> [f32; 3] {
    match ramp {
        Ramp::Sequential => sample(&VIRIDIS, t),
        Ramp::Diverging => sample(&RDBU, t),
    }
}

/// `v` on `[lo, hi]` as `[0, 1]`, clamped.
fn unit(v: f64, lo: f64, hi: f64) -> f32 {
    if hi > lo { ((v - lo) / (hi - lo)).clamp(0.0, 1.0) as f32 } else { 0.5 }
}

/// Ramp position for a diverging ramp: 0 stays at the neutral middle however
/// lopsided the range is, with each side stretched over its own half. A range
/// that does not straddle 0 has no neutral and is spread over the whole ramp.
fn diverging_t(v: f64, lo: f64, hi: f64) -> f32 {
    if lo < 0.0 && hi > 0.0 {
        if v < 0.0 { 0.5 - 0.5 * (v / lo).min(1.0) as f32 } else { 0.5 + 0.5 * (v / hi).min(1.0) as f32 }
    } else {
        unit(v, lo, hi)
    }
}

/// Linear interpolation between evenly spaced stops.
fn sample(stops: &[[f32; 3]], t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
    let i = (x.floor() as usize).min(stops.len() - 2);
    let f = x - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

const fn hex(c: u32) -> [f32; 3] {
    [((c >> 16) & 0xff) as f32 / 255.0, ((c >> 8) & 0xff) as f32 / 255.0, (c & 0xff) as f32 / 255.0]
}

/// matplotlib's viridis at 0, 0.1, …, 1: perceptually even, and without greys,
/// so the beyond-max-dist grey cannot be mistaken for a value.
const VIRIDIS: [[f32; 3]; 11] = [
    hex(0x440154), hex(0x482475), hex(0x414487), hex(0x355f8d), hex(0x2a788e), hex(0x21918c),
    hex(0x22a884), hex(0x44bf70), hex(0x7ad151), hex(0xbddf26), hex(0xfde725),
];

/// ColorBrewer RdBu (9 classes), blue for negative and red for positive. The
/// 9-class ends rather than the 11-class ones, which are too dark to see on
/// the default black background. The near-white neutral is far lighter than
/// the beyond grey.
const RDBU: [[f32; 3]; 9] = [
    hex(0x2166ac), hex(0x4393c3), hex(0x92c5de), hex(0xd1e5f0), hex(0xf7f7f7),
    hex(0xfddbc7), hex(0xf4a582), hex(0xd6604d), hex(0xb2182b),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor as copc_compare writes it: i32, scale, min/max set.
    fn i32_desc(name: &str, scale: f64, min: i64, max: i64) -> [u8; DESCRIPTOR_SIZE] {
        let mut d = [0u8; DESCRIPTOR_SIZE];
        d[OFF_DATA_TYPE] = 6;
        d[OFF_OPTIONS] = OPT_SCALE | OPT_MIN | OPT_MAX;
        d[OFF_NAME..OFF_NAME + name.len()].copy_from_slice(name.as_bytes());
        d[OFF_MIN..OFF_MIN + 8].copy_from_slice(&min.to_le_bytes());
        d[OFF_MAX..OFF_MAX + 8].copy_from_slice(&max.to_le_bytes());
        d[OFF_SCALE..OFF_SCALE + 8].copy_from_slice(&scale.to_le_bytes());
        d
    }

    fn undocumented(n: u8) -> [u8; DESCRIPTOR_SIZE] {
        let mut d = [0u8; DESCRIPTOR_SIZE];
        d[OFF_OPTIONS] = n;
        d[OFF_NAME..OFF_NAME + 3].copy_from_slice(b"pre");
        d
    }

    fn payload(ds: &[[u8; DESCRIPTOR_SIZE]]) -> Vec<u8> {
        ds.iter().flatten().copied().collect()
    }

    /// Against real copc_compare c2c output: `HAIVIZ_C2C=/path/x.copc.laz
    /// cargo test --release real_c2c -- --ignored --nocapture`. Reads the
    /// first chunks' points and checks the fields against each other: beyond
    /// points hold the cap everywhere, the others have a distance within
    /// max-dist that agrees with their components.
    #[test]
    #[ignore]
    fn real_c2c_fields_agree() {
        let Ok(p) = std::env::var("HAIVIZ_C2C") else { return };
        let mut r = las::Reader::from_path(&p).unwrap();
        let x = ExtraFields::from_header(r.header());
        for f in &x.fields {
            println!("{:<14} start {:>2} len {} type {:>2} scale {} min {:?} max {:?} decimals {}",
                f.name, f.start, f.len, f.data_type, f.scale, f.min(), f.max(), f.decimals());
        }
        let cap = x.c2c_cap_raw.expect("no c2c cap: no point beyond max-dist, or not c2c output");
        let dist = x.get(C2C_DIST).unwrap();
        let comps: Vec<&ExtraField> = ["c2c_dx", "c2c_dy", "c2c_dz"].iter().filter_map(|n| x.get(n)).collect();
        let plane = x.get("c2c_plane");
        let max_dist = dist.scaled(cap) / 5.0;
        println!("cap raw {cap} = {} m, max-dist {max_dist} m", dist.scaled(cap));
        let (mut n, mut beyond, mut worst_comp, mut worst_plane) = (0usize, 0usize, 0.0f64, 0.0f64);
        // `HAIVIZ_C2C_N` points (default half a million), a million at a time.
        let limit: usize = std::env::var("HAIVIZ_C2C_N").ok().and_then(|v| v.parse().ok()).unwrap_or(500_000);
        while n < limit {
            let data = r.read_points(1_000_000.min((limit - n) as u64)).unwrap();
            if data.is_empty() {
                break;
            }
            for pt in data.points() {
                let pt = pt.unwrap();
                let e = &pt.extra_bytes;
                n += 1;
                let raw = dist.raw(e).unwrap();
                if x.is_beyond(dist, raw) {
                    beyond += 1;
                    for f in x.fields.iter().filter(|f| f.is_c2c()) {
                        assert_eq!(f.raw(e), Some(cap), "{} of a beyond point", f.name);
                    }
                    let lines = x.describe(e);
                    assert!(lines.iter().filter(|(n, _)| C2C_FIELDS.contains(&n.as_str())).all(|(_, v)| v == "beyond max-dist"));
                    println!("beyond at {:.3} {:.3} {:.3}: {lines:?}", pt.x, pt.y, pt.z);
                    continue;
                }
                let d = dist.value(e).unwrap();
                assert!((0.0..=max_dist + dist.scale).contains(&d), "c2c_dist {d}");
                if comps.len() == 3 {
                    let c: Vec<f64> = comps.iter().map(|f| f.value(e).unwrap()).collect();
                    let h = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                    worst_comp = worst_comp.max((h - d).abs());
                }
                if let Some(pl) = plane {
                    let v = pl.value(e).unwrap();
                    assert!(!x.is_beyond(pl, pl.raw(e).unwrap()));
                    assert!(v.abs() <= max_dist + pl.scale, "c2c_plane {v}");
                    worst_plane = worst_plane.max(v.abs());
                }
            }
        }
        println!("{n} points, {beyond} beyond; |hypot(dx,dy,dz) - dist| <= {worst_comp:.6} m; |plane| <= {worst_plane:.4} m");
        // Each stored value is rounded to half a step.
        assert!(worst_comp <= 2.0 * dist.scale, "components disagree with c2c_dist by {worst_comp}");
    }

    #[test]
    fn offsets_follow_earlier_fields() {
        // 3 bytes of the input's own, then c2c_dist and c2c_dz.
        let p = payload(&[undocumented(3), i32_desc("c2c_dist", 1e-4, 0, 50000), i32_desc("c2c_dz", 1e-4, -9000, 50000)]);
        let f = ExtraFields::parse(Some(&p), 13);
        let names: Vec<_> = f.fields.iter().map(|f| (f.name.as_str(), f.start, f.len)).collect();
        assert_eq!(names, [("pre", 0, 3), ("c2c_dist", 3, 4), ("c2c_dz", 7, 4), ("(undescribed)", 11, 2)]);
        let mut rec = vec![0xaa, 0xbb, 0xcc];
        rec.extend_from_slice(&1234i32.to_le_bytes());
        rec.extend_from_slice(&(-250i32).to_le_bytes());
        rec.extend_from_slice(&[1, 2]);
        let dz = f.get("c2c_dz").unwrap();
        assert!((dz.value(&rec).unwrap() + 0.025).abs() < 1e-12);
        assert_eq!(f.c2c_cap_raw, Some(50000.0));
        assert!((f.c2c_cap().unwrap() - 5.0).abs() < 1e-12);
        let lines = f.describe(&rec);
        assert_eq!(lines[0], ("pre".into(), "aabbcc".into()));
        assert_eq!(lines[1], ("c2c_dist".into(), "0.1234 m".into()));
        assert_eq!(lines[2], ("c2c_dz".into(), "-0.0250 m".into()));
        assert_eq!(lines[3], ("(undescribed)".into(), "0102".into()));
    }

    #[test]
    fn cap_is_beyond_in_every_field() {
        let p = payload(&[i32_desc("c2c_dist", 1e-3, 0, 5000), i32_desc("c2c_dx", 1e-3, -1000, 5000)]);
        let f = ExtraFields::parse(Some(&p), 8);
        let mut rec = 5000i32.to_le_bytes().to_vec();
        rec.extend_from_slice(&5000i32.to_le_bytes());
        assert!(f.describe(&rec).iter().all(|(_, v)| v == "beyond max-dist"));
        let sel = ExtraColor { name: "c2c_dx".into(), lo: -1.0, hi: 1.0, ramp: Ramp::Diverging };
        let r = ExtraRamp::resolve(Some(&sel), &f).unwrap();
        assert_eq!(r.color(&rec), BEYOND_COLOR);
        // 0 is neutral, however lopsided the range.
        let mut zero = 0i32.to_le_bytes().to_vec();
        zero.extend_from_slice(&0i32.to_le_bytes());
        let lop = ExtraRamp::resolve(Some(&ExtraColor { lo: -0.1, hi: 0.9, ..sel.clone() }), &f).unwrap();
        assert_eq!(lop.color(&zero), RDBU[4]);
    }

    #[test]
    fn no_cap_when_components_disagree() {
        // Nothing beyond max-dist: dist's max is just the largest distance.
        let p = payload(&[i32_desc("c2c_dist", 1e-3, 0, 870), i32_desc("c2c_dx", 1e-3, -600, 700)]);
        let f = ExtraFields::parse(Some(&p), 8);
        assert_eq!(f.c2c_cap_raw, None);
        let s = summarise(std::iter::once(&f));
        let d = ExtraColor::default_for(&s[0], None);
        assert_eq!((d.lo, d.ramp), (0.0, Ramp::Sequential));
        assert!((d.hi - 0.87).abs() < 1e-12);
        let x = ExtraColor::default_for(&s[1], None);
        assert_eq!(x.ramp, Ramp::Diverging);
        assert!((x.lo + 0.7).abs() < 1e-12 && (x.hi - 0.7).abs() < 1e-12);
    }

    #[test]
    fn c2c_defaults_are_a_fifth_of_the_cap() {
        let p = payload(&[i32_desc("c2c_dist", 1e-4, 0, 50000), i32_desc("c2c_plane", 1e-4, -8000, 50000)]);
        let f = ExtraFields::parse(Some(&p), 8);
        let s = summarise(std::iter::once(&f));
        assert_eq!(default_field(&s).unwrap().name, "c2c_dist");
        let d = ExtraColor::default_for(&s[0], None);
        assert!((d.lo - 0.0).abs() < 1e-12 && (d.hi - 1.0).abs() < 1e-12);
        let pl = ExtraColor::default_for(&s[1], None);
        assert!((pl.lo + 1.0).abs() < 1e-12 && (pl.hi - 1.0).abs() < 1e-12);
        assert_eq!(pl.ramp, Ramp::Diverging);
    }

    #[test]
    fn decimals_from_scale() {
        assert_eq!(decimals_of(1e-4), 4);
        assert_eq!(decimals_of(0.001), 3);
        assert_eq!(decimals_of(0.25), 2);
        assert_eq!(decimals_of(1.0), 0);
        assert_eq!(decimals_of(0.0), 0);
    }

    #[test]
    fn general_field_without_stats() {
        // u16 with scale 0.5 and no min/max: range falls back to the loaded points.
        let mut d = [0u8; DESCRIPTOR_SIZE];
        d[OFF_DATA_TYPE] = 3;
        d[OFF_OPTIONS] = OPT_SCALE;
        d[OFF_NAME..OFF_NAME + 4].copy_from_slice(b"temp");
        d[OFF_SCALE..OFF_SCALE + 8].copy_from_slice(&0.5f64.to_le_bytes());
        let f = ExtraFields::parse(Some(&d), 2);
        let recs: Vec<Vec<u8>> = [10u16, 40, 20].iter().map(|v| v.to_le_bytes().to_vec()).collect();
        let r = f.fields[0].value_range(recs.iter().map(|r| r.as_slice()), None);
        assert_eq!(r, Some((5.0, 20.0)));
        let s = summarise(std::iter::once(&f));
        let c = ExtraColor::default_for(&s[0], r);
        assert_eq!((c.lo, c.hi, c.ramp), (5.0, 20.0, Ramp::Sequential));
        assert_eq!(f.describe(&recs[1])[0].1, "20.0");
    }

    #[test]
    fn ramps_hit_their_ends() {
        assert_eq!(sample(&VIRIDIS, 0.0), VIRIDIS[0]);
        assert_eq!(sample(&VIRIDIS, 1.0), VIRIDIS[10]);
        assert_eq!(diverging_t(-5.0, -1.0, 2.0), 0.0);
        assert_eq!(diverging_t(1.0, -1.0, 2.0), 0.75);
        assert_eq!(diverging_t(3.0, 1.0, 5.0), 0.5);
    }
}
