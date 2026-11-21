//! Gaussian splats, from the PLY layout 3DGS trainers write (Inria, Brush,
//! gsplat, nerfstudio): position, log-scale, rotation quaternion, logit
//! opacity and spherical-harmonic colour per vertex.
//!
//! Rendering is the usual: each splat is a screen-space ellipse from its 3D
//! covariance, alpha-blended back to front. Two things make it cheaper here
//! than in a perspective viewer.
//!
//! - **The projection is exact.** Under an orthographic camera the projection
//!   is affine, so the 2D covariance is J Σ Jᵀ with J the camera's linear
//!   part — no per-splat Jacobian approximation.
//! - **The sort depends only on the view direction.** Depth along a fixed
//!   direction does not change when the camera pans or zooms, so only
//!   rotation re-sorts. That runs on a worker thread; the order on screen
//!   lags a rotation by a frame or two, which shows as nothing at all.
//!
//! Splat data lives in float textures, fetched in the vertex shader by an
//! index that is the only per-instance attribute. A re-sort then uploads 4
//! bytes per splat, not the 50 the splat itself takes.
//!
//! Colour is the SH DC term only; the view-dependent bands are read past.
//! For an inspection viewer that loses glints, not geometry.
//!
//! The same renderer draws splats streamed from 3D Tiles: [`SplatAtlas`]
//! keeps every resident tile's splats in one set of textures, so a single
//! sort and a single draw cover all the tiles on screen and nothing seams
//! where two tiles meet. Tiles carry SPZ-compressed splats
//! ([`decode_spz`]) or plain glTF accessors.
//!
//! Georeferencing: a `scene.json` with `origin_rd` (and optionally
//! `horizontal_scale`) beside the PLY or up to three folders above it — the
//! file the haires splat pipeline trains from — places the splats in RD/NAP,
//! by `local = diag(s, s, 1) · (RD − origin)`. Without one the PLY's
//! coordinates are taken as world coordinates.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use glam::{DVec3, Mat3, Quat, Vec3};
use miniquad::*;
use rayon::prelude::*;

use crate::raw_miniquad::ZClip;

/// Width of the data textures. A power of two, so the index → texel
/// arithmetic in the shader is exact in `float`.
const TEX_W: usize = 4096;
const SH_C0: f32 = 0.282_094_8;

#[derive(Clone, Debug, PartialEq)]
pub struct Georef {
    pub offset: [f64; 3],
    pub horizontal_scale: f64,
    /// The file it came from; `None` means the PLY's own coordinates.
    pub source: Option<PathBuf>,
}

impl Georef {
    pub const IDENTITY: Georef = Georef { offset: [0.0; 3], horizontal_scale: 1.0, source: None };

    pub fn to_world(&self, p: [f32; 3]) -> DVec3 {
        let s = self.horizontal_scale;
        DVec3::new(
            p[0] as f64 / s + self.offset[0],
            p[1] as f64 / s + self.offset[1],
            p[2] as f64 + self.offset[2],
        )
    }

    /// Look for a `scene.json` with `origin_rd` beside `ply` or above it.
    pub fn find(ply: &Path) -> Georef {
        let mut dir = ply.parent();
        for _ in 0..4 {
            let Some(d) = dir else { break };
            let p = d.join("scene.json");
            if let Some(g) = std::fs::read(&p)
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|j| Georef::from_scene_json(&j, &p))
            {
                return g;
            }
            dir = d.parent();
        }
        Georef::IDENTITY
    }

    fn from_scene_json(j: &serde_json::Value, path: &Path) -> Option<Georef> {
        let o = j.get("origin_rd")?.as_array()?;
        if o.len() < 2 {
            return None;
        }
        let z = o.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let s = j.get("horizontal_scale").and_then(|v| v.as_f64()).filter(|s| *s > 0.0).unwrap_or(1.0);
        Some(Georef { offset: [o[0].as_f64()?, o[1].as_f64()?, z], horizontal_scale: s, source: Some(path.to_path_buf()) })
    }
}

/// Splats as read from the file, in its own frame.
pub struct SplatFile {
    pub path: PathBuf,
    pub local: Vec<[f32; 3]>,
    /// Upper triangle of the 3D covariance: xx, xy, xz, yy, yz, zz.
    pub cov: Vec<[f32; 6]>,
    pub rgba: Vec<[u8; 4]>,
    /// Splats skipped as fully transparent.
    pub dropped: usize,
    pub georef: Georef,
}

#[derive(Clone, Copy)]
enum PlyType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyType {
    fn parse(s: &str) -> Option<PlyType> {
        Some(match s {
            "char" | "int8" => PlyType::I8,
            "uchar" | "uint8" => PlyType::U8,
            "short" | "int16" => PlyType::I16,
            "ushort" | "uint16" => PlyType::U16,
            "int" | "int32" => PlyType::I32,
            "uint" | "uint32" => PlyType::U32,
            "float" | "float32" => PlyType::F32,
            "double" | "float64" => PlyType::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            PlyType::I8 | PlyType::U8 => 1,
            PlyType::I16 | PlyType::U16 => 2,
            PlyType::I32 | PlyType::U32 | PlyType::F32 => 4,
            PlyType::F64 => 8,
        }
    }
    fn read(self, b: &[u8]) -> f32 {
        match self {
            PlyType::I8 => b[0] as i8 as f32,
            PlyType::U8 => b[0] as f32,
            PlyType::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            PlyType::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            PlyType::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            PlyType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            PlyType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            PlyType::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        }
    }
}

struct PlyElement {
    name: String,
    count: usize,
    /// (name, type, byte offset in the record)
    props: Vec<(String, PlyType, usize)>,
    stride: usize,
    has_list: bool,
}

impl PlyElement {
    fn prop(&self, name: &str) -> Option<(PlyType, usize)> {
        self.props.iter().find(|p| p.0 == name).map(|p| (p.1, p.2))
    }
}

/// Whether a PLY is Gaussian splats rather than an ordinary point cloud or
/// mesh — what decides which loader a dropped `.ply` goes to.
pub fn is_splat_ply(path: &Path) -> bool {
    let Ok(f) = std::fs::File::open(path) else { return false };
    let mut r = BufReader::new(f);
    match read_header(&mut r) {
        Ok(elements) => elements
            .iter()
            .any(|e| e.name == "vertex" && e.prop("scale_0").is_some() && e.prop("rot_0").is_some() && e.prop("opacity").is_some()),
        Err(_) => false,
    }
}

fn read_header(r: &mut impl BufRead) -> Result<Vec<PlyElement>, String> {
    let mut line = String::new();
    let next = |r: &mut dyn BufRead, line: &mut String| -> Result<(), String> {
        line.clear();
        let n = r.read_line(line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("PLY header ends early".into());
        }
        Ok(())
    };
    next(r, &mut line)?;
    if line.trim() != "ply" {
        return Err("not a PLY file".into());
    }
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut binary_le = false;
    loop {
        next(r, &mut line)?;
        let mut w = line.split_whitespace();
        match w.next() {
            Some("format") => {
                let f = w.next().unwrap_or("");
                if f != "binary_little_endian" {
                    return Err(format!("PLY format {f} is not supported, only binary_little_endian"));
                }
                binary_le = true;
            }
            Some("element") => {
                let name = w.next().unwrap_or("").to_string();
                let count = w.next().and_then(|c| c.parse().ok()).ok_or("bad PLY element line")?;
                elements.push(PlyElement { name, count, props: Vec::new(), stride: 0, has_list: false });
            }
            Some("property") => {
                let e = elements.last_mut().ok_or("PLY property before any element")?;
                let t = w.next().unwrap_or("");
                if t == "list" {
                    e.has_list = true;
                    continue;
                }
                let ty = PlyType::parse(t).ok_or_else(|| format!("unknown PLY type {t}"))?;
                let name = w.next().unwrap_or("").to_string();
                e.props.push((name, ty, e.stride));
                e.stride += ty.size();
            }
            Some("end_header") => break,
            _ => {}
        }
    }
    if !binary_le {
        return Err("PLY has no format line".into());
    }
    Ok(elements)
}

/// One decoded splat: centre, covariance, colour.
/// One splat: centre, covariance (xx, xy, xz, yy, yz, zz) and sRGB colour
/// with linear alpha.
pub type SplatRow = ([f32; 3], [f32; 6], [u8; 4]);

/// Σ = (R S)(R S)ᵀ, as its upper triangle. `scale` is linear, not log.
pub fn covariance(scale: Vec3, q: Quat) -> [f32; 6] {
    let q = if q.length_squared() > 0.0 { q.normalize() } else { Quat::IDENTITY };
    let m = Mat3::from_quat(q) * Mat3::from_diagonal(scale);
    let c = m * m.transpose();
    [c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z]
}

/// `L Σ Lᵀ`: a covariance carried through a linear map.
pub fn transform_covariance(l: &Mat3, v: [f32; 6]) -> [f32; 6] {
    let s = Mat3::from_cols(Vec3::new(v[0], v[1], v[2]), Vec3::new(v[1], v[3], v[4]), Vec3::new(v[2], v[4], v[5]));
    let c = *l * s * l.transpose();
    [c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z]
}

/// Splats in the layout the shader reads: three texels each.
/// `t0` = centre (viewer space) and colour packed as r·65536 + g·256 + b;
/// `t1` = xx, xy, xz, alpha; `t2` = yy, yz, zz.
#[derive(Default)]
pub struct SplatTexels {
    pub t0: Vec<[f32; 4]>,
    pub t1: Vec<[f32; 4]>,
    pub t2: Vec<[f32; 4]>,
}

impl SplatTexels {
    pub fn len(&self) -> usize {
        self.t0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t0.is_empty()
    }
    pub fn push(&mut self, center: [f32; 3], v: [f32; 6], rgba: [u8; 4]) {
        let [r, g, b, a] = rgba;
        let packed = r as f32 * 65536.0 + g as f32 * 256.0 + b as f32;
        self.t0.push([center[0], center[1], center[2], packed]);
        self.t1.push([v[0], v[1], v[2], a as f32 / 255.0]);
        self.t2.push([v[3], v[4], v[5], 0.0]);
    }
    pub fn append(&mut self, mut other: SplatTexels) {
        self.t0.append(&mut other.t0);
        self.t1.append(&mut other.t1);
        self.t2.append(&mut other.t2);
    }
}

/// Decode an SPZ blob (Niantic's compressed splats, gzipped), as
/// `KHR_gaussian_splatting_compression_spz` embeds it, into the file's own
/// frame — the glTF accessor space, with no axis change. Versions 2 and 3;
/// the spherical-harmonic bands are skipped like the PLY's.
pub fn decode_spz(gz: &[u8]) -> Result<Vec<SplatRow>, String> {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(gz).read_to_end(&mut raw).map_err(|e| format!("SPZ: {e}"))?;
    if raw.len() < 16 || raw[0..4] != 0x5053_474e_u32.to_le_bytes() {
        return Err("SPZ: not an SPZ stream".into());
    }
    let version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let n = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let (sh_degree, frac) = (raw[12], raw[13]);
    if !(2..=3).contains(&version) {
        return Err(format!("SPZ version {version} is not supported (2 and 3 are)"));
    }
    let rot_size = if version == 2 { 3 } else { 4 };
    let sh_dim = match sh_degree {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        d => return Err(format!("SPZ: spherical-harmonic degree {d}")),
    };
    let need = 16 + n * (9 + 1 + 3 + 3 + rot_size + sh_dim * 3);
    if raw.len() < need {
        return Err(format!("SPZ: {} bytes for {n} splats, {need} needed", raw.len()));
    }
    let pos = &raw[16..16 + 9 * n];
    let alpha = &raw[16 + 9 * n..16 + 10 * n];
    let color = &raw[16 + 10 * n..16 + 13 * n];
    let scale = &raw[16 + 13 * n..16 + 16 * n];
    let rot = &raw[16 + 16 * n..16 + (16 + rot_size) * n];
    let step = 1.0 / (1u32 << frac) as f32;
    let rows = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let a = alpha[i];
            if a == 0 {
                return None;
            }
            let p = [0, 1, 2].map(|k| {
                let b = &pos[9 * i + 3 * k..];
                // 24-bit two's complement, sign-extended through the top byte.
                (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 * step
            });
            let s = Vec3::new(
                (scale[3 * i] as f32 / 16.0 - 10.0).exp(),
                (scale[3 * i + 1] as f32 / 16.0 - 10.0).exp(),
                (scale[3 * i + 2] as f32 / 16.0 - 10.0).exp(),
            );
            let q = if version == 2 {
                let r = &rot[3 * i..3 * i + 3];
                let v = Vec3::new(r[0] as f32, r[1] as f32, r[2] as f32) / 127.5 - Vec3::ONE;
                Quat::from_xyzw(v.x, v.y, v.z, (1.0 - v.length_squared()).max(0.0).sqrt())
            } else {
                // Smallest three: 2 bits name the largest component, the
                // others are 9 bits of magnitude over 1/√2 and a sign bit.
                let r = &rot[4 * i..4 * i + 4];
                let mut comp = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                let largest = (comp >> 30) as usize;
                let mut q = [0f32; 4];
                let mut sum = 0.0;
                for k in (0..4).rev() {
                    if k == largest {
                        continue;
                    }
                    let mag = (comp & 511) as f32 / 511.0 * std::f32::consts::FRAC_1_SQRT_2;
                    q[k] = if (comp >> 9) & 1 == 1 { -mag } else { mag };
                    sum += q[k] * q[k];
                    comp >>= 10;
                }
                q[largest] = (1.0 - sum).max(0.0).sqrt();
                Quat::from_xyzw(q[0], q[1], q[2], q[3])
            };
            // Colour is the DC coefficient over a fixed range: f = (c − ½)/0.15.
            let rgb = [0, 1, 2].map(|k| {
                let f = (color[3 * i + k] as f32 / 255.0 - 0.5) / 0.15;
                ((0.5 + SH_C0 * f).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            });
            Some((p, covariance(s, q), [rgb[0], rgb[1], rgb[2], a]))
        })
        .collect();
    Ok(rows)
}

pub fn read_ply(path: &Path) -> Result<SplatFile, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 22, f);
    let elements = read_header(&mut r)?;
    if elements.iter().any(|e| e.name == "chunk") {
        return Err("compressed PLY (SuperSplat) is not supported yet — export an uncompressed PLY".into());
    }
    // Elements before the vertices are skipped; only fixed-size ones can be.
    for e in &elements {
        if e.name == "vertex" {
            break;
        }
        if e.has_list {
            return Err(format!("PLY element {} before the vertices has list properties", e.name));
        }
        std::io::copy(&mut (&mut r).take((e.count * e.stride) as u64), &mut std::io::sink()).map_err(|e| e.to_string())?;
    }
    let v = elements.iter().find(|e| e.name == "vertex").ok_or("PLY has no vertex element")?;
    if v.has_list {
        return Err("PLY vertices have list properties".into());
    }
    let need = |n: &str| v.prop(n).ok_or_else(|| format!("not a Gaussian-splat PLY: no {n} property"));
    let pos = [need("x")?, need("y")?, need("z")?];
    let scale = [need("scale_0")?, need("scale_1")?, need("scale_2")?];
    let rot = [need("rot_0")?, need("rot_1")?, need("rot_2")?, need("rot_3")?];
    let opacity = need("opacity")?;
    // DC colour; plain 8-bit colour as a fallback some exporters write.
    let dc = match (v.prop("f_dc_0"), v.prop("f_dc_1"), v.prop("f_dc_2")) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };
    let rgb8 = match (v.prop("red"), v.prop("green"), v.prop("blue")) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };

    let mut local = Vec::with_capacity(v.count);
    let mut cov = Vec::with_capacity(v.count);
    let mut rgba = Vec::with_capacity(v.count);
    let mut dropped = 0;
    const BATCH: usize = 1 << 16;
    let mut buf = vec![0u8; BATCH * v.stride];
    let mut left = v.count;
    while left > 0 {
        let n = left.min(BATCH);
        let bytes = &mut buf[..n * v.stride];
        r.read_exact(bytes).map_err(|e| format!("{}: {e} (file shorter than its header says)", path.display()))?;
        let rows: Vec<Option<SplatRow>> = bytes
            .par_chunks_exact(v.stride)
            .map(|rec| {
                let g = |(t, o): (PlyType, usize)| t.read(&rec[o..]);
                let alpha = 1.0 / (1.0 + (-g(opacity)).exp());
                // NaN opacity is dropped along with the transparent.
                if alpha.is_nan() || alpha < 1.0 / 255.0 {
                    return None;
                }
                let p = [g(pos[0]), g(pos[1]), g(pos[2])];
                if !p.iter().all(|c| c.is_finite()) {
                    return None;
                }
                let s = Vec3::new(g(scale[0]).exp(), g(scale[1]).exp(), g(scale[2]).exp());
                let q = Quat::from_xyzw(g(rot[1]), g(rot[2]), g(rot[3]), g(rot[0]));
                let rgb = match (dc, rgb8) {
                    (Some(dc), _) => [0, 1, 2].map(|k| ((0.5 + SH_C0 * g(dc[k])).clamp(0.0, 1.0) * 255.0 + 0.5) as u8),
                    (None, Some(c8)) => [0, 1, 2].map(|k| g(c8[k]).clamp(0.0, 255.0) as u8),
                    _ => [200, 200, 200],
                };
                Some((p, covariance(s, q), [rgb[0], rgb[1], rgb[2], (alpha * 255.0 + 0.5) as u8]))
            })
            .collect();
        for row in rows {
            match row {
                Some((p, c, col)) => {
                    local.push(p);
                    cov.push(c);
                    rgba.push(col);
                }
                None => dropped += 1,
            }
        }
        left -= n;
    }
    if local.is_empty() {
        return Err("the PLY has no visible splats".into());
    }
    Ok(SplatFile { path: path.to_path_buf(), local, cov, rgba, dropped, georef: Georef::find(path) })
}

impl SplatFile {
    /// World box of the bulk of the splats, 0.5th to 99.5th percentile per
    /// axis. Trainers leave a haze of floaters kilometres out; framing on the
    /// full extent would shrink the scene to a dot.
    pub fn core_bounds(&self) -> (DVec3, DVec3) {
        let mut lo = [0f32; 3];
        let mut hi = [0f32; 3];
        let n = self.local.len();
        let (a, b) = (n / 200, (n - 1) - n / 200);
        for k in 0..3 {
            let mut v: Vec<f32> = self.local.iter().map(|p| p[k]).collect();
            lo[k] = *v.select_nth_unstable_by(a, f32::total_cmp).1;
            hi[k] = *v.select_nth_unstable_by(b, f32::total_cmp).1;
        }
        (self.georef.to_world(lo), self.georef.to_world(hi))
    }
}

/// An RGBA32F texture, made by miniquad (which owns and frees it) and
/// allocated through raw GL, since miniquad offers only half floats.
/// Positions in half float would be centimetres off a few tens of metres out.
/// Its contents are undefined until [`upload_texels`] fills them.
fn float_texture(ctx: &mut dyn RenderingBackend, w: usize, h: usize) -> TextureId {
    let tex = ctx.new_texture(
        TextureAccess::Static,
        TextureSource::Empty,
        TextureParams {
            format: TextureFormat::RGBA16F,
            wrap: TextureWrap::Clamp,
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::None,
            width: w as u32,
            height: h as u32,
            ..Default::default()
        },
    );
    with_raw_texture(ctx, tex, |_| unsafe {
        use miniquad::gl::*;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F as GLint, w as GLint, h as GLint, 0, GL_RGBA, GL_FLOAT, std::ptr::null());
    });
    tex
}

/// Run raw GL with `tex` bound, then put the binding back exactly as found:
/// miniquad caches it and would otherwise skip a bind it believes is already
/// in place.
fn with_raw_texture(ctx: &mut dyn RenderingBackend, tex: TextureId, f: impl FnOnce(u32)) {
    const GL_TEXTURE_BINDING_2D: u32 = 0x8069;
    // Irrefutable where miniquad is built without Metal.
    #[allow(irrefutable_let_patterns)]
    if let RawId::OpenGl(id) = unsafe { ctx.texture_raw_id(tex) } {
        unsafe {
            use miniquad::gl::*;
            let mut prev: GLint = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &mut prev);
            glBindTexture(GL_TEXTURE_2D, id);
            f(id);
            glBindTexture(GL_TEXTURE_2D, prev as GLuint);
        }
    }
}

/// Write texels `start..start + data.len()` of a `TEX_W`-wide float texture:
/// a partial first row, whole rows, a partial last row.
fn upload_texels(ctx: &mut dyn RenderingBackend, tex: TextureId, start: usize, data: &[[f32; 4]]) {
    with_raw_texture(ctx, tex, |_| {
        let mut i = 0;
        while i < data.len() {
            let (row, col) = ((start + i) / TEX_W, (start + i) % TEX_W);
            let (w, h) = if col == 0 && data.len() - i >= TEX_W {
                (TEX_W, (data.len() - i) / TEX_W)
            } else {
                ((TEX_W - col).min(data.len() - i), 1)
            };
            unsafe {
                use miniquad::gl::*;
                glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    col as GLint,
                    row as GLint,
                    w as GLint,
                    h as GLint,
                    GL_RGBA,
                    GL_FLOAT,
                    data[i..].as_ptr() as *const _,
                );
            }
            i += w * h;
        }
    });
}

/// Back-to-front order along `front`, as the `float` indices the shader
/// reads. Keys pack the depth's sortable bits above the index, so one
/// parallel `u64` sort does it.
fn sort_order(centers: &[[f32; 3]], front: [f32; 3]) -> Vec<f32> {
    let f = Vec3::from(front);
    let mut keys: Vec<u64> = centers
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            // Farthest first: ascending in −depth.
            let d = -Vec3::from(*c).dot(f);
            let b = d.to_bits();
            let ord = if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 };
            ((ord as u64) << 32) | i as u64
        })
        .collect();
    keys.par_sort_unstable();
    keys.par_iter().map(|k| (*k & 0xFFFF_FFFF) as f32).collect()
}

/// [`sort_order`] over several tiles' splats at once, each at its own base
/// in the atlas.
fn sort_segments(segs: &[(Arc<SplatTexels>, usize)], front: [f32; 3]) -> Vec<f32> {
    let f = Vec3::from(front);
    let mut keys: Vec<u64> = Vec::with_capacity(segs.iter().map(|(t, _)| t.len()).sum());
    for (t, base) in segs {
        keys.par_extend(t.t0.par_iter().enumerate().map(|(i, c)| {
            let d = -Vec3::new(c[0], c[1], c[2]).dot(f);
            let b = d.to_bits();
            let ord = if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 };
            ((ord as u64) << 32) | (base + i) as u64
        }));
    }
    keys.par_sort_unstable();
    keys.par_iter().map(|k| (*k & 0xFFFF_FFFF) as f32).collect()
}

/// The most rows the atlas grows to: 4096 × 4096 splats is where a `float`
/// index stops being exact (2²⁴).
const ATLAS_MAX_ROWS: usize = 4096;

/// Free runs of atlas slots, `(start, len)`, sorted by start and merged.
#[derive(Default)]
struct Slots {
    free: Vec<(usize, usize)>,
}

impl Slots {
    /// First fit.
    fn alloc(&mut self, n: usize) -> Option<usize> {
        let k = self.free.iter().position(|&(_, len)| len >= n)?;
        let (start, len) = self.free[k];
        if len == n {
            self.free.remove(k);
        } else {
            self.free[k] = (start + n, len - n);
        }
        Some(start)
    }

    fn free(&mut self, start: usize, n: usize) {
        if n == 0 {
            return;
        }
        let k = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(k, (start, n));
        // Merge with the neighbours it touches.
        if k + 1 < self.free.len() && self.free[k].0 + self.free[k].1 == self.free[k + 1].0 {
            self.free[k].1 += self.free[k + 1].1;
            self.free.remove(k + 1);
        }
        if k > 0 && self.free[k - 1].0 + self.free[k - 1].1 == self.free[k].0 {
            self.free[k - 1].1 += self.free[k].1;
            self.free.remove(k);
        }
    }
}

struct SortReq {
    front: [f32; 3],
    /// Tile ids, sorted, that the order covers.
    sel: Vec<usize>,
    segs: Vec<(Arc<SplatTexels>, usize)>,
}

struct SortRes {
    order: Vec<f32>,
    front: [f32; 3],
    sel: Vec<usize>,
}

/// Splats streamed from tiles, in one set of textures with one draw order.
///
/// Every resident tile owns a run of texel slots; a sort over the tiles on
/// screen produces one order for one instanced draw, so splats blend in the
/// right order across tile borders. The textures grow by doubling (re-filled
/// from the tiles' CPU copies) up to [`ATLAS_MAX_ROWS`].
///
/// The order on screen refers to slots, so a tile it covers must keep its
/// slot until a newer order no longer does: [`SplatAtlas::protects`] tells
/// eviction which tiles those are.
pub struct SplatAtlas {
    textures: Option<[TextureId; 3]>,
    rows: usize,
    slots: Slots,
    quad: BufferId,
    quad_idx: BufferId,
    order: Option<(BufferId, usize)>,
    pipeline: Pipeline,
    sort_tx: Option<mpsc::Sender<SortReq>>,
    sort_rx: mpsc::Receiver<SortRes>,
    /// What the order on screen covers, and how many splats it draws.
    shown_sel: Vec<usize>,
    shown_count: usize,
    shown_front: [f32; 3],
    pending_sel: Option<Vec<usize>>,
}

impl SplatAtlas {
    pub fn new(ctx: &mut dyn RenderingBackend) -> SplatAtlas {
        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let quad = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&corners));
        let quad_idx = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&[0u16, 1, 2, 0, 2, 3]));
        let (sort_tx, req_rx) = mpsc::channel::<SortReq>();
        let (res_tx, sort_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("tile-splat-sort".into())
            .spawn(move || {
                while let Ok(mut r) = req_rx.recv() {
                    while let Ok(newer) = req_rx.try_recv() {
                        r = newer;
                    }
                    let order = sort_segments(&r.segs, r.front);
                    if res_tx.send(SortRes { order, front: r.front, sel: r.sel }).is_err() {
                        break;
                    }
                }
            })
            .expect("cannot spawn the splat sort thread");
        SplatAtlas {
            textures: None,
            rows: 0,
            slots: Slots::default(),
            quad,
            quad_idx,
            order: None,
            pipeline: splat_pipeline(ctx),
            sort_tx: Some(sort_tx),
            sort_rx,
            shown_sel: Vec::new(),
            shown_count: 0,
            shown_front: [0.0; 3],
            pending_sel: None,
        }
    }

    /// Splats drawn by the order on screen.
    pub fn shown(&self) -> usize {
        self.shown_count
    }

    /// `n` slots, first fit. `None` when no run is long enough: grow, or
    /// evict.
    pub fn alloc(&mut self, n: usize) -> Option<usize> {
        self.slots.alloc(n)
    }

    pub fn free(&mut self, start: usize, n: usize) {
        self.slots.free(start, n);
    }

    /// Double the textures (at least enough for `n` more splats), refilling
    /// what is resident at the same slots. False at the size limit.
    pub fn grow(&mut self, ctx: &mut dyn RenderingBackend, n: usize, resident: &[(usize, Arc<SplatTexels>)]) -> bool {
        let old_cap = self.rows * TEX_W;
        let mut rows = (self.rows * 2).max(64);
        while rows * TEX_W < old_cap + n && rows < ATLAS_MAX_ROWS {
            rows *= 2;
        }
        let rows = rows.min(ATLAS_MAX_ROWS);
        if rows <= self.rows {
            return false;
        }
        let textures = [0, 1, 2].map(|_| float_texture(ctx, TEX_W, rows));
        for (start, t) in resident {
            for (k, data) in [&t.t0, &t.t1, &t.t2].into_iter().enumerate() {
                upload_texels(ctx, textures[k], *start, data);
            }
        }
        if let Some(old) = self.textures.replace(textures) {
            for t in old {
                ctx.delete_texture(t);
            }
        }
        self.rows = rows;
        self.free(old_cap, rows * TEX_W - old_cap);
        true
    }

    pub fn upload(&self, ctx: &mut dyn RenderingBackend, start: usize, t: &SplatTexels) {
        if let Some(tex) = self.textures {
            for (k, data) in [&t.t0, &t.t1, &t.t2].into_iter().enumerate() {
                upload_texels(ctx, tex[k], start, data);
            }
        }
    }

    /// Whether the order on screen, or the one being sorted, covers tile `i`.
    pub fn protects(&self, i: usize) -> bool {
        self.shown_sel.binary_search(&i).is_ok() || self.pending_sel.as_ref().is_some_and(|p| p.binary_search(&i).is_ok())
    }

    /// Take a finished sort; ask for another when the tiles on screen have
    /// changed or the view has turned. `sel` is `(tile, slot, splats)`.
    pub fn update(&mut self, ctx: &mut dyn RenderingBackend, front: Vec3, sel: &[(usize, usize, Arc<SplatTexels>)]) {
        let mut latest = None;
        while let Ok(r) = self.sort_rx.try_recv() {
            latest = Some(r);
        }
        if let Some(r) = latest {
            let n = r.order.len();
            if self.order.is_none_or(|(_, cap)| cap < n) {
                if let Some((b, _)) = self.order.take() {
                    ctx.delete_buffer(b);
                }
                let cap = n.next_power_of_two().max(1024);
                self.order = Some((ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Stream, BufferSource::empty::<f32>(cap)), cap));
            }
            if n > 0
                && let Some((b, _)) = self.order
            {
                ctx.buffer_update(b, BufferSource::slice(&r.order));
            }
            self.shown_count = n;
            self.shown_sel = r.sel;
            self.shown_front = r.front;
            self.pending_sel = None;
        }
        if self.pending_sel.is_some() {
            return;
        }
        let mut ids: Vec<usize> = sel.iter().map(|s| s.0).collect();
        ids.sort_unstable();
        // About a tenth of a degree.
        let turned = Vec3::from(self.shown_front).dot(front) < 0.999_998;
        if (ids != self.shown_sel || turned)
            && let Some(tx) = &self.sort_tx
        {
            let segs = sel.iter().map(|(_, start, t)| (t.clone(), *start)).collect();
            if tx.send(SortReq { front: front.to_array(), sel: ids.clone(), segs }).is_ok() {
                self.pending_sel = Some(ids);
            }
        }
    }

    pub fn draw(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, viewport: (f32, f32), z_clip: ZClip) {
        let (Some(textures), Some((order, _))) = (self.textures, self.order) else { return };
        if self.shown_count == 0 {
            return;
        }
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&Bindings { vertex_buffers: vec![self.quad, order], index_buffer: self.quad_idx, images: textures.to_vec() });
        ctx.apply_uniforms(UniformsSource::table(&splat_shader::Uniforms {
            mvp: *view_proj,
            viewport: [viewport.0, viewport.1],
            tex_size: [TEX_W as f32, self.rows as f32],
            z_clip_min: z_clip.min,
            z_clip_max: z_clip.max,
            crop_lo: [-1.0e9; 3],
            crop_hi: [1.0e9; 3],
        }));
        // As in `SplatCloud::draw`.
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_FALSE as _) };
        ctx.draw(0, 6, self.shown_count as i32);
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_TRUE as _) };
    }

    pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
        self.sort_tx = None;
        if let Some(t) = self.textures {
            for t in t {
                ctx.delete_texture(t);
            }
        }
        if let Some((b, _)) = self.order {
            ctx.delete_buffer(b);
        }
        ctx.delete_buffer(self.quad);
        ctx.delete_buffer(self.quad_idx);
        ctx.delete_pipeline(self.pipeline);
    }
}

pub struct SplatCloud {
    pub path: PathBuf,
    pub visible: bool,
    pub count: usize,
    pub dropped: usize,
    pub georef: Georef,
    /// Hide splats centred outside the scene's box; see `crop_box`.
    pub crop: bool,
    world_lo: DVec3,
    world_hi: DVec3,
    /// Viewer-space box splats must be centred in while `crop` is on.
    crop_lo: Vec3,
    crop_hi: Vec3,
    textures: [TextureId; 3],
    tex_h: usize,
    quad: BufferId,
    quad_idx: BufferId,
    order: BufferId,
    pipeline: Pipeline,
    sort_tx: Option<mpsc::Sender<[f32; 3]>>,
    sort_rx: mpsc::Receiver<(Vec<f32>, [f32; 3])>,
    sorted_for: [f32; 3],
    sort_pending: bool,
}

impl SplatCloud {
    pub fn attach(ctx: &mut dyn RenderingBackend, file: SplatFile, origin: [f64; 3], front: Vec3) -> SplatCloud {
        let (world_lo, world_hi) = file.core_bounds();
        let n = file.local.len();
        let o = DVec3::from(origin);
        // Trainers surround the scene with large, faint background splats.
        // A perspective viewer sits inside that shell and sees it as sky;
        // under an orthographic camera every one of them along the line of
        // sight lands in front of the scene. The crop box is the core box
        // grown by a tenth of its horizontal size on every side — upwards
        // too, so roofs and trees above the 99.5th height percentile stay.
        let ext = world_hi - world_lo;
        let margin = DVec3::splat(0.1 * ext.x.max(ext.y));
        let (crop_lo, crop_hi) = ((world_lo - margin - o).as_vec3(), (world_hi + margin - o).as_vec3());
        let g = &file.georef;
        let centers: Vec<[f32; 3]> = file
            .local
            .par_iter()
            .map(|p| {
                let w = g.to_world(*p) - o;
                [w.x as f32, w.y as f32, w.z as f32]
            })
            .collect();

        let tex_h = n.div_ceil(TEX_W);
        let mut texels = SplatTexels::default();
        // The horizontal scale shrinks x and y on the way to world; the
        // covariance follows as D Σ Dᵀ.
        let s = g.horizontal_scale as f32;
        let (sxx, sxz) = (1.0 / (s * s), 1.0 / s);
        for ((c, v), col) in centers.iter().zip(&file.cov).zip(&file.rgba) {
            texels.push(*c, [v[0] * sxx, v[1] * sxx, v[2] * sxz, v[3] * sxx, v[4] * sxz, v[5]], *col);
        }
        let textures = [&texels.t0, &texels.t1, &texels.t2].map(|t| {
            let tex = float_texture(ctx, TEX_W, tex_h);
            upload_texels(ctx, tex, 0, t);
            tex
        });

        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let quad = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&corners));
        let quad_idx = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&[0u16, 1, 2, 0, 2, 3]));
        let first = sort_order(&centers, front.to_array());
        let order = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Dynamic, BufferSource::slice(&first));

        let centers = Arc::new(centers);
        let (sort_tx, req_rx) = mpsc::channel::<[f32; 3]>();
        let (res_tx, sort_rx) = mpsc::channel();
        {
            let centers = centers.clone();
            std::thread::Builder::new()
                .name("splat-sort".into())
                .spawn(move || {
                    while let Ok(mut f) = req_rx.recv() {
                        // Only the latest direction matters.
                        while let Ok(newer) = req_rx.try_recv() {
                            f = newer;
                        }
                        if res_tx.send((sort_order(&centers, f), f)).is_err() {
                            break;
                        }
                    }
                })
                .expect("cannot spawn the splat sort thread");
        }

        ::log::info!(
            "{} splats from {} ({} transparent ones skipped){}",
            n,
            file.path.display(),
            file.dropped,
            match &g.source {
                Some(p) => format!(", placed by {}", p.display()),
                None => String::new(),
            }
        );

        SplatCloud {
            path: file.path,
            visible: true,
            count: n,
            dropped: file.dropped,
            georef: file.georef,
            crop: true,
            world_lo,
            world_hi,
            crop_lo,
            crop_hi,
            textures,
            tex_h,
            quad,
            quad_idx,
            order,
            pipeline: splat_pipeline(ctx),
            sort_tx: Some(sort_tx),
            sort_rx,
            sorted_for: front.to_array(),
            sort_pending: false,
        }
    }

    pub fn name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    }

    /// World box of the bulk of the splats; see [`SplatFile::core_bounds`].
    pub fn world_bounds(&self) -> (DVec3, DVec3) {
        (self.world_lo, self.world_hi)
    }

    /// Take a finished sort, and ask for another if the view has turned.
    pub fn update(&mut self, ctx: &mut dyn RenderingBackend, front: Vec3) {
        let mut latest = None;
        while let Ok(r) = self.sort_rx.try_recv() {
            latest = Some(r);
        }
        if let Some((order, f)) = latest {
            ctx.buffer_update(self.order, BufferSource::slice(&order));
            self.sorted_for = f;
            self.sort_pending = false;
        }
        if !self.visible || self.sort_pending {
            return;
        }
        // About a tenth of a degree.
        if Vec3::from(self.sorted_for).dot(front) < 0.999_998
            && let Some(tx) = &self.sort_tx
            && tx.send(front.to_array()).is_ok()
        {
            self.sort_pending = true;
        }
    }

    /// `viewport` is the view's size in physical pixels.
    pub fn draw(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, viewport: (f32, f32), z_clip: ZClip) {
        if !self.visible {
            return;
        }
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&Bindings {
            vertex_buffers: vec![self.quad, self.order],
            index_buffer: self.quad_idx,
            images: self.textures.to_vec(),
        });
        ctx.apply_uniforms(UniformsSource::table(&splat_shader::Uniforms {
            mvp: *view_proj,
            viewport: [viewport.0, viewport.1],
            tex_size: [TEX_W as f32, self.tex_h as f32],
            z_clip_min: z_clip.min,
            z_clip_max: z_clip.max,
            crop_lo: if self.crop { self.crop_lo.to_array() } else { [-1.0e9; 3] },
            crop_hi: if self.crop { self.crop_hi.to_array() } else { [1.0e9; 3] },
        }));
        // Test depth, do not write it; see `splat_pipeline`. Put back
        // straight after: with the mask off, the next frame's depth clear
        // would silently do nothing.
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_FALSE as _) };
        ctx.draw(0, 6, self.count as i32);
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_TRUE as _) };
    }

    pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
        self.sort_tx = None;
        for t in self.textures {
            ctx.delete_texture(t);
        }
        ctx.delete_buffer(self.quad);
        ctx.delete_buffer(self.quad_idx);
        ctx.delete_buffer(self.order);
        ctx.delete_pipeline(self.pipeline);
    }
}

fn splat_pipeline(ctx: &mut dyn RenderingBackend) -> Pipeline {
    let shader = ctx
        .new_shader(ShaderSource::Glsl { vertex: splat_shader::VERTEX, fragment: splat_shader::FRAGMENT }, splat_shader::meta())
        .expect("splat shader");
    ctx.new_pipeline(
        &[BufferLayout::default(), BufferLayout { step_func: VertexStep::PerInstance, ..Default::default() }],
        &[
            VertexAttribute::with_buffer("corner", VertexFormat::Float2, 0),
            VertexAttribute::with_buffer("splat_idx", VertexFormat::Float1, 1),
        ],
        shader,
        PipelineParams {
            primitive_type: PrimitiveType::Triangles,
            // Hidden behind points and meshes, but never hiding each other:
            // the sort and the blend do that. So depth is tested and not
            // written — but miniquad turns the depth *test* off along with
            // the write for a pipeline that does not write, so this one says
            // it writes and `draw` masks the writes off around the draw call.
            depth_test: Comparison::LessOrEqual,
            depth_write: true,
            cull_face: CullFace::Nothing,
            color_blend: Some(BlendState::new(
                Equation::Add,
                BlendFactor::Value(BlendValue::SourceAlpha),
                BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
            )),
            // Leave the framebuffer's alpha alone.
            alpha_blend: Some(BlendState::new(Equation::Add, BlendFactor::Zero, BlendFactor::One)),
            ..Default::default()
        },
    )
}

mod splat_shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
        attribute vec2 corner;
        attribute float splat_idx;

        uniform sampler2D tex_center;
        uniform sampler2D tex_cov_a;
        uniform sampler2D tex_cov_b;
        uniform mat4 mvp;
        uniform vec2 viewport;
        uniform vec2 tex_size;
        uniform float z_clip_min;
        uniform float z_clip_max;
        uniform vec3 crop_lo;
        uniform vec3 crop_hi;

        varying mediump vec4 v_color;
        varying mediump vec2 v_pos;

        void main() {
            v_color = vec4(0.0);
            v_pos = vec2(0.0);
            highp float row = floor((splat_idx + 0.5) / tex_size.x);
            highp float col = splat_idx - row * tex_size.x;
            highp vec2 uv = vec2((col + 0.5) / tex_size.x, (row + 0.5) / tex_size.y);
            highp vec4 t0 = texture2D(tex_center, uv);
            if (t0.z < z_clip_min || t0.z > z_clip_max
                || any(lessThan(t0.xyz, crop_lo)) || any(greaterThan(t0.xyz, crop_hi))) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }
            highp vec4 t1 = texture2D(tex_cov_a, uv);
            highp vec4 t2 = texture2D(tex_cov_b, uv);
            highp vec4 clip = mvp * vec4(t0.xyz, 1.0);

            // Orthographic, so the screen covariance is exactly J S J^T, J
            // the first two rows of the camera's linear part, in pixels.
            highp vec3 jx = vec3(mvp[0][0], mvp[1][0], mvp[2][0]) * (0.5 * viewport.x);
            highp vec3 jy = vec3(mvp[0][1], mvp[1][1], mvp[2][1]) * (0.5 * viewport.y);
            highp mat3 S = mat3(t1.x, t1.y, t1.z,
                                t1.y, t2.x, t2.y,
                                t1.z, t2.y, t2.z);
            highp vec3 sy = S * jy;
            // 0.3 px^2: the low-pass every 3DGS renderer adds, so a splat
            // never falls between pixels.
            highp float a = dot(jx, S * jx) + 0.3;
            highp float b = dot(jx, sy);
            highp float d = dot(jy, sy) + 0.3;
            highp float mid = 0.5 * (a + d);
            highp float rad = length(vec2(0.5 * (a - d), b));
            highp float l1 = mid + rad;
            highp float l2 = max(mid - rad, 0.1);
            // Eigenvector of the larger eigenvalue, from whichever of the two
            // textbook forms is better conditioned.
            highp vec2 va = vec2(b, l1 - a);
            highp vec2 vb = vec2(l1 - d, b);
            highp vec2 e1 = dot(va, va) > dot(vb, vb) ? va : vb;
            e1 = dot(e1, e1) > 1e-20 ? normalize(e1) : vec2(1.0, 0.0);
            highp vec2 e2 = vec2(-e1.y, e1.x);
            // Three sigma, capped so a splat filling the screen when zoomed
            // right in cannot cost the frame.
            highp float r1 = min(3.0 * sqrt(l1), 2048.0);
            highp float r2 = min(3.0 * sqrt(l2), 2048.0);
            if (abs(clip.x) > 1.0 + 2.0 * r1 / viewport.x || abs(clip.y) > 1.0 + 2.0 * r1 / viewport.y) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }
            highp vec2 off = (corner.x * r1) * e1 + (corner.y * r2) * e2;
            gl_Position = vec4(clip.xy + off * 2.0 / viewport, clip.z, 1.0);
            v_pos = corner * 3.0;

            // Colour is packed as r * 65536 + g * 256 + b: exact in a float.
            highp float p = t0.w;
            highp float cr = floor(p / 65536.0);
            highp float cg = floor((p - cr * 65536.0) / 256.0);
            highp float cb = p - cr * 65536.0 - cg * 256.0;
            v_color = vec4(cr / 255.0, cg / 255.0, cb / 255.0, t1.w);
        }"#;

    pub const FRAGMENT: &str = r#"#version 100
        precision mediump float;
        varying mediump vec4 v_color;
        varying mediump vec2 v_pos;

        void main() {
            float r2 = dot(v_pos, v_pos);
            if (r2 > 9.0) discard;
            float a = v_color.a * exp(-0.5 * r2);
            if (a < 1.0 / 255.0) discard;
            gl_FragColor = vec4(v_color.rgb, a);
        }"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["tex_center".to_string(), "tex_cov_a".to_string(), "tex_cov_b".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("mvp", UniformType::Mat4),
                    UniformDesc::new("viewport", UniformType::Float2),
                    UniformDesc::new("tex_size", UniformType::Float2),
                    UniformDesc::new("z_clip_min", UniformType::Float1),
                    UniformDesc::new("z_clip_max", UniformType::Float1),
                    UniformDesc::new("crop_lo", UniformType::Float3),
                    UniformDesc::new("crop_hi", UniformType::Float3),
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
        pub viewport: [f32; 2],
        pub tex_size: [f32; 2],
        pub z_clip_min: f32,
        pub z_clip_max: f32,
        pub crop_lo: [f32; 3],
        pub crop_hi: [f32; 3],
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn write_ply(path: &Path, rows: &[[f32; 14]]) {
        let names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"];
        let mut h = format!("ply\nformat binary_little_endian 1.0\ncomment test\nelement vertex {}\n", rows.len());
        for n in names {
            h += &format!("property float {n}\n");
        }
        h += "end_header\n";
        let mut b = h.into_bytes();
        for r in rows {
            for v in r {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, b).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("haiviz-splats-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn splat_attributes_are_activated() {
        let d = scratch("attrs");
        let p = d.join("s.ply");
        // Scale e^0 = 1 along x, e^-2 elsewhere, rotated 90° about z: the long
        // axis ends up along y. Opacity logit 0 is alpha one half.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        write_ply(&p, &[
            [1.0, 2.0, 3.0, 0.0, -2.0, -2.0, 0.0, h, 0.0, 0.0, h, 0.0, 1.0, -10.0],
            // Fully transparent: dropped.
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        assert!(is_splat_ply(&p));
        let f = read_ply(&p).unwrap();
        assert_eq!(f.local, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(f.dropped, 1);
        let c = f.cov[0];
        assert!((c[3] - 1.0).abs() < 1e-5, "yy {c:?}");
        assert!((c[0] - (-4.0f32).exp()).abs() < 1e-5, "xx {c:?}");
        assert!(c[1].abs() < 1e-5);
        assert_eq!(f.rgba[0][3], 128);
        assert_eq!(f.rgba[0][0], 128); // 0.5 + C0·0
        assert_eq!(f.rgba[0][2], 0); // clamped
        assert_eq!(f.georef, Georef::IDENTITY);
    }

    #[test]
    fn scene_json_above_the_ply_places_it_in_rd() {
        let d = scratch("georef");
        let run = d.join("runs/brush");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(d.join("scene.json"), r#"{"origin_rd":[149730.0,464777.0,0.0],"horizontal_scale":2.0}"#).unwrap();
        let g = Georef::find(&run.join("export.ply"));
        assert_eq!(g.offset, [149730.0, 464777.0, 0.0]);
        assert_eq!(g.to_world([10.0, -4.0, 12.0]), DVec3::new(149735.0, 464775.0, 12.0));
    }

    /// Against a real PLY: `HAIVIZ_SPLATS=/path/export.ply cargo test
    /// --release real_splats -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_splats_read_and_sort() {
        let Ok(p) = std::env::var("HAIVIZ_SPLATS") else { return };
        let t = std::time::Instant::now();
        let f = read_ply(Path::new(&p)).unwrap();
        println!("{} splats ({} dropped) read in {:?}, georef {:?}", f.local.len(), f.dropped, t.elapsed(), f.georef);
        let t = std::time::Instant::now();
        let (lo, hi) = f.core_bounds();
        println!("core bounds {lo} .. {hi} in {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let order = sort_order(&f.local, [0.3, 0.2, -0.93]);
        println!("sorted in {:?}", t.elapsed());
        assert_eq!(order.len(), f.local.len());
    }

    #[test]
    fn sort_is_back_to_front() {
        let c = [[0.0, 0.0, 5.0], [0.0, 0.0, -3.0], [0.0, 0.0, 1.0]];
        // Looking down: the lowest splat is farthest and drawn first.
        assert_eq!(sort_order(&c, [0.0, 0.0, -1.0]), vec![1.0, 2.0, 0.0]);
        assert_eq!(sort_order(&c, [0.0, 0.0, 1.0]), vec![0.0, 2.0, 1.0]);
    }

    /// An SPZ stream, gzipped, for splats given as (fixed-point position,
    /// alpha, colour, log-scale and rotation bytes).
    pub(crate) fn spz(version: u32, rows: &[([i32; 3], u8, [u8; 3], [u8; 3], Vec<u8>)]) -> Vec<u8> {
        use std::io::Write;
        let n = rows.len();
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x5053_474e_u32.to_le_bytes());
        raw.extend_from_slice(&version.to_le_bytes());
        raw.extend_from_slice(&(n as u32).to_le_bytes());
        raw.extend_from_slice(&[0, 12, 0, 0]);
        for r in rows {
            for v in r.0 {
                raw.extend_from_slice(&v.to_le_bytes()[..3]);
            }
        }
        rows.iter().for_each(|r| raw.push(r.1));
        rows.iter().for_each(|r| raw.extend_from_slice(&r.2));
        rows.iter().for_each(|r| raw.extend_from_slice(&r.3));
        rows.iter().for_each(|r| raw.extend_from_slice(&r.4));
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&raw).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn spz_v3_positions_colours_scales_and_rotations() {
        // Scale byte 160 is log-scale 0 (1 m), 144 is −1. The rotation word
        // names w largest with x, y, z zero: identity. 224 is colour 1.0.
        let ident = (3u32 << 30).to_le_bytes().to_vec();
        // 90° about z: w = z = 1/√2, w largest, z at full 9-bit magnitude.
        let quarter = ((3u32 << 30) | 511).to_le_bytes().to_vec();
        let rows = vec![
            ([4096, -8192, 2048], 255, [224, 128, 0], [160, 144, 144], ident.clone()),
            ([0, 0, 0], 128, [128, 128, 128], [160, 144, 144], quarter),
            ([0, 0, 0], 0, [128, 128, 128], [160, 160, 160], ident),
        ];
        let out = decode_spz(&spz(3, &rows)).unwrap();
        assert_eq!(out.len(), 2, "the transparent one is dropped");
        let (p, c, col) = out[0];
        assert_eq!(p, [1.0, -2.0, 0.5]);
        assert!((c[0] - 1.0).abs() < 1e-5 && (c[3] - (-2.0f32).exp()).abs() < 1e-5 && c[1].abs() < 1e-6);
        assert_eq!(col[3], 255);
        assert!(col[0] > 250 && (126..=129).contains(&col[1]) && col[2] < 15, "{col:?}");
        // Turned a quarter about z, x and y trade variances.
        let (_, c, _) = out[1];
        assert!((c[0] - (-2.0f32).exp()).abs() < 1e-3 && (c[3] - 1.0).abs() < 1e-3, "{c:?}");
    }

    #[test]
    fn spz_v2_rotations_are_xyz_with_w_derived() {
        let rows = vec![([0, 0, 0], 255, [128; 3], [160, 144, 144], vec![128, 128, 128])];
        let out = decode_spz(&spz(2, &rows)).unwrap();
        let c = out[0].1;
        assert!((c[0] - 1.0).abs() < 1e-3 && (c[3] - (-2.0f32).exp()).abs() < 1e-3, "{c:?}");
        let e = decode_spz(&spz(4, &rows)).err().unwrap();
        assert!(e.contains("version 4"), "{e}");
    }

    #[test]
    fn atlas_slots_are_reused_and_merged() {
        let mut s = Slots::default();
        s.free(0, 100);
        assert_eq!(s.alloc(30), Some(0));
        assert_eq!(s.alloc(30), Some(30));
        assert_eq!(s.alloc(50), None);
        s.free(0, 30);
        s.free(30, 30);
        assert_eq!(s.free, vec![(0, 100)]);
        assert_eq!(s.alloc(100), Some(0));
    }

    #[test]
    fn plain_point_cloud_ply_is_not_a_splat() {
        let d = scratch("plain");
        let p = d.join("pc.ply");
        std::fs::write(&p, b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n").unwrap();
        assert!(!is_splat_ply(&p));
    }
}
