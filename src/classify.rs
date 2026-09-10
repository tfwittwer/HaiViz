//! Manual classification edits.
//!
//! Edits are stored as an ordered list of *region operations* rather than as
//! per-point assignments. Each op is a prism — a polygon drawn in section space
//! (metres along the section line × absolute Z), extruded across the corridor
//! depth — plus the classification to assign inside it.
//!
//! Storing the operation instead of the affected points is what makes this work
//! against a streamed COPC: the viewer only ever holds a subset of the file's
//! points, so a list of point ids gathered from the screen would silently miss
//! everything that was not resident when the lasso was drawn. An op is applied
//! wherever points are seen — in `CopcStage::ingest_node` as nodes stream in,
//! and again over every hierarchy entry at export — so coverage does not depend
//! on what the LOD happened to load.
//!
//! Ops are stored in absolute world coordinates, independent of the viewer
//! origin and of the file's chunk layout, so a sidecar stays valid even if the
//! source is recompressed or re-tiled.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ::log::{info, warn};
use las::Point;

/// The classification id a point currently carries.
///
/// LAS 1.4 moved "overlap" out of the classification field into its own flag,
/// and the `las` crate follows that, so class 12 has to be recovered from
/// `is_overlap`. Mirrors the display logic in `class_label`.
pub fn effective_class(p: &Point) -> u8 {
    if p.is_overlap { 12 } else { p.classification.into() }
}

/// Assign a classification id, handling the overlap flag as `effective_class` reads it.
pub fn set_class(p: &mut Point, id: u8) {
    if id == 12 {
        p.is_overlap = true;
    } else {
        p.is_overlap = false;
        if let Ok(c) = las::point::Classification::new(id) {
            p.classification = c;
        }
    }
}

/// The shape an op selects. Both variants are pure geometric predicates,
/// which is what lets the same op be replayed over a COPC node whenever it
/// streams in and again over every node at export.
#[derive(Clone, Debug)]
pub enum ClassRegion {
    /// A corridor prism: a polygon drawn in section space (metres along the
    /// section line × absolute Z), extruded across the corridor depth.
    Corridor {
        /// Section line start in absolute world XY.
        start: (f64, f64),
        /// Section line end in absolute world XY.
        end: (f64, f64),
        /// Corridor half-width in metres, measured perpendicular to the line.
        depth: f64,
        /// Lasso polygon in section space: `(metres along the line, absolute Z)`.
        polygon: Vec<(f64, f64)>,
    },
    /// Everything standing above the local ground inside a plan-view footprint.
    ///
    /// The ground is a plane fitted once, when the op is made, and then frozen
    /// into the op — so it stays a predicate rather than something that has to
    /// be re-derived from whatever points happen to be loaded later.
    AboveGround {
        /// Lasso footprint in absolute world XY.
        polygon: Vec<(f64, f64)>,
        /// Local origin the plane is expressed against. Fitting against raw
        /// survey coordinates (hundreds of thousands of metres) loses most of
        /// the available precision in the normal equations.
        origin: (f64, f64),
        /// Ground plane as `z = a*(x-ox) + b*(y-oy) + c`.
        plane: (f64, f64, f64),
        /// How far above the plane a point must be to be selected.
        offset: f64,
        /// Height band in force when the op was made, in absolute Z. Recorded
        /// so the op means the same thing at export as it did on screen.
        z_min: f64,
        z_max: f64,
    },
    /// Everything inside a plan-view footprint, between two absolute heights.
    ///
    /// The ground-free counterpart to `AboveGround`: nothing is fitted and
    /// nothing is excluded from below, so the height band is the only vertical
    /// limit. Use it where the ground is not a meaningful reference — indoors,
    /// on a bridge deck, or wherever the thing to be reclassified sits on top
    /// of something that must not be swept up with it.
    Volume {
        /// Lasso footprint in absolute world XY.
        polygon: Vec<(f64, f64)>,
        /// Height band in force when the op was made, in absolute Z.
        z_min: f64,
        z_max: f64,
    },
}

/// One manual classification operation.
#[derive(Clone, Debug)]
pub struct ClassOp {
    /// Monotonic id, used for ordering and for the undo stack.
    pub seq: u64,
    /// What the op selects.
    pub region: ClassRegion,
    /// Classification to assign inside the region.
    pub target_class: u8,
    /// When set, only points that currently carry this class are changed.
    pub from_class: Option<u8>,
    /// Cached world-space AABB, for cheap node rejection.
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
}

impl ClassOp {
    pub fn new(
        seq: u64,
        region: ClassRegion,
        target_class: u8,
        from_class: Option<u8>,
    ) -> Self {
        let mut op = ClassOp {
            seq, region, target_class, from_class,
            bbox_min: [f64::MAX; 3],
            bbox_max: [f64::MIN; 3],
        };
        op.recompute_bbox();
        op
    }

    /// Convenience constructor for a section-view corridor op.
    #[allow(clippy::too_many_arguments)]
    pub fn corridor(
        seq: u64,
        start: (f64, f64),
        end: (f64, f64),
        depth: f64,
        polygon: Vec<(f64, f64)>,
        target_class: u8,
        from_class: Option<u8>,
    ) -> Self {
        ClassOp::new(
            seq,
            ClassRegion::Corridor { start, end, depth, polygon },
            target_class,
            from_class,
        )
    }

    /// Unit direction along a section line, and its left normal.
    fn corridor_axes(start: (f64, f64), end: (f64, f64)) -> Option<((f64, f64), (f64, f64))> {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 { return None; }
        let dir = (dx / len, dy / len);
        Some((dir, (-dir.1, dir.0)))
    }

    fn recompute_bbox(&mut self) {
        match &self.region {
            ClassRegion::Corridor { start, end, depth, polygon } => {
                let Some((dir, normal)) = Self::corridor_axes(*start, *end) else { return };
                if polygon.is_empty() { return; }
                let (mut a0, mut a1) = (f64::MAX, f64::MIN);
                let (mut z0, mut z1) = (f64::MAX, f64::MIN);
                for &(a, z) in polygon {
                    a0 = a0.min(a); a1 = a1.max(a);
                    z0 = z0.min(z); z1 = z1.max(z);
                }
                let (mut xmin, mut ymin) = (f64::MAX, f64::MAX);
                let (mut xmax, mut ymax) = (f64::MIN, f64::MIN);
                for &a in &[a0, a1] {
                    for &p in &[-*depth, *depth] {
                        let x = start.0 + dir.0 * a + normal.0 * p;
                        let y = start.1 + dir.1 * a + normal.1 * p;
                        xmin = xmin.min(x); xmax = xmax.max(x);
                        ymin = ymin.min(y); ymax = ymax.max(y);
                    }
                }
                self.bbox_min = [xmin, ymin, z0];
                self.bbox_max = [xmax, ymax, z1];
            }
            ClassRegion::AboveGround { polygon, z_min, z_max, .. }
            | ClassRegion::Volume { polygon, z_min, z_max } => {
                if polygon.is_empty() { return; }
                let (mut xmin, mut ymin) = (f64::MAX, f64::MAX);
                let (mut xmax, mut ymax) = (f64::MIN, f64::MIN);
                for &(x, y) in polygon {
                    xmin = xmin.min(x); xmax = xmax.max(x);
                    ymin = ymin.min(y); ymax = ymax.max(y);
                }
                self.bbox_min = [xmin, ymin, *z_min];
                self.bbox_max = [xmax, ymax, *z_max];
            }
        }
    }

    /// True if this op could touch the given world-space AABB.
    /// Conservative — a cheap reject before the per-point test.
    pub fn intersects_aabb(&self, min: &[f64; 3], max: &[f64; 3]) -> bool {
        (0..3).all(|i| self.bbox_min[i] <= max[i] && self.bbox_max[i] >= min[i])
    }

    /// True if the absolute world point falls inside the region.
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        match &self.region {
            ClassRegion::Corridor { start, end, depth, polygon } => {
                let Some((dir, normal)) = Self::corridor_axes(*start, *end) else { return false };
                let rx = x - start.0;
                let ry = y - start.1;
                let perp = rx * normal.0 + ry * normal.1;
                if perp.abs() > *depth { return false; }
                let along = rx * dir.0 + ry * dir.1;
                point_in_polygon(polygon, along, z)
            }
            ClassRegion::AboveGround { polygon, origin, plane, offset, z_min, z_max } => {
                if z < *z_min || z > *z_max { return false; }
                let ground = plane.0 * (x - origin.0) + plane.1 * (y - origin.1) + plane.2;
                if z < ground + *offset { return false; }
                point_in_polygon(polygon, x, y)
            }
            ClassRegion::Volume { polygon, z_min, z_max } => {
                if z < *z_min || z > *z_max { return false; }
                point_in_polygon(polygon, x, y)
            }
        }
    }

    /// Apply to a single point. Returns true if the classification changed.
    pub fn apply_point(&self, p: &mut Point) -> bool {
        if let Some(from) = self.from_class && effective_class(p) != from {
            return false;
        }
        if effective_class(p) == self.target_class { return false; }
        if !self.contains(p.x, p.y, p.z) { return false; }
        set_class(p, self.target_class);
        true
    }
}

/// Ray-casting point-in-polygon over `(along, z)` section coordinates.
fn point_in_polygon(poly: &[(f64, f64)], px: f64, py: f64) -> bool {
    if poly.len() < 3 { return false; }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The full set of manual edits for one data source.
#[derive(Clone, Default, Debug)]
pub struct ClassEdits {
    pub ops: Vec<ClassOp>,
    next_seq: u64,
    /// Set when `ops` has changed since the last successful save.
    pub dirty: bool,
}

impl ClassEdits {
    pub fn is_empty(&self) -> bool { self.ops.is_empty() }

    pub fn push(&mut self, mut op: ClassOp) {
        op.seq = self.next_seq;
        self.next_seq += 1;
        self.ops.push(op);
        self.dirty = true;
    }

    /// Drop the most recent op. Callers must re-apply the remaining ops from
    /// the source data — an op cannot be reversed in place, since it does not
    /// record what the points were before.
    pub fn undo(&mut self) -> Option<ClassOp> {
        let op = self.ops.pop();
        if op.is_some() { self.dirty = true; }
        op
    }

    /// Apply every op, in order, to a slab of points known to lie within
    /// `[node_min, node_max]`. Returns the number of points changed.
    pub fn apply_to_node(
        &self,
        points: &mut [Point],
        node_min: &[f64; 3],
        node_max: &[f64; 3],
    ) -> usize {
        self.apply_from(0, points, node_min, node_max)
    }

    /// Apply only `ops[from_idx..]`. Appending one op to a large resident cloud
    /// should cost one pass, not one pass per op ever recorded.
    pub fn apply_from(
        &self,
        from_idx: usize,
        points: &mut [Point],
        node_min: &[f64; 3],
        node_max: &[f64; 3],
    ) -> usize {
        if from_idx >= self.ops.len() { return 0; }
        let relevant: Vec<&ClassOp> = self.ops[from_idx..].iter()
            .filter(|op| op.intersects_aabb(node_min, node_max))
            .collect();
        if relevant.is_empty() { return 0; }
        let mut changed = 0;
        for p in points.iter_mut() {
            for op in &relevant {
                if op.apply_point(p) { changed += 1; }
            }
        }
        changed
    }

    /// Apply every op to an arbitrary slab, with no bounds hint.
    pub fn apply_all(&self, points: &mut [Point]) -> usize {
        if self.ops.is_empty() { return 0; }
        let mut changed = 0;
        for p in points.iter_mut() {
            for op in &self.ops {
                if op.apply_point(p) { changed += 1; }
            }
        }
        changed
    }

    // ---- Sidecar persistence ----

    /// Sidecar path for a source file: `<source>.haiviz-class.txt`.
    pub fn sidecar_path(source: &str) -> PathBuf {
        let mut s = source.to_string();
        s.push_str(".haiviz-class.txt");
        PathBuf::from(s)
    }

    /// Read the sidecar beside `source`, if one exists. A malformed line is
    /// skipped with a warning rather than failing the whole load.
    pub fn load(source: &str) -> ClassEdits {
        let path = Self::sidecar_path(source);
        if !path.exists() { return ClassEdits::default(); }
        let Ok(content) = std::fs::read_to_string(&path) else {
            warn!("Cannot read {}", path.display());
            return ClassEdits::default();
        };
        let mut edits = ClassEdits::default();
        for (n, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            match parse_op(line) {
                Some(op) => {
                    edits.next_seq = edits.next_seq.max(op.seq + 1);
                    edits.ops.push(op);
                }
                None => warn!("{}:{}: malformed op, skipped", path.display(), n + 1),
            }
        }
        edits.ops.sort_by_key(|o| o.seq);
        if !edits.ops.is_empty() {
            info!("Loaded {} classification ops from {}", edits.ops.len(), path.display());
        }
        edits
    }

    /// Write the sidecar beside `source` and clear the dirty flag.
    pub fn save(&mut self, source: &str) -> std::io::Result<PathBuf> {
        let path = Self::sidecar_path(source);
        let mut out = String::new();
        out.push_str("# haiviz manual classification ops, v1\n");
        out.push_str(&format!("# source: {}\n", source));
        out.push_str("# op;seq;from(*=any);target;sx;sy;ex;ey;depth;along,z;...\n");
        out.push_str("# gop;seq;from(*=any);target;zmin;zmax;ox;oy;a;b;c;offset;x,y;...\n");
        out.push_str("# vop;seq;from(*=any);target;zmin;zmax;x,y;...\n");
        for op in &self.ops {
            out.push_str(&format_op(op));
            out.push('\n');
        }
        std::fs::write(&path, out)?;
        self.dirty = false;
        info!("Saved {} classification ops to {}", self.ops.len(), path.display());
        Ok(path)
    }
}

fn format_op(op: &ClassOp) -> String {
    let from = match op.from_class {
        Some(c) => c.to_string(),
        None => "*".to_string(),
    };
    match &op.region {
        ClassRegion::Corridor { start, end, depth, polygon } => {
            let mut s = format!(
                "op;{};{};{};{};{};{};{};{}",
                op.seq, from, op.target_class,
                start.0, start.1, end.0, end.1, depth,
            );
            for (a, z) in polygon {
                s.push_str(&format!(";{a},{z}"));
            }
            s
        }
        ClassRegion::AboveGround { polygon, origin, plane, offset, z_min, z_max } => {
            let mut s = format!(
                "gop;{};{};{};{};{};{};{};{};{};{};{}",
                op.seq, from, op.target_class,
                z_min, z_max, origin.0, origin.1,
                plane.0, plane.1, plane.2, offset,
            );
            for (x, y) in polygon {
                s.push_str(&format!(";{x},{y}"));
            }
            s
        }
        ClassRegion::Volume { polygon, z_min, z_max } => {
            let mut s = format!(
                "vop;{};{};{};{};{}",
                op.seq, from, op.target_class, z_min, z_max,
            );
            for (x, y) in polygon {
                s.push_str(&format!(";{x},{y}"));
            }
            s
        }
    }
}

fn parse_op(line: &str) -> Option<ClassOp> {
    let parts: Vec<&str> = line.split(';').collect();
    if parts.len() < 4 { return None; }
    let kind = parts[0];
    let seq = parts[1].trim().parse::<u64>().ok()?;
    let from_class = match parts[2].trim() {
        "*" => None,
        other => Some(other.parse::<u8>().ok()?),
    };
    let target_class = parts[3].trim().parse::<u8>().ok()?;
    let f = |i: usize| parts.get(i)?.trim().parse::<f64>().ok();

    let read_polygon = |from: usize| -> Option<Vec<(f64, f64)>> {
        let mut polygon = Vec::new();
        for part in &parts[from..] {
            let (a, b) = part.split_once(',')?;
            polygon.push((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?));
        }
        if polygon.len() < 3 { None } else { Some(polygon) }
    };

    let region = match kind {
        "op" => {
            if parts.len() < 12 { return None; }
            ClassRegion::Corridor {
                start: (f(4)?, f(5)?),
                end: (f(6)?, f(7)?),
                depth: f(8)?,
                polygon: read_polygon(9)?,
            }
        }
        "gop" => {
            if parts.len() < 15 { return None; }
            ClassRegion::AboveGround {
                z_min: f(4)?,
                z_max: f(5)?,
                origin: (f(6)?, f(7)?),
                plane: (f(8)?, f(9)?, f(10)?),
                offset: f(11)?,
                polygon: read_polygon(12)?,
            }
        }
        "vop" => {
            if parts.len() < 9 { return None; }
            ClassRegion::Volume {
                z_min: f(4)?,
                z_max: f(5)?,
                polygon: read_polygon(6)?,
            }
        }
        _ => return None,
    };
    Some(ClassOp::new(seq, region, target_class, from_class))
}

/// A ground plane fitted to a lasso footprint.
pub struct GroundFit {
    pub origin: (f64, f64),
    /// `z = a*(x-ox) + b*(y-oy) + c`
    pub plane: (f64, f64, f64),
    /// Grid cells that survived outlier rejection and defined the final plane.
    pub cells_used: usize,
    /// RMS residual of those cells, in metres — a usable confidence readout.
    pub rms: f64,
    /// Points considered before gridding.
    pub points_considered: usize,
}

impl GroundFit {
    pub fn height_at(&self, x: f64, y: f64) -> f64 {
        self.plane.0 * (x - self.origin.0) + self.plane.1 * (y - self.origin.1) + self.plane.2
    }

    /// Slope of the fitted plane, in degrees from horizontal.
    pub fn slope_degrees(&self) -> f64 {
        (self.plane.0 * self.plane.0 + self.plane.1 * self.plane.1).sqrt().atan().to_degrees()
    }
}

/// Solve a 3x3 system by Gaussian elimination with partial pivoting.
/// Returns `None` if the matrix is singular — a degenerate footprint, or
/// cell centres that all fall on one line.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let (pivot, _) = (col..3)
            .map(|r| (r, m[r][col].abs()))
            .fold((col, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        if m[pivot][col].abs() < 1e-12 { return None; }
        m.swap(col, pivot);
        for r in 0..3 {
            if r == col { continue; }
            let factor = m[r][col] / m[col][col];
            let (row, pivot_row) = (m[r], m[col]);
            let mut updated = row;
            for (c, v) in updated.iter_mut().enumerate().skip(col) {
                *v -= factor * pivot_row[c];
            }
            m[r] = updated;
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

/// Least-squares plane through `(dx, dy, z)` samples.
fn fit_plane(samples: &[(f64, f64, f64)]) -> Option<(f64, f64, f64)> {
    if samples.len() < 3 { return None; }
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    let (mut sxz, mut syz) = (0.0, 0.0);
    for &(x, y, z) in samples {
        sxx += x * x; sxy += x * y; syy += y * y;
        sx += x; sy += y; sz += z;
        sxz += x * z; syz += y * z;
    }
    let n = samples.len() as f64;
    solve3([
        [sxx, sxy, sx, sxz],
        [sxy, syy, sy, syz],
        [sx,  sy,  n,  sz],
    ])
    .map(|v| (v[0], v[1], v[2]))
}

/// Minimum grid cells needed before a fit is trusted.
const MIN_GROUND_CELLS: usize = 8;

/// Fit a ground plane to the points inside a plan-view footprint.
///
/// Fits to the *lowest* point in each grid cell rather than to everything:
/// a footprint chosen for classification is mostly the stuff standing on the
/// ground, so a plain least-squares fit over all points would be dragged up
/// into the canopy or the roof. Two rejection passes then drop cells that sit
/// far off the plane — vegetation the grid never saw under, and noise below
/// the surface, which matters because taking minima is exactly what makes low
/// noise dangerous.
pub fn fit_ground_plane(
    points: &[(f64, f64, f64)],
    polygon: &[(f64, f64)],
    z_min: f64,
    z_max: f64,
) -> Option<GroundFit> {
    if polygon.len() < 3 { return None; }

    let (mut xmin, mut ymin) = (f64::MAX, f64::MAX);
    let (mut xmax, mut ymax) = (f64::MIN, f64::MIN);
    for &(x, y) in polygon {
        xmin = xmin.min(x); xmax = xmax.max(x);
        ymin = ymin.min(y); ymax = ymax.max(y);
    }
    let origin = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5);

    // Aim for roughly a 40x40 grid over the footprint, but never so fine that
    // most cells are empty nor so coarse that a slope is averaged away.
    let extent = (xmax - xmin).max(ymax - ymin);
    let cell = (extent / 40.0).clamp(0.25, 5.0);

    let mut minima: HashMap<(i64, i64), (f64, f64, f64)> = HashMap::new();
    let mut considered = 0usize;
    for &(x, y, z) in points {
        if z < z_min || z > z_max { continue; }
        if !point_in_polygon(polygon, x, y) { continue; }
        considered += 1;
        let key = (((x - origin.0) / cell).floor() as i64, ((y - origin.1) / cell).floor() as i64);
        let e = minima.entry(key).or_insert((x - origin.0, y - origin.1, z));
        if z < e.2 { *e = (x - origin.0, y - origin.1, z); }
    }
    if minima.len() < MIN_GROUND_CELLS { return None; }

    let mut samples: Vec<(f64, f64, f64)> = minima.into_values().collect();
    let mut plane = fit_plane(&samples)?;

    for _ in 0..2 {
        let rms = residual_rms(&samples, plane);
        if rms <= 1e-9 { break; }
        let keep: Vec<(f64, f64, f64)> = samples.iter()
            .copied()
            .filter(|&(x, y, z)| (z - (plane.0 * x + plane.1 * y + plane.2)).abs() <= 2.0 * rms)
            .collect();
        if keep.len() < MIN_GROUND_CELLS { break; }
        samples = keep;
        plane = fit_plane(&samples)?;
    }

    let rms = residual_rms(&samples, plane);
    Some(GroundFit {
        origin,
        plane,
        cells_used: samples.len(),
        rms,
        points_considered: considered,
    })
}

fn residual_rms(samples: &[(f64, f64, f64)], plane: (f64, f64, f64)) -> f64 {
    if samples.is_empty() { return 0.0; }
    let sum: f64 = samples.iter()
        .map(|&(x, y, z)| {
            let r = z - (plane.0 * x + plane.1 * y + plane.2);
            r * r
        })
        .sum();
    (sum / samples.len() as f64).sqrt()
}

/// Classes to omit entirely when exporting — the noise-filtering half of the
/// workflow. Kept alongside the edits so a session's intent travels together.
#[derive(Clone, Default, Debug)]
pub struct ExportFilter {
    pub drop_classes: HashSet<u8>,
}

impl ExportFilter {
    pub fn keeps(&self, p: &Point) -> bool {
        !self.drop_classes.contains(&effective_class(p))
    }
}

/// Default output path for an export: `<stem>_classified.laz` beside the source.
pub fn default_export_path(source: &str) -> PathBuf {
    let p = Path::new(source);
    let dir = p.parent().unwrap_or(Path::new("."));
    // Strip the full `.copc.laz` / `.laz` / `.las` suffix, not just the last one.
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("cloud");
    let stem = name
        .strip_suffix(".copc.laz")
        .or_else(|| name.strip_suffix(".laz"))
        .or_else(|| name.strip_suffix(".las"))
        .unwrap_or(name);
    dir.join(format!("{stem}_classified.laz"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64, z: f64, class: u8) -> Point {
        let mut p = Point { x, y, z, ..Default::default() };
        set_class(&mut p, class);
        p
    }

    /// A 20 m box around a section line running due east from (100, 200).
    fn box_op(target: u8, from: Option<u8>) -> ClassOp {
        ClassOp::corridor(
            0,
            (100.0, 200.0),
            (200.0, 200.0),
            5.0,
            vec![(10.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0)],
            target,
            from,
        )
    }

    #[test]
    fn contains_respects_along_depth_and_height() {
        let op = box_op(7, None);
        // Inside: 20 m along, on the line, 10 m up.
        assert!(op.contains(120.0, 200.0, 10.0));
        // Outside along the line.
        assert!(!op.contains(105.0, 200.0, 10.0));
        // Outside in height.
        assert!(!op.contains(120.0, 200.0, 25.0));
        // Inside the corridor half-width, but only just.
        assert!(op.contains(120.0, 204.9, 10.0));
        assert!(!op.contains(120.0, 205.1, 10.0));
    }

    #[test]
    fn diagonal_line_maps_section_space_correctly() {
        // A line at 45°: 10 m "along" is (√50, √50) from the start.
        let op = ClassOp::corridor(
            0, (0.0, 0.0), (100.0, 100.0), 2.0,
            vec![(9.0, -1.0), (11.0, -1.0), (11.0, 1.0), (9.0, 1.0)],
            7, None,
        );
        let d = (50.0f64).sqrt();
        assert!(op.contains(d, d, 0.0));
        assert!(!op.contains(d, d + 5.0, 0.0));
    }

    #[test]
    fn from_class_restricts_the_edit() {
        let op = box_op(7, Some(1));
        let mut ground = pt(120.0, 200.0, 10.0, 2);
        let mut unclassified = pt(120.0, 200.0, 10.0, 1);
        assert!(!op.apply_point(&mut ground));
        assert_eq!(effective_class(&ground), 2);
        assert!(op.apply_point(&mut unclassified));
        assert_eq!(effective_class(&unclassified), 7);
    }

    #[test]
    fn applying_twice_is_a_no_op() {
        let op = box_op(7, None);
        let mut p = pt(120.0, 200.0, 10.0, 2);
        assert!(op.apply_point(&mut p));
        assert!(!op.apply_point(&mut p), "second apply must not report a change");
        assert_eq!(effective_class(&p), 7);
    }

    #[test]
    fn overlap_class_round_trips_through_the_flag() {
        let mut p = pt(0.0, 0.0, 0.0, 2);
        set_class(&mut p, 12);
        assert!(p.is_overlap);
        assert_eq!(effective_class(&p), 12);
        set_class(&mut p, 2);
        assert!(!p.is_overlap);
        assert_eq!(effective_class(&p), 2);
    }

    #[test]
    fn bbox_rejects_distant_nodes_but_keeps_touching_ones() {
        let op = box_op(7, None);
        assert!(!op.intersects_aabb(&[0.0, 0.0, 0.0], &[50.0, 50.0, 50.0]));
        assert!(op.intersects_aabb(&[115.0, 195.0, 5.0], &[125.0, 205.0, 15.0]));
        // Overlapping in X/Y but far below in Z.
        assert!(!op.intersects_aabb(&[115.0, 195.0, -100.0], &[125.0, 205.0, -50.0]));
    }

    #[test]
    fn sidecar_round_trips() {
        let mut edits = ClassEdits::default();
        edits.push(box_op(7, None));
        edits.push(box_op(18, Some(1)));

        let text: Vec<String> = edits.ops.iter().map(format_op).collect();
        let parsed: Vec<ClassOp> = text.iter().map(|l| parse_op(l).expect("parses")).collect();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].target_class, 7);
        assert_eq!(parsed[0].from_class, None);
        assert_eq!(parsed[1].target_class, 18);
        assert_eq!(parsed[1].from_class, Some(1));
        match (&parsed[1].region, &edits.ops[1].region) {
            (
                ClassRegion::Corridor { polygon: a, depth: da, .. },
                ClassRegion::Corridor { polygon: b, depth: db, .. },
            ) => {
                assert_eq!(a, b);
                assert_eq!(da, db);
            }
            _ => panic!("expected corridor regions"),
        }
        // A parsed op must select the same points as the one that was written.
        assert!(parsed[0].contains(120.0, 200.0, 10.0));
    }

    /// The sidecar is what carries manual classification across sessions, so
    /// the on-disk round trip is worth exercising for real, not just the
    /// line format in memory.
    #[test]
    fn sidecar_survives_a_save_load_cycle_on_disk() {
        let dir = std::env::temp_dir().join("haiviz-sidecar-test");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("cloud.copc.laz");
        let source_str = source.to_str().unwrap().to_string();
        let sidecar = ClassEdits::sidecar_path(&source_str);
        let _ = std::fs::remove_file(&sidecar);

        // It lives beside the source, keyed on the source's full path — not in
        // the working directory, so it is found again however haiviz is
        // launched.
        assert_eq!(sidecar.parent(), source.parent());
        assert_eq!(sidecar.file_name().unwrap(), "cloud.copc.laz.haiviz-class.txt");

        // Nothing on disk yet: a fresh cloud must open with no edits.
        assert!(ClassEdits::load(&source_str).is_empty());

        let mut edits = ClassEdits::default();
        edits.push(box_op(7, None));
        edits.push(box_op(2, Some(7)));
        assert!(edits.dirty);
        edits.save(&source_str).expect("saves");
        assert!(!edits.dirty, "saving clears the dirty flag");
        assert!(sidecar.exists());

        // Reopening the cloud in a later session reads them back...
        let reloaded = ClassEdits::load(&source_str);
        assert_eq!(reloaded.ops.len(), 2);
        assert!(!reloaded.dirty);
        assert_eq!(reloaded.ops[0].seq, 0);
        assert_eq!(reloaded.ops[1].seq, 1);
        assert_eq!(reloaded.ops[1].from_class, Some(7));

        // ...and they still select and reclassify the same points.
        let mut pts = vec![pt(120.0, 200.0, 10.0, 1), pt(0.0, 0.0, 0.0, 1)];
        let changed = reloaded.apply_all(&mut pts);
        assert_eq!(changed, 2, "op 1 sets class 7, op 2 then moves it to 2");
        assert_eq!(effective_class(&pts[0]), 2);
        assert_eq!(effective_class(&pts[1]), 1, "point outside the prism is untouched");

        // A further op appended in the new session keeps numbering going,
        // rather than colliding with the sequence already on disk.
        let mut reloaded = reloaded;
        reloaded.push(box_op(18, None));
        assert_eq!(reloaded.ops[2].seq, 2);
        reloaded.save(&source_str).unwrap();
        assert_eq!(ClassEdits::load(&source_str).ops.len(), 3);

        let _ = std::fs::remove_file(&sidecar);
    }

    #[test]
    fn malformed_lines_are_rejected_not_panicked_on() {
        assert!(parse_op("").is_none());
        assert!(parse_op("op;0;*;7").is_none());
        assert!(parse_op("notanop;0;*;7;0;0;1;0;1;0,0;1,0;1,1").is_none());
        // Fewer than three vertices is not a polygon.
        assert!(parse_op("op;0;*;7;0;0;1;0;1;0,0;1,0").is_none());
    }

    #[test]
    fn ops_apply_in_order() {
        let mut edits = ClassEdits::default();
        edits.push(box_op(7, None));
        edits.push(box_op(2, None));
        let mut pts = vec![pt(120.0, 200.0, 10.0, 1)];
        edits.apply_all(&mut pts);
        assert_eq!(effective_class(&pts[0]), 2, "later op wins");
    }

    #[test]
    fn apply_from_skips_earlier_ops() {
        let mut edits = ClassEdits::default();
        edits.push(box_op(7, None));
        let mut pts = vec![pt(120.0, 200.0, 10.0, 1)];
        let big = [f64::MAX; 3];
        let small = [f64::MIN; 3];
        // Skipping the only op must leave the point alone.
        assert_eq!(edits.apply_from(1, &mut pts, &small, &big), 0);
        assert_eq!(effective_class(&pts[0]), 1);
        assert_eq!(edits.apply_from(0, &mut pts, &small, &big), 1);
        assert_eq!(effective_class(&pts[0]), 7);
    }

    /// A square footprint centred on the origin, `half` metres each way.
    fn square(half: f64) -> Vec<(f64, f64)> {
        vec![(-half, -half), (half, -half), (half, half), (-half, half)]
    }

    /// Ground samples on `z = a*x + b*y + c`, on a 1 m grid.
    fn ground(a: f64, b: f64, c: f64, half: f64) -> Vec<(f64, f64, f64)> {
        let mut v = Vec::new();
        let n = (half * 2.0) as i32;
        for i in 0..=n {
            for j in 0..=n {
                let (x, y) = (-half + i as f64, -half + j as f64);
                v.push((x, y, a * x + b * y + c));
            }
        }
        v
    }

    #[test]
    fn fits_a_flat_ground_under_vegetation() {
        let mut pts = ground(0.0, 0.0, 10.0, 20.0);
        // A canopy over most of the footprint: a plain fit through everything
        // would land somewhere between the two surfaces.
        for i in 0..30 {
            for j in 0..30 {
                pts.push((-15.0 + i as f64, -15.0 + j as f64, 15.0));
            }
        }
        let fit = fit_ground_plane(&pts, &square(20.0), -100.0, 100.0).expect("fits");
        assert!((fit.height_at(0.0, 0.0) - 10.0).abs() < 0.01, "got {}", fit.height_at(0.0, 0.0));
        assert!((fit.height_at(18.0, -18.0) - 10.0).abs() < 0.01);
        assert!(fit.slope_degrees() < 0.1);
        assert!(fit.rms < 0.01);
    }

    #[test]
    fn recovers_a_tilted_ground() {
        let pts = ground(0.10, -0.05, 5.0, 20.0);
        let fit = fit_ground_plane(&pts, &square(20.0), -100.0, 100.0).expect("fits");
        for &(x, y) in &[(0.0, 0.0), (10.0, 10.0), (-15.0, 5.0)] {
            let expected = 0.10 * x - 0.05 * y + 5.0;
            assert!((fit.height_at(x, y) - expected).abs() < 0.01,
                "at ({x},{y}) got {} want {expected}", fit.height_at(x, y));
        }
    }

    #[test]
    fn low_noise_does_not_drag_the_plane_down() {
        let mut pts = ground(0.0, 0.0, 10.0, 20.0);
        // Points well below the surface — the danger case, since the fit takes
        // the minimum in each cell.
        pts.push((-10.0, -10.0, 2.0));
        pts.push((5.0, 5.0, 1.0));
        pts.push((12.0, -3.0, 0.0));
        let fit = fit_ground_plane(&pts, &square(20.0), -100.0, 100.0).expect("fits");
        assert!((fit.height_at(0.0, 0.0) - 10.0).abs() < 0.05,
            "low noise pulled the plane to {}", fit.height_at(0.0, 0.0));
    }

    #[test]
    fn the_height_band_bounds_what_is_fitted() {
        let mut pts = ground(0.0, 0.0, 10.0, 20.0);
        // A basement slab below the band under test.
        for i in 0..40 {
            for j in 0..40 {
                pts.push((-20.0 + i as f64, -20.0 + j as f64, 2.0));
            }
        }
        // With no band the fit lands on the slab...
        let all = fit_ground_plane(&pts, &square(20.0), -100.0, 100.0).expect("fits");
        assert!((all.height_at(0.0, 0.0) - 2.0).abs() < 0.05);
        // ...and with one it lands on the floor above it.
        let banded = fit_ground_plane(&pts, &square(20.0), 8.0, 100.0).expect("fits");
        assert!((banded.height_at(0.0, 0.0) - 10.0).abs() < 0.05);
    }

    #[test]
    fn refuses_a_fit_with_too_little_to_go_on() {
        assert!(fit_ground_plane(&[], &square(20.0), -100.0, 100.0).is_none());
        // Three points make a plane but not evidence.
        let sparse = vec![(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)];
        assert!(fit_ground_plane(&sparse, &square(20.0), -100.0, 100.0).is_none());
        // Points outside the footprint do not count.
        let outside = ground(0.0, 0.0, 10.0, 20.0);
        assert!(fit_ground_plane(&outside, &square(1.0), -100.0, 100.0).is_none());
    }

    #[test]
    fn above_ground_op_selects_only_what_stands_up() {
        let pts = ground(0.05, 0.0, 10.0, 20.0);
        let fit = fit_ground_plane(&pts, &square(20.0), -100.0, 100.0).expect("fits");
        let op = ClassOp::new(
            0,
            ClassRegion::AboveGround {
                polygon: square(20.0),
                origin: fit.origin,
                plane: fit.plane,
                offset: 0.5,
                z_min: 0.0,
                z_max: 100.0,
            },
            7,
            None,
        );
        let g = |x: f64| 0.05 * x + 10.0;
        // On the ground: untouched, including on the sloped far side.
        assert!(!op.contains(0.0, 0.0, g(0.0)));
        assert!(!op.contains(18.0, 4.0, g(18.0)));
        // Just under the offset: still untouched.
        assert!(!op.contains(0.0, 0.0, g(0.0) + 0.4));
        // Clearly above: selected.
        assert!(op.contains(0.0, 0.0, g(0.0) + 3.0));
        assert!(op.contains(-18.0, 12.0, g(-18.0) + 3.0));
        // Outside the footprint, however high: untouched.
        assert!(!op.contains(60.0, 0.0, g(60.0) + 3.0));
        // Above the recorded band: untouched.
        assert!(!op.contains(0.0, 0.0, 200.0));
    }

    #[test]
    fn above_ground_op_survives_the_sidecar() {
        let op = ClassOp::new(
            0,
            ClassRegion::AboveGround {
                polygon: vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
                origin: (20.0, 30.0),
                plane: (0.01, -0.02, 5.5),
                offset: 0.25,
                z_min: 1.0,
                z_max: 40.0,
            },
            18,
            Some(1),
        );
        let parsed = parse_op(&format_op(&op)).expect("parses");
        assert_eq!(parsed.target_class, 18);
        assert_eq!(parsed.from_class, Some(1));
        match parsed.region {
            ClassRegion::AboveGround { origin, plane, offset, z_min, z_max, ref polygon } => {
                assert_eq!(origin, (20.0, 30.0));
                assert_eq!(plane, (0.01, -0.02, 5.5));
                assert_eq!(offset, 0.25);
                assert_eq!((z_min, z_max), (1.0, 40.0));
                assert_eq!(polygon.len(), 4);
            }
            _ => panic!("wrong region kind"),
        }
        // And it still selects the same points either side of the round trip.
        for probe in [(20.0, 30.0, 6.0), (20.0, 30.0, 12.0), (0.0, 0.0, 12.0)] {
            assert_eq!(
                op.contains(probe.0, probe.1, probe.2),
                parsed.contains(probe.0, probe.1, probe.2),
            );
        }
    }

    #[test]
    fn volume_op_selects_everything_inside_the_footprint() {
        let op = ClassOp::new(
            0,
            ClassRegion::Volume {
                polygon: square(20.0),
                z_min: 5.0,
                z_max: 30.0,
            },
            7,
            None,
        );
        // Inside the footprint and inside the band: selected, at any height —
        // this is the whole difference from `AboveGround`, which would leave
        // whatever the ground plane sits under untouched.
        assert!(op.contains(0.0, 0.0, 5.0));
        assert!(op.contains(0.0, 0.0, 17.0));
        assert!(op.contains(-18.0, 12.0, 30.0));
        // Outside the band: untouched.
        assert!(!op.contains(0.0, 0.0, 4.9));
        assert!(!op.contains(0.0, 0.0, 30.1));
        // Outside the footprint: untouched.
        assert!(!op.contains(60.0, 0.0, 17.0));
    }

    #[test]
    fn volume_op_survives_the_sidecar() {
        let op = ClassOp::new(
            0,
            ClassRegion::Volume {
                polygon: vec![(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
                z_min: 1.0,
                z_max: 40.0,
            },
            18,
            Some(1),
        );
        let line = format_op(&op);
        assert!(line.starts_with("vop;"));
        let parsed = parse_op(&line).expect("parses");
        assert_eq!(parsed.target_class, 18);
        assert_eq!(parsed.from_class, Some(1));
        match parsed.region {
            ClassRegion::Volume { z_min, z_max, ref polygon } => {
                assert_eq!((z_min, z_max), (1.0, 40.0));
                assert_eq!(polygon.len(), 4);
            }
            _ => panic!("wrong region kind"),
        }
        for probe in [(20.0, 30.0, 6.0), (20.0, 30.0, 0.0), (0.0, 0.0, 6.0)] {
            assert_eq!(
                op.contains(probe.0, probe.1, probe.2),
                parsed.contains(probe.0, probe.1, probe.2),
            );
        }
    }

    #[test]
    fn corridor_and_above_ground_ops_coexist_in_one_sidecar() {
        let mut edits = ClassEdits::default();
        edits.push(box_op(7, None));
        edits.push(ClassOp::new(
            0,
            ClassRegion::AboveGround {
                polygon: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                origin: (5.0, 5.0),
                plane: (0.0, 0.0, 1.0),
                offset: 0.2,
                z_min: 0.0,
                z_max: 50.0,
            },
            2,
            None,
        ));
        let text: Vec<String> = edits.ops.iter().map(format_op).collect();
        assert!(text[0].starts_with("op;"));
        assert!(text[1].starts_with("gop;"));
        let back: Vec<ClassOp> = text.iter().map(|l| parse_op(l).expect("parses")).collect();
        assert!(matches!(back[0].region, ClassRegion::Corridor { .. }));
        assert!(matches!(back[1].region, ClassRegion::AboveGround { .. }));
        assert_eq!(back[1].seq, 1);
    }

    #[test]
    fn export_filter_drops_by_effective_class() {
        let mut filter = ExportFilter::default();
        filter.drop_classes.insert(7);
        filter.drop_classes.insert(12);
        assert!(filter.keeps(&pt(0.0, 0.0, 0.0, 2)));
        assert!(!filter.keeps(&pt(0.0, 0.0, 0.0, 7)));
        // Class 12 lives in the overlap flag, not the classification field.
        assert!(!filter.keeps(&pt(0.0, 0.0, 0.0, 12)));
    }

    #[test]
    fn export_path_strips_the_full_copc_suffix() {
        assert_eq!(
            default_export_path("/data/cloud.copc.laz").file_name().unwrap(),
            "cloud_classified.laz",
        );
        assert_eq!(
            default_export_path("/data/tile.laz").file_name().unwrap(),
            "tile_classified.laz",
        );
    }
}
