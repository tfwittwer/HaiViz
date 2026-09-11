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

/// A set of classification ids: 256 bits, one per possible class.
///
/// One type serves two jobs that the plan kept apart, because they meet:
/// which classes an op is allowed to touch, and which classes are visible.
/// An op freezes the intersection of the two at the moment it is made (see
/// [`ClassOp::from`]), so a single representation is what makes that
/// intersection one `&`.
///
/// `[u64; 4]` rather than `[bool; 256]`: 32 bytes is cheap to copy into the
/// per-point filter closures the snap queries and the pick sites run, and a
/// membership test is a shift and a mask.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassMask([u64; 4]);

impl ClassMask {
    /// Every class. The "no restriction" value, and the default visibility.
    pub const ALL: ClassMask = ClassMask([u64::MAX; 4]);
    /// No class at all. Only reachable by hiding everything, or by
    /// intersecting a restriction with a visibility that excludes it — which
    /// is a mistake the UI warns about rather than committing.
    pub const NONE: ClassMask = ClassMask([0; 4]);

    pub fn single(id: u8) -> ClassMask {
        let mut m = ClassMask::NONE;
        m.insert(id);
        m
    }

    #[inline]
    pub fn contains(&self, id: u8) -> bool {
        self.0[(id >> 6) as usize] & (1u64 << (id & 63)) != 0
    }

    pub fn insert(&mut self, id: u8) {
        self.0[(id >> 6) as usize] |= 1u64 << (id & 63);
    }

    pub fn remove(&mut self, id: u8) {
        self.0[(id >> 6) as usize] &= !(1u64 << (id & 63));
    }

    pub fn set(&mut self, id: u8, on: bool) {
        if on { self.insert(id) } else { self.remove(id) }
    }

    pub fn intersect(self, other: ClassMask) -> ClassMask {
        let mut out = ClassMask::NONE;
        for i in 0..4 { out.0[i] = self.0[i] & other.0[i]; }
        out
    }

    pub fn inverted(self) -> ClassMask {
        let mut out = ClassMask::NONE;
        for i in 0..4 { out.0[i] = !self.0[i]; }
        out
    }

    pub fn is_all(&self) -> bool { *self == ClassMask::ALL }
    pub fn is_empty(&self) -> bool { *self == ClassMask::NONE }
    pub fn count(&self) -> u32 { self.0.iter().map(|w| w.count_ones()).sum() }

    /// The one class in the set, if there is exactly one.
    pub fn only(&self) -> Option<u8> {
        if self.count() != 1 { return None; }
        (0..=255u8).find(|&c| self.contains(c))
    }

    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u8).filter(move |&c| self.contains(c))
    }
}

impl Default for ClassMask {
    fn default() -> Self { ClassMask::ALL }
}

/// The UI carries "only from class N" as an `Option<u8>`; this is the bridge.
impl From<Option<u8>> for ClassMask {
    fn from(o: Option<u8>) -> ClassMask {
        match o { Some(c) => ClassMask::single(c), None => ClassMask::ALL }
    }
}

impl std::fmt::Debug for ClassMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_all() { return f.write_str("ClassMask(*)"); }
        if self.is_empty() { return f.write_str("ClassMask(-)"); }
        write!(f, "ClassMask({})", format_mask(self))
    }
}

/// The camera a lasso was drawn through, frozen into the op.
///
/// haiviz's 3D camera is orthographic, so every pixel looks in the same
/// direction and an outline drawn on screen selects a prism: the outline laid
/// in the view plane and swept along the view direction through the whole depth
/// of the scene. `origin`, `right` and `up` place that plane; a point is inside
/// when its coordinates along `right` and `up` fall inside the outline. That is
/// exactly "projects inside what was drawn", and it is still a pure geometric
/// predicate, so it replays at export like every other region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewPlane {
    /// A point on the plane, absolute world coordinates.
    pub origin: [f64; 3],
    /// Unit screen-right and screen-up, in world space; perpendicular.
    pub right: [f64; 3],
    pub up: [f64; 3],
}

impl ViewPlane {
    /// A world point's coordinates in the plane: `(along right, along up)`.
    pub fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let d = [x - self.origin[0], y - self.origin[1], z - self.origin[2]];
        (dot3(d, self.right), dot3(d, self.up))
    }

    /// The world point at plane coordinates `(u, v)`.
    pub fn point(&self, u: f64, v: f64) -> [f64; 3] {
        [0, 1, 2].map(|i| self.origin[i] + u * self.right[i] + v * self.up[i])
    }

    /// The direction the prism extends in. Its sign does not matter: the prism
    /// runs both ways.
    pub fn depth_dir(&self) -> [f64; 3] {
        let (r, u) = (self.right, self.up);
        [r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0]]
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Is a world point inside a lasso outline? In world XY for a plan outline
/// (`view` is `None`), in the view plane for a screen outline.
fn outline_contains(polygon: &[(f64, f64)], view: Option<&ViewPlane>, x: f64, y: f64, z: f64) -> bool {
    match view {
        None => point_in_polygon(polygon, x, y),
        Some(v) => {
            let (u, w) = v.project(x, y, z);
            point_in_polygon(polygon, u, w)
        }
    }
}

/// World-XY bounds of what a lasso outline can select inside a height band.
///
/// For a plan outline, the outline's own bounds. For a screen outline, each
/// corner's line of sight cut at the band's floor and ceiling. Looking
/// horizontally a line of sight never leaves the band, and the bounds are
/// unlimited — conservative, since they only ever reject.
pub fn outline_xy_bounds(
    polygon: &[(f64, f64)],
    view: Option<&ViewPlane>,
    z_min: f64,
    z_max: f64,
) -> ((f64, f64), (f64, f64)) {
    const UNBOUNDED: ((f64, f64), (f64, f64)) = ((f64::MIN, f64::MIN), (f64::MAX, f64::MAX));
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    let mut include = |x: f64, y: f64| {
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    };
    match view {
        None => for &(x, y) in polygon { include(x, y); },
        Some(v) => {
            let d = v.depth_dir();
            if d[2].abs() < 1e-9 { return UNBOUNDED; }
            for &(u, w) in polygon {
                let p = v.point(u, w);
                for z in [z_min, z_max] {
                    let t = (z - p[2]) / d[2];
                    let (x, y) = (p[0] + t * d[0], p[1] + t * d[1]);
                    if !(x.is_finite() && y.is_finite()) { return UNBOUNDED; }
                    include(x, y);
                }
            }
        }
    }
    (lo, hi)
}

/// The shape an op selects. Every variant is a pure geometric predicate,
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
    /// Everything standing above the local ground inside a lasso outline.
    ///
    /// The ground is a plane fitted once, when the op is made, and then frozen
    /// into the op — so it stays a predicate rather than something that has to
    /// be re-derived from whatever points happen to be loaded later.
    AboveGround {
        /// Lasso outline: absolute world XY when `view` is `None`, view-plane
        /// coordinates when the outline was drawn away from top view.
        polygon: Vec<(f64, f64)>,
        /// The camera a screen outline was drawn through; see [`ViewPlane`].
        view: Option<ViewPlane>,
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
    /// Everything inside a lasso outline, between two absolute heights.
    ///
    /// The ground-free counterpart to `AboveGround`: nothing is fitted and
    /// nothing is excluded from below, so the height band is the only vertical
    /// limit. Use it where the ground is not a meaningful reference — indoors,
    /// on a bridge deck, or wherever the thing to be reclassified sits on top
    /// of something that must not be swept up with it.
    Volume {
        /// Lasso outline, as for `AboveGround`.
        polygon: Vec<(f64, f64)>,
        view: Option<ViewPlane>,
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
    /// The classes this op is allowed to touch, frozen when the op was made.
    ///
    /// Two things are folded in here: an explicit "only from class N", and the
    /// set of classes that were visible at the time. Ops are replayed on COPC
    /// nodes as they stream in and again on every node at export, so
    /// *"the exported file agrees with the display by construction"* — and
    /// that guarantee only holds if the meaning of an op is fixed when it is
    /// drawn. If it depended on live visibility, an export an hour later with
    /// different classes toggled would reclassify different points.
    pub from: ClassMask,
    /// Cached world-space AABB, for cheap node rejection.
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
}

impl ClassOp {
    pub fn new(
        seq: u64,
        region: ClassRegion,
        target_class: u8,
        from: ClassMask,
    ) -> Self {
        let mut op = ClassOp {
            seq, region, target_class, from,
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
        from: ClassMask,
    ) -> Self {
        ClassOp::new(
            seq,
            ClassRegion::Corridor { start, end, depth, polygon },
            target_class,
            from,
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
            ClassRegion::AboveGround { polygon, view, z_min, z_max, .. }
            | ClassRegion::Volume { polygon, view, z_min, z_max } => {
                if polygon.is_empty() { return; }
                let (lo, hi) = outline_xy_bounds(polygon, view.as_ref(), *z_min, *z_max);
                self.bbox_min = [lo.0, lo.1, *z_min];
                self.bbox_max = [hi.0, hi.1, *z_max];
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
            ClassRegion::AboveGround { polygon, view, origin, plane, offset, z_min, z_max } => {
                if z < *z_min || z > *z_max { return false; }
                let ground = plane.0 * (x - origin.0) + plane.1 * (y - origin.1) + plane.2;
                if z < ground + *offset { return false; }
                outline_contains(polygon, view.as_ref(), x, y, z)
            }
            ClassRegion::Volume { polygon, view, z_min, z_max } => {
                if z < *z_min || z > *z_max { return false; }
                outline_contains(polygon, view.as_ref(), x, y, z)
            }
        }
    }

    /// Apply to a single point. Returns true if the classification changed.
    pub fn apply_point(&self, p: &mut Point) -> bool {
        if !self.from.contains(effective_class(p)) { return false; }
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
        out.push_str("# haiviz manual classification ops, v3\n");
        out.push_str(&format!("# source: {}\n", source));
        out.push_str("# from: * = any class, 5 = only class 5, 2|5|6 = only those,\n");
        out.push_str("#       !3|7 = every class except those. The set and complement\n");
        out.push_str("#       forms are v2 and appear only when classes were hidden.\n");
        out.push_str("# op;seq;from;target;sx;sy;ex;ey;depth;along,z;...\n");
        out.push_str("# gop;seq;from;target;zmin;zmax;ox;oy;a;b;c;offset;x,y;...\n");
        out.push_str("# vop;seq;from;target;zmin;zmax;x,y;...\n");
        out.push_str("# v3, only for outlines drawn away from top view: the outline is in\n");
        out.push_str("# the view plane at o, along unit vectors r (right) and u (up).\n");
        out.push_str("# gvop;seq;from;target;zmin;zmax;ox;oy;oz;rx;ry;rz;ux;uy;uz;gx;gy;a;b;c;offset;u,v;...\n");
        out.push_str("# vvop;seq;from;target;zmin;zmax;ox;oy;oz;rx;ry;rz;ux;uy;uz;u,v;...\n");
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

/// Render a class mask as the `from` token.
///
/// Ordinary sessions keep writing byte-identical v1 sidecars: with no
/// visibility filter in force the mask is either everything (`*`) or the
/// single class the operator restricted to (`5`), which is exactly what v1
/// wrote. The set forms only appear once something was hidden, so an older
/// haiviz — which skips lines it cannot parse — degrades only for the files
/// that genuinely carry the new meaning.
///
/// Two set forms rather than one: a visible set is usually "everything except
/// the two classes I switched off", and spelling that out as 254 ids would put
/// a kilobyte on every line. Whichever of the set and its complement is
/// shorter wins.
fn format_mask(m: &ClassMask) -> String {
    if m.is_all() { return "*".to_string(); }
    // Not reachable from the UI, which refuses to commit an op that can never
    // match. Written explicitly rather than as an empty token so a file that
    // somehow contains one round-trips instead of failing to parse.
    if m.is_empty() { return "-".to_string(); }
    if let Some(c) = m.only() { return c.to_string(); }
    let join = |ids: Vec<u8>| ids.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("|");
    let set: Vec<u8> = m.iter().collect();
    if set.len() <= 128 {
        join(set)
    } else {
        format!("!{}", join(m.inverted().iter().collect()))
    }
}

fn parse_mask(token: &str) -> Option<ClassMask> {
    let token = token.trim();
    match token {
        "*" => Some(ClassMask::ALL),
        "-" => Some(ClassMask::NONE),
        _ => {
            let (base, body) = match token.strip_prefix('!') {
                Some(rest) => (ClassMask::ALL, rest),
                None => (ClassMask::NONE, token),
            };
            let mut m = base;
            for part in body.split('|') {
                let id = part.trim().parse::<u8>().ok()?;
                if base.is_all() { m.remove(id) } else { m.insert(id) }
            }
            Some(m)
        }
    }
}

/// `;ox;oy;oz;rx;ry;rz;ux;uy;uz` for a view plane.
fn format_view(v: &ViewPlane) -> String {
    let [o, r, u] = [v.origin, v.right, v.up];
    format!(";{};{};{};{};{};{};{};{};{}", o[0], o[1], o[2], r[0], r[1], r[2], u[0], u[1], u[2])
}

fn format_op(op: &ClassOp) -> String {
    let from = format_mask(&op.from);
    let outline = |polygon: &[(f64, f64)]| -> String {
        polygon.iter().map(|(a, b)| format!(";{a},{b}")).collect()
    };
    match &op.region {
        // A screen outline gets its own line kinds. A plan outline — anything
        // drawn from straight above — keeps writing gop/vop exactly as before,
        // so an ordinary session's sidecar stays readable by older builds,
        // which skip line kinds they do not know.
        ClassRegion::AboveGround { polygon, view: Some(v), origin, plane, offset, z_min, z_max } => {
            format!(
                "gvop;{};{};{};{};{}{};{};{};{};{};{};{}{}",
                op.seq, from, op.target_class, z_min, z_max, format_view(v),
                origin.0, origin.1, plane.0, plane.1, plane.2, offset, outline(polygon),
            )
        }
        ClassRegion::Volume { polygon, view: Some(v), z_min, z_max } => {
            format!(
                "vvop;{};{};{};{};{}{}{}",
                op.seq, from, op.target_class, z_min, z_max, format_view(v), outline(polygon),
            )
        }
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
        ClassRegion::AboveGround { polygon, view: None, origin, plane, offset, z_min, z_max } => {
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
        ClassRegion::Volume { polygon, view: None, z_min, z_max } => {
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
    let from = parse_mask(parts[2])?;
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
                view: None,
            }
        }
        "vop" => {
            if parts.len() < 9 { return None; }
            ClassRegion::Volume {
                z_min: f(4)?,
                z_max: f(5)?,
                polygon: read_polygon(6)?,
                view: None,
            }
        }
        "gvop" | "vvop" => {
            let view = ViewPlane {
                origin: [f(6)?, f(7)?, f(8)?],
                right: [f(9)?, f(10)?, f(11)?],
                up: [f(12)?, f(13)?, f(14)?],
            };
            if kind == "vvop" {
                ClassRegion::Volume { z_min: f(4)?, z_max: f(5)?, polygon: read_polygon(15)?, view: Some(view) }
            } else {
                ClassRegion::AboveGround {
                    z_min: f(4)?,
                    z_max: f(5)?,
                    origin: (f(15)?, f(16)?),
                    plane: (f(17)?, f(18)?, f(19)?),
                    offset: f(20)?,
                    polygon: read_polygon(21)?,
                    view: Some(view),
                }
            }
        }
        _ => return None,
    };
    Some(ClassOp::new(seq, region, target_class, from))
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
    let (lo, hi) = outline_xy_bounds(polygon, None, z_min, z_max);
    fit_ground_cells(points, &|x, y, _| point_in_polygon(polygon, x, y), lo, hi, z_min, z_max)
}

/// [`fit_ground_plane`] for any lasso outline, plan or screen.
///
/// A screen outline's footprint on the ground is not the outline — from an
/// oblique view it is the stretch of ground the prism crosses inside the band —
/// so the fitting grid is laid over the points the prism actually caught
/// rather than over the outline's bounds.
pub fn fit_ground_plane_in(
    points: &[(f64, f64, f64)],
    polygon: &[(f64, f64)],
    view: Option<&ViewPlane>,
    z_min: f64,
    z_max: f64,
) -> Option<GroundFit> {
    let Some(v) = view else { return fit_ground_plane(points, polygon, z_min, z_max) };
    if polygon.len() < 3 { return None; }
    let inside = |x: f64, y: f64, z: f64| outline_contains(polygon, Some(v), x, y, z);
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for &(x, y, z) in points {
        if z < z_min || z > z_max || !inside(x, y, z) { continue; }
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    }
    if lo.0 > hi.0 { return None; }
    fit_ground_cells(points, &inside, lo, hi, z_min, z_max)
}

/// The fit itself, over the points `inside` accepts, gridded over the XY box
/// `lo`..`hi`.
fn fit_ground_cells(
    points: &[(f64, f64, f64)],
    inside: &dyn Fn(f64, f64, f64) -> bool,
    lo: (f64, f64),
    hi: (f64, f64),
    z_min: f64,
    z_max: f64,
) -> Option<GroundFit> {
    let ((xmin, ymin), (xmax, ymax)) = (lo, hi);
    let origin = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5);

    // Aim for roughly a 40x40 grid over the footprint, but never so fine that
    // most cells are empty nor so coarse that a slope is averaged away.
    let extent = (xmax - xmin).max(ymax - ymin);
    let cell = (extent / 40.0).clamp(0.25, 5.0);

    let mut minima: HashMap<(i64, i64), (f64, f64, f64)> = HashMap::new();
    let mut considered = 0usize;
    for &(x, y, z) in points {
        if z < z_min || z > z_max { continue; }
        if !inside(x, y, z) { continue; }
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
        box_op_mask(target, from.into())
    }

    fn box_op_mask(target: u8, from: ClassMask) -> ClassOp {
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
            7, ClassMask::ALL,
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
    fn class_mask_set_algebra() {
        assert!(ClassMask::ALL.contains(0) && ClassMask::ALL.contains(255));
        assert!(!ClassMask::NONE.contains(0));
        assert_eq!(ClassMask::ALL.count(), 256);
        assert_eq!(ClassMask::NONE.count(), 0);

        let five = ClassMask::single(5);
        assert!(five.contains(5) && !five.contains(4));
        assert_eq!(five.only(), Some(5));
        assert_eq!(ClassMask::ALL.only(), None);
        assert_eq!(ClassMask::ALL.inverted(), ClassMask::NONE);

        // "Only from 5" while 5 is visible keeps 5; while 5 is hidden it
        // empties, which is the case §6.5 makes the UI refuse.
        let mut visible = ClassMask::ALL;
        visible.remove(3);
        assert_eq!(five.intersect(visible), five);
        visible.remove(5);
        assert!(five.intersect(visible).is_empty());

        assert_eq!(ClassMask::from(None::<u8>), ClassMask::ALL);
        assert_eq!(ClassMask::from(Some(7u8)), ClassMask::single(7));
        assert_eq!(ClassMask::default(), ClassMask::ALL);

        let mut m = ClassMask::NONE;
        for id in [2u8, 5, 6] { m.insert(id); }
        assert_eq!(m.iter().collect::<Vec<u8>>(), vec![2, 5, 6]);
    }

    #[test]
    fn from_token_round_trips_in_every_form() {
        let mut set = ClassMask::NONE;
        for id in [2u8, 5, 6] { set.insert(id); }
        let mut complement = ClassMask::ALL;
        for id in [3u8, 7] { complement.remove(id); }

        for (mask, want) in [
            (ClassMask::ALL, "*"),
            (ClassMask::NONE, "-"),
            (ClassMask::single(5), "5"),
            (set, "2|5|6"),
            (complement, "!3|7"),
        ] {
            assert_eq!(format_mask(&mask), want, "formatting {mask:?}");
            assert_eq!(parse_mask(want), Some(mask), "parsing {want}");
        }
        assert_eq!(parse_mask("nonsense"), None);
        assert_eq!(parse_mask("2|x"), None);
    }

    #[test]
    fn a_session_without_a_visibility_filter_writes_v1_op_lines() {
        // Risk 3: an older haiviz drops any line it cannot parse. The
        // mitigation is that the new token only ever appears when classes were
        // actually hidden, so an ordinary session's sidecar stays readable by
        // older builds. That is a property of `format_mask`, so assert it on
        // the op lines rather than trusting the reasoning.
        let all_visible = ClassMask::ALL;
        let no_restriction = ClassMask::from(None::<u8>).intersect(all_visible);
        let only_five = ClassMask::from(Some(5u8)).intersect(all_visible);

        let a = format_op(&box_op_mask(7, no_restriction));
        let b = format_op(&box_op_mask(7, only_five));
        assert_eq!(a.split(';').nth(2), Some("*"));
        assert_eq!(b.split(';').nth(2), Some("5"));

        // And with a filter in force the line does change — otherwise the
        // freeze would not be recorded at all.
        let mut visible = ClassMask::ALL;
        visible.remove(3);
        let c = format_op(&box_op_mask(7, ClassMask::from(None::<u8>).intersect(visible)));
        assert_eq!(c.split(';').nth(2), Some("!3"));
    }

    #[test]
    fn the_shorter_of_the_set_and_its_complement_is_written() {
        // A visible set is usually "everything except the two I switched off".
        // Spelling that out would put a kilobyte on every op line.
        let mut visible = ClassMask::ALL;
        for id in [1u8, 3, 5] { visible.remove(id); }
        let token = format_mask(&visible);
        assert_eq!(token, "!1|3|5");
        assert!(token.len() < 10, "complement form must stay short: {token}");
        assert_eq!(parse_mask(&token), Some(visible));
    }

    #[test]
    fn the_frozen_mask_decides_which_points_an_op_touches() {
        // Ground visible, vegetation hidden: the op reclassifies the ground
        // under the lasso and leaves the vegetation alone, whatever the
        // visibility is when it is replayed.
        let mut visible = ClassMask::ALL;
        visible.remove(5);
        let op = box_op_mask(7, ClassMask::from(None::<u8>).intersect(visible));

        let mut ground = pt(120.0, 200.0, 10.0, 2);
        let mut veg = pt(120.0, 200.0, 10.0, 5);
        assert!(op.apply_point(&mut ground));
        assert_eq!(effective_class(&ground), 7);
        assert!(!op.apply_point(&mut veg));
        assert_eq!(effective_class(&veg), 5, "a hidden class must not be touched");

        // Survives the sidecar, or the freeze would only hold for one session.
        let reparsed = parse_op(&format_op(&op)).expect("parses");
        let mut veg2 = pt(120.0, 200.0, 10.0, 5);
        assert!(!reparsed.apply_point(&mut veg2));
    }

    #[test]
    fn an_empty_mask_matches_nothing() {
        let op = box_op_mask(7, ClassMask::NONE);
        for class in [0u8, 2, 5, 12, 255] {
            let mut p = pt(120.0, 200.0, 10.0, class);
            assert!(!op.apply_point(&mut p), "class {class} must not be touched");
        }
        // And it round-trips rather than corrupting the file it lands in.
        assert_eq!(parse_op(&format_op(&op)).expect("parses").from, ClassMask::NONE);
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
        assert_eq!(parsed[0].from, ClassMask::ALL);
        assert_eq!(parsed[1].target_class, 18);
        assert_eq!(parsed[1].from, ClassMask::single(1));
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
        assert_eq!(reloaded.ops[1].from, ClassMask::single(7));

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
                view: None,
                origin: fit.origin,
                plane: fit.plane,
                offset: 0.5,
                z_min: 0.0,
                z_max: 100.0,
            },
            7,
            ClassMask::ALL,
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
                view: None,
                origin: (20.0, 30.0),
                plane: (0.01, -0.02, 5.5),
                offset: 0.25,
                z_min: 1.0,
                z_max: 40.0,
            },
            18,
            ClassMask::single(1),
        );
        let parsed = parse_op(&format_op(&op)).expect("parses");
        assert_eq!(parsed.target_class, 18);
        assert_eq!(parsed.from, ClassMask::single(1));
        match parsed.region {
            ClassRegion::AboveGround { origin, plane, offset, z_min, z_max, ref polygon, view } => {
                assert_eq!(view, None, "a plan footprint stays a plan footprint");
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
                view: None,
                z_min: 5.0,
                z_max: 30.0,
            },
            7,
            ClassMask::ALL,
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
                view: None,
                z_min: 1.0,
                z_max: 40.0,
            },
            18,
            ClassMask::single(1),
        );
        let line = format_op(&op);
        assert!(line.starts_with("vop;"));
        let parsed = parse_op(&line).expect("parses");
        assert_eq!(parsed.target_class, 18);
        assert_eq!(parsed.from, ClassMask::single(1));
        match parsed.region {
            ClassRegion::Volume { z_min, z_max, ref polygon, view } => {
                assert_eq!((z_min, z_max), (1.0, 40.0));
                assert_eq!(polygon.len(), 4);
                assert_eq!(view, None);
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
                view: None,
                origin: (5.0, 5.0),
                plane: (0.0, 0.0, 1.0),
                offset: 0.2,
                z_min: 0.0,
                z_max: 50.0,
            },
            2,
            ClassMask::ALL,
        ));
        let text: Vec<String> = edits.ops.iter().map(format_op).collect();
        assert!(text[0].starts_with("op;"));
        assert!(text[1].starts_with("gop;"));
        let back: Vec<ClassOp> = text.iter().map(|l| parse_op(l).expect("parses")).collect();
        assert!(matches!(back[0].region, ClassRegion::Corridor { .. }));
        assert!(matches!(back[1].region, ClassRegion::AboveGround { .. }));
        assert_eq!(back[1].seq, 1);
    }

    /// Looking north and 45° down, from anywhere: right is east, up is north
    /// and up in equal measure.
    fn looking_north_45() -> ViewPlane {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        ViewPlane { origin: [0.0, 0.0, 0.0], right: [1.0, 0.0, 0.0], up: [0.0, s, s] }
    }

    #[test]
    fn a_screen_outline_selects_what_projects_inside_it() {
        let op = ClassOp::new(
            0,
            ClassRegion::Volume {
                polygon: square(1.0),
                view: Some(looking_north_45()),
                z_min: -50.0,
                z_max: 50.0,
            },
            7,
            ClassMask::ALL,
        );
        // Straight down the line of sight, however far along it.
        assert!(op.contains(0.0, 0.0, 0.0));
        assert!(op.contains(0.5, 20.0, -20.0));
        assert!(op.contains(-0.5, -30.0, 30.0));
        // Beside it, or above it on screen.
        assert!(!op.contains(2.0, 20.0, -20.0));
        assert!(!op.contains(0.0, 0.0, 3.0));
        // The column a top-view outline of the same square would take is not
        // what was drawn from this angle.
        assert!(!op.contains(0.0, 0.0, -20.0));
        // And the band still bounds it.
        assert!(!op.contains(0.0, 60.0, -60.0));
    }

    #[test]
    fn a_screen_outlines_bounds_are_its_lines_of_sight_cut_by_the_band() {
        let op = ClassOp::new(
            0,
            ClassRegion::Volume { polygon: square(1.0), view: Some(looking_north_45()), z_min: -20.0, z_max: 0.0 },
            7,
            ClassMask::ALL,
        );
        // Where the prism runs through the band: from about y = -1.4 at the
        // top of the band to y = 21.4 at its floor.
        assert!(op.intersects_aabb(&[-0.5, 9.5, -10.5], &[0.5, 10.5, -9.5]));
        assert!(!op.intersects_aabb(&[-0.5, -30.0, -10.5], &[0.5, -25.0, -9.5]));
        assert!(!op.intersects_aabb(&[5.0, 9.5, -10.5], &[6.0, 10.5, -9.5]));

        // Looking horizontally the prism never leaves the band, so nothing
        // in XY can be ruled out.
        let side = ViewPlane { origin: [0.0; 3], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] };
        let op = ClassOp::new(
            0,
            ClassRegion::Volume { polygon: square(1.0), view: Some(side), z_min: -20.0, z_max: 20.0 },
            7,
            ClassMask::ALL,
        );
        assert!(op.intersects_aabb(&[-0.5, 1.0e6, -0.5], &[0.5, 1.0e6 + 1.0, 0.5]));
        assert!(op.contains(0.0, 1.0e6, 0.0));
    }

    #[test]
    fn screen_outline_ops_survive_the_sidecar_and_plan_ones_keep_their_old_lines() {
        let view = ViewPlane {
            origin: [208_800.5, 373_300.25, 12.0],
            right: [0.6, 0.8, 0.0],
            up: [-0.4, 0.3, 0.866_025_403_784_438_6],
        };
        let polygon = vec![(-2.0, -1.0), (3.0, -1.0), (3.0, 4.0), (-2.0, 4.0)];
        let volume = ClassOp::new(
            3,
            ClassRegion::Volume { polygon: polygon.clone(), view: Some(view), z_min: 1.0, z_max: 40.0 },
            18,
            ClassMask::single(1),
        );
        let ground = ClassOp::new(
            4,
            ClassRegion::AboveGround {
                polygon: polygon.clone(),
                view: Some(view),
                origin: (208_801.0, 373_301.0),
                plane: (0.01, -0.02, 11.5),
                offset: 0.25,
                z_min: 1.0,
                z_max: 40.0,
            },
            7,
            ClassMask::ALL,
        );
        for (op, kind) in [(&volume, "vvop;"), (&ground, "gvop;")] {
            let line = format_op(op);
            assert!(line.starts_with(kind), "{line}");
            let back = parse_op(&line).expect("parses");
            assert_eq!(back.seq, op.seq);
            // Selects the same points either side of the round trip, probed
            // along and across the prism.
            for k in -5..=5 {
                for (u, w) in [(0.0, 0.0), (2.9, 3.9), (3.1, 0.0), (0.0, -1.1)] {
                    let p = view.point(u, w);
                    let d = view.depth_dir();
                    let t = k as f64 * 3.0;
                    let (x, y, z) = (p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]);
                    assert_eq!(op.contains(x, y, z), back.contains(x, y, z), "{kind} at ({u}, {w}, {t})");
                }
            }
        }

        // An outline from straight above is still a plan outline, written the
        // way an older haiviz can read.
        let plan = ClassOp::new(
            0,
            ClassRegion::Volume { polygon: square(5.0), view: None, z_min: 0.0, z_max: 1.0 },
            7,
            ClassMask::ALL,
        );
        assert!(format_op(&plan).starts_with("vop;"));
    }

    #[test]
    fn the_ground_fits_under_an_oblique_outline() {
        let mut pts = ground(0.05, 0.0, 10.0, 40.0);
        // A canopy over the middle, where the outline's line of sight crosses
        // both it and the ground under it.
        for i in 0..31 {
            for j in 0..21 {
                pts.push((-15.0 + i as f64, -10.0 + j as f64, 16.0));
            }
        }
        let v = ViewPlane { origin: [0.0, 0.0, 10.0], ..looking_north_45() };
        let polygon = square(8.0);
        let fit = fit_ground_plane_in(&pts, &polygon, Some(&v), 0.0, 30.0).expect("fits");
        for &(x, y) in &[(0.0, 0.0), (5.0, 10.0), (-6.0, -8.0)] {
            let want = 0.05 * x + 10.0;
            assert!((fit.height_at(x, y) - want).abs() < 0.05, "at ({x},{y}) {} want {want}", fit.height_at(x, y));
        }
        // Nothing caught, nothing fitted.
        assert!(fit_ground_plane_in(&pts, &polygon, Some(&v), 100.0, 200.0).is_none());
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
