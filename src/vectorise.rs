//! Digitising mode: drawing polylines by hand onto the cloud.
//!
//! `main.rs` holds only the mode wiring — the key handling next to the lasso
//! block, the panel call in each of the three per-mode egui blocks, and the
//! overlay draw. Everything about what a vertex *is* lives here.
//!
//! ## Where a vertex comes from
//!
//! A vertex is pulled onto real data by a [`SnapMode`], within an absolute
//! radius in metres — see [`DEFAULT_SNAP_RADIUS`] for why absolute. Every snap
//! query runs through the spatial index with a filter that applies the height
//! band *and* class visibility inside the grid walk, which is what makes class
//! visibility the primary control over what you snap to: hide everything but
//! ground and "lowest Z in radius" means the ground.
//!
//! Shift-click places a vertex freely instead, on a horizontal plane through
//! the previous one — the only well-defined answer when the operator has said
//! the data should not decide, and what makes drawing across a gap possible.
//!
//! A snap never silently degrades. It is [`SnapQuality::Good`] and placeable,
//! or it is refused and says why.
//!
//! ## Where a line goes
//!
//! A layer has a destination from the moment it exists, because layers
//! auto-save: there is no "save this somewhere" step to hang a file dialog on.
//! A new layer lands in `<cloud>.haiviz-vectors.gpkg`, mirroring the
//! `<cloud>.haiviz-class.txt` convention the classification sidecar already
//! uses. *Save As* in the Polylines panel moves it afterwards.
//!
//! Saves fire on a finished line, not on a vertex: `ClassEdits` saves after
//! each lasso rather than each lasso point, and the same reasoning applies. A
//! half-drawn line is not a meaningful thing to persist, and a whole-file
//! rewrite per vertex would be waste. A crash therefore loses at most the line
//! currently being drawn, which is a far better trade than journalling
//! in-progress geometry.

// Edge following (phase 5) and vertex editing (phase 6) build on this same
// state; a few accessors here are theirs.
#![allow(dead_code)]

use macroquad::prelude::*;
use std::path::Path;

use crate::polylines::{Layer, Polyline, PolylineFile, VectorFormat};
use crate::spatial::{NO_FILTER, SpatialGrid, percentile_by_z};

/// The sidecar a new layer lands in, beside the cloud it was drawn on.
const VECTOR_SIDECAR_SUFFIX: &str = ".haiviz-vectors.gpkg";

/// Default snap radius, in metres.
///
/// Absolute rather than zoom-relative: a metre means the same thing at every
/// zoom level, which is what makes a snap reproducible. A zoom-relative radius
/// would quietly change what a click meant every time the camera moved.
pub const DEFAULT_SNAP_RADIUS: f32 = 0.2;
pub const MIN_SNAP_RADIUS: f32 = 0.02;
pub const MAX_SNAP_RADIUS: f32 = 5.0;

/// What a vertex is pulled onto.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapMode {
    /// The nearest real point to the cursor ray. True 3D, so it works from any
    /// camera angle rather than only from directly above.
    Nearest,
    /// A low percentile of Z within the radius: gutters, ditch invert, toe of
    /// slope. Percentile rather than the raw minimum because a raw minimum on
    /// LiDAR is usually one stray low return.
    LowZ,
    /// A high percentile of Z within the radius: kerb top, crest, cable, roof
    /// ridge.
    HighZ,
    /// The previous vertex's height, at the cursor's XY. Contours and
    /// breaklines, where the elevation is the thing being held constant.
    SameZ,
    /// A vertex or segment of an existing polyline, for closing loops and
    /// tying into neighbouring lines.
    Existing,
}

impl SnapMode {
    pub const ALL: [SnapMode; 5] =
        [SnapMode::Nearest, SnapMode::LowZ, SnapMode::HighZ, SnapMode::SameZ, SnapMode::Existing];

    pub fn label(&self) -> &'static str {
        match self {
            SnapMode::Nearest => "Nearest point",
            SnapMode::LowZ => "Low Z (percentile)",
            SnapMode::HighZ => "High Z (percentile)",
            SnapMode::SameZ => "Same Z as previous",
            SnapMode::Existing => "Existing polyline",
        }
    }

    pub fn next(&self) -> SnapMode {
        let i = SnapMode::ALL.iter().position(|m| m == self).unwrap_or(0);
        SnapMode::ALL[(i + 1) % SnapMode::ALL.len()]
    }

    /// Whether a whole line can be re-draped in this mode.
    ///
    /// A drape keeps every vertex's plan position and asks the cloud for its
    /// height, so it needs a mode that answers "how high is the data here".
    /// Same-Z and existing-polyline are both about *other vertices*, not the
    /// data, and have nothing to say about a vertex on its own.
    pub fn drapes(&self) -> bool {
        matches!(self, SnapMode::Nearest | SnapMode::LowZ | SnapMode::HighZ)
    }
}

/// How much to trust a snap.
///
/// There is deliberately no "fell back to something else" state. A snap that
/// silently degrades is worse than one that refuses: the operator would keep
/// clicking and only find out at export that half the line came from a
/// thinned LOD sample.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapQuality {
    /// Full-resolution data under the cursor; the vertex is trustworthy.
    Good,
    /// The resident LOD is coarser than the snap radius, so a percentile here
    /// would be taken over whatever sparse subset happened to be loaded. The
    /// marker says so and the click is refused.
    Coarse,
    /// Nothing within the radius. Also refused.
    Empty,
}

/// A candidate vertex under the cursor.
#[derive(Clone, Debug)]
pub struct SnapHit {
    pub world: [f64; 3],
    pub quality: SnapQuality,
    /// Why, when the quality is not `Good`.
    pub note: Option<String>,
}

impl SnapHit {
    pub fn good(world: [f64; 3]) -> SnapHit {
        SnapHit { world, quality: SnapQuality::Good, note: None }
    }
    pub fn is_placeable(&self) -> bool { self.quality == SnapQuality::Good }
}

/// What the panel is asking `main.rs` to do. Returned rather than acted on so
/// this module never has to know about `AppMode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Request {
    Start,
    Cancel,
    Finish,
    /// Finish, joining the last vertex back to the first.
    FinishRing,
    UndoVertex,
}

pub struct Digitise {
    /// Where finished lines go, as `(file index, layer index)`. Resolved when
    /// digitising starts and held until the operator points it somewhere else.
    pub target: Option<(usize, usize)>,
    /// The line being drawn, in absolute world coordinates — the same space
    /// `Polyline` stores, so committing is a move rather than a conversion.
    pub pending: Vec<[f64; 3]>,
    /// Name for the next finished line. `{n}` is replaced by its number in the
    /// layer, so a run of lines does not need renaming one at a time.
    pub name_template: String,
    /// Attribute written to the `code` column, for whoever consumes the file.
    pub code: String,
    pub color: Color,
    /// Drawn width of the line in progress, in screen pixels. Matches
    /// `Layer::width_px`, and seeds it for a layer haiviz creates.
    pub width_px: f32,
    pub status: Option<String>,
    pub snap_mode: SnapMode,
    /// Snap radius in metres. See [`DEFAULT_SNAP_RADIUS`].
    pub radius: f32,
    /// Percentile for [`SnapMode::LowZ`] / [`SnapMode::HighZ`]. `0` and `100`
    /// give the raw minimum and maximum back, which is the escape hatch when
    /// the data really is clean.
    pub low_pct: f32,
    pub high_pct: f32,
    /// The snap under the cursor this frame, for the marker and the readout.
    pub hover: Option<SnapHit>,
    /// Vertices placed so far this session, for the readout.
    pub committed_lines: usize,
}

impl Default for Digitise {
    fn default() -> Self {
        Digitise {
            target: None,
            pending: Vec::new(),
            name_template: "Line {n}".to_string(),
            code: String::new(),
            color: Color::new(1.0, 0.85, 0.2, 1.0),
            width_px: 2.0,
            status: None,
            snap_mode: SnapMode::Nearest,
            radius: DEFAULT_SNAP_RADIUS,
            low_pct: 5.0,
            high_pct: 95.0,
            hover: None,
            committed_lines: 0,
        }
    }
}

impl Digitise {
    pub fn is_drawing(&self) -> bool { !self.pending.is_empty() }

    pub fn push(&mut self, v: [f64; 3]) { self.pending.push(v); }

    pub fn pop(&mut self) -> Option<[f64; 3]> { self.pending.pop() }

    pub fn clear(&mut self) { self.pending.clear(); }

    /// The vertex a rubber band or a "same Z as previous" snap works from.
    pub fn last(&self) -> Option<[f64; 3]> { self.pending.last().copied() }

    /// Length of the line so far, and of its last segment, in metres.
    pub fn lengths(&self) -> (f64, f64) {
        let mut total = 0.0;
        for w in self.pending.windows(2) { total += dist3(w[0], w[1]); }
        let last = if self.pending.len() >= 2 {
            dist3(self.pending[self.pending.len() - 2], self.pending[self.pending.len() - 1])
        } else {
            0.0
        };
        (total, last)
    }

    /// Slope of the last segment in percent, and its ΔZ. `None` until there is
    /// a segment to measure, or when it is exactly vertical.
    pub fn last_slope(&self) -> Option<(f64, f64)> {
        if self.pending.len() < 2 { return None; }
        let a = self.pending[self.pending.len() - 2];
        let b = self.pending[self.pending.len() - 1];
        let dz = b[2] - a[2];
        let run = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if run <= 0.0 { return None; }
        Some((dz / run * 100.0, dz))
    }

    /// Commit the pending vertices as a line in the target layer — as a closed
    /// ring when `closed`, for outlines.
    ///
    /// Returns the file that now needs saving, so the caller can auto-save it
    /// without this module reaching for the filesystem.
    pub fn finish(&mut self, files: &mut [PolylineFile], closed: bool) -> Option<usize> {
        let needed = crate::vector_edit::min_vertices(closed);
        if self.pending.len() < needed {
            self.status = Some(if closed {
                "A ring needs at least three vertices.".to_string()
            } else {
                "A line needs at least two vertices.".to_string()
            });
            return None;
        }
        let Some((fi, li)) = self.target else {
            self.status = Some("No layer to put the line in.".to_string());
            return None;
        };
        let Some(n) = files.get(fi).and_then(|pf| pf.layers.get(li)).map(|l| l.polylines.len() + 1) else {
            // The target's file was removed from under the tool. Keep the
            // vertices: the operator can point "Into:" somewhere else and
            // finish again, rather than redrawing the line.
            self.status = Some("The target layer is no longer loaded — choose another in Into:.".to_string());
            self.target = None;
            return None;
        };
        let name = self.name_template.replace("{n}", &n.to_string());

        let mut line = Polyline::from_points(name, self.color, std::mem::take(&mut self.pending));
        line.closed = closed;
        if !self.code.is_empty() {
            line.attrs.insert("code".to_string(), self.code.clone());
        }
        files[fi].add_line(li, line)?;
        self.committed_lines += 1;
        Some(fi)
    }

    /// Keep `target` pointing at the same layer after file `fi` is unloaded.
    pub fn file_removed(&mut self, fi: usize) {
        self.target = match self.target {
            Some((f, _)) if f == fi => None,
            Some((f, l)) if f > fi => Some((f - 1, l)),
            t => t,
        };
    }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

/// Where a new layer lands: beside the cloud, named after it.
///
/// Mirrors `ClassEdits::sidecar_path`. Deliberately not a file dialog: layers
/// auto-save, so a layer without a destination is not a state that can exist,
/// and asking at creation time would put a modal in front of the first click.
pub fn default_vector_path(cloud_path: &str) -> String {
    format!("{cloud_path}{VECTOR_SIDECAR_SUFFIX}")
}

/// Find or create the layer finished lines go into.
///
/// If the sidecar is already open it is reused, so restarting digitising in one
/// session does not fork the file. Otherwise a fresh in-memory GeoPackage is
/// created; it reaches disk on the first finished line.
pub fn ensure_target(
    files: &mut Vec<PolylineFile>,
    cloud_path: &str,
    srs_id: i32,
    width_px: f32,
) -> (usize, usize) {
    let path = default_vector_path(cloud_path);
    if let Some(fi) = files.iter().position(|f| f.path == path) {
        if files[fi].layers.is_empty() {
            files[fi].layers.push(new_layer(srs_id, width_px));
        }
        return (fi, files[fi].layers.len() - 1);
    }
    let mut pf = PolylineFile::new(&path, VectorFormat::GeoPackage);
    pf.layers.push(new_layer(srs_id, width_px));
    files.push(pf);
    (files.len() - 1, 0)
}

fn new_layer(srs_id: i32, width_px: f32) -> Layer {
    let mut l = Layer::new("digitised");
    l.srs_id = srs_id;
    // The layer inherits the width the line was drawn at, so a finished line
    // does not change thickness the moment it is committed.
    l.width_px = width_px;
    l
}

/// Label for a `(file, layer)` target, for the panel's combo box.
pub fn target_label(files: &[PolylineFile], fi: usize, li: usize) -> String {
    match files.get(fi).and_then(|f| f.layers.get(li).map(|l| (f, l))) {
        Some((f, l)) => format!("{} › {}", f.file_name(), l.name),
        None => "—".to_string(),
    }
}

/// Draw the line in progress, the rubber band, and the snap marker.
///
/// Under the 3D camera, before `set_default_camera()`, so the overlay is
/// depth-tested against the cloud and still composites beneath the panel.
pub fn draw_overlay(
    dig: &Digitise,
    origin: [f64; 3],
    view_dir: Vec3,
    metres_per_pixel: f32,
    cursor: Option<[f64; 3]>,
    drawing: bool,
) {
    let to_view = |p: [f64; 3]| vec3(
        (p[0] - origin[0]) as f32,
        (p[1] - origin[1]) as f32,
        (p[2] - origin[2]) as f32,
    );
    let half = dig.width_px.max(0.5) * 0.5 * metres_per_pixel;

    let pts: Vec<Vec3> = dig.pending.iter().map(|p| to_view(*p)).collect();
    crate::polylines::draw_ribbon_3d(&pts, false, half, dig.color, view_dir);

    // Every committed vertex gets a marker: on a dense cloud the line alone
    // does not show where the operator actually clicked.
    let tick = (half * 3.0).max(metres_per_pixel * 3.0);
    for p in &pts {
        draw_line_3d(*p - vec3(tick, 0.0, 0.0), *p + vec3(tick, 0.0, 0.0), dig.color);
        draw_line_3d(*p - vec3(0.0, tick, 0.0), *p + vec3(0.0, tick, 0.0), dig.color);
        draw_line_3d(*p - vec3(0.0, 0.0, tick), *p + vec3(0.0, 0.0, tick), dig.color);
    }

    if drawing
        && let (Some(&last), Some(c)) = (pts.last(), cursor) {
        let faint = Color::new(dig.color.r, dig.color.g, dig.color.b, 0.55);
        crate::polylines::draw_ribbon_3d(&[last, to_view(c)], false, half, faint, view_dir);
    }

    if drawing { draw_snap_marker(dig, origin); }
}

/// The marker colour for a snap: the line colour when it can be placed, and a
/// visibly different one — not merely a dimmer one — when it is refused. A
/// coarse snap is a refusal, and it must not read as "nearly fine".
fn marker_color(dig: &Digitise, quality: SnapQuality) -> Color {
    match quality {
        SnapQuality::Good => dig.color,
        SnapQuality::Coarse => Color::new(1.0, 0.4, 0.1, 1.0),
        SnapQuality::Empty => Color::new(0.9, 0.25, 0.25, 1.0),
    }
}

/// Draw the snap marker and the radius circle at the candidate vertex.
///
/// A cross, like `measure_a` / `measure_b`, so it reads the same way as the
/// other precision tools. The circle sits at the *snapped* Z rather than at the
/// cursor's, so what it encloses is what the percentile was taken over — and
/// it is built in world units, so it keeps showing the true radius in metres.
///
/// Shared with vertex editing, which snaps a dragged vertex the same way.
pub fn draw_snap_marker(dig: &Digitise, origin: [f64; 3]) {
    let Some(hit) = &dig.hover else { return };
    let to_view = |p: [f64; 3]| vec3(
        (p[0] - origin[0]) as f32,
        (p[1] - origin[1]) as f32,
        (p[2] - origin[2]) as f32,
    );
    let color = marker_color(dig, hit.quality);

    let c = to_view(hit.world);
    let r = dig.radius;
    let arm = r * 0.6;
    draw_line_3d(c - vec3(arm, 0.0, 0.0), c + vec3(arm, 0.0, 0.0), color);
    draw_line_3d(c - vec3(0.0, arm, 0.0), c + vec3(0.0, arm, 0.0), color);
    draw_line_3d(c - vec3(0.0, 0.0, arm), c + vec3(0.0, 0.0, arm), color);

    const SEGMENTS: usize = 40;
    let faint = Color::new(color.r, color.g, color.b, 0.5);
    let mut prev = vec3(c.x + r, c.y, c.z);
    for i in 1..=SEGMENTS {
        let a = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
        let next = vec3(c.x + r * a.cos(), c.y + r * a.sin(), c.z);
        draw_line_3d(prev, next, faint);
        prev = next;
    }
}

// ---- The section view ----
//
// Digitising in a section works in (along, height): the space the operator is
// looking at. A vertex lands **on the section plane**, at the along-position
// and height it was snapped to — even when the point it snapped to sits some
// way off the plane inside the corridor. The section view shows nothing about
// that offset, so a vertex that kept it would be a coordinate the operator
// never saw. The existing-polyline snap is the one exception; see
// `polylines::nearest_in_section`.
//
// The snap queries run over the section's own points, which are exactly the
// ones on screen: the height band and hidden classes were already applied when
// the section was cut.

/// A vertex on the section plane, in world coordinates, from a section-space
/// position. `start` / `dir` are the section line in viewer space.
pub fn section_to_world(
    start: (f32, f32),
    dir: (f32, f32),
    along: f32,
    height: f32,
    origin: [f64; 3],
) -> [f64; 3] {
    [
        start.0 as f64 + dir.0 as f64 * along as f64 + origin[0],
        start.1 as f64 + dir.1 as f64 * along as f64 + origin[1],
        height as f64 + origin[2],
    ]
}

/// A world position in section space, as `(along, height)`.
pub fn world_to_section(
    start: (f32, f32),
    dir: (f32, f32),
    p: [f64; 3],
    origin: [f64; 3],
) -> (f32, f32) {
    let rx = (p[0] - origin[0]) as f32 - start.0;
    let ry = (p[1] - origin[1]) as f32 - start.1;
    (rx * dir.0 + ry * dir.1, (p[2] - origin[2]) as f32)
}

/// The section point nearest to `(along, height)`, within `r` metres of it on
/// screen — the section view's counterpart of the nearest-to-ray snap.
pub fn section_nearest(
    grid: &SpatialGrid,
    pts: &[[f32; 3]],
    along: f32,
    height: f32,
    r: f32,
) -> Option<(f32, f32)> {
    let mut cands = Vec::new();
    grid.points_in_radius(pts, along, height, r, NO_FILTER, &mut cands);
    let d2 = |p: &[f32; 3]| (p[0] - along).powi(2) + (p[1] - height).powi(2);
    cands.iter()
        .map(|&i| pts[i as usize])
        .min_by(|a, b| d2(a).partial_cmp(&d2(b)).unwrap_or(std::cmp::Ordering::Equal))
        .map(|p| (p[0], p[1]))
}

/// The `pct`-th percentile of height among the section points within `r`
/// metres of `along`, whatever their height.
///
/// A vertical strip, which is what the 3D view's radius column becomes when the
/// view is a slice: "the lowest data within 0.2 m of here" means the whole
/// strip, not a disc around the cursor, or a low percentile would miss the
/// ground whenever the cursor was held above it.
pub fn section_percentile(
    grid: &SpatialGrid,
    pts: &[[f32; 3]],
    along: f32,
    r: f32,
    pct: f32,
) -> Option<(f32, f32)> {
    let (lo, hi) = grid.bounds_xy();
    let mut zs: Vec<(f32, f32)> = Vec::new();
    grid.cells_in_box([along - r, lo[1]], [along + r, hi[1]], |cell| {
        for &i in cell.points {
            let p = pts[i as usize];
            if (p[0] - along).abs() <= r { zs.push((p[1], p[0])); }
        }
    });
    percentile_by_z(&mut zs, pct).map(|(h, a)| (a, h))
}

/// The line in progress, the rubber band and the snap marker, in the section
/// view. Screen space, under the default camera; `to_screen` maps
/// `(along, height)` to window pixels and `px_per_m` is the view's scale.
pub fn draw_section_overlay(
    dig: &Digitise,
    start: (f32, f32),
    dir: (f32, f32),
    origin: [f64; 3],
    to_screen: impl Fn(f32, f32) -> (f32, f32),
    px_per_m: f32,
) {
    let pts: Vec<(f32, f32)> = dig.pending.iter()
        .map(|p| {
            let (a, h) = world_to_section(start, dir, *p, origin);
            to_screen(a, h)
        })
        .collect();
    let width = dig.width_px.max(1.0);
    for w in pts.windows(2) {
        draw_line(w[0].0, w[0].1, w[1].0, w[1].1, width, dig.color);
    }
    for (x, y) in &pts {
        draw_rectangle_lines(x - 4.0, y - 4.0, 8.0, 8.0, 1.5, dig.color);
    }

    let Some(hit) = &dig.hover else { return };
    let (a, h) = world_to_section(start, dir, hit.world, origin);
    let (cx, cy) = to_screen(a, h);
    if hit.is_placeable()
        && let Some(&(lx, ly)) = pts.last() {
        let faint = Color::new(dig.color.r, dig.color.g, dig.color.b, 0.55);
        draw_line(lx, ly, cx, cy, width, faint);
    }
    let color = marker_color(dig, hit.quality);
    let arm = 9.0;
    draw_line(cx - arm, cy, cx + arm, cy, 1.5, color);
    draw_line(cx, cy - arm, cx, cy + arm, 1.5, color);
    // The radius at the view's scale, so what it encloses is what a snap can
    // reach. For the percentile modes the strip is what counts, so mark its
    // edges too.
    let r_px = dig.radius * px_per_m;
    draw_circle_lines(cx, cy, r_px.max(2.0), 1.0, Color::new(color.r, color.g, color.b, 0.5));
    if matches!(dig.snap_mode, SnapMode::LowZ | SnapMode::HighZ) {
        let faint = Color::new(color.r, color.g, color.b, 0.25);
        for x in [cx - r_px, cx + r_px] {
            draw_line(x, cy - 3.0 * r_px.max(8.0), x, cy + 3.0 * r_px.max(8.0), 1.0, faint);
        }
    }
}

/// Distance and height difference to the previous vertex, drawn beside the
/// cursor.
///
/// The panel carries the same numbers, but reading them means looking away from
/// the point being placed — and the whole job is placing it precisely. Drawn
/// under the default camera, in screen space, so the text stays the same size
/// at any zoom.
pub fn draw_cursor_readout(dig: &Digitise, mx: f32, my: f32) {
    let Some(hit) = &dig.hover else { return };

    const SIZE: f32 = 16.0;
    let mut lines: Vec<(String, Color)> = Vec::new();
    let warn = Color::new(1.0, 0.65, 0.25, 1.0);

    if hit.is_placeable() {
        if let Some(prev) = dig.last() {
            let run = ((hit.world[0] - prev[0]).powi(2)
                     + (hit.world[1] - prev[1]).powi(2)).sqrt();
            let dz = hit.world[2] - prev[2];
            let d = (run * run + dz * dz).sqrt();
            lines.push((format!("{d:.3} m   ΔZ {dz:+.3} m"), WHITE));
            if run > 0.0 {
                lines.push((format!("{:+.1}%   plan {run:.3} m", dz / run * 100.0), LIGHTGRAY));
            } else {
                lines.push(("vertical".to_string(), LIGHTGRAY));
            }
            let (total, _) = dig.lengths();
            lines.push((
                format!("total {:.3} m ({} pts)", total + d, dig.pending.len() + 1),
                LIGHTGRAY,
            ));
        } else {
            // Nothing to measure from yet, but the height being committed to is
            // still worth seeing before the click.
            lines.push((format!("Z {:.3} m", hit.world[2]), WHITE));
        }
    } else {
        // A refusal is more use beside the cursor than buried in the panel.
        let note = hit.note.clone().unwrap_or_else(|| "Cannot snap here.".to_string());
        for chunk in wrap(&note, 44) { lines.push((chunk, warn)); }
    }
    if lines.is_empty() { return; }

    let w = lines.iter()
        .map(|(t, _)| measure_text(t, None, SIZE as u16, 1.0).width)
        .fold(0.0_f32, f32::max);
    let h = lines.len() as f32 * (SIZE + 2.0);

    // Keep the box on screen and out from under the cursor.
    let mut x = mx + 18.0;
    let mut y = my + 18.0;
    if x + w + 12.0 > screen_width() { x = mx - w - 24.0; }
    if y + h + 10.0 > screen_height() { y = my - h - 18.0; }

    draw_rectangle(x - 6.0, y - 4.0, w + 12.0, h + 8.0, Color::new(0.0, 0.0, 0.0, 0.62));
    for (i, (text, color)) in lines.iter().enumerate() {
        draw_text(text, x, y + SIZE + i as f32 * (SIZE + 2.0) - 3.0, SIZE, *color);
    }
}

/// Break a message into lines of at most `width` characters, on word
/// boundaries.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() { line.push(' '); }
        line.push_str(word);
    }
    if !line.is_empty() { out.push(line); }
    out
}

/// The digitising panel, called from all three per-mode egui blocks.
///
/// Shaped like `ground_tool_controls`: one function, three call sites, no
/// triplication.
pub fn controls(
    ui: &mut egui_macroquad::egui::Ui,
    dig: &mut Digitise,
    files: &[PolylineFile],
    drawing: bool,
    have_cloud: bool,
) -> Option<Request> {
    use egui_macroquad::egui;
    let mut request = None;

    ui.separator();
    ui.label("Digitise polylines:");

    // Target layer. Every layer of every loaded file is a candidate, so a line
    // can be added to a GeoPackage that came from QGIS as easily as to the
    // sidecar.
    ui.horizontal(|ui| {
        ui.label("Into:");
        let selected = match dig.target {
            Some((fi, li)) => target_label(files, fi, li),
            None => format!("new layer beside the cloud{}", if have_cloud { "" } else { " (none loaded)" }),
        };
        egui::ComboBox::from_id_salt("digitise_target")
            .selected_text(selected)
            .width(220.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut dig.target, None, "new layer beside the cloud");
                for (fi, f) in files.iter().enumerate() {
                    for (li, _) in f.layers.iter().enumerate() {
                        ui.selectable_value(
                            &mut dig.target, Some((fi, li)), target_label(files, fi, li),
                        );
                    }
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("Name:");
        ui.add(egui::TextEdit::singleline(&mut dig.name_template).desired_width(120.0))
            .on_hover_text("{n} becomes the line's number in the layer");
        ui.label("Code:");
        ui.add(egui::TextEdit::singleline(&mut dig.code).desired_width(60.0))
            .on_hover_text("Written to the `code` attribute");
    });

    ui.horizontal(|ui| {
        ui.label("Snap:");
        egui::ComboBox::from_id_salt("digitise_snap")
            .selected_text(dig.snap_mode.label())
            .width(160.0)
            .show_ui(ui, |ui| {
                for m in SnapMode::ALL {
                    ui.selectable_value(&mut dig.snap_mode, m, m.label());
                }
            });
        ui.label("(N)");
    });
    ui.horizontal(|ui| {
        ui.label("Radius:");
        ui.add(egui::DragValue::new(&mut dig.radius)
            .speed(0.01)
            .range(MIN_SNAP_RADIUS..=MAX_SNAP_RADIUS)
            .max_decimals(2)
            .suffix(" m"))
            .on_hover_text("Ctrl + mouse wheel while digitising");
    });
    if matches!(dig.snap_mode, SnapMode::LowZ | SnapMode::HighZ) {
        ui.horizontal(|ui| {
            let (label, pct) = match dig.snap_mode {
                SnapMode::LowZ => ("Low percentile:", &mut dig.low_pct),
                _ => ("High percentile:", &mut dig.high_pct),
            };
            ui.label(label);
            ui.add(egui::Slider::new(pct, 0.0..=100.0).max_decimals(0))
                .on_hover_text("0 and 100 give the raw minimum and maximum back");
        });
    }

    ui.horizontal(|ui| {
        let mut c = egui::Color32::from_rgb(
            (dig.color.r * 255.0) as u8, (dig.color.g * 255.0) as u8, (dig.color.b * 255.0) as u8,
        );
        if ui.color_edit_button_srgba(&mut c).changed() {
            dig.color = Color::new(
                c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, 1.0,
            );
        }
        ui.label("line colour");
        ui.add(egui::DragValue::new(&mut dig.width_px)
            .speed(0.2).range(0.5..=12.0).max_decimals(1).suffix(" px"))
            .on_hover_text("Drawn width in the 3D view");
    });

    if drawing {
        let (total, last) = dig.lengths();
        ui.label(format!("{} vertices, {total:.3} m total", dig.pending.len()));
        if dig.pending.len() >= 2 {
            match dig.last_slope() {
                Some((slope, dz)) => ui.label(format!(
                    "last segment {last:.3} m, ΔZ {dz:+.3} m, {slope:+.1}%",
                )),
                None => ui.label(format!("last segment {last:.3} m, vertical")),
            };
        }
        if let Some(hit) = &dig.hover {
            match hit.quality {
                SnapQuality::Good => {
                    let z = hit.world[2];
                    if let Some(prev) = dig.last() {
                        let run = ((hit.world[0] - prev[0]).powi(2)
                                 + (hit.world[1] - prev[1]).powi(2)).sqrt();
                        let dz = z - prev[2];
                        let d = (run * run + dz * dz).sqrt();
                        let slope = if run > 0.0 {
                            format!("{:+.1}%", dz / run * 100.0)
                        } else {
                            "vertical".to_string()
                        };
                        ui.label(format!("→ Z {z:.3} m, {d:.3} m away, ΔZ {dz:+.3} m, {slope}"));
                    } else {
                        ui.label(format!("→ Z {z:.3} m"));
                    }
                }
                _ => {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 140, 60),
                        hit.note.clone().unwrap_or_else(|| "Cannot snap here.".to_string()),
                    );
                }
            }
        }
        ui.label("Click to add a vertex, Shift-click to place one freely.");
        ui.label("Backspace undoes one, Enter finishes, Esc cancels.");
        ui.horizontal(|ui| {
            if ui.add_enabled(dig.pending.len() >= 2, egui::Button::new("Finish (Enter)")).clicked() {
                request = Some(Request::Finish);
            }
            if ui.add_enabled(!dig.pending.is_empty(), egui::Button::new("Undo vertex")).clicked() {
                request = Some(Request::UndoVertex);
            }
            if ui.button("Cancel (Esc)").clicked() { request = Some(Request::Cancel); }
        });
    } else {
        if dig.committed_lines > 0 {
            ui.label(format!("{} line(s) drawn this session", dig.committed_lines));
        }
        if ui.add_enabled(have_cloud, egui::Button::new("Digitise (P)")).clicked() {
            request = Some(Request::Start);
        }
    }

    if let Some(msg) = &dig.status {
        ui.label(msg.as_str());
    }
    request
}

/// Strip the vector-sidecar suffix, for a status line that says where a layer
/// went without repeating the whole path.
pub fn short_path(path: &str) -> &str {
    Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files_with_layer() -> Vec<PolylineFile> {
        let mut pf = PolylineFile::new("/tmp/haiviz-vectorise-test.gpkg", VectorFormat::GeoPackage);
        pf.layers.push(Layer::new("digitised"));
        pf.refresh_view([0.0; 3]);
        vec![pf]
    }

    #[test]
    fn a_finished_line_lands_in_the_target_layer() {
        let mut files = files_with_layer();
        let mut dig = Digitise { target: Some((0, 0)), ..Default::default() };
        dig.push([208_800.0, 373_300.0, 12.0]);
        dig.push([208_810.0, 373_300.0, 12.5]);

        assert_eq!(dig.finish(&mut files, false), Some(0), "the caller is told which file to save");
        assert_eq!(files[0].layers[0].polylines.len(), 1);
        assert_eq!(files[0].layers[0].polylines[0].name, "Line 1");
        assert!(dig.pending.is_empty(), "finishing clears the pending line");
        assert!(files[0].dirty, "and marks the file for the auto-save");
    }

    #[test]
    fn the_name_template_numbers_lines_within_the_layer() {
        let mut files = files_with_layer();
        let mut dig = Digitise {
            target: Some((0, 0)),
            name_template: "kerb {n}".to_string(),
            code: "KRB".to_string(),
            ..Default::default()
        };
        for _ in 0..3 {
            dig.push([0.0, 0.0, 0.0]);
            dig.push([1.0, 1.0, 1.0]);
            dig.finish(&mut files, false);
        }
        let names: Vec<&str> = files[0].layers[0].polylines.iter()
            .map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["kerb 1", "kerb 2", "kerb 3"]);
        assert_eq!(files[0].layers[0].polylines[0].attrs.get("code").map(String::as_str), Some("KRB"));
        assert_eq!(dig.committed_lines, 3);
    }

    #[test]
    fn a_single_vertex_is_not_a_line() {
        let mut files = files_with_layer();
        let mut dig = Digitise { target: Some((0, 0)), ..Default::default() };
        dig.push([0.0, 0.0, 0.0]);
        assert_eq!(dig.finish(&mut files, false), None);
        assert!(dig.status.is_some(), "and the operator is told why");
        assert_eq!(files[0].layers[0].polylines.len(), 0);
        // The vertex is kept, not discarded — Enter by accident must not cost
        // the click that was already made.
        assert_eq!(dig.pending.len(), 1);
    }

    #[test]
    fn the_readout_measures_what_was_drawn() {
        let mut dig = Digitise::default();
        dig.push([0.0, 0.0, 0.0]);
        dig.push([3.0, 4.0, 0.0]);
        dig.push([3.0, 4.0, 1.0]);
        // 5 m in plan, then 1 m straight up.
        let (total, last) = dig.lengths();
        assert!((total - 6.0).abs() < 1e-9, "{total}");
        assert!((last - 1.0).abs() < 1e-9);
        assert_eq!(dig.last_slope(), None, "a vertical segment has no slope");

        dig.pop();
        let (_, last) = dig.lengths();
        assert!((last - 5.0).abs() < 1e-9);
        let (slope, dz) = dig.last_slope().unwrap();
        assert!(slope.abs() < 1e-9 && dz.abs() < 1e-9);
    }

    #[test]
    fn cycling_snap_modes_visits_every_one_and_comes_back() {
        let mut m = SnapMode::Nearest;
        let mut seen = vec![m];
        for _ in 0..SnapMode::ALL.len() - 1 {
            m = m.next();
            assert!(!seen.contains(&m), "{m:?} repeated before the cycle closed");
            seen.push(m);
        }
        assert_eq!(seen.len(), SnapMode::ALL.len());
        assert_eq!(m.next(), SnapMode::Nearest, "N must wrap, not stop");
    }

    #[test]
    fn a_snap_that_is_not_good_is_not_placeable() {
        // The whole point of the quality flag: a coarse snap still has a
        // position, and drawing the marker there is what tells the operator
        // where the problem is — but it must not be clickable.
        let coarse = SnapHit {
            world: [1.0, 2.0, 3.0],
            quality: SnapQuality::Coarse,
            note: Some("too sparse".to_string()),
        };
        assert!(!coarse.is_placeable());
        assert!(SnapHit::good([1.0, 2.0, 3.0]).is_placeable());
    }

    #[test]
    fn a_new_layer_lands_beside_the_cloud() {
        let cloud = "/data/site/scan.copc.laz";
        assert_eq!(default_vector_path(cloud), "/data/site/scan.copc.laz.haiviz-vectors.gpkg");

        let mut files: Vec<PolylineFile> = Vec::new();
        let (fi, li) = ensure_target(&mut files, cloud, 28992, 2.0);
        assert_eq!((fi, li), (0, 0));
        assert_eq!(files[0].format, VectorFormat::GeoPackage);
        assert_eq!(files[0].layers[0].srs_id, 28992);

        // Asking again reuses it rather than forking the file.
        let again = ensure_target(&mut files, cloud, 28992, 2.0);
        assert_eq!(again, (0, 0));
        assert_eq!(files.len(), 1);
    }
}
