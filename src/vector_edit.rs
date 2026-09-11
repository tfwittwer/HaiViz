//! Editing lines that already exist.
//!
//! Digitising (`vectorise.rs`) makes new lines; this changes the ones already
//! there — moving, inserting and deleting vertices, deleting whole lines,
//! closing and opening rings, and re-draping a line onto the cloud. As with
//! digitising, `main.rs` holds only the wiring, and a vertex placed here goes
//! through exactly the same snap as a new one.
//!
//! ## Drag, not click
//!
//! A vertex is moved by dragging it, and a new one is pulled out of a segment
//! the same way. A press released without travelling only *selects*: otherwise
//! clicking a vertex to read its coordinates would re-snap it to whatever sits a
//! pixel away, and quietly change the line.
//!
//! ## Undo and saving
//!
//! Every committed edit goes on its file's own inverse stack (`polylines.rs`)
//! and is auto-saved at once — one save per completed change, the granularity
//! digitising already uses. A drag is one change however many frames it lasted,
//! because nothing reaches the model until the button comes up.
//!
//! Each file keeps its own stack, so this module keeps the *order* edits were
//! made across files. Ctrl+Z undoes the most recent change wherever it was,
//! not the most recent change in whichever file happens to be selected.

use macroquad::prelude::*;

use crate::polylines::{PolyRef, PolylineFile, SnapExclude};
use crate::vectorise::SnapMode;

/// How close, in screen pixels, the cursor must be to grab a vertex.
pub const VERTEX_PICK_PX: f32 = 10.0;
/// How close to grab a segment. Tighter than a vertex, so that near a corner
/// the vertex wins without the operator having to aim for it.
pub const SEGMENT_PICK_PX: f32 = 7.0;
/// How far the pointer must travel with the button held before a press becomes
/// a drag. Below it, releasing is a click, which only selects.
pub const DRAG_THRESHOLD_PX: f32 = 4.0;

/// Fewest vertices a line may keep: two for an open line, three for a ring. A
/// two-vertex ring is a line drawn back over itself, and a GeoPackage stores it
/// as exactly that.
pub fn min_vertices(closed: bool) -> usize {
    if closed { 3 } else { 2 }
}

/// Something under the cursor that can be grabbed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Handle {
    Vertex { line: PolyRef, idx: usize },
    /// A segment by its two vertex indices, and the fraction along it that was
    /// grabbed. A vertex pulled out of it is inserted at `i + 1` — which is
    /// also right for a ring's closing segment `(n - 1, 0)`, where it appends.
    Segment { line: PolyRef, i: usize, j: usize, t: f32 },
}

impl Handle {
    pub fn line(&self) -> PolyRef {
        match *self {
            Handle::Vertex { line, .. } | Handle::Segment { line, .. } => line,
        }
    }

    /// The part of the line a snap made while dragging this must ignore.
    pub fn exclude(&self) -> SnapExclude {
        match *self {
            Handle::Vertex { line, idx } => SnapExclude { line, vertex: Some(idx), segment: None },
            Handle::Segment { line, i, j, .. } => SnapExclude { line, vertex: None, segment: Some((i, j)) },
        }
    }

    /// Where the handle is in the world: the vertex, or the grabbed point on
    /// the segment.
    pub fn world(&self, files: &[PolylineFile]) -> Option<[f64; 3]> {
        let pl = files.get(self.line().file)?.get(self.line().line_ref())?;
        match *self {
            Handle::Vertex { idx, .. } => pl.points().get(idx).copied(),
            Handle::Segment { i, j, t, .. } => {
                let (a, b) = (*pl.points().get(i)?, *pl.points().get(j)?);
                let t = t as f64;
                Some([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])])
            }
        }
    }
}

/// The handle under the cursor, if any. Vertices first: a corner is where both
/// a vertex and two segments are in reach, and the corner is what is meant.
///
/// `mx` / `my` are relative to the view rect, `sw` / `sh` its size.
pub fn hit_test(
    files: &[PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    sw: f32,
    sh: f32,
) -> Option<Handle> {
    if let Some((line, idx)) =
        crate::polylines::pick_vertex(files, view_proj, mx, my, sw, sh, VERTEX_PICK_PX) {
        return Some(Handle::Vertex { line, idx });
    }
    crate::polylines::pick_segment(files, view_proj, mx, my, sw, sh, SEGMENT_PICK_PX)
        .map(|(line, i, j, t)| Handle::Segment { line, i, j, t })
}

/// A press on a handle, possibly turning into a drag.
#[derive(Clone, Debug)]
pub struct Drag {
    pub handle: Handle,
    /// Where the button went down, in window pixels.
    pub press: (f32, f32),
    /// Past [`DRAG_THRESHOLD_PX`]. Until then, releasing is a click.
    pub moving: bool,
    /// The handle's world position when grabbed. Shift-drag keeps its height,
    /// and "same Z as previous" holds it.
    pub origin: [f64; 3],
}

impl Drag {
    /// Follow the pointer. Once a press has become a drag it stays one, even if
    /// the pointer comes back to where it started.
    pub fn track(&mut self, mx: f32, my: f32) {
        let (dx, dy) = (mx - self.press.0, my - self.press.1);
        if dx * dx + dy * dy > DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX {
            self.moving = true;
        }
    }
}

/// What the panel is asking `main.rs` to do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Request {
    Start,
    Stop,
    DeleteVertex,
    DeleteLine,
    ToggleRing,
    Redrape,
    Undo,
}

#[derive(Default)]
pub struct Editor {
    /// The selected vertex, with the line it belongs to. The selected *line* is
    /// `selected_poly` in `main.rs`, shared with the Polylines panel so that
    /// selecting a line there and editing it here are one selection; carrying
    /// the line here too is what stops a vertex index from silently moving to
    /// another line when the selection changes under it.
    pub vertex: Option<(PolyRef, usize)>,
    pub hover: Option<Handle>,
    pub drag: Option<Drag>,
    /// File indices in the order edits were committed. See the module note.
    undo_order: Vec<usize>,
    pub status: Option<String>,
}

impl Editor {
    /// Note that file `fi` just took an edit, so Ctrl+Z can find it.
    pub fn record(&mut self, fi: usize) {
        self.undo_order.push(fi);
    }

    pub fn can_undo(&self, files: &[PolylineFile]) -> bool {
        self.undo_order.iter().any(|&fi| files.get(fi).is_some_and(|f| f.can_undo()))
    }

    /// Undo the most recent change in any file. Returns the file that changed,
    /// for the auto-save.
    pub fn undo(&mut self, files: &mut [PolylineFile]) -> Option<usize> {
        while let Some(fi) = self.undo_order.pop() {
            if files.get_mut(fi).is_some_and(|f| f.undo()) {
                return Some(fi);
            }
        }
        None
    }

    /// Keep the undo order pointing at the right files after file `fi` is
    /// unloaded. Its own history goes with it.
    pub fn file_removed(&mut self, fi: usize) {
        self.undo_order.retain(|&f| f != fi);
        for f in &mut self.undo_order {
            if *f > fi { *f -= 1; }
        }
        self.hover = None;
        self.drag = None;
    }

    /// Drop anything that no longer points at a line: after an undo removed
    /// the line, a vertex delete shortened it, or its file was unloaded.
    pub fn validate(&mut self, files: &[PolylineFile], selected: &mut Option<PolyRef>) {
        let len_of = |at: PolyRef| files.get(at.file).and_then(|f| f.get(at.line_ref())).map(|l| l.len());
        if selected.is_some_and(|at| len_of(at).is_none()) { *selected = None; }
        if self.vertex.is_some_and(|(line, i)| Some(line) != *selected || len_of(line).is_none_or(|n| i >= n)) {
            self.vertex = None;
        }
        let handle_valid = |h: &Handle| match *h {
            Handle::Vertex { line, idx } => len_of(line).is_some_and(|n| idx < n),
            Handle::Segment { line, i, j, .. } => len_of(line).is_some_and(|n| i < n && j < n),
        };
        if self.hover.as_ref().is_some_and(|h| !handle_valid(h)) { self.hover = None; }
        if self.drag.as_ref().is_some_and(|d| !handle_valid(&d.handle)) { self.drag = None; }
    }

    /// Commit a drag: move the vertex, or insert a new one, at `v`.
    pub fn commit_drag(&mut self, files: &mut [PolylineFile], handle: Handle, v: [f64; 3]) -> Option<usize> {
        let line = handle.line();
        let pf = files.get_mut(line.file)?;
        pf.get(line.line_ref())?;
        match handle {
            Handle::Vertex { idx, .. } => {
                pf.move_vertex(line.line_ref(), idx, v);
                self.vertex = Some((line, idx));
            }
            Handle::Segment { i, .. } => {
                pf.insert_vertex(line.line_ref(), i + 1, v);
                self.vertex = Some((line, i + 1));
            }
        }
        self.record(line.file);
        Some(line.file)
    }

    /// Delete one vertex, refusing to take a line below [`min_vertices`] —
    /// deleting the line is a separate, deliberate action.
    pub fn delete_vertex(
        &mut self,
        files: &mut [PolylineFile],
        line: PolyRef,
        idx: usize,
    ) -> Result<usize, String> {
        let pf = files.get_mut(line.file).ok_or("That line is no longer loaded.")?;
        let pl = pf.get(line.line_ref()).ok_or("That line is no longer loaded.")?;
        if pl.len() <= min_vertices(pl.closed) {
            return Err(format!(
                "A {} needs at least {} vertices — use Delete line to remove it.",
                if pl.closed { "ring" } else { "line" },
                min_vertices(pl.closed),
            ));
        }
        pf.remove_vertex(line.line_ref(), idx).ok_or("No such vertex.")?;
        self.vertex = None;
        self.record(line.file);
        Ok(line.file)
    }

    pub fn delete_line(&mut self, files: &mut [PolylineFile], line: PolyRef) -> Option<usize> {
        files.get_mut(line.file)?.remove_line(line.line_ref())?;
        self.vertex = None;
        self.record(line.file);
        Some(line.file)
    }

    /// Close an open line into a ring, or open a ring.
    pub fn toggle_ring(&mut self, files: &mut [PolylineFile], line: PolyRef) -> Result<usize, String> {
        let pf = files.get_mut(line.file).ok_or("That line is no longer loaded.")?;
        let pl = pf.get(line.line_ref()).ok_or("That line is no longer loaded.")?;
        let closing = !pl.closed;
        if closing && pl.len() < min_vertices(true) {
            return Err("A ring needs at least three vertices.".to_string());
        }
        pf.set_closed(line.line_ref(), closing);
        self.record(line.file);
        Ok(line.file)
    }

    /// Replace a line's vertices as one undoable change — a re-drape.
    pub fn replace_points(
        &mut self,
        files: &mut [PolylineFile],
        line: PolyRef,
        points: Vec<[f64; 3]>,
    ) -> Option<usize> {
        let pf = files.get_mut(line.file)?;
        pf.get(line.line_ref())?;
        pf.set_points(line.line_ref(), points);
        self.record(line.file);
        Some(line.file)
    }
}

// ---- Re-drape ----

/// Why a vertex kept its old height in a re-drape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DrapeMiss {
    /// Nothing within the snap radius.
    Empty,
    /// The resident COPC data is thinner than the radius; see the spacing guard
    /// in `resolve_snap`.
    Coarse,
}

/// The result of re-draping a line.
#[derive(Clone, Debug, PartialEq)]
pub struct Drape {
    pub points: Vec<[f64; 3]>,
    /// Vertices whose height actually changed.
    pub moved: usize,
    pub empty: usize,
    pub coarse: usize,
}

/// Re-drape a line: keep every vertex's **plan position** and ask `z_at` for
/// its height.
///
/// Plan position, because it is what the operator placed and what a change of
/// snap mode is not about — a line switched from low to high percentile should
/// rise onto the kerb top, not wander sideways by up to a radius onto whichever
/// point happened to realise the percentile. A vertex `z_at` cannot answer for
/// keeps its height and is counted, so the result says what it skipped rather
/// than leaving a line half-draped in silence.
pub fn drape(
    points: &[[f64; 3]],
    mut z_at: impl FnMut([f64; 3]) -> Result<f64, DrapeMiss>,
) -> Drape {
    let mut out = Drape { points: Vec::with_capacity(points.len()), moved: 0, empty: 0, coarse: 0 };
    for &p in points {
        match z_at(p) {
            Ok(z) => {
                if z != p[2] { out.moved += 1; }
                out.points.push([p[0], p[1], z]);
            }
            Err(miss) => {
                match miss {
                    DrapeMiss::Empty => out.empty += 1,
                    DrapeMiss::Coarse => out.coarse += 1,
                }
                out.points.push(p);
            }
        }
    }
    out
}

impl Drape {
    pub fn summary(&self, mode: SnapMode) -> String {
        let n = self.points.len();
        let mut s = format!("Re-draped to {}: {} of {n} vertices moved", mode.label(), self.moved);
        if self.empty > 0 {
            s.push_str(&format!("; {} had nothing within the radius", self.empty));
        }
        if self.coarse > 0 {
            s.push_str(&format!(
                "; {} kept their height because the loaded data there is too sparse — \
                 zoom in along the line and try again", self.coarse,
            ));
        }
        s.push('.');
        s
    }
}

/// The height a snap mode gives one vertex, from the candidates within the
/// radius of its plan position (viewer space, as `(z, position)`).
///
/// `Nearest` takes the candidate closest in **3D**, not in plan: under a bridge
/// both decks are within the radius in plan, and the one the vertex already
/// sits on is the one that was meant.
pub fn drape_z(
    mode: SnapMode,
    vertex: [f32; 3],
    candidates: &mut [(f32, [f32; 3])],
    low_pct: f32,
    high_pct: f32,
) -> Option<f32> {
    match mode {
        SnapMode::Nearest => {
            let d2 = |p: &[f32; 3]| (p[0] - vertex[0]).powi(2)
                + (p[1] - vertex[1]).powi(2) + (p[2] - vertex[2]).powi(2);
            candidates.iter()
                .min_by(|a, b| d2(&a.1).partial_cmp(&d2(&b.1)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|c| c.0)
        }
        SnapMode::LowZ => crate::spatial::percentile_by_z(candidates, low_pct).map(|c| c.0),
        SnapMode::HighZ => crate::spatial::percentile_by_z(candidates, high_pct).map(|c| c.0),
        SnapMode::SameZ | SnapMode::Existing => None,
    }
}

// ---- Drawing ----

/// Window pixel for a viewer-space position. `rect` is the view's
/// `(x, y, w, h)`.
fn to_screen(vp: &glam::Mat4, p: [f32; 3], rect: (f32, f32, f32, f32)) -> Option<(f32, f32)> {
    let clip = *vp * glam::Vec4::new(p[0], p[1], p[2], 1.0);
    if clip.w.abs() < 1e-6 { return None; }
    let (nx, ny) = (clip.x / clip.w, clip.y / clip.w);
    Some((rect.0 + (nx + 1.0) * 0.5 * rect.2, rect.1 + (1.0 - ny) * 0.5 * rect.3))
}

/// The line as it will be when the drag is released, under the 3D camera, so
/// it is depth-tested like every other line.
pub fn draw_drag_preview(
    ed: &Editor,
    files: &[PolylineFile],
    origin: [f64; 3],
    view_dir: glam::Vec3,
    metres_per_pixel: f32,
    target: Option<[f64; 3]>,
) {
    let (Some(drag), Some(target)) = (&ed.drag, target) else { return };
    if !drag.moving { return; }
    let line = drag.handle.line();
    let Some(pf) = files.get(line.file) else { return };
    let Some(pl) = pf.get(line.line_ref()) else { return };
    let width_px = pf.layers.get(line.layer).map(|l| l.width_px).unwrap_or(2.0).max(0.5);

    let mut pts: Vec<[f64; 3]> = pl.points().to_vec();
    match drag.handle {
        Handle::Vertex { idx, .. } => if let Some(p) = pts.get_mut(idx) { *p = target; },
        Handle::Segment { i, .. } => pts.insert((i + 1).min(pts.len()), target),
    }
    let view: Vec<glam::Vec3> = pts.iter().map(|p| glam::Vec3::new(
        (p[0] - origin[0]) as f32, (p[1] - origin[1]) as f32, (p[2] - origin[2]) as f32,
    )).collect();
    let c = pl.color;
    crate::polylines::draw_ribbon_3d(
        &view, pl.closed, width_px * metres_per_pixel,
        Color::new(c.r, c.g, c.b, 0.55), view_dir,
    );
}

/// Vertex handles, in screen space under the default camera, so they stay one
/// size at any zoom and are never hidden by the cloud — a handle that can be
/// occluded cannot be grabbed.
pub fn draw_handles(
    ed: &Editor,
    files: &[PolylineFile],
    selected: Option<PolyRef>,
    view_proj: &glam::Mat4,
    rect: (f32, f32, f32, f32),
) {
    let fill = Color::new(0.0, 0.0, 0.0, 0.6);
    let accent = Color::new(1.0, 0.85, 0.2, 1.0);

    if let Some(at) = selected
        && let Some(pl) = files.get(at.file).and_then(|f| f.get(at.line_ref())) {
        for (i, v) in pl.view().iter().enumerate() {
            let Some((x, y)) = to_screen(view_proj, *v, rect) else { continue };
            let selected_vertex = ed.vertex == Some((at, i));
            let s = if selected_vertex { 5.0 } else { 3.5 };
            draw_rectangle(x - s, y - s, 2.0 * s, 2.0 * s, if selected_vertex { accent } else { fill });
            draw_rectangle_lines(x - s, y - s, 2.0 * s, 2.0 * s, 1.5, WHITE);
        }
    }

    // What a press would grab. Not while dragging: the snap marker is what
    // matters then.
    if ed.drag.as_ref().is_some_and(|d| d.moving) { return; }
    let Some(handle) = ed.hover else { return };
    let Some(pl) = files.get(handle.line().file).and_then(|f| f.get(handle.line().line_ref())) else { return };
    match handle {
        Handle::Vertex { idx, .. } => {
            if let Some((x, y)) = pl.view().get(idx).and_then(|v| to_screen(view_proj, *v, rect)) {
                draw_circle_lines(x, y, 8.0, 2.0, accent);
            }
        }
        Handle::Segment { i, j, t, .. } => {
            let (Some(a), Some(b)) = (
                pl.view().get(i).and_then(|v| to_screen(view_proj, *v, rect)),
                pl.view().get(j).and_then(|v| to_screen(view_proj, *v, rect)),
            ) else { return };
            let (x, y) = (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1));
            // A plus: pulling here makes a new vertex.
            draw_circle(x, y, 7.0, fill);
            draw_line(x - 5.0, y, x + 5.0, y, 2.0, accent);
            draw_line(x, y - 5.0, x, y + 5.0, 2.0, accent);
        }
    }
}

// ---- Panel ----

/// The editing controls, called from the same place as the digitising panel
/// in all three per-mode blocks.
pub fn controls(
    ui: &mut egui_macroquad::egui::Ui,
    ed: &Editor,
    files: &[PolylineFile],
    selected: Option<PolyRef>,
    editing: bool,
    snap_mode: SnapMode,
) -> Option<Request> {
    use egui_macroquad::egui;
    let mut request = None;

    ui.separator();
    ui.label("Edit polylines:");
    if editing {
        ui.label("Drag a vertex to move it, or a segment to add one.");
        ui.label("Both snap like digitising; hold Shift to place freely.");
        ui.label("Click to select. Delete removes the vertex, or the line.");
        if ui.button("Stop editing (M)").clicked() { request = Some(Request::Stop); }
    } else if ui.button("Edit lines (M)").clicked() {
        request = Some(Request::Start);
    }

    let line = selected.and_then(|at| files.get(at.file).and_then(|f| f.get(at.line_ref())));
    match line {
        Some(pl) => {
            ui.label(format!(
                "{} — {} vertices, {:.3} m{}",
                pl.name, pl.len(), pl.length_3d(), if pl.closed { ", ring" } else { "" },
            ));
            let vertex = ed.vertex
                .filter(|(l, _)| Some(*l) == selected)
                .and_then(|(_, i)| pl.points().get(i).map(|p| (i, p)));
            if let Some((i, p)) = vertex {
                ui.label(format!("vertex {}: {:.3}  {:.3}  {:.3}", i + 1, p[0], p[1], p[2]));
            }
            ui.horizontal(|ui| {
                if ui.add_enabled(vertex.is_some(), egui::Button::new("Delete vertex")).clicked() {
                    request = Some(Request::DeleteVertex);
                }
                if ui.button("Delete line").clicked() { request = Some(Request::DeleteLine); }
                let ring = if pl.closed { "Open ring" } else { "Close ring" };
                if ui.button(ring).clicked() { request = Some(Request::ToggleRing); }
            });
            let drape = ui.add_enabled(
                snap_mode.drapes(),
                egui::Button::new(format!("Re-drape to {}", snap_mode.label())),
            );
            let drape = if snap_mode.drapes() {
                drape.on_hover_text("Keep every vertex's plan position and take its height from the \
                                     cloud, using the snap mode and radius in Digitise")
            } else {
                drape.on_disabled_hover_text("Choose Nearest, Low Z or High Z in Digitise — this \
                                              mode has no height of its own to drape to")
            };
            if drape.clicked() { request = Some(Request::Redrape); }
        }
        None => {
            ui.label("No line selected — click one in the view or in Polylines.");
        }
    }
    if ui.add_enabled(ed.can_undo(files), egui::Button::new("Undo last change (Ctrl+Z)")).clicked() {
        request = Some(Request::Undo);
    }
    if let Some(msg) = &ed.status {
        ui.label(msg.as_str());
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polylines::{Layer, Polyline, VectorFormat};

    fn file(name: &str, lines: &[&[[f64; 3]]]) -> PolylineFile {
        let mut pf = PolylineFile::new(format!("/tmp/haiviz-edit-test-{name}"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        for pts in lines {
            pf.add_line(0, Polyline::from_points("a", WHITE, pts.to_vec()));
        }
        pf
    }

    fn at(file: usize, line: usize) -> PolyRef { PolyRef { file, layer: 0, line } }

    #[test]
    fn a_press_that_does_not_travel_is_a_click() {
        let mut d = Drag {
            handle: Handle::Vertex { line: at(0, 0), idx: 0 },
            press: (100.0, 100.0),
            moving: false,
            origin: [0.0; 3],
        };
        d.track(102.0, 101.0);
        assert!(!d.moving, "a hand tremor must not re-snap the vertex");
        d.track(106.0, 100.0);
        assert!(d.moving);
        d.track(100.0, 100.0);
        assert!(d.moving, "and once dragging, coming back does not turn it into a click");
    }

    #[test]
    fn dragging_a_vertex_moves_it_and_a_segment_inserts_one() {
        let mut files = vec![file("drag", &[&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])];
        let mut ed = Editor::default();

        ed.commit_drag(&mut files, Handle::Vertex { line: at(0, 0), idx: 1 }, [10.0, 2.0, 1.0]);
        assert_eq!(files[0].layers[0].polylines[0].points()[1], [10.0, 2.0, 1.0]);

        let seg = Handle::Segment { line: at(0, 0), i: 0, j: 1, t: 0.5 };
        assert_eq!(seg.world(&files), Some([5.0, 1.0, 0.5]), "the grabbed point on the segment");
        ed.commit_drag(&mut files, seg, [5.0, 3.0, 0.5]);
        assert_eq!(files[0].layers[0].polylines[0].points().len(), 3);
        assert_eq!(files[0].layers[0].polylines[0].points()[1], [5.0, 3.0, 0.5]);
        assert_eq!(ed.vertex, Some((at(0, 0), 1)), "the new vertex is selected");
    }

    #[test]
    fn inserting_into_a_rings_closing_segment_appends() {
        let mut files = vec![file("ring", &[&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])];
        let mut ed = Editor::default();
        ed.toggle_ring(&mut files, at(0, 0)).unwrap();
        ed.commit_drag(&mut files, Handle::Segment { line: at(0, 0), i: 2, j: 0, t: 0.5 }, [0.5, 0.5, 0.0]);
        let pts = files[0].layers[0].polylines[0].points();
        assert_eq!(pts.len(), 4);
        assert_eq!(pts[3], [0.5, 0.5, 0.0], "between the last vertex and the first");
    }

    #[test]
    fn undo_follows_the_order_edits_were_made_across_files() {
        let two = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let mut files = vec![file("u0", &[&two]), file("u1", &[&two])];
        let mut ed = Editor::default();

        ed.commit_drag(&mut files, Handle::Vertex { line: at(0, 0), idx: 0 }, [0.0, 5.0, 0.0]);
        ed.commit_drag(&mut files, Handle::Vertex { line: at(1, 0), idx: 0 }, [0.0, 7.0, 0.0]);
        ed.commit_drag(&mut files, Handle::Vertex { line: at(0, 0), idx: 2 }, [2.0, 9.0, 0.0]);

        assert_eq!(ed.undo(&mut files), Some(0));
        assert_eq!(files[0].layers[0].polylines[0].points()[2], [2.0, 0.0, 0.0]);
        assert_eq!(files[0].layers[0].polylines[0].points()[0], [0.0, 5.0, 0.0], "only the last edit");
        assert_eq!(ed.undo(&mut files), Some(1), "then the other file's, not this file's older one");
        assert_eq!(ed.undo(&mut files), Some(0));
        assert!(!ed.can_undo(&files), "the lines' own additions were never recorded here");
        assert_eq!(ed.undo(&mut files), None);
    }

    #[test]
    fn unloading_a_file_keeps_undo_pointed_at_the_right_files() {
        let two = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let mut files = vec![file("r0", &[&two]), file("r1", &[&two]), file("r2", &[&two])];
        let mut ed = Editor::default();
        ed.commit_drag(&mut files, Handle::Vertex { line: at(2, 0), idx: 0 }, [0.0, 1.0, 0.0]);
        ed.commit_drag(&mut files, Handle::Vertex { line: at(1, 0), idx: 0 }, [0.0, 1.0, 0.0]);

        files.remove(1);
        ed.file_removed(1);
        // What was file 2 is now file 1, and its edit is still the one undone.
        assert_eq!(ed.undo(&mut files), Some(1));
        assert_eq!(files[1].layers[0].polylines[0].points()[0], [0.0, 0.0, 0.0]);
        assert_eq!(ed.undo(&mut files), None);
    }

    #[test]
    fn a_line_cannot_be_whittled_below_a_segment() {
        let mut files = vec![file("min", &[&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])];
        let mut ed = Editor::default();
        assert!(ed.delete_vertex(&mut files, at(0, 0), 1).is_ok());
        let refused = ed.delete_vertex(&mut files, at(0, 0), 1);
        assert!(refused.as_ref().is_err_and(|e| e.contains("Delete line")), "{refused:?}");
        assert_eq!(files[0].layers[0].polylines[0].len(), 2);

        // Nor can two vertices become a ring.
        assert!(ed.toggle_ring(&mut files, at(0, 0)).is_err());
        assert_eq!(ed.delete_line(&mut files, at(0, 0)), Some(0));
        assert!(files[0].layers[0].polylines.is_empty());
    }

    #[test]
    fn a_selection_that_no_longer_exists_is_dropped() {
        let mut files = vec![file("val", &[&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])];
        let mut ed = Editor::default();
        let mut selected = Some(at(0, 0));
        ed.delete_vertex(&mut files, at(0, 0), 0).unwrap();
        ed.vertex = Some((at(0, 0), 2));
        ed.validate(&files, &mut selected);
        assert_eq!((selected, ed.vertex), (Some(at(0, 0)), None), "vertex 2 is gone, the line is not");

        // A vertex of a line that is no longer the selected one is dropped too.
        ed.vertex = Some((at(0, 0), 0));
        let mut other = None;
        ed.validate(&files, &mut other);
        assert_eq!(ed.vertex, None);

        ed.undo(&mut files);
        ed.delete_line(&mut files, at(0, 0));
        ed.validate(&files, &mut selected);
        assert_eq!(selected, None);
    }

    #[test]
    fn a_drape_keeps_plan_position_and_says_what_it_skipped() {
        let line = [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0], [3.0, 0.0, 5.0]];
        let d = drape(&line, |p| match p[0] as i32 {
            0 => Ok(6.0),
            1 => Ok(5.0),
            2 => Err(DrapeMiss::Empty),
            _ => Err(DrapeMiss::Coarse),
        });
        assert_eq!(d.points, vec![[0.0, 0.0, 6.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0], [3.0, 0.0, 5.0]]);
        assert_eq!((d.moved, d.empty, d.coarse), (1, 1, 1));
        let s = d.summary(SnapMode::LowZ);
        assert!(s.contains("1 of 4") && s.contains("nothing within") && s.contains("sparse"), "{s}");
    }

    #[test]
    fn a_nearest_drape_stays_on_the_deck_it_was_on() {
        // A bridge deck at 8 m over a road at 0 m: both within the radius in
        // plan. A vertex already on the deck must not drop to the road.
        let mut cands = vec![(0.0, [0.1, 0.0, 0.0]), (8.2, [0.3, 0.0, 8.2]), (0.05, [0.0, 0.1, 0.05])];
        assert_eq!(drape_z(SnapMode::Nearest, [0.0, 0.0, 7.9], &mut cands, 5.0, 95.0), Some(8.2));
        assert_eq!(drape_z(SnapMode::LowZ, [0.0, 0.0, 7.9], &mut cands, 0.0, 95.0), Some(0.0));
        assert_eq!(drape_z(SnapMode::HighZ, [0.0, 0.0, 7.9], &mut cands, 0.0, 100.0), Some(8.2));
        assert_eq!(drape_z(SnapMode::SameZ, [0.0, 0.0, 7.9], &mut cands, 5.0, 95.0), None);
    }
}
