//! Digitising mode: drawing polylines by hand onto the cloud.
//!
//! `main.rs` holds only the mode wiring — the key handling next to the lasso
//! block, the panel call in each of the three per-mode egui blocks, and the
//! overlay draw. Everything about what a vertex *is* lives here.
//!
//! ## Where a vertex comes from
//!
//! This phase places vertices on real points: the click resolves through the
//! same screen-space pick that Measure and Point Info use, so a vertex is
//! always something the cloud actually contains. Shift-click places one freely
//! instead, on a horizontal plane through the previous vertex — the only
//! well-defined answer when the operator explicitly does not want the data to
//! decide.
//!
//! Snap modes, the radius, and the COPC spacing guard arrive in the next
//! phase and replace `place_from_pick` without changing anything else here.
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

// Phases 4-6 fill in snapping, edge following and vertex editing against this
// same state; a few accessors here are theirs.
#![allow(dead_code)]

use macroquad::prelude::*;
use std::path::Path;

use crate::polylines::{Layer, Polyline, PolylineFile, VectorFormat};

/// The sidecar a new layer lands in, beside the cloud it was drawn on.
const VECTOR_SIDECAR_SUFFIX: &str = ".haiviz-vectors.gpkg";

/// What the panel is asking `main.rs` to do. Returned rather than acted on so
/// this module never has to know about `AppMode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Request {
    Start,
    Cancel,
    Finish,
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
    pub status: Option<String>,
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
            status: None,
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

    /// Commit the pending vertices as a line in the target layer.
    ///
    /// Returns the file that now needs saving, so the caller can auto-save it
    /// without this module reaching for the filesystem.
    pub fn finish(&mut self, files: &mut [PolylineFile]) -> Option<usize> {
        if self.pending.len() < 2 {
            self.status = Some("A line needs at least two vertices.".to_string());
            return None;
        }
        let (fi, li) = self.target?;
        let pf = files.get_mut(fi)?;
        let n = pf.layers.get(li)?.polylines.len() + 1;
        let name = self.name_template.replace("{n}", &n.to_string());

        let mut line = Polyline::from_points(name, self.color, std::mem::take(&mut self.pending));
        if !self.code.is_empty() {
            line.attrs.insert("code".to_string(), self.code.clone());
        }
        pf.add_line(li, line)?;
        self.committed_lines += 1;
        Some(fi)
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
) -> (usize, usize) {
    let path = default_vector_path(cloud_path);
    if let Some(fi) = files.iter().position(|f| f.path == path) {
        if files[fi].layers.is_empty() {
            files[fi].layers.push(new_layer(srs_id));
        }
        return (fi, files[fi].layers.len() - 1);
    }
    let mut pf = PolylineFile::new(&path, VectorFormat::GeoPackage);
    pf.layers.push(new_layer(srs_id));
    files.push(pf);
    (files.len() - 1, 0)
}

fn new_layer(srs_id: i32) -> Layer {
    let mut l = Layer::new("digitised");
    l.srs_id = srs_id;
    l
}

/// Label for a `(file, layer)` target, for the panel's combo box.
pub fn target_label(files: &[PolylineFile], fi: usize, li: usize) -> String {
    match files.get(fi).and_then(|f| f.layers.get(li).map(|l| (f, l))) {
        Some((f, l)) => format!("{} › {}", f.file_name(), l.name),
        None => "—".to_string(),
    }
}

/// Draw the line in progress and the rubber band to the cursor.
///
/// `origin` converts the world-space pending vertices into the viewer space
/// everything else in the 3D view is drawn in.
pub fn draw_3d(dig: &Digitise, origin: [f64; 3], cursor: Option<[f64; 3]>, drawing: bool) {
    if dig.pending.is_empty() { return; }
    let to_view = |p: [f64; 3]| vec3(
        (p[0] - origin[0]) as f32,
        (p[1] - origin[1]) as f32,
        (p[2] - origin[2]) as f32,
    );

    let pts: Vec<Vec3> = dig.pending.iter().map(|p| to_view(*p)).collect();
    for w in pts.windows(2) {
        draw_line_3d(w[0], w[1], dig.color);
    }
    // Every committed vertex gets a marker: on a dense cloud the line alone
    // does not show where the operator actually clicked.
    for p in &pts {
        draw_cube_wires(*p, vec3(0.08, 0.08, 0.08), dig.color);
    }
    if drawing
        && let (Some(&last), Some(c)) = (pts.last(), cursor) {
        let faint = Color::new(dig.color.r, dig.color.g, dig.color.b, 0.5);
        draw_line_3d(last, to_view(c), faint);
    }
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
        let mut c = egui::Color32::from_rgb(
            (dig.color.r * 255.0) as u8, (dig.color.g * 255.0) as u8, (dig.color.b * 255.0) as u8,
        );
        if ui.color_edit_button_srgba(&mut c).changed() {
            dig.color = Color::new(
                c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, 1.0,
            );
        }
        ui.label("line colour");
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

        assert_eq!(dig.finish(&mut files), Some(0), "the caller is told which file to save");
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
            dig.finish(&mut files);
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
        assert_eq!(dig.finish(&mut files), None);
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
    fn a_new_layer_lands_beside_the_cloud() {
        let cloud = "/data/site/scan.copc.laz";
        assert_eq!(default_vector_path(cloud), "/data/site/scan.copc.laz.haiviz-vectors.gpkg");

        let mut files: Vec<PolylineFile> = Vec::new();
        let (fi, li) = ensure_target(&mut files, cloud, 28992);
        assert_eq!((fi, li), (0, 0));
        assert_eq!(files[0].format, VectorFormat::GeoPackage);
        assert_eq!(files[0].layers[0].srs_id, 28992);

        // Asking again reuses it rather than forking the file.
        let again = ensure_target(&mut files, cloud, 28992);
        assert_eq!(again, (0, 0));
        assert_eq!(files.len(), 1);
    }
}
