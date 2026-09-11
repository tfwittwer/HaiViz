//! Polyline overlays: the data model, Salome Hydro ASCII I/O, and drawing.
//!
//! ## Coordinates
//!
//! The master representation is **absolute world f64**, with a cached
//! viewer-space f32 copy for drawing. That is the opposite of the choice made
//! for the spatial index, and for the opposite reason: f32-relative-to-origin
//! is fine for a cloud-local overlay — sub-millimetre within a few km — but a
//! polyline is a *stored* coordinate. Round-tripping an external GeoPackage
//! through f32 would quantise every vertex it wrote back, and no amount of care
//! in the snapping would recover it.
//!
//! The cache is rebuilt from the file's `cached_origin`. The viewer origin is
//! pinned for as long as anything is loaded but is re-pinned when everything is
//! unloaded and new data arrives somewhere else, so `refresh_view` is called
//! before drawing rather than assumed to have been done.
//!
//! ## Files, layers, lines
//!
//! GeoPackage is a container of named feature tables and Salome is one flat
//! file, so the model has a layer level between the file and the lines. A
//! Salome file loads as a single layer named after the file; a GeoPackage
//! round-trips one layer per feature table.
//!
//! ## Undo
//!
//! A plain `Vec` of inverse edits, deliberately *not* modelled on
//! `ClassEdits`. That design stores geometric ops in a sidecar because
//! classification has to be replayable against streaming COPC nodes and again
//! at export. A polyline is just data: the file on disk is the state, and undo
//! only has to walk back to the previous one.

// The model, the mutation API and the undo stack land a phase ahead of the
// tool that drives them: digitising (phase 3) appends and pops vertices,
// editing (phase 6) moves and deletes them, and the GeoPackage codec
// (phase 2b) is what reads and writes `attrs`. Until then most of this module
// has only its tests for callers.
#![allow(dead_code)]

use macroquad::models::{Mesh, Vertex, draw_mesh};
use macroquad::prelude::*;
use crate::gpkg;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

/// The file format a layer came from, and writes back to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorFormat {
    /// Salome Hydro ASCII: `x y z` per line, blank line between polylines.
    /// No attributes, no layers, no CRS.
    Salome,
    /// OGC GeoPackage — a SQLite container of feature tables.
    GeoPackage,
}

impl VectorFormat {
    pub fn label(&self) -> &'static str {
        match self {
            VectorFormat::Salome => "Salome ASCII",
            VectorFormat::GeoPackage => "GeoPackage",
        }
    }
}

/// Address of one line within a file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LineRef {
    pub layer: usize,
    pub line: usize,
}

/// Address of one line across all loaded files.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PolyRef {
    pub file: usize,
    pub layer: usize,
    pub line: usize,
}

impl PolyRef {
    pub fn line_ref(&self) -> LineRef {
        LineRef { layer: self.layer, line: self.line }
    }
}

#[derive(Clone)]
pub struct Polyline {
    /// Master geometry, absolute world coordinates.
    points: Vec<[f64; 3]>,
    /// Viewer-space copy for drawing and picking. Derived from `points`; never
    /// written to directly, which is what stops the two drifting apart.
    view: Vec<[f32; 3]>,
    pub color: Color,
    /// Cylinder radius in world units; controls drawn width in the section view.
    pub radius: f32,
    pub name: String,
    /// A closed ring draws the segment from the last vertex back to the first.
    pub closed: bool,
    /// Non-geometry columns, kept verbatim so a file written by QGIS round-trips
    /// without losing the attributes haiviz does not understand. Ordered so a
    /// rewrite is byte-stable.
    pub attrs: BTreeMap<String, String>,
}

impl Polyline {
    pub fn new(name: impl Into<String>, color: Color) -> Polyline {
        Polyline {
            points: Vec::new(),
            view: Vec::new(),
            color,
            radius: 0.01,
            name: name.into(),
            closed: false,
            attrs: BTreeMap::new(),
        }
    }

    pub fn from_points(name: impl Into<String>, color: Color, points: Vec<[f64; 3]>) -> Polyline {
        let mut pl = Polyline::new(name, color);
        pl.points = points;
        pl
    }

    /// World-space vertices. Read-only: mutation goes through `PolylineFile`,
    /// which is what records the undo entry and refreshes the draw cache.
    pub fn points(&self) -> &[[f64; 3]] { &self.points }

    /// Viewer-space vertices, valid as of the last `refresh_view`.
    pub fn view(&self) -> &[[f32; 3]] { &self.view }

    pub fn len(&self) -> usize { self.points.len() }
    pub fn is_empty(&self) -> bool { self.points.is_empty() }

    fn rebuild_view(&mut self, origin: [f64; 3]) {
        self.view.clear();
        self.view.extend(self.points.iter().map(|p| [
            (p[0] - origin[0]) as f32,
            (p[1] - origin[1]) as f32,
            (p[2] - origin[2]) as f32,
        ]));
    }

    /// Total 3D length, in metres. The readout the digitising tool shows, and
    /// the `length_3d` attribute written to a GeoPackage.
    pub fn length_3d(&self) -> f64 {
        let mut total = 0.0;
        for w in self.points.windows(2) {
            total += dist3(w[0], w[1]);
        }
        if self.closed && self.points.len() > 2 {
            total += dist3(self.points[self.points.len() - 1], self.points[0]);
        }
        total
    }

    /// Segment endpoints in draw order, including the closing one for a ring.
    fn segments(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let n = self.view.len();
        let closing = if self.closed && n > 2 { Some((n - 1, 0)) } else { None };
        (0..n.saturating_sub(1)).map(|i| (i, i + 1)).chain(closing)
    }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// One named group of lines. A GeoPackage feature table, or a whole Salome file.
pub struct Layer {
    pub name: String,
    pub polylines: Vec<Polyline>,
    /// Whether the layer is drawn. Purely a view setting.
    pub visible: bool,
    /// EPSG code the coordinates are in, `0` when the file did not say.
    /// Salome carries no CRS at all, which is one of the things a GeoPackage →
    /// Salome conversion drops.
    pub srs_id: i32,
    /// The geometry column's name, so a file from QGIS keeps it.
    pub geom_column: String,
    /// Non-geometry columns as declared, in table order. Empty means "a layer
    /// haiviz made", which gets the default set from §7.2 on the way out.
    pub columns: Vec<(String, String)>,
    /// Drawn width in the 3D view, in **screen pixels**.
    ///
    /// Pixels rather than metres, and per layer rather than per line. The
    /// section view draws a cross-section of a physical cylinder, so its width
    /// is `Polyline::radius` in metres and has to scale with zoom; the 3D view
    /// draws a schematic line, where what matters is that it stays legible at
    /// any zoom. Those are different quantities and conflating them is why a
    /// 1 cm radius drew a sub-pixel line.
    pub width_px: f32,
}

impl Layer {
    pub fn new(name: impl Into<String>) -> Layer {
        Layer {
            name: name.into(),
            polylines: Vec::new(),
            visible: true,
            srs_id: 0,
            geom_column: "geom".to_string(),
            columns: Vec::new(),
            width_px: 2.0,
        }
    }
}

/// The inverse of one committed change, for the undo stack.
///
/// Each variant records what is needed to put things back, not what was done —
/// so undo is a straight apply, with no re-derivation.
#[derive(Clone)]
enum Edit {
    /// Undo by truncating the line back to `len` vertices.
    Truncate { at: LineRef, len: usize },
    /// Undo by inserting `v` back at `idx`.
    Reinsert { at: LineRef, idx: usize, v: [f64; 3] },
    /// Undo by removing the vertex at `idx`.
    Uninsert { at: LineRef, idx: usize },
    /// Undo by restoring the vertex at `idx` to `v`.
    Restore { at: LineRef, idx: usize, v: [f64; 3] },
    /// Undo by removing the line.
    Unadd { at: LineRef },
    /// Undo by putting the whole line back where it was.
    Readd { at: LineRef, line: Box<Polyline> },
    /// Undo by setting `closed` back to `was`.
    Unclose { at: LineRef, was: bool },
}

pub struct PolylineFile {
    pub path: String,
    pub format: VectorFormat,
    pub layers: Vec<Layer>,
    /// Unsaved changes. Auto-save (§8.2) clears it on the way out.
    pub dirty: bool,
    /// Modification time when the file was read, or `None` for a layer that has
    /// never been written. Compared before an auto-save so a file edited in
    /// QGIS since it was loaded is not silently overwritten.
    pub mtime: Option<SystemTime>,
    /// The origin the `view` caches were built against.
    cached_origin: [f64; 3],
    undo: Vec<Edit>,
}

impl PolylineFile {
    pub fn new(path: impl Into<String>, format: VectorFormat) -> PolylineFile {
        PolylineFile {
            path: path.into(),
            format,
            layers: Vec::new(),
            dirty: false,
            mtime: None,
            cached_origin: [f64::NAN; 3],
            undo: Vec::new(),
        }
    }

    pub fn file_name(&self) -> &str {
        Path::new(&self.path).file_name().and_then(|n| n.to_str()).unwrap_or(&self.path)
    }

    pub fn line_count(&self) -> usize {
        self.layers.iter().map(|l| l.polylines.len()).sum()
    }

    pub fn get(&self, at: LineRef) -> Option<&Polyline> {
        self.layers.get(at.layer)?.polylines.get(at.line)
    }

    fn get_mut(&mut self, at: LineRef) -> Option<&mut Polyline> {
        self.layers.get_mut(at.layer)?.polylines.get_mut(at.line)
    }

    /// Rebuild the viewer-space caches if the origin has moved. Cheap to call
    /// every frame: the common case is one float comparison per file.
    pub fn refresh_view(&mut self, origin: [f64; 3]) {
        if self.cached_origin == origin { return; }
        self.cached_origin = origin;
        for layer in &mut self.layers {
            for pl in &mut layer.polylines {
                pl.rebuild_view(origin);
            }
        }
    }

    /// Refresh one line's cache after it was edited, without touching the rest.
    fn touch(&mut self, at: LineRef) {
        let origin = self.cached_origin;
        // Before the first `refresh_view` the origin is NaN and there is
        // nothing meaningful to project into; the next refresh does the work.
        if origin[0].is_nan() { return; }
        if let Some(pl) = self.get_mut(at) { pl.rebuild_view(origin); }
        self.dirty = true;
    }

    // ---- Mutation. Every one records its inverse. ----

    pub fn append_vertex(&mut self, at: LineRef, v: [f64; 3]) {
        let Some(pl) = self.get_mut(at) else { return };
        let len = pl.points.len();
        pl.points.push(v);
        self.undo.push(Edit::Truncate { at, len });
        self.touch(at);
    }

    pub fn insert_vertex(&mut self, at: LineRef, idx: usize, v: [f64; 3]) {
        let Some(pl) = self.get_mut(at) else { return };
        if idx > pl.points.len() { return; }
        pl.points.insert(idx, v);
        self.undo.push(Edit::Uninsert { at, idx });
        self.touch(at);
    }

    pub fn remove_vertex(&mut self, at: LineRef, idx: usize) -> Option<[f64; 3]> {
        let pl = self.get_mut(at)?;
        if idx >= pl.points.len() { return None; }
        let v = pl.points.remove(idx);
        self.undo.push(Edit::Reinsert { at, idx, v });
        self.touch(at);
        Some(v)
    }

    /// Drop the last vertex — what Backspace does while digitising.
    pub fn pop_vertex(&mut self, at: LineRef) -> Option<[f64; 3]> {
        let n = self.get(at)?.points.len();
        if n == 0 { return None; }
        self.remove_vertex(at, n - 1)
    }

    pub fn move_vertex(&mut self, at: LineRef, idx: usize, v: [f64; 3]) {
        let Some(pl) = self.get_mut(at) else { return };
        let Some(slot) = pl.points.get_mut(idx) else { return };
        let was = *slot;
        *slot = v;
        self.undo.push(Edit::Restore { at, idx, v: was });
        self.touch(at);
    }

    pub fn set_closed(&mut self, at: LineRef, closed: bool) {
        let Some(pl) = self.get_mut(at) else { return };
        let was = pl.closed;
        if was == closed { return; }
        pl.closed = closed;
        self.undo.push(Edit::Unclose { at, was });
        self.touch(at);
    }

    /// Append a line to a layer and return its address.
    pub fn add_line(&mut self, layer: usize, line: Polyline) -> Option<LineRef> {
        let l = self.layers.get_mut(layer)?;
        let at = LineRef { layer, line: l.polylines.len() };
        l.polylines.push(line);
        self.undo.push(Edit::Unadd { at });
        self.touch(at);
        Some(at)
    }

    pub fn remove_line(&mut self, at: LineRef) -> Option<Polyline> {
        let l = self.layers.get_mut(at.layer)?;
        if at.line >= l.polylines.len() { return None; }
        let line = l.polylines.remove(at.line);
        self.undo.push(Edit::Readd { at, line: Box::new(line.clone()) });
        self.dirty = true;
        Some(line)
    }

    pub fn can_undo(&self) -> bool { !self.undo.is_empty() }

    /// Reverse the last committed change. Returns false when there is nothing
    /// to undo.
    pub fn undo(&mut self) -> bool {
        let Some(edit) = self.undo.pop() else { return false };
        let at = match &edit {
            Edit::Truncate { at, len } => {
                if let Some(pl) = self.get_mut(*at) { pl.points.truncate(*len); }
                *at
            }
            Edit::Reinsert { at, idx, v } => {
                if let Some(pl) = self.get_mut(*at)
                    && *idx <= pl.points.len() { pl.points.insert(*idx, *v); }
                *at
            }
            Edit::Uninsert { at, idx } => {
                if let Some(pl) = self.get_mut(*at)
                    && *idx < pl.points.len() { pl.points.remove(*idx); }
                *at
            }
            Edit::Restore { at, idx, v } => {
                if let Some(pl) = self.get_mut(*at)
                    && let Some(slot) = pl.points.get_mut(*idx) { *slot = *v; }
                *at
            }
            Edit::Unclose { at, was } => {
                if let Some(pl) = self.get_mut(*at) { pl.closed = *was; }
                *at
            }
            Edit::Unadd { at } => {
                if let Some(l) = self.layers.get_mut(at.layer)
                    && at.line < l.polylines.len() { l.polylines.remove(at.line); }
                self.dirty = true;
                return true;
            }
            Edit::Readd { at, line } => {
                if let Some(l) = self.layers.get_mut(at.layer) {
                    let idx = at.line.min(l.polylines.len());
                    l.polylines.insert(idx, (**line).clone());
                }
                *at
            }
        };
        // `touch` would push a new undo entry if it went through the mutators;
        // it does not, so refreshing the cache here is safe.
        self.touch(at);
        true
    }

    /// Has the file on disk changed since it was loaded or last written?
    ///
    /// Auto-save rewrites the whole file (§8.2), so an unnoticed external edit
    /// would be destroyed rather than merely conflicted with. Stop and ask.
    pub fn changed_on_disk(&self) -> bool {
        let Some(known) = self.mtime else { return false };
        match fs::metadata(&self.path).and_then(|m| m.modified()) {
            Ok(now) => now != known,
            // Gone, or unreadable: not an external edit, and the save will
            // report its own error if it cannot write.
            Err(_) => false,
        }
    }

    /// Write the file back in the format it came from.
    ///
    /// Always a whole-file rewrite: there is no incremental path, so a save
    /// goes to a temporary file beside the target and is renamed over it. A
    /// crash mid-write must not be able to destroy the previous version, and
    /// with auto-save the write path is exercised constantly.
    pub fn save(&mut self) -> Result<(), String> {
        match self.format {
            VectorFormat::Salome => {
                let body = self.to_salome();
                atomic_write(&self.path, body.as_bytes())?;
            }
            VectorFormat::GeoPackage => {
                let layers: Vec<gpkg::GpkgLayer> =
                    self.layers.iter().map(to_gpkg_layer).collect();
                let bytes = gpkg::encode(&layers)?;
                atomic_write(&self.path, &bytes)?;
            }
        }
        self.mtime = fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        self.dirty = false;
        Ok(())
    }

    /// Everything a Salome save would silently discard, for the warning in
    /// §7.3. Empty means the conversion is lossless.
    pub fn salome_losses(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.layers.iter().any(|l| l.srs_id != 0) {
            out.push("the coordinate reference system is not recorded".to_string());
        }
        if self.layers.len() > 1 {
            out.push(format!(
                "{} layers are flattened into one file", self.layers.len(),
            ));
        }
        let with_attrs = self.layers.iter()
            .flat_map(|l| &l.polylines)
            .filter(|p| !p.attrs.is_empty())
            .count();
        if with_attrs > 0 {
            out.push(format!("{with_attrs} lines lose their attribute columns"));
        }
        if self.layers.iter().flat_map(|l| &l.polylines).any(|p| p.closed) {
            out.push("closed rings are written as open lines".to_string());
        }
        if self.layers.iter().flat_map(|l| &l.polylines).any(|p| !p.name.is_empty()) {
            out.push("line names are not stored".to_string());
        }
        out
    }

    fn to_salome(&self) -> String {
        let mut out = String::new();
        for layer in &self.layers {
            for pl in &layer.polylines {
                if pl.points.len() < 2 { continue; }
                for p in &pl.points {
                    // Millimetre resolution, which is well past what any survey
                    // this reads carries, and short enough to stay readable.
                    out.push_str(&format!("{:.3} {:.3} {:.3}\n", p[0], p[1], p[2]));
                }
                // A ring is written as an open line that returns to its start,
                // since the format has no way to say "closed".
                if pl.closed && pl.points.len() > 2 {
                    let p = pl.points[0];
                    out.push_str(&format!("{:.3} {:.3} {:.3}\n", p[0], p[1], p[2]));
                }
                out.push('\n');
            }
        }
        out
    }
}

/// Write `bytes` to `path` via a temporary file in the same directory, then
/// rename over the target. Same directory so the rename stays within one
/// filesystem, which is what makes it atomic.
fn atomic_write(path: &str, bytes: &[u8]) -> Result<(), String> {
    let target = Path::new(path);
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let stem = target.file_name().and_then(|n| n.to_str()).unwrap_or("out");
    let tmp = dir.join(format!(".{stem}.haiviz-tmp"));

    let write = || -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        // The rename is atomic, but only orders against data that has actually
        // reached the disk. Without this a crash can leave the new name over
        // empty content — worse than either version.
        f.sync_all()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(format!("{path}: {e}"));
    }
    fs::rename(&tmp, target).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("{path}: {e}")
    })
}

/// Which format a file holds, decided by its first bytes rather than its
/// extension: `.gpkg` is not guaranteed, and the Salome files in the wild use
/// `.txt`, `.xyz`, `.poly` or no extension at all.
pub fn detect_format(path: &str) -> VectorFormat {
    use std::io::Read;
    let mut head = [0u8; 16];
    if let Ok(mut f) = fs::File::open(path)
        && f.read_exact(&mut head).is_ok()
        && &head == b"SQLite format 3\0" {
        return VectorFormat::GeoPackage;
    }
    VectorFormat::Salome
}

/// Load a vector file, picking the reader by magic bytes.
pub fn load(path: &str, origin: [f64; 3], default_color: Color) -> Result<PolylineFile, String> {
    let mut pf = match detect_format(path) {
        VectorFormat::Salome => load_salome(path, default_color)?,
        VectorFormat::GeoPackage => load_gpkg(path, default_color)?,
    };
    pf.mtime = fs::metadata(path).and_then(|m| m.modified()).ok();
    pf.refresh_view(origin);
    Ok(pf)
}

/// Parse Salome Hydro Polyline ASCII: one `x y z` per line, blank lines
/// separating polylines. The whole file becomes one layer named after it.
fn load_salome(path: &str, default_color: Color) -> Result<PolylineFile, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut pf = PolylineFile::new(path, VectorFormat::Salome);
    let layer_name = Path::new(path).file_stem().and_then(|n| n.to_str()).unwrap_or("lines");
    let mut layer = Layer::new(layer_name);
    let mut current: Vec<[f64; 3]> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            flush(&mut current, &mut layer.polylines, default_color);
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 3
            && let (Ok(x), Ok(y), Ok(z)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
            ) {
            current.push([x, y, z]);
        }
    }
    flush(&mut current, &mut layer.polylines, default_color);

    pf.layers.push(layer);
    Ok(pf)
}

/// Read a GeoPackage: one `Layer` per feature table, one `Polyline` per part.
///
/// A multi-linestring row becomes several lines sharing the row's attributes,
/// so nothing is dropped and nothing is invented.
fn load_gpkg(path: &str, default_color: Color) -> Result<PolylineFile, String> {
    let mut pf = PolylineFile::new(path, VectorFormat::GeoPackage);
    for gl in gpkg::read(path)? {
        let mut layer = Layer::new(&gl.name);
        layer.srs_id = gl.srs_id;
        layer.geom_column = gl.geom_column.clone();
        layer.columns = gl.columns.clone();
        for f in gl.features {
            // A `name` column is what the panel shows; anything else keeps its
            // place in `attrs` and is written back untouched.
            let name = f.attrs.get("name")
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("Polyline {}", layer.polylines.len() + 1));
            let mut pl = Polyline::from_points(name, default_color, f.points);
            pl.closed = f.closed;
            pl.attrs = f.attrs;
            layer.polylines.push(pl);
        }
        pf.layers.push(layer);
    }
    if pf.layers.is_empty() {
        return Err(format!(
            "{}: no line or polygon layers in this GeoPackage",
            Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path),
        ));
    }
    Ok(pf)
}

fn to_gpkg_layer(layer: &Layer) -> gpkg::GpkgLayer {
    let mut gl = gpkg::GpkgLayer::new(&layer.name, layer.srs_id);
    gl.geom_column = layer.geom_column.clone();
    // A layer read from a file keeps the columns it arrived with; one haiviz
    // created gets the default set.
    if !layer.columns.is_empty() { gl.columns = layer.columns.clone(); }

    let has = |c: &str| gl.columns.iter().any(|(n, _)| n == c);
    for pl in &layer.polylines {
        // Start from what came out of the file, so columns haiviz knows nothing
        // about survive the round trip verbatim.
        let mut attrs = pl.attrs.clone();
        // The derived ones are refreshed rather than preserved: after an edit
        // the stored vertex count and length are simply wrong, and a stale
        // number is worse than none.
        if has("name") { attrs.insert("name".into(), pl.name.clone()); }
        if has("n_vertices") { attrs.insert("n_vertices".into(), pl.len().to_string()); }
        if has("length_3d") { attrs.insert("length_3d".into(), format!("{:.4}", pl.length_3d())); }
        if has("created") { attrs.entry("created".into()).or_insert_with(gpkg::iso8601_now); }
        attrs.retain(|k, _| has(k));

        gl.features.push(gpkg::Feature {
            fid: 0,
            points: pl.points.clone(),
            closed: pl.closed,
            attrs,
        });
    }
    gl
}

fn flush(current: &mut Vec<[f64; 3]>, polylines: &mut Vec<Polyline>, color: Color) {
    if current.len() >= 2 {
        let name = format!("Polyline {}", polylines.len() + 1);
        polylines.push(Polyline::from_points(name, color, std::mem::take(current)));
    } else {
        current.clear();
    }
}

/// Every visible line across all files, as `(address, line)`.
fn visible_lines(files: &[PolylineFile]) -> impl Iterator<Item = (PolyRef, &Polyline)> {
    files.iter().enumerate().flat_map(|(fi, pf)| {
        pf.layers.iter().enumerate()
            .filter(|(_, l)| l.visible)
            .flat_map(move |(li, l)| {
                l.polylines.iter().enumerate().map(move |(pi, pl)| {
                    (PolyRef { file: fi, layer: li, line: pi }, pl)
                })
            })
    })
}

/// Draw a polyline as a ribbon of camera-facing quads.
///
/// `draw_line_3d` has no width and a screen-space 2D line has no depth, and
/// both of those matter here: a thin line is hard to see against a dense cloud,
/// and a line that draws over the points in front of it gives no clue where it
/// actually sits. Real geometry under the 3D camera gets both — macroquad's
/// 3D pipeline depth-tests, and the cloud pass writes depth.
///
/// `half_width` is in world units; the caller converts from pixels so the line
/// keeps its apparent thickness at any zoom.
pub fn draw_ribbon_3d(
    pts: &[glam::Vec3],
    closed: bool,
    half_width: f32,
    color: Color,
    view_dir: glam::Vec3,
) {
    if pts.len() < 2 { return; }
    let segments: Vec<(usize, usize)> = (0..pts.len() - 1).map(|i| (i, i + 1))
        .chain(if closed && pts.len() > 2 { Some((pts.len() - 1, 0)) } else { None })
        .collect();

    let mut mesh = Mesh { vertices: Vec::new(), indices: Vec::new(), texture: None };
    // `indices` is u16, so a mesh holds at most 65535 vertices — four per
    // segment. Flush well short of that rather than wrapping round silently.
    const MAX_QUADS: usize = 8_000;

    for (a, b) in segments {
        let (p, q) = (pts[a], pts[b]);
        let along = q - p;
        if along.length_squared() <= 0.0 { continue; }
        // Perpendicular to both the segment and the view direction, so the
        // ribbon always faces the camera and never collapses to a hairline.
        let mut off = along.cross(view_dir);
        if off.length_squared() < 1e-12 {
            // Segment pointing straight at the camera: any perpendicular will
            // do, and this one is stable.
            off = along.cross(glam::Vec3::Z).normalize_or_zero();
            if off.length_squared() < 1e-12 { off = glam::Vec3::X; }
        }
        let off = off.normalize_or_zero() * half_width;

        let base = mesh.vertices.len() as u16;
        for v in [p - off, p + off, q + off, q - off] {
            mesh.vertices.push(Vertex::new(v.x, v.y, v.z, 0.0, 0.0, color));
        }
        mesh.indices.extend_from_slice(&[
            base, base + 1, base + 2, base, base + 2, base + 3,
        ]);
        if mesh.vertices.len() >= MAX_QUADS * 4 {
            draw_mesh(&mesh);
            mesh.vertices.clear();
            mesh.indices.clear();
        }
    }
    if !mesh.vertices.is_empty() { draw_mesh(&mesh); }
}

/// Draw all visible polylines in the 3D view.
///
/// Call with the 3D camera active, before `set_default_camera()` — that is what
/// keeps the lines behind the panel and in front of only the points they are
/// actually in front of.
///
/// `metres_per_pixel` converts each layer's pixel width into world units.
pub fn draw_3d(files: &[PolylineFile], view_dir: glam::Vec3, metres_per_pixel: f32) {
    for (at, pl) in visible_lines(files) {
        let width_px = files[at.file].layers[at.layer].width_px.max(0.5);
        let pts: Vec<glam::Vec3> = pl.view.iter().map(|p| glam::Vec3::from(*p)).collect();
        draw_ribbon_3d(
            &pts, pl.closed, width_px * 0.5 * metres_per_pixel, pl.color, view_dir,
        );
    }
}

/// Draw polylines projected into section view 2D space.
/// Only the parts of each segment within `|perp| <= depth` of the section plane are drawn,
/// where perp is the perpendicular distance from the section line.
#[allow(clippy::too_many_arguments)]
pub fn draw_section(
    files: &[PolylineFile],
    section_start: (f32, f32),
    section_end: (f32, f32),
    depth: f32,
    section_pan: glam::Vec2,
    section_zoom: f32,
    // The (x, y, w, h) rect the section view renders into — not the window,
    // which the docked panel no longer fills.
    view: (f32, f32, f32, f32),
) {
    let (sw, sh) = (view.2, view.3);
    let dx = section_end.0 - section_start.0;
    let dy = section_end.1 - section_start.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 { return; }
    let ux = dx / len;
    let uy = dy / len;
    // perpendicular direction (rotate 90°)
    let px = -uy;
    let py = ux;

    let aspect = sw / sh;
    let half_h = section_zoom;
    let half_w = half_h * aspect;

    let to_sx = |wx: f32| view.0 + (wx - (section_pan.x - half_w)) / (2.0 * half_w) * sw;
    let to_sy = |wy: f32| view.1 + (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * sh;

    for (_, pl) in visible_lines(files) {
        if pl.view.len() < 2 { continue; }

        // Compute (along, perp, height) for each point
        let proj: Vec<(f32, f32, f32)> = pl.view.iter().map(|p| {
            let rx = p[0] - section_start.0;
            let ry = p[1] - section_start.1;
            let along = rx * ux + ry * uy;
            let perp  = rx * px + ry * py;
            (along, perp, p[2])
        }).collect();

        // Convert world-unit radius to screen pixels (x and y scales are equal)
        let r_px = (pl.radius * sh / (2.0 * half_h)).max(0.5);

        for (i, j) in pl.segments() {
            let (a_along, a_perp, a_h) = proj[i];
            let (b_along, b_perp, b_h) = proj[j];
            if let Some((sa, sb)) = clip_segment(a_along, a_perp, a_h, b_along, b_perp, b_h, depth) {
                let (sax, say) = (to_sx(sa.0), to_sy(sa.1));
                let (sbx, sby) = (to_sx(sb.0), to_sy(sb.1));
                draw_line(sax, say, sbx, sby, r_px * 2.0, pl.color);
                // Rounded endcaps to complete the capsule shape
                draw_circle(sax, say, r_px, pl.color);
                draw_circle(sbx, sby, r_px, pl.color);
            }
        }
    }
}

/// Clip a segment (given as along/perp/height at each endpoint) to |perp| <= depth.
/// Returns the (along, height) pairs of the clipped endpoints, or None if fully outside.
fn clip_segment(
    a_along: f32, a_perp: f32, a_h: f32,
    b_along: f32, b_perp: f32, b_h: f32,
    depth: f32,
) -> Option<((f32, f32), (f32, f32))> {
    let mut t0: f32 = 0.0;
    let mut t1: f32 = 1.0;

    let d_perp = b_perp - a_perp;

    // Clip to perp >= -depth  =>  t*d_perp >= -depth - a_perp
    if d_perp.abs() < 1e-9 {
        if a_perp < -depth { return None; }
    } else {
        let t = (-depth - a_perp) / d_perp;
        if d_perp > 0.0 { t0 = t0.max(t); } else { t1 = t1.min(t); }
    }

    // Clip to perp <= +depth  =>  t*d_perp <= depth - a_perp
    if d_perp.abs() < 1e-9 {
        if a_perp > depth { return None; }
    } else {
        let t = (depth - a_perp) / d_perp;
        if d_perp > 0.0 { t1 = t1.min(t); } else { t0 = t0.max(t); }
    }

    if t0 > t1 { return None; }

    let interp = |a: f32, b: f32, t: f32| a + t * (b - a);
    let sa = (interp(a_along, b_along, t0), interp(a_h, b_h, t0));
    let sb = (interp(a_along, b_along, t1), interp(a_h, b_h, t1));
    Some((sa, sb))
}

/// The visible polyline closest to `(mx, my)` in screen space, within
/// `threshold` pixels.
pub fn pick(
    files: &[PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    sw: f32,
    sh: f32,
    threshold: f32,
) -> Option<PolyRef> {
    let mut best = threshold;
    let mut result = None;
    for (at, pl) in visible_lines(files) {
        for (i, j) in pl.segments() {
            let a = project(view_proj, pl.view[i], sw, sh);
            let b = project(view_proj, pl.view[j], sw, sh);
            let d = seg_dist(mx, my, a, b);
            if d < best {
                best = d;
                result = Some(at);
            }
        }
    }
    result
}

/// The visible **vertex** closest to `(mx, my)` in screen space, within
/// `threshold` pixels, as `(line, vertex index)`.
///
/// `pick` only resolves to a whole line, which is enough to select one but not
/// to move a corner. This is what vertex editing and snapping to an existing
/// line are built on.
pub fn pick_vertex(
    files: &[PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    sw: f32,
    sh: f32,
    threshold: f32,
) -> Option<(PolyRef, usize)> {
    let mut best = threshold;
    let mut result = None;
    for (at, pl) in visible_lines(files) {
        for (i, v) in pl.view.iter().enumerate() {
            let (px, py) = project(view_proj, *v, sw, sh);
            let d = ((px - mx).powi(2) + (py - my).powi(2)).sqrt();
            if d < best {
                best = d;
                result = Some((at, i));
            }
        }
    }
    result
}

/// The nearest point on any visible polyline to `p`, in world space, within
/// `r` metres.
///
/// Vertices win ties against segments at the same distance, so tying into the
/// end of an existing line lands exactly on its endpoint rather than a
/// millimetre along it — which is the difference between a closed network and
/// one with hairline gaps in it.
pub fn nearest_on_lines(
    files: &[PolylineFile],
    p: [f64; 3],
    r: f64,
) -> Option<([f64; 3], PolyRef)> {
    let mut best: Option<(f64, [f64; 3], PolyRef)> = None;
    let mut consider = |d: f64, at: [f64; 3], who: PolyRef| {
        if d <= r && best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
            best = Some((d, at, who));
        }
    };

    for (at, pl) in visible_lines(files) {
        for v in pl.points() {
            consider(dist3(*v, p), *v, at);
        }
        // Segments are offered at a hair more than their true distance, so an
        // endpoint at the same place always wins.
        for (i, j) in pl.segments() {
            let (a, b) = (pl.points()[i], pl.points()[j]);
            let (proj, d) = closest_on_segment(a, b, p);
            consider(d * (1.0 + 1e-9) + 1e-9, proj, at);
        }
    }
    best.map(|(_, at, who)| (at, who))
}

/// The nearest point on any visible polyline to a ray, within `r` metres of it.
///
/// The ray form rather than a point form, because there is no sensible point to
/// search around when the cursor is over a gap in the cloud — which is exactly
/// when tying into an existing line matters most. Under an orthographic camera
/// the ray origin sits far behind the data, so distances along it are large and
/// positive; only the perpendicular distance is compared.
pub fn nearest_on_lines_to_ray(
    files: &[PolylineFile],
    origin: [f64; 3],
    dir: [f64; 3],
    r: f64,
) -> Option<([f64; 3], PolyRef)> {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    // NaN as well as zero: normalising by either poisons every comparison below.
    if len <= 0.0 || len.is_nan() { return None; }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];

    let mut best: Option<(f64, f64, [f64; 3], PolyRef)> = None;
    let mut consider = |perp: f64, along: f64, at: [f64; 3], who: PolyRef| {
        if perp > r || along < 0.0 { return; }
        // Frontmost wins, like the point-cloud ray pick: within the tolerance
        // the operator means the line facing them.
        if best.as_ref().is_none_or(|(_, ba, _, _)| along < *ba) {
            best = Some((perp, along, at, who));
        }
    };

    for (at, pl) in visible_lines(files) {
        for v in pl.points() {
            let (perp, along) = ray_distance_to_point(origin, d, *v);
            consider(perp, along, *v, at);
        }
        for (i, j) in pl.segments() {
            let (a, b) = (pl.points()[i], pl.points()[j]);
            if let Some((q, perp, along)) = ray_closest_on_segment(origin, d, a, b) {
                // A hair worse than a vertex at the same place, so tying to an
                // endpoint lands exactly on it.
                consider(perp * (1.0 + 1e-9) + 1e-9, along, q, at);
            }
        }
    }
    best.map(|(_, _, at, who)| (at, who))
}

/// Perpendicular distance from a normalised ray to a point, and how far along
/// the ray the foot of that perpendicular lies.
fn ray_distance_to_point(o: [f64; 3], d: [f64; 3], p: [f64; 3]) -> (f64, f64) {
    let v = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    let t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
    let perp = [v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2]];
    ((perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt(), t)
}

/// Closest approach between a ray and a segment: the point on the segment, its
/// distance to the ray, and how far along the ray that happens.
fn ray_closest_on_segment(
    o: [f64; 3], d: [f64; 3], a: [f64; 3], b: [f64; 3],
) -> Option<([f64; 3], f64, f64)> {
    let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    if ee <= 0.0 { return None; }
    let w = [a[0] - o[0], a[1] - o[1], a[2] - o[2]];
    let de = d[0] * e[0] + d[1] * e[1] + d[2] * e[2];
    let dw = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
    let ew = e[0] * w[0] + e[1] * w[1] + e[2] * w[2];

    // `d` is normalised, so the usual denominator |d|²|e|² - (d·e)² is
    // ee - de².
    let denom = ee - de * de;
    let s = if denom.abs() < 1e-12 {
        // Parallel: any point will do, so take the segment's start.
        0.0
    } else {
        ((de * dw - ew) / denom).clamp(0.0, 1.0)
    };
    let q = [a[0] + s * e[0], a[1] + s * e[1], a[2] + s * e[2]];
    let (perp, along) = ray_distance_to_point(o, d, q);
    Some((q, perp, along))
}

/// Closest point on segment `a`–`b` to `p`, and its distance.
fn closest_on_segment(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> ([f64; 3], f64) {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 <= 0.0 { return (a, dist3(a, p)); }
    let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / len2)
        .clamp(0.0, 1.0);
    let q = [a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]];
    (q, dist3(q, p))
}

fn project(vp: &glam::Mat4, p: [f32; 3], sw: f32, sh: f32) -> (f32, f32) {
    let clip = *vp * glam::Vec4::new(p[0], p[1], p[2], 1.0);
    if clip.w.abs() < 1e-6 { return (-1e9, -1e9); }
    let nx = clip.x / clip.w;
    let ny = clip.y / clip.w;
    ((nx + 1.0) / 2.0 * sw, (1.0 - ny) / 2.0 * sh)
}

fn seg_dist(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        return ((px - a.0).powi(2) + (py - a.1).powi(2)).sqrt();
    }
    let t = ((px - a.0) * dx + (py - a.1) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let cx = a.0 + t * dx - px;
    let cy = a.1 + t * dy - py;
    (cx * cx + cy * cy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("haiviz-poly-test-{}-{}", std::process::id(), name));
        p.to_string_lossy().into_owned()
    }

    fn file_with_one_line() -> (PolylineFile, LineRef) {
        let mut pf = PolylineFile::new(tmp("model"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0, 0.0, 0.0]);
        let at = pf.add_line(0, Polyline::from_points("a", WHITE, vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        ])).unwrap();
        (pf, at)
    }

    #[test]
    fn world_coordinates_survive_the_viewer_cache() {
        // The whole reason the master is f64: an easting near 200 km with a
        // millimetre on the end is representable in f64 and not in f32.
        let mut pf = PolylineFile::new(tmp("prec"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        let v = [208_802.001_f64, 373_320.002, 12.003];
        let at = pf.add_line(0, Polyline::from_points("a", WHITE, vec![v, [0.0; 3]])).unwrap();
        pf.refresh_view([208_800.0, 373_300.0, 0.0]);

        assert_eq!(pf.get(at).unwrap().points()[0], v, "the master must not be quantised");
        let seen = pf.get(at).unwrap().view()[0];
        assert!((seen[0] - 2.001).abs() < 1e-4, "viewer copy is close enough to draw: {seen:?}");
    }

    #[test]
    fn the_view_cache_follows_the_origin() {
        let (mut pf, at) = file_with_one_line();
        assert_eq!(pf.get(at).unwrap().view()[1], [1.0, 0.0, 0.0]);
        pf.refresh_view([1.0, 0.0, 0.0]);
        assert_eq!(pf.get(at).unwrap().view()[1], [0.0, 0.0, 0.0]);
        assert_eq!(pf.get(at).unwrap().points()[1], [1.0, 0.0, 0.0], "master unchanged");
    }

    #[test]
    fn every_mutation_is_undoable() {
        let (mut pf, at) = file_with_one_line();
        let before: Vec<[f64; 3]> = pf.get(at).unwrap().points().to_vec();

        pf.append_vertex(at, [2.0, 2.0, 2.0]);
        pf.insert_vertex(at, 0, [-1.0, 0.0, 0.0]);
        pf.move_vertex(at, 2, [9.0, 9.0, 9.0]);
        pf.remove_vertex(at, 1);
        pf.set_closed(at, true);
        assert!(pf.get(at).unwrap().closed);

        // Unwind the five edits, but not the `add_line` underneath them.
        for _ in 0..5 { assert!(pf.undo()); }
        assert_eq!(pf.get(at).unwrap().points(), &before[..],
                   "undoing everything must reach the state before the edits");
        assert!(!pf.get(at).unwrap().closed);

        // The line itself went on the same stack, so one more step removes it
        // and the stack is then empty.
        assert!(pf.undo());
        assert_eq!(pf.layers[0].polylines.len(), 0);
        assert!(!pf.can_undo());
        assert!(!pf.undo(), "undoing past the start is a no-op, not a panic");
    }

    #[test]
    fn undo_refreshes_the_draw_cache() {
        let (mut pf, at) = file_with_one_line();
        pf.append_vertex(at, [5.0, 5.0, 5.0]);
        assert_eq!(pf.get(at).unwrap().view().len(), 4);
        pf.undo();
        assert_eq!(pf.get(at).unwrap().view().len(), 3,
                   "a stale cache would draw a vertex that no longer exists");
        assert_eq!(pf.get(at).unwrap().view()[2], [1.0, 1.0, 0.0]);
    }

    #[test]
    fn deleting_a_line_and_undoing_puts_it_back_where_it_was() {
        let mut pf = PolylineFile::new(tmp("del"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        for i in 0..3 {
            pf.add_line(0, Polyline::from_points(
                format!("line {i}"), WHITE, vec![[i as f64, 0.0, 0.0], [i as f64, 1.0, 0.0]],
            ));
        }
        pf.remove_line(LineRef { layer: 0, line: 1 });
        assert_eq!(pf.layers[0].polylines[1].name, "line 2");
        pf.undo();
        assert_eq!(pf.layers[0].polylines.len(), 3);
        assert_eq!(pf.layers[0].polylines[1].name, "line 1", "restored in place, not appended");
    }

    #[test]
    fn pop_vertex_is_what_backspace_needs() {
        let (mut pf, at) = file_with_one_line();
        assert_eq!(pf.pop_vertex(at), Some([1.0, 1.0, 0.0]));
        assert_eq!(pf.get(at).unwrap().len(), 2);
        pf.pop_vertex(at);
        pf.pop_vertex(at);
        assert_eq!(pf.pop_vertex(at), None, "an empty line pops nothing rather than panicking");
    }

    #[test]
    fn length_counts_the_closing_segment_only_for_a_ring() {
        let (mut pf, at) = file_with_one_line();
        assert!((pf.get(at).unwrap().length_3d() - 2.0).abs() < 1e-9);
        pf.set_closed(at, true);
        let closed = pf.get(at).unwrap().length_3d();
        assert!((closed - (2.0 + 2.0_f64.sqrt())).abs() < 1e-9, "{closed}");
    }

    #[test]
    fn salome_round_trips_through_a_file() {
        let path = tmp("roundtrip.txt");
        let mut pf = PolylineFile::new(&path, VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![
            [208_802.001, 373_320.002, 12.003], [208_812.5, 373_320.0, 12.25],
        ]));
        pf.add_line(0, Polyline::from_points("b", WHITE, vec![
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
        ]));
        pf.save().expect("saves");
        assert!(!pf.dirty);

        let back = load(&path, [0.0; 3], WHITE).expect("loads");
        assert_eq!(back.format, VectorFormat::Salome);
        assert_eq!(back.layers.len(), 1);
        assert_eq!(back.layers[0].polylines.len(), 2);
        assert_eq!(back.layers[0].polylines[0].points()[0], [208_802.001, 373_320.002, 12.003]);
        assert_eq!(back.layers[0].polylines[1].points().len(), 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_atomic_write_leaves_no_temporary_behind() {
        let path = tmp("atomic.txt");
        let mut pf = PolylineFile::new(&path, VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![[0.0; 3], [1.0, 1.0, 1.0]]));
        pf.save().expect("saves");

        let dir = Path::new(&path).parent().unwrap();
        let stem = Path::new(&path).file_name().unwrap().to_str().unwrap();
        assert!(!dir.join(format!(".{stem}.haiviz-tmp")).exists());
        assert!(Path::new(&path).exists());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_external_edit_is_noticed_before_it_can_be_clobbered() {
        let path = tmp("mtime.txt");
        let mut pf = PolylineFile::new(&path, VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![[0.0; 3], [1.0, 1.0, 1.0]]));
        pf.save().expect("saves");
        assert!(!pf.changed_on_disk());

        // Someone else writes the file. Filesystem mtime resolution varies, so
        // set it explicitly rather than racing it.
        fs::write(&path, "0 0 0\n9 9 9\n").unwrap();
        let f = fs::File::open(&path).unwrap();
        f.set_times(fs::FileTimes::new()
            .set_modified(SystemTime::now() + std::time::Duration::from_secs(10))).unwrap();
        assert!(pf.changed_on_disk(), "an auto-save here would destroy their work");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_salome_save_says_what_it_would_drop() {
        let mut pf = PolylineFile::new(tmp("loss"), VectorFormat::Salome);
        pf.layers.push(Layer::new("roads"));
        pf.refresh_view([0.0; 3]);
        assert!(pf.salome_losses().is_empty());

        let mut line = Polyline::from_points("kerb", WHITE, vec![[0.0; 3], [1.0, 0.0, 0.0]]);
        line.attrs.insert("code".to_string(), "KRB".to_string());
        pf.add_line(0, line);
        pf.layers.push(Layer::new("kerbs"));
        let losses = pf.salome_losses();
        assert!(losses.iter().any(|l| l.contains("layers")), "{losses:?}");
        assert!(losses.iter().any(|l| l.contains("attribute")), "{losses:?}");
    }

    #[test]
    fn a_hidden_layer_is_not_drawn_and_not_pickable() {
        let mut pf = PolylineFile::new(tmp("vis"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![[0.0; 3], [1.0, 0.0, 0.0]]));
        let files = vec![pf];
        assert_eq!(visible_lines(&files).count(), 1);

        let mut files = files;
        files[0].layers[0].visible = false;
        assert_eq!(visible_lines(&files).count(), 0);
        let vp = glam::Mat4::IDENTITY;
        assert_eq!(pick(&files, &vp, 0.0, 0.0, 100.0, 100.0, 1e6), None);
        assert_eq!(pick_vertex(&files, &vp, 0.0, 0.0, 100.0, 100.0, 1e6), None);
    }

    #[test]
    fn a_geopackage_round_trips_through_the_model() {
        let path = tmp("model.gpkg");
        let mut pf = PolylineFile::new(&path, VectorFormat::GeoPackage);
        let mut layer = Layer::new("kerbs");
        layer.srs_id = 28992;
        pf.layers.push(layer);
        pf.refresh_view([0.0; 3]);

        let a = pf.add_line(0, Polyline::from_points("left kerb", WHITE, vec![
            [208_802.001, 373_320.002, 12.003],
            [208_812.5, 373_320.0, 12.25],
            [208_820.0, 373_325.0, 13.0],
        ])).unwrap();
        let b = pf.add_line(0, Polyline::from_points("island", WHITE, vec![
            [208_830.0, 373_310.0, 14.0], [208_840.0, 373_310.0, 14.0], [208_840.0, 373_320.0, 14.0],
        ])).unwrap();
        pf.set_closed(b, true);
        pf.save().expect("saves");
        assert!(!pf.dirty);

        let back = load(&path, [208_800.0, 373_300.0, 0.0], WHITE).expect("loads");
        assert_eq!(back.format, VectorFormat::GeoPackage);
        assert_eq!(back.layers.len(), 1);
        assert_eq!(back.layers[0].name, "kerbs");
        assert_eq!(back.layers[0].srs_id, 28992);

        let lines = &back.layers[0].polylines;
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].name, "left kerb", "the name column drives the panel label");
        assert_eq!(lines[0].points(), pf.get(a).unwrap().points(),
                   "world coordinates must be bit-exact through the file");
        assert!(lines[1].closed, "a ring must not come back as an open line");
        assert_eq!(lines[1].points().len(), 3, "and not with a repeated vertex either");
        // Derived attributes are refreshed on write, not carried over stale.
        assert_eq!(lines[0].attrs.get("n_vertices").map(String::as_str), Some("3"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn columns_haiviz_knows_nothing_about_survive_a_rewrite() {
        // The point of keeping `attrs` verbatim: a file from QGIS carrying its
        // own columns must not lose them just because haiviz opened it.
        let path = tmp("extra.gpkg");
        let mut pf = PolylineFile::new(&path, VectorFormat::GeoPackage);
        let mut layer = Layer::new("survey");
        layer.srs_id = 28992;
        layer.columns = vec![
            ("name".to_string(), "TEXT".to_string()),
            ("surveyor".to_string(), "TEXT".to_string()),
            ("epoch".to_string(), "INTEGER".to_string()),
        ];
        pf.layers.push(layer);
        pf.refresh_view([0.0; 3]);
        let mut line = Polyline::from_points("a", WHITE, vec![[0.0; 3], [1.0, 1.0, 1.0]]);
        line.attrs.insert("surveyor".to_string(), "TW".to_string());
        line.attrs.insert("epoch".to_string(), "2026".to_string());
        pf.add_line(0, line);
        pf.save().expect("saves");

        let back = load(&path, [0.0; 3], WHITE).expect("loads");
        let pl = &back.layers[0].polylines[0];
        assert_eq!(pl.attrs.get("surveyor").map(String::as_str), Some("TW"));
        assert_eq!(pl.attrs.get("epoch").map(String::as_str), Some("2026"));
        assert!(!back.layers[0].columns.iter().any(|(n, _)| n == "code"),
                "haiviz must not graft its own columns onto someone else's table");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_geopackage_says_what_a_salome_save_would_cost() {
        let mut pf = PolylineFile::new(tmp("cost.gpkg"), VectorFormat::GeoPackage);
        let mut layer = Layer::new("kerbs");
        layer.srs_id = 28992;
        pf.layers.push(layer);
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![[0.0; 3], [1.0, 1.0, 1.0]]));
        let losses = pf.salome_losses();
        assert!(losses.iter().any(|l| l.contains("coordinate reference system")), "{losses:?}");
        assert!(losses.iter().any(|l| l.contains("names")), "{losses:?}");
    }

    #[test]
    fn snapping_to_an_existing_line_prefers_its_vertices() {
        let mut pf = PolylineFile::new(tmp("snap"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        ]));
        let files = vec![pf];

        // Mid-segment: the projection onto the line.
        let (at, _) = nearest_on_lines(&files, [5.0, 0.3, 0.0], 1.0).expect("in range");
        assert_eq!(at, [5.0, 0.0, 0.0]);

        // Near an endpoint the vertex wins, so tying two lines together closes
        // exactly rather than leaving a hairline gap.
        let (at, _) = nearest_on_lines(&files, [10.05, 0.02, 0.0], 1.0).expect("in range");
        assert_eq!(at, [10.0, 0.0, 0.0]);

        // Out of range is nothing, not the nearest thing anyway.
        assert!(nearest_on_lines(&files, [5.0, 50.0, 0.0], 1.0).is_none());

        // A hidden layer is not snappable, same as it is not pickable.
        let mut files = files;
        files[0].layers[0].visible = false;
        assert!(nearest_on_lines(&files, [5.0, 0.3, 0.0], 1.0).is_none());
    }

    #[test]
    fn snapping_to_a_line_works_from_a_ray_with_no_cloud_under_it() {
        // The case the point form could not serve: the cursor is over a gap,
        // so there is no surface point to search around — which is exactly
        // when tying into an existing line matters.
        let mut pf = PolylineFile::new(tmp("rayline"), VectorFormat::Salome);
        pf.layers.push(Layer::new("l"));
        pf.refresh_view([0.0; 3]);
        pf.add_line(0, Polyline::from_points("a", WHITE, vec![
            [0.0, 0.0, 5.0], [10.0, 0.0, 5.0],
        ]));
        let files = vec![pf];

        // Straight down onto the middle of the segment.
        let (at, _) = nearest_on_lines_to_ray(&files, [5.0, 0.0, 100.0], [0.0, 0.0, -1.0], 0.5)
            .expect("hits");
        assert_eq!(at, [5.0, 0.0, 5.0]);

        // Straight down near the endpoint: the vertex wins.
        let (at, _) = nearest_on_lines_to_ray(&files, [10.02, 0.0, 100.0], [0.0, 0.0, -1.0], 0.5)
            .expect("hits");
        assert_eq!(at, [10.0, 0.0, 5.0]);

        // Oblique, which the point form could not do at all.
        let (at, _) = nearest_on_lines_to_ray(&files, [5.0, -50.0, 55.0], [0.0, 1.0, -1.0], 0.5)
            .expect("hits");
        assert!((at[0] - 5.0).abs() < 1e-6 && (at[2] - 5.0).abs() < 1e-6, "{at:?}");

        // Too far off the line, and pointing away, both give nothing.
        assert!(nearest_on_lines_to_ray(&files, [5.0, 20.0, 100.0], [0.0, 0.0, -1.0], 0.5).is_none());
        assert!(nearest_on_lines_to_ray(&files, [5.0, 0.0, 100.0], [0.0, 0.0, 1.0], 0.5).is_none());
        // A degenerate direction is refused rather than dividing by zero.
        assert!(nearest_on_lines_to_ray(&files, [5.0, 0.0, 100.0], [0.0; 3], 0.5).is_none());
    }

    #[test]
    fn detect_format_reads_the_bytes_not_the_extension() {
        let path = tmp("notreally.txt");
        fs::write(&path, b"SQLite format 3\0rest of a database").unwrap();
        assert_eq!(detect_format(&path), VectorFormat::GeoPackage);
        fs::write(&path, b"1.0 2.0 3.0\n4.0 5.0 6.0\n").unwrap();
        assert_eq!(detect_format(&path), VectorFormat::Salome);
        // A file too short to hold the magic is not a database.
        fs::write(&path, b"1 2 3").unwrap();
        assert_eq!(detect_format(&path), VectorFormat::Salome);
        let _ = fs::remove_file(&path);
    }
}
