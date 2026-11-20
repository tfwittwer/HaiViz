//! Meshes and splats: what haiviz draws beside the point clouds without them
//! being points.
//!
//! These layers are for looking at. They share the clouds' viewer origin, so a
//! tileset or a splat in the same CRS lands on the cloud, and they are
//! depth-tested against it both ways. They are not part of anything that works
//! on points: picking, snapping, sections, classification and export all see
//! only the clouds.
//!
//! Draw order: a tileset's points go in the cloud's own point pass, so EDL
//! and the point size apply to them as to the clouds. After that pass (and
//! after EDL's composite and depth replay, when EDL is on): opaque meshes,
//! then splats blended back to front over everything, testing depth but not
//! writing it.

use std::path::{Path, PathBuf};

use egui_macroquad::egui;
use glam::Vec3;
use miniquad::*;

use crate::raw_miniquad::ZClip;
use crate::splats::{self, SplatCloud, SplatFile};
use crate::tiles3d::{Lighting, Tileset, TilesetDesc};

pub enum LoadKind {
    /// LAS / LAZ / COPC, or a folder of tiles: the existing loaders.
    PointCloud,
    Tileset(PathBuf),
    Splats(PathBuf),
    Unsupported(String),
}

fn dir_has_las(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x.eq_ignore_ascii_case("las") || x.eq_ignore_ascii_case("laz"))
            })
        })
        .unwrap_or(false)
}

/// Which loader a path belongs to. A folder holding a `tileset.json` and no
/// LAS/LAZ is the tileset, so the folder can be passed or dropped as is.
pub fn classify(p: &Path) -> LoadKind {
    if p.is_dir() {
        let ts = p.join("tileset.json");
        if ts.is_file() && !dir_has_las(p) {
            return LoadKind::Tileset(ts);
        }
        return LoadKind::PointCloud;
    }
    let ext = p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("json") => LoadKind::Tileset(p.to_path_buf()),
        Some("ply") if splats::is_splat_ply(p) => LoadKind::Splats(p.to_path_buf()),
        Some("ply") => LoadKind::Unsupported(format!(
            "{}: not a Gaussian-splat PLY (no scale_0 / rot_0 / opacity)",
            p.display()
        )),
        _ => LoadKind::PointCloud,
    }
}

/// A layer read from disk, before the viewer origin is known.
pub enum Prepared {
    Tileset(TilesetDesc),
    Splats(SplatFile),
}

impl Prepared {
    pub fn world_bounds(&self) -> ([f64; 3], [f64; 3]) {
        let (lo, hi) = match self {
            Prepared::Tileset(d) => d.world_bounds(),
            Prepared::Splats(s) => s.core_bounds(),
        };
        (lo.to_array(), hi.to_array())
    }
}

/// Read a layer's file. Blocking, as LAS loads are: a 4-million-splat PLY
/// takes a second or two.
pub fn prepare(kind: LoadKind) -> Result<Prepared, String> {
    match kind {
        LoadKind::Tileset(p) => Tileset::open(&p).map(Prepared::Tileset),
        LoadKind::Splats(p) => splats::read_ply(&p).map(Prepared::Splats),
        LoadKind::Unsupported(e) => Err(e),
        LoadKind::PointCloud => Err("not a mesh or splat file".into()),
    }
}

#[derive(Clone, Copy)]
enum LayerRef {
    Tileset(usize),
    Splats(usize),
}

pub struct Layers {
    pub tilesets: Vec<Tileset>,
    pub splats: Vec<SplatCloud>,
    /// Refine a tile while its geometric error spans more than this many
    /// physical pixels. 3D Tiles viewers default to 16 for perspective views;
    /// the orthographic view shows the whole screen at the nearest LOD, where
    /// 16 looked blobby on `haires amesh` tiles, so the default is sharper.
    pub max_sse: f32,
    /// The same for point tiles. Their geometric error is about their point
    /// spacing, so 1 px streams them as densely as a COPC cloud is.
    pub point_sse: f32,
    pub budget_mb: f32,
    pub lighting: Lighting,
    /// The last load that failed, for the panel.
    pub status: Option<String>,
    frame: u64,
    unload: Option<LayerRef>,
}

impl Default for Layers {
    fn default() -> Self {
        Layers {
            tilesets: Vec::new(),
            splats: Vec::new(),
            max_sse: 8.0,
            point_sse: 1.0,
            budget_mb: 2048.0,
            lighting: Lighting::Material,
            status: None,
            frame: 0,
            unload: None,
        }
    }
}

impl Layers {
    pub fn is_empty(&self) -> bool {
        self.tilesets.is_empty() && self.splats.is_empty()
    }

    pub fn attach(&mut self, ctx: &mut dyn RenderingBackend, p: Prepared, origin: [f64; 3], front: Vec3) {
        match p {
            Prepared::Tileset(d) => self.tilesets.push(Tileset::attach(ctx, d, origin)),
            Prepared::Splats(s) => self.splats.push(SplatCloud::attach(ctx, s, origin, front)),
        }
        self.status = None;
    }

    /// Viewer-space box over every layer, for Zoom All.
    pub fn aabb_viewer(&self, origin: [f64; 3]) -> Option<(Vec3, Vec3)> {
        let boxes = self
            .tilesets
            .iter()
            .map(|t| t.world_bounds())
            .chain(self.splats.iter().map(|s| s.world_bounds()));
        let o = glam::DVec3::from(origin);
        let mut out: Option<(Vec3, Vec3)> = None;
        for (lo, hi) in boxes {
            let (lo, hi) = ((lo - o).as_vec3(), (hi - o).as_vec3());
            out = Some(match out {
                Some((a, b)) => (a.min(lo), b.max(hi)),
                None => (lo, hi),
            });
        }
        out
    }

    /// Apply an Unload pressed in last frame's panel.
    pub fn apply_unload(&mut self, ctx: &mut dyn RenderingBackend) {
        match self.unload.take() {
            Some(LayerRef::Tileset(i)) if i < self.tilesets.len() => {
                let t = self.tilesets.remove(i);
                ::log::info!("Unloading tileset {}", t.path.display());
                t.unload(ctx);
            }
            Some(LayerRef::Splats(i)) if i < self.splats.len() => {
                let s = self.splats.remove(i);
                ::log::info!("Unloading splats {}", s.path.display());
                s.unload(ctx);
            }
            _ => {}
        }
    }

    /// Stream tiles and re-sort splats for this frame's camera. `mpp` is
    /// metres per physical pixel.
    pub fn update(&mut self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, front: Vec3, mpp: f32) {
        self.frame += 1;
        // The budget is shared: each tileset gets its share of it.
        let budget = (self.budget_mb as usize) << 20;
        let per = budget / self.tilesets.len().max(1);
        for t in &mut self.tilesets {
            t.update(ctx, view_proj, mpp as f64, self.max_sse as f64, self.point_sse as f64, per, self.frame);
        }
        for s in &mut self.splats {
            s.update(ctx, front);
        }
    }

    /// Draw the tilesets' points into the point pass that is current — the
    /// EDL target or the default framebuffer — beside the clouds' own.
    pub fn draw_points(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, point_size: f32, z_clip: ZClip, depth_only: bool) {
        for t in &self.tilesets {
            t.draw_points(ctx, view_proj, point_size, z_clip, depth_only);
        }
    }

    /// Draw into the default framebuffer over the cloud. `vp` is the view
    /// rect in physical pixels, as `ViewRect::gl` gives it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, front: Vec3, right: Vec3, up: Vec3, vp: (i32, i32, i32, i32), z_clip: ZClip) {
        if self.is_empty() {
            return;
        }
        ctx.begin_default_pass(PassAction::Nothing);
        ctx.apply_viewport(vp.0, vp.1, vp.2, vp.3);
        // A key light over the viewer's right shoulder, so it turns with
        // the view and a face turned to the camera is never black.
        let light = (-front + up * 0.6 + right * 0.4).normalize();
        for t in &self.tilesets {
            t.draw(ctx, view_proj, light, self.lighting, z_clip);
        }
        for s in &self.splats {
            s.draw(ctx, view_proj, (vp.2 as f32, vp.3 as f32), z_clip);
        }
        ctx.end_render_pass();
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, pending_loads: &mut Vec<String>) {
        if ui.button("Open tileset / splats…").clicked()
            && let Some(paths) = rfd::FileDialog::new()
                .add_filter("Meshes and splats", &["json", "ply"])
                .add_filter("3D Tiles tileset", &["json"])
                .add_filter("Gaussian splats", &["ply"])
                .pick_files()
        {
            pending_loads.extend(paths.into_iter().map(|p| p.to_string_lossy().to_string()));
        }
        if let Some(s) = &self.status {
            ui.colored_label(egui::Color32::LIGHT_RED, s);
        }
        if self.is_empty() {
            ui.label("A 3D Tiles tileset.json (or its folder) or a Gaussian-splat .ply.");
            return;
        }
        for (i, t) in self.tilesets.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut t.visible, "").on_hover_text("Show / hide this tileset");
                ui.label(t.name()).on_hover_text(t.path.display().to_string());
                if ui.button("Unload").clicked() {
                    self.unload = Some(LayerRef::Tileset(i));
                }
            });
            let st = &t.stats;
            let mut drawn = Vec::new();
            if st.drawn_triangles > 0 || !t.has_points {
                drawn.push(format!("{:.1} M triangles", st.drawn_triangles as f64 / 1e6));
            }
            if st.drawn_points > 0 || t.has_points {
                drawn.push(format!("{:.1} M points", st.drawn_points as f64 / 1e6));
            }
            ui.label(format!(
                "  {} tiles drawn, {} | {} loaded ({:.0} MB), {} queued",
                st.drawn_tiles,
                drawn.join(", "),
                st.loaded,
                st.gpu_bytes as f64 / (1 << 20) as f64,
                st.queued,
            ));
            if st.budget_full {
                ui.colored_label(egui::Color32::YELLOW, "  GPU budget full: loading paused — raise the budget or lower the detail");
            }
            if st.failed > 0 {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("  {} failed: {}", st.failed, st.first_error.as_deref().unwrap_or("")),
                );
            }
            if st.warnings > 0 {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("  {} warnings: {}", st.warnings, st.first_warning.as_deref().unwrap_or("")),
                );
            }
            if t.ecef {
                ui.colored_label(egui::Color32::YELLOW, "  ECEF: shown in a local east-north-up frame, not aligned with projected clouds");
            }
        }
        for (i, s) in self.splats.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.visible, "").on_hover_text("Show / hide these splats");
                ui.label(s.name()).on_hover_text(s.path.display().to_string());
                if ui.button("Unload").clicked() {
                    self.unload = Some(LayerRef::Splats(i));
                }
            });
            let placed = match &s.georef.source {
                Some(p) => format!(
                    ", placed by {} at ({:.0}, {:.0})",
                    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                    s.georef.offset[0],
                    s.georef.offset[1]
                ),
                None => ", file coordinates".to_string(),
            };
            ui.label(format!("  {:.2} M splats{placed}", s.count as f64 / 1e6))
                .on_hover_text(format!("{} fully transparent splats skipped", s.dropped));
            ui.checkbox(&mut s.crop, "Crop floaters").on_hover_text(
                "Hide splats centred well outside the scene: the large faint background splats \
                 training leaves around it, which an orthographic view puts in front of everything.",
            );
        }
        let has_mesh = self.tilesets.iter().any(|t| t.has_mesh);
        let has_points = self.tilesets.iter().any(|t| t.has_points);
        if has_mesh {
            ui.horizontal(|ui| {
                ui.label("Mesh detail:");
                ui.add(egui::Slider::new(&mut self.max_sse, 1.0..=64.0).logarithmic(true).suffix(" px"))
                    .on_hover_text("Screen-space error allowed before a finer tile is loaded. Lower is sharper and loads more.");
            });
        }
        if has_points {
            ui.horizontal(|ui| {
                ui.label("Point detail:");
                ui.add(egui::Slider::new(&mut self.point_sse, 0.25..=64.0).logarithmic(true).suffix(" px"))
                    .on_hover_text(
                        "Screen-space error allowed before a finer point tile is loaded — about the point spacing \
                         on screen. Lower is denser and loads more. Point size and EDL are the clouds' settings.",
                    );
            });
        }
        if !self.tilesets.is_empty() {
            ui.horizontal(|ui| {
                ui.label("GPU budget:");
                ui.add(egui::Slider::new(&mut self.budget_mb, 256.0..=16384.0).logarithmic(true).suffix(" MB"))
                    .on_hover_text(
                        "Tiles not on screen are released beyond this. Tiles on screen are never released; \
                         when they fill it, loading pauses.",
                    );
            });
        }
        if has_mesh {
            ui.horizontal(|ui| {
                ui.label("Lighting:");
                egui::ComboBox::from_id_salt("mesh_lighting")
                    .selected_text(self.lighting.label())
                    .show_ui(ui, |ui| {
                        for l in Lighting::ALL {
                            ui.selectable_value(&mut self.lighting, l, l.label());
                        }
                    });
            });
        }
    }
}
