use std::env;
use std::path::Path;
//use rfd;
use log::{info,error};
//use console::style;
use crate::glam::vec3;

use macroquad::prelude::*;

use crate::raw_miniquad::{ClassDefs, ClassVis, CopcStage, EdlRenderer, SectionData, Stage, TileStageManager, build_section_from_raw_stages, build_section_from_stages, classification_color, default_class_label, load_class_defs, make_section_data, save_class_defs, scan_dir_for_copc};
use crate::classify::{ClassMask, effective_class};
use las::Point as LasPoint;

const MOVE_SPEED: f32 = 0.5;
const LOOK_SPEED: f32 = 0.5;

/// Upper bound on the points a force-loaded section corridor may hold.
/// Pinning bypasses the LOD point budget by design, so it needs its own guard:
/// a long corridor through a dense cloud can otherwise pull the whole file into
/// memory at once.
const CORRIDOR_POINT_LIMIT: i64 = 60_000_000;

/// Side of the box of COPC nodes pinned around the cursor while digitising,
/// in metres. Widened to four snap radii when the radius is large enough to
/// need it, so the pinned neighbourhood always covers what a snap can reach.
const SNAP_NEIGHBOURHOOD_M: f64 = 20.0;

mod raw_miniquad;
mod spatial;
mod polylines;
mod gpkg;
mod vectorise;
mod gcp;
mod classify;
mod export;

mod view_cube {
    use macroquad::prelude::*;
    use std::f32::consts::PI;

    const SIZE: f32 = 70.0;
    const MARGIN: f32 = 20.0;

    const CORNERS: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0], // 0
        [ 1.0, -1.0, -1.0], // 1
        [ 1.0,  1.0, -1.0], // 2
        [-1.0,  1.0, -1.0], // 3
        [-1.0, -1.0,  1.0], // 4
        [ 1.0, -1.0,  1.0], // 5
        [ 1.0,  1.0,  1.0], // 6
        [-1.0,  1.0,  1.0], // 7
    ];

    struct FaceDef {
        corners: [usize; 4],
        label: &'static str,
        normal: [f32; 3],
        snap_yaw: f32,
        snap_pitch: f32,
        color: Color,
    }

    fn faces() -> [FaceDef; 6] {
        [
            FaceDef { corners: [4, 5, 6, 7], label: "Top",    normal: [ 0.0, 0.0, 1.0], snap_yaw: PI/2.0, snap_pitch: -PI/2.0, color: Color::new(0.3, 0.5, 0.8, 0.85) },
            FaceDef { corners: [3, 2, 1, 0], label: "Bot",    normal: [ 0.0, 0.0,-1.0], snap_yaw: PI/2.0, snap_pitch:  PI/2.0, color: Color::new(0.2, 0.3, 0.5, 0.85) },
            FaceDef { corners: [1, 2, 6, 5], label: "E",      normal: [ 1.0, 0.0, 0.0], snap_yaw: 0.0,    snap_pitch:  0.0,     color: Color::new(0.8, 0.3, 0.3, 0.85) },
            FaceDef { corners: [0, 3, 7, 4], label: "W",      normal: [-1.0, 0.0, 0.0], snap_yaw: PI,     snap_pitch:  0.0,     color: Color::new(0.5, 0.2, 0.2, 0.85) },
            FaceDef { corners: [3, 2, 6, 7], label: "N",      normal: [ 0.0, 1.0, 0.0], snap_yaw: PI/2.0, snap_pitch:  0.0,     color: Color::new(0.3, 0.7, 0.3, 0.85) },
            FaceDef { corners: [0, 1, 5, 4], label: "S",      normal: [ 0.0,-1.0, 0.0], snap_yaw: -PI/2.0, snap_pitch: 0.0,     color: Color::new(0.2, 0.4, 0.2, 0.85) },
        ]
    }

    fn rotate(p: &[f32; 3], right: Vec3, up: Vec3, front: Vec3) -> [f32; 3] {
        [
            p[0] * right.x + p[1] * right.y + p[2] * right.z,
            p[0] * up.x    + p[1] * up.y    + p[2] * up.z,
            -(p[0] * front.x + p[1] * front.y + p[2] * front.z),
        ]
    }

    fn to_screen(rotated: &[f32; 3], cx: f32, cy: f32, scale: f32) -> (f32, f32) {
        (cx + rotated[0] * scale, cy - rotated[1] * scale)
    }

    fn point_in_quad(px: f32, py: f32, quad: &[(f32, f32); 4]) -> bool {
        let mut sign = None;
        for i in 0..4 {
            let (x1, y1) = quad[i];
            let (x2, y2) = quad[(i + 1) % 4];
            let cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
            let s = cross > 0.0;
            match sign {
                None => sign = Some(s),
                Some(prev) => if prev != s { return false; }
            }
        }
        true
    }

    /// Snap targets for the keyboard view shortcuts, as `(yaw, pitch)`.
    ///
    /// These reuse the cube's own face snaps so the keys and the cube always
    /// agree. Naming follows the usual CAD convention, where the view is named
    /// for where the camera sits, not for the direction it looks:
    /// Front looks north from the south, Right looks west from the east.
    pub const VIEW_TOP: (f32, f32) = (PI / 2.0, -PI / 2.0);
    pub const VIEW_FRONT: (f32, f32) = (PI / 2.0, 0.0);
    pub const VIEW_BACK: (f32, f32) = (-PI / 2.0, 0.0);
    pub const VIEW_LEFT: (f32, f32) = (0.0, 0.0);
    pub const VIEW_RIGHT: (f32, f32) = (PI, 0.0);

    /// `keys` gates the keyboard shortcuts separately from `interactive`, so a
    /// letter typed into a class label does not fling the camera around.
    pub fn draw_and_interact(front: Vec3, right: Vec3, up: Vec3, interactive: bool, keys: bool) -> Option<(f32, f32)> {
        let cx = screen_width() - MARGIN - SIZE;
        let cy = MARGIN + SIZE;
        let scale = SIZE * 0.45;

        let face_defs = faces();

        let rotated_corners: Vec<[f32; 3]> = CORNERS.iter()
            .map(|c| rotate(c, right, up, front))
            .collect();

        struct RotatedFace {
            screen_corners: [(f32, f32); 4],
            center_z: f32,
            center_screen: (f32, f32),
            label: &'static str,
            snap_yaw: f32,
            snap_pitch: f32,
            color: Color,
            facing_camera: bool,
        }

        let mut rot_faces: Vec<RotatedFace> = face_defs.iter().map(|f| {
            let rc: Vec<[f32; 3]> = f.corners.iter().map(|&i| rotated_corners[i]).collect();
            let screen_corners = [
                to_screen(&rc[0], cx, cy, scale),
                to_screen(&rc[1], cx, cy, scale),
                to_screen(&rc[2], cx, cy, scale),
                to_screen(&rc[3], cx, cy, scale),
            ];
            let rn = rotate(&f.normal, right, up, front);
            let center_z = rc.iter().map(|c| c[2]).sum::<f32>() / 4.0;
            let center_screen = (
                screen_corners.iter().map(|c| c.0).sum::<f32>() / 4.0,
                screen_corners.iter().map(|c| c.1).sum::<f32>() / 4.0,
            );
            RotatedFace {
                screen_corners,
                center_z,
                center_screen,
                label: f.label,
                snap_yaw: f.snap_yaw,
                snap_pitch: f.snap_pitch,
                color: f.color,
                facing_camera: rn[2] > 0.0,
            }
        }).collect();

        rot_faces.sort_by(|a, b| a.center_z.partial_cmp(&b.center_z).unwrap());

        draw_circle(cx, cy, SIZE * 0.7, Color::new(0.15, 0.15, 0.15, 0.6));

        for face in &rot_faces {
            let c = face.screen_corners;
            draw_triangle(
                Vec2::new(c[0].0, c[0].1),
                Vec2::new(c[1].0, c[1].1),
                Vec2::new(c[2].0, c[2].1),
                face.color,
            );
            draw_triangle(
                Vec2::new(c[0].0, c[0].1),
                Vec2::new(c[2].0, c[2].1),
                Vec2::new(c[3].0, c[3].1),
                face.color,
            );

            for i in 0..4 {
                let j = (i + 1) % 4;
                draw_line(c[i].0, c[i].1, c[j].0, c[j].1, 1.0, Color::new(0.8, 0.8, 0.8, 0.6));
            }

            if face.facing_camera {
                let font_size = 14.0;
                let text_w = face.label.len() as f32 * font_size * 0.35;
                draw_text(
                    face.label,
                    face.center_screen.0 - text_w / 2.0,
                    face.center_screen.1 + font_size * 0.25,
                    font_size,
                    WHITE,
                );
            }
        }

        if interactive && keys {
            for (key, view) in [
                (KeyCode::T, VIEW_TOP),
                (KeyCode::F, VIEW_FRONT),
                (KeyCode::B, VIEW_BACK),
                (KeyCode::L, VIEW_LEFT),
                (KeyCode::R, VIEW_RIGHT),
            ] {
                if is_key_pressed(key) {
                    return Some(view);
                }
            }
        }


        if interactive && is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = macroquad::input::mouse_position();
            let dx = mx - cx;
            let dy = my - cy;
            if dx * dx + dy * dy < (SIZE * 0.8) * (SIZE * 0.8) {
                for face in rot_faces.iter().rev() {
                    if face.facing_camera && point_in_quad(mx, my, &face.screen_corners) {
                        return Some((face.snap_yaw, face.snap_pitch));
                    }
                }
            }
        }

        None
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum ColorMode {
    Height,
    Intensity,
    Classification,
    Rgb,
    PointSourceId,
    Time,
}

impl ColorMode {
    fn label(&self) -> &'static str {
        match self {
            ColorMode::Height => "Height",
            ColorMode::Intensity => "Intensity",
            ColorMode::Classification => "Classification",
            ColorMode::Rgb => "RGB",
            ColorMode::PointSourceId => "PointSourceId",
            ColorMode::Time => "Time",
        }
    }

    const ALL: [ColorMode; 6] = [
        ColorMode::Height,
        ColorMode::Intensity,
        ColorMode::Classification,
        ColorMode::Rgb,
        ColorMode::PointSourceId,
        ColorMode::Time,
    ];
}


#[derive(Clone, Copy, PartialEq)]
enum AppMode {
    View3D,
    PointPick,
    SectionPickStart,
    SectionPickEnd,
    SectionAdjustDepth,
    SectionView,
    MeasureFirst,
    MeasureSecond,
    SectionMeasureFirst,
    SectionMeasureSecond,
    SectionPointPick,
    /// Waiting for the force-loaded section corridor to finish streaming.
    /// Only COPC reaches this state; other modes have all points resident.
    SectionCorridorLoad,
    /// Drawing a lasso in the section view to reclassify the points inside it.
    SectionLasso,
    /// Drawing a plan-view lasso to reclassify what falls inside it. What
    /// "inside" means is `LassoKind`, kept separate so both flavours share
    /// one drawing mode.
    GroundLasso,
    /// Drawing a polyline onto the cloud, vertex by vertex.
    ///
    /// Unlike the lassos this works from any camera angle — an explicit
    /// exception to the `TOP_VIEW_HINT` rule. A lasso in an oblique view is
    /// ambiguous because a screen polygon says nothing about depth; a
    /// digitised vertex comes from a real point and is unambiguous from
    /// anywhere.
    Digitise,
}

/// What a plan-view lasso selects once it is closed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LassoKind {
    /// Fit the local ground and take only what stands above it.
    AboveGround,
    /// Take everything in the column, bounded only by the height band.
    Inside,
}

/// How far off straight-down the camera may sit and still count as top view.
///
/// The footprint is unprojected onto one horizontal plane, so any tilt shears
/// it in proportion to a point's height above that plane — at one degree a
/// point 20 m up already lands a third of a metre from where it was clicked.
/// The snaps (`T`, or the cube's top face) land exactly on zero, so this is a
/// tolerance for floating point, not an invitation to eyeball it.
const TOP_VIEW_TOL: f32 = std::f32::consts::PI / 180.0;

/// Shown wherever a plan-view lasso is refused for want of the top view.
const TOP_VIEW_HINT: &str = "Top view only — press T or click the cube's Top face.";

/// True when the camera is looking straight down, which is the only attitude
/// the plan-view lassos are meaningful from. Yaw does not matter: the view
/// direction is vertical whatever it is.
fn is_top_view(pitch: f32) -> bool {
    (pitch - view_cube::VIEW_TOP.1).abs() <= TOP_VIEW_TOL
}

impl LassoKind {
    fn label(self) -> &'static str {
        match self {
            LassoKind::AboveGround => "Above-ground lasso",
            LassoKind::Inside => "Inside lasso",
        }
    }
}

/*impl AppMode {
    fn label(&self) -> &'static str {
        match self {
            AppMode::View3D => "View3D",
            AppMode::PointPick => "PointPick",
            AppMode::SectionPickStart => "SectionPickStart",
            AppMode::SectionPickEnd => "SectionPickEnd",
            AppMode::SectionAdjustDepth => "SectionAdjustDepth",
            AppMode::SectionView => "SectionView",
            AppMode::MeasureFirst => "MeasureFirst",
            AppMode::MeasureSecond => "MeasureSecond",
            AppMode::SectionMeasureFirst => "SectionMeasureFirst",
            AppMode::SectionMeasureSecond => "SectionMeasureSecond",
        }
    }
}*/

/// Unproject screen coordinates to world XY plane at given Z height.
fn screen_to_world_xy(sx: f32, sy: f32, view_proj: &glam::Mat4, z_plane: f32) -> Option<(f32, f32)> {
    let inv = view_proj.inverse();
    let ndc_x = sx / screen_width() * 2.0 - 1.0;
    let ndc_y = 1.0 - sy / screen_height() * 2.0;

    let near = inv * glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far = inv * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near = glam::Vec3::new(near.x / near.w, near.y / near.w, near.z / near.w);
    let far = glam::Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);

    let dir = far - near;
    if dir.z.abs() < 1e-6 {
        return Some((near.x, near.y));
    }
    let t = (z_plane - near.z) / dir.z;
    Some((near.x + dir.x * t, near.y + dir.y * t))
}

/// The file a new vector layer should sit beside.
///
/// The first loaded source wins. Digitising onto several clouds at once is
/// possible, but the layer has to land somewhere, and "beside the first one"
/// is at least predictable.
fn cloud_source_path(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
) -> Option<String> {
    if let Some(s) = stages.first() { return Some(s.filepath.clone()); }
    if let Some(c) = copc_stages.first() { return Some(c.filepath.clone()); }
    // Tile mode has a directory rather than a file; the sidecar goes inside it.
    tile_mgr.as_ref()
        .and_then(|m| m.tiles.first())
        .and_then(|t| Path::new(&t.filepath).parent().map(|p| p.join("tiles").to_string_lossy().into_owned()))
}

/// The cursor as a ray in viewer space.
///
/// Under an orthographic camera the near plane sits far behind the data, so
/// every point in front of the camera has a positive distance along this ray —
/// which is what lets `nearest_ray` reject everything behind the operator.
fn cursor_ray(view_proj: &glam::Mat4, mx: f32, my: f32) -> ([f32; 3], [f32; 3]) {
    let inv = view_proj.inverse();
    let ndc_x = mx / screen_width() * 2.0 - 1.0;
    let ndc_y = 1.0 - my / screen_height() * 2.0;
    let unproject = |z: f32| {
        let p = inv * glam::Vec4::new(ndc_x, ndc_y, z, 1.0);
        glam::Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
    };
    let near = unproject(-1.0);
    let dir = (unproject(1.0) - near).normalize_or_zero();
    ([near.x, near.y, near.z], [dir.x, dir.y, dir.z])
}

/// Resolve the vertex under the cursor for the active snap mode.
///
/// The percentile and same-Z modes need an XY to work from, and it comes from
/// the ray hit rather than from a fixed plane: that is what makes them mean
/// "within 0.2 m of the surface I am pointing at" from an oblique view as well
/// as from directly above.
#[allow(clippy::too_many_arguments)]
fn resolve_snap(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    poly_files: &[polylines::PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    data_offset: [f64; 3],
    dig: &vectorise::Digitise,
) -> vectorise::SnapHit {
    use vectorise::{SnapHit, SnapMode, SnapQuality};
    let to_world = |v: [f32; 3]| [
        v[0] as f64 + data_offset[0],
        v[1] as f64 + data_offset[1],
        v[2] as f64 + data_offset[2],
    ];
    let empty = |note: &str| SnapHit {
        world: [0.0; 3], quality: SnapQuality::Empty, note: Some(note.to_string()),
    };

    let (origin, dir) = cursor_ray(view_proj, mx, my);
    let r = dig.radius;

    // Where on the surface the cursor is pointing. Every mode except
    // `Existing` starts from this.
    let mut anchor: Option<[f32; 3]> = None;
    let mut best_t = f32::INFINITY;
    for st in stages {
        if !st.visible { continue; }
        if let Some(h) = st.snap_nearest_ray(origin, dir, r)
            && h.t < best_t { best_t = h.t; anchor = Some(h.pos); }
    }
    for cs in copc_stages {
        if !cs.visible { continue; }
        if let Some((h, _)) = cs.snap_nearest_ray(origin, dir, r)
            && h.t < best_t { best_t = h.t; anchor = Some(h.pos); }
    }
    if let Some(mgr) = tile_mgr
        && let Some((_, h)) = mgr.snap_nearest_ray(origin, dir, r)
        && h.t < best_t { anchor = Some(h.pos); }

    if dig.snap_mode == SnapMode::Existing {
        // Anchor on the cloud if there is one, so the search is around the
        // surface; otherwise use the ray at the height band's middle.
        let probe = match anchor {
            Some(a) => to_world(a),
            None => {
                let z = origin[2] + dir[2] * 0.0;
                to_world([origin[0], origin[1], z])
            }
        };
        return match polylines::nearest_on_lines(poly_files, probe, r as f64) {
            Some((w, _)) => SnapHit::good(w),
            None => empty("No polyline within the snap radius."),
        };
    }

    let Some(anchor) = anchor else {
        return empty("Nothing within the snap radius. Widen it with Ctrl + wheel,                       or Shift-click to place a vertex freely.");
    };

    // The COPC spacing guard. Zoomed out the resident set is a thinned sample,
    // so a percentile taken here would be over whatever subset happened to be
    // loaded and wrong by decimetres with nothing looking wrong. Refuse rather
    // than guess — and only for the modes that actually aggregate, since the
    // nearest point is a real point at any LOD.
    if matches!(dig.snap_mode, SnapMode::LowZ | SnapMode::HighZ) {
        for cs in copc_stages {
            if !cs.visible { continue; }
            if let Some(spacing) = cs.lod_spacing_near(anchor[0], anchor[1], r)
                && spacing > r as f64 {
                return SnapHit {
                    world: to_world(anchor),
                    quality: SnapQuality::Coarse,
                    note: Some(format!(
                        "Point spacing here is {spacing:.2} m, wider than the {r:.2} m                          snap radius — a percentile would be taken over whatever the                          LOD happened to load. Zoom in, or widen the radius.",
                    )),
                };
            }
        }
    }

    match dig.snap_mode {
        SnapMode::Nearest => SnapHit::good(to_world(anchor)),
        SnapMode::SameZ => match dig.last() {
            // The XY the operator pointed at, the Z they already committed to.
            Some(prev) => SnapHit::good([
                anchor[0] as f64 + data_offset[0],
                anchor[1] as f64 + data_offset[1],
                prev[2],
            ]),
            None => SnapHit::good(to_world(anchor)),
        },
        SnapMode::LowZ | SnapMode::HighZ => {
            let pct = if dig.snap_mode == SnapMode::LowZ { dig.low_pct } else { dig.high_pct };
            // Candidates are pooled across every backend before the rank is
            // taken: a percentile over a union is not the union of
            // percentiles, and a snap radius straddling a tile edge is the
            // normal case.
            let mut zs: Vec<(f32, [f32; 3])> = Vec::new();
            for st in stages {
                if st.visible { st.snap_z_candidates(anchor[0], anchor[1], r, &mut zs); }
            }
            for cs in copc_stages {
                if cs.visible { cs.snap_z_candidates(anchor[0], anchor[1], r, &mut zs); }
            }
            if let Some(mgr) = tile_mgr {
                mgr.snap_z_candidates(anchor[0], anchor[1], r, &mut zs);
            }
            match crate::spatial::percentile_by_z(&mut zs, pct) {
                Some((_, p)) => SnapHit::good(to_world(p)),
                None => empty("Nothing within the snap radius."),
            }
        }
        SnapMode::Existing => unreachable!("handled above"),
    }
}

/// Project a world position to screen coordinates.
fn world_to_screen(pos: glam::Vec3, view_proj: &glam::Mat4) -> (f32, f32) {
    let clip = *view_proj * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let sx = (ndc_x + 1.0) / 2.0 * screen_width();
    let sy = (1.0 - ndc_y) / 2.0 * screen_height();
    (sx, sy)
}

/// Draw the section corridor as 3D wireframe box.
fn draw_section_corridor(
    start: (f32, f32),
    end: (f32, f32),
    depth: f32,
    height_range: f32,
) {
    //println!("{} {} {} {}",start.0,start.1,end.0,end.1);
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 { return; }
    let nx = -dy / len * depth;
    let ny = dx / len * depth;

    // 4 bottom corners (z=0), 4 top corners (z=height_range)
    let b = [
        vec3(start.0 + nx, start.1 + ny, 0.0),
        vec3(start.0 - nx, start.1 - ny, 0.0),
        vec3(end.0 - nx,   end.1 - ny,   0.0),
        vec3(end.0 + nx,   end.1 + ny,   0.0),
    ];
    let t = [
        vec3(start.0 + nx, start.1 + ny, height_range),
        vec3(start.0 - nx, start.1 - ny, height_range),
        vec3(end.0 - nx,   end.1 - ny,   height_range),
        vec3(end.0 + nx,   end.1 + ny,   height_range),
    ];

    let col = YELLOW;
    for i in 0..4 {
        //println!("{} {}",b[i],t[i]);
        draw_line_3d(b[i], b[(i + 1) % 4], col);
    }
    for i in 0..4 {
        draw_line_3d(t[i], t[(i + 1) % 4], col);
    }
    for i in 0..4 {
        draw_line_3d(b[i], t[i], col);
    }
    draw_line_3d(
        vec3(start.0, start.1, height_range * 0.5),
        vec3(end.0, end.1, height_range * 0.5),
        RED,
    );
}

/// Draw the editable height colour-bar endpoints. Shown only when colouring by
/// height. Values are metres relative to the dataset minimum (0 = lowest point).
/// Returns `true` if either endpoint was edited this frame.
/// Endpoints of the height colour ramp.
///
/// `h_min`/`h_max` are viewer-space (metres above the dataset minimum), which
/// is what `color_for_point` compares against, but they are shown as absolute
/// elevations so they read the same as Point Info and the height filter.
fn height_color_controls(
    ui: &mut egui_macroquad::egui::Ui,
    color_mode: ColorMode,
    h_min: &mut f32,
    h_max: &mut f32,
    z_offset: f64,
) -> bool {
    use egui_macroquad::egui;
    if color_mode != ColorMode::Height {
        return false;
    }
    let mut changed = false;
    // Same rounding trap as the height band sliders: compare the stored value,
    // not egui's `.changed()`, or the ramp rebuilds every frame.
    ui.horizontal(|ui| {
        ui.label("Height color:");
        let mut abs_min = *h_min as f64 + z_offset;
        let mut abs_max = *h_max as f64 + z_offset;
        ui.add(egui::DragValue::new(&mut abs_min).speed(0.1).max_decimals(2).suffix(" m"));
        let new_min = (abs_min - z_offset) as f32;
        if new_min != *h_min {
            *h_min = new_min;
            changed = true;
        }
        ui.label("to");
        ui.add(egui::DragValue::new(&mut abs_max).speed(0.1).max_decimals(2).suffix(" m"));
        let new_max = (abs_max - z_offset) as f32;
        if new_max != *h_max {
            *h_max = new_max;
            changed = true;
        }
    });
    changed
}

/// Measure markers and the cross-section corridor preview, in 3D.
///
/// Must be called *after* the point cloud has been composited, not before.
/// With EDL on, the cloud renders to an offscreen texture and is then blitted
/// back with a full-screen quad; anything drawn beforehand survives only where
/// the quad discards, which is exactly where there are no points. Drawn first,
/// these overlays vanish over the cloud — precisely where they are needed.
#[allow(clippy::too_many_arguments)]
fn draw_3d_overlays(
    app_mode: AppMode,
    measure_a: Option<Vec3>,
    measure_b: Option<Vec3>,
    section_start: Option<(f32, f32)>,
    section_end: Option<(f32, f32)>,
    cursor_world: Option<(f32, f32)>,
    section_depth: f32,
    mid_z: f32,
    height_range: f32,
    ortho_scale: f32,
    ground_lasso: &[glam::Vec2],
    ground_color: Color,
    digitise: &vectorise::Digitise,
    // Viewer origin, to project the digitised vertices — which are stored in
    // world coordinates — into the space everything else here is drawn in.
    data_offset: [f64; 3],
    // World-space point the next vertex would land on, for the rubber band.
    digitise_cursor: Option<[f64; 3]>,
) {
    vectorise::draw_3d(digitise, data_offset, digitise_cursor, app_mode == AppMode::Digitise);
    if app_mode == AppMode::Digitise {
        vectorise::draw_snap_marker(digitise, data_offset);
    }
    if !ground_lasso.is_empty() {
        let pts: Vec<Vec3> = ground_lasso.iter().map(|v| vec3(v.x, v.y, mid_z)).collect();
        for w in pts.windows(2) {
            draw_line_3d(w[0], w[1], ground_color);
        }
        if let Some(&last) = pts.last() {
            if app_mode == AppMode::GroundLasso
                && let Some((wx, wy)) = cursor_world {
                    draw_line_3d(last, vec3(wx, wy, mid_z), ground_color);
            }
            if pts.len() >= 2 {
                let faint = Color::new(ground_color.r, ground_color.g, ground_color.b, 0.4);
                let end = if app_mode == AppMode::GroundLasso {
                    cursor_world.map(|(x, y)| vec3(x, y, mid_z)).unwrap_or(last)
                } else {
                    last
                };
                draw_line_3d(end, pts[0], faint);
            }
        }
        // A short vertical tick at each vertex, so the footprint stays legible
        // once the view is tilted off top-down.
        let tick = height_range * 0.03;
        for p in &pts {
            draw_line_3d(*p, vec3(p.x, p.y, p.z + tick), ground_color);
        }
    }

    let cross_size = ortho_scale * 0.015;
    if let Some(a) = measure_a {
        draw_line_3d(vec3(a.x - cross_size, a.y, a.z), vec3(a.x + cross_size, a.y, a.z), YELLOW);
        draw_line_3d(vec3(a.x, a.y - cross_size, a.z), vec3(a.x, a.y + cross_size, a.z), YELLOW);
        draw_line_3d(vec3(a.x, a.y, a.z - cross_size), vec3(a.x, a.y, a.z + cross_size), YELLOW);
    }
    if let Some(b) = measure_b {
        draw_line_3d(vec3(b.x - cross_size, b.y, b.z), vec3(b.x + cross_size, b.y, b.z), ORANGE);
        draw_line_3d(vec3(b.x, b.y - cross_size, b.z), vec3(b.x, b.y + cross_size, b.z), ORANGE);
        draw_line_3d(vec3(b.x, b.y, b.z - cross_size), vec3(b.x, b.y, b.z + cross_size), ORANGE);
    }
    if let (Some(a), Some(b)) = (measure_a, measure_b) {
        draw_line_3d(a, b, WHITE);
    }

    match app_mode {
        AppMode::SectionPickEnd => {
            if let (Some(start), Some((wx, wy))) = (section_start, cursor_world) {
                draw_line_3d(vec3(start.0, start.1, mid_z), vec3(wx, wy, mid_z), YELLOW);
            }
        }
        AppMode::SectionAdjustDepth => {
            if let (Some(start), Some(end)) = (section_start, section_end) {
                draw_section_corridor(start, end, section_depth, height_range);
            }
        }
        _ => {}
    }
}

/// Height-band filter: pick the lowest and highest Z that is drawn, to cut
/// irrelevant levels out of a multi-storey cloud.
///
/// Presented in absolute elevation — the same numbers Point Info reports —
/// while `z_lo`/`z_hi` stay in viewer space, which is what the shader and the
/// section extraction work in.
fn z_clip_controls(
    ui: &mut egui_macroquad::egui::Ui,
    enabled: &mut bool,
    z_lo: &mut f32,
    z_hi: &mut f32,
    z_offset: f64,
    height_range: f32,
) -> bool {
    use egui_macroquad::egui;
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(enabled, "Height filter").changed();
        if *enabled && ui.button("Full range").clicked() {
            *z_lo = 0.0;
            *z_hi = height_range;
            changed = true;
        }
    });
    if !*enabled {
        return changed;
    }
    let lo_bound = z_offset;
    let hi_bound = z_offset + height_range as f64;
    // `max_decimals` makes egui round the value it displays and report that
    // rounding as a change — on every frame, not just the ones the operator
    // touched. Taking `.changed()` at face value here re-coloured and
    // re-uploaded the whole cloud 60 times a second for as long as the filter
    // was open, so compare the band actually stored instead. The stored value
    // round-trips to the same rounding next frame, so this settles after one.
    ui.horizontal(|ui| {
        ui.label("  Top:");
        let mut abs_hi = *z_hi as f64 + z_offset;
        ui.add(egui::Slider::new(&mut abs_hi, lo_bound..=hi_bound).max_decimals(2).suffix(" m"));
        let new_hi = (abs_hi - z_offset) as f32;
        if new_hi != *z_hi {
            *z_hi = new_hi;
            // Dragging one end past the other would blank the view; push
            // rather than swap, so the drag stays predictable.
            *z_lo = z_lo.min(*z_hi);
            changed = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("  Bot:");
        let mut abs_lo = *z_lo as f64 + z_offset;
        ui.add(egui::Slider::new(&mut abs_lo, lo_bound..=hi_bound).max_decimals(2).suffix(" m"));
        let new_lo = (abs_lo - z_offset) as f32;
        if new_lo != *z_lo {
            *z_lo = new_lo;
            *z_hi = z_hi.max(*z_lo);
            changed = true;
        }
    });
    changed
}

/// Swap in a new cross section, parking the old one's GPU buffers in `trash`.
/// They cannot be freed here: most callers are already holding the GL context
/// to build the replacement. The per-frame drain in the main loop releases
/// them, at a point where nothing else has the context borrowed.
fn set_section(
    section_data: &mut Option<SectionData>,
    trash: &mut Vec<SectionData>,
    new: Option<SectionData>,
) {
    if let Some(old) = section_data.take() {
        trash.push(old);
    }
    *section_data = new;
}

/// Push the height band onto every loaded data source.
fn apply_z_clip(
    stages: &mut [Stage],
    copc_stages: &mut [CopcStage],
    tile_mgr: &mut Option<TileStageManager>,
    z_clip: crate::raw_miniquad::ZClip,
) {
    for s in stages.iter_mut() { s.z_clip = z_clip; }
    for cs in copc_stages.iter_mut() { cs.z_clip = z_clip; }
    if let Some(mgr) = tile_mgr.as_mut() { mgr.z_clip = z_clip; }
}

/// The class set an op made right now is allowed to touch: the explicit
/// "only from class N" restriction intersected with what is currently visible.
///
/// The visible set is *frozen into the op* rather than consulted at replay
/// time. Ops are replayed on COPC nodes as they stream in and again on every
/// node at export, which is what makes the exported file agree with the
/// display by construction — and that only holds if an op's meaning is fixed
/// when it is drawn. If it followed live visibility, an export an hour later
/// with different classes toggled would reclassify a different set of points.
///
/// An empty result means the restriction and the visibility exclude each
/// other — "only from class 5" while class 5 is hidden. The op would match
/// nothing, so the callers refuse it and say so rather than committing a
/// no-op that looks like a failure later.
fn op_from_mask(classify_from: Option<u8>, class_vis: ClassVis) -> ClassMask {
    ClassMask::from(classify_from).intersect(class_vis)
}

/// Explain an empty `op_from_mask`, for the status line.
fn empty_from_mask_reason(classify_from: Option<u8>) -> String {
    match classify_from {
        Some(c) => format!(
            "Nothing to do: \"only from class {c}\" is set, but class {c} is hidden. \
             Show it in the Classes window, or clear the restriction.",
        ),
        None => "Nothing to do: every class is hidden. \
                 Show at least one in the Classes window.".to_string(),
    }
}

/// Push the visible class set onto every loaded data source, and report
/// whether any of them changed.
///
/// The return value is what drives the colour rebuild, and it is deliberately
/// "did any backend's value change" rather than "did the operator just toggle
/// something": a stage or tile loaded while a filter is already in force
/// arrives unfiltered, and this is what notices and repaints it.
fn apply_class_vis(
    stages: &mut [Stage],
    copc_stages: &mut [CopcStage],
    tile_mgr: &mut Option<TileStageManager>,
    class_vis: ClassVis,
) -> bool {
    let mut changed = false;
    for s in stages.iter_mut() {
        if s.class_vis != class_vis { s.class_vis = class_vis; changed = true; }
    }
    for cs in copc_stages.iter_mut() {
        if cs.class_vis != class_vis { cs.class_vis = class_vis; changed = true; }
    }
    if let Some(mgr) = tile_mgr.as_mut()
        && mgr.class_vis != class_vis { mgr.class_vis = class_vis; changed = true; }
    changed
}

/// Push the height colour-bar endpoints onto every loaded data source so the
/// next colour rebuild uses them.
fn apply_height_color(
    stages: &mut [Stage],
    copc_stages: &mut [CopcStage],
    tile_mgr: &mut Option<TileStageManager>,
    h_min: f32,
    h_max: f32,
) {
    for s in stages.iter_mut() {
        s.color_h_min = h_min;
        s.color_h_max = h_max;
    }
    for cs in copc_stages.iter_mut() {
        cs.color_h_min = h_min;
        cs.color_h_max = h_max;
    }
    if let Some(mgr) = tile_mgr.as_mut() {
        mgr.color_h_min = h_min;
        mgr.color_h_max = h_max;
    }
}

/// A readout line with a copy button beside it, so the value can be lifted
/// out without retyping it. `clip` is what lands on the clipboard, which is
/// not always what is displayed — the coordinates are copied tab separated
/// for pasting into a spreadsheet.
fn copyable_line(ui: &mut egui_macroquad::egui::Ui, text: String, clip: String, hint: &str) {
    ui.horizontal(|ui| {
        ui.label(text);
        if ui.small_button("Copy").on_hover_text(hint).clicked() {
            set_clipboard(clip);
        }
    });
}

/// Put `text` on the system clipboard.
///
/// Deliberately bypasses egui's `copy_text`, which routes through miniquad.
/// Miniquad's X11 clipboard answers only `UTF8_STRING` selection requests and
/// refuses everything else, `TARGETS` included, so a copy reaches only those
/// apps that happen to ask for `UTF8_STRING` directly. Under Wayland it
/// reaches nothing at all: we run on XWayland, and the compositor asks the
/// selection owner for `TARGETS` to learn which MIME types to bridge across.
///
/// `arboard` talks the Wayland data-control protocol directly when
/// `WAYLAND_DISPLAY` is set, and falls back to a complete X11 implementation
/// otherwise.
fn set_clipboard(text: String) {
    // The first call sets up arboard's clipboard serving, which stays alive for
    // the rest of the process; later calls reuse it.
    if let Err(e) = arboard::Clipboard::new().and_then(|mut c| c.set_text(text)) {
        error!("Could not copy to the clipboard: {e}");
    }
}

/// Point Info block, shared by every viewer mode. `title` lets the tile
/// browser name the file the point came from.
fn point_info_ui(
    ui: &mut egui_macroquad::egui::Ui,
    title: &str,
    p: &LasPoint,
    class_defs: &Option<ClassDefs>,
) {
    ui.label(title);
    copyable_line(
        ui,
        format!("  X: {:.3}  Y: {:.3}  Z: {:.3}", p.x, p.y, p.z),
        format!("{:.3}	{:.3}	{:.3}", p.x, p.y, p.z),
        "Copy the coordinates to the clipboard, tab separated",
    );
    ui.label(format!("  Intensity: {}", p.intensity));
    ui.label(format!("  Classification: {}", class_label(p, class_defs)));
    ui.label(format!("  Return: {}/{}", p.return_number, p.number_of_returns));
    if let Some(ref color) = p.color {
        ui.label(format!("  RGB: {} {} {}", color.red, color.green, color.blue));
    }
    if let Some(time) = p.gps_time {
        copyable_line(
            ui,
            format!("  GPS time: {:.6}", time),
            format!("{:.6}", time),
            "Copy the GPS time to the clipboard",
        );
    }
    ui.label(format!("  Point Source Id: {}", p.point_source_id));
}

/// Format classification label using custom class_defs or default debug format.
/// Recovers class 12 (Overlap) from the `is_overlap` flag, since the `las` crate
/// strips it from the classification field per LAS 1.4 semantics.
fn class_label(point: &las::Point, class_defs: &Option<crate::raw_miniquad::ClassDefs>) -> String {
    let id: u8 = if point.is_overlap { 12 } else { point.classification.into() };
    if let Some(defs) = class_defs &&
        let Some((label, _)) = defs.get(&id) {
            return format!("{} ({})", id, label);

    }
    if point.is_overlap {
        format!("{} (OverlapPoints)", id)
    } else {
        format!("{} ({:?})", id, point.classification)
    }
}

/// Display colour for a classification id, honouring any custom definition
/// from `classes.txt` so the lasso is drawn in the colour the reclassified
/// points will take.
fn class_color(id: u8, class_defs: &Option<ClassDefs>) -> [f32; 3] {
    if let Some(defs) = class_defs
        && let Some((_, c)) = defs.get(&id) {
            return *c;
    }
    classification_color(id)
}

/// `class_color` as a macroquad colour, for the 3D overlays.
fn class_color_mq(id: u8, class_defs: &Option<ClassDefs>) -> Color {
    let c = class_color(id, class_defs);
    Color::new(c[0], c[1], c[2], 1.0)
}

/// Find the LAS point nearest to a section-view click, searching whichever
/// sources contributed points to the current section. `wx`/`wy` are the click
/// position in section space (metres along the line, height). Returns the
/// point plus its section-space position.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pick_section_point(
    stages: &[Stage],
    copc_section_raw: &Option<(Vec<Vec<(LasPoint, f32, f32)>>, f32, f32, f32)>,
    tile_mgr: &Option<TileStageManager>,
    start: (f32, f32),
    end: (f32, f32),
    depth: f32,
    z_clip: crate::raw_miniquad::ZClip,
    class_vis: ClassVis,
    wx: f32,
    wy: f32,
) -> Option<(LasPoint, glam::Vec2)> {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 0.0 {
        return None;
    }
    let dir = (dx / len, dy / len);
    let normal = (-dir.1, dir.0);

    let mut best: Option<(LasPoint, glam::Vec2)> = None;
    let mut best_dist = f32::MAX;
    let mut consider = |p: &LasPoint, along: f32, z: f32| {
        if !z_clip.contains(z) { return; }
        // A hidden class is unpickable in the section view too — the same rule
        // the 3D pick, the lasso and the ground fit follow.
        if !class_vis.contains(effective_class(p)) { return; }
        let d = (along - wx).powi(2) + (z - wy).powi(2);
        if d < best_dist {
            best_dist = d;
            best = Some((p.clone(), glam::Vec2::new(along, z)));
        }
    };

    for s in stages {
        for (i, pos) in s.positions.iter().enumerate() {
            let rel_x = pos[0] - start.0;
            let rel_y = pos[1] - start.1;
            let along = rel_x * dir.0 + rel_y * dir.1;
            let perp = rel_x * normal.0 + rel_y * normal.1;
            if along >= 0.0 && along <= len && perp.abs() <= depth {
                consider(&s.cloud.points[i], along, pos[2]);
            }
        }
    }
    if let Some((per_stage_raw, _, _, _)) = copc_section_raw {
        for raw in per_stage_raw {
            for (p, along, z) in raw {
                consider(p, *along, *z);
            }
        }
    }
    if let Some(mgr) = tile_mgr {
        for loaded in mgr.loaded.iter().flatten() {
            for (i, pos) in loaded.positions.iter().enumerate() {
                let rel_x = pos[0] - start.0;
                let rel_y = pos[1] - start.1;
                let along = rel_x * dir.0 + rel_y * dir.1;
                let perp = rel_x * normal.0 + rel_y * normal.1;
                if along >= 0.0 && along <= len && perp.abs() <= depth {
                    consider(&loaded.cloud.points[i], along, pos[2]);
                }
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn polylines_window(
    egui_ctx: &egui_macroquad::egui::Context,
    poly_files: &mut Vec<polylines::PolylineFile>,
    selected_poly: &mut Option<polylines::PolyRef>,
    poly_load_error: &mut Option<String>,
    poly_save_status: &mut Option<String>,
    // A Salome save that would drop something, waiting for the operator to
    // say yes: (file index, target path, what would be lost).
    pending_salome: &mut Option<(usize, String, Vec<String>)>,
    data_offset: [f64; 3],
    background: Color,
) {
    egui_macroquad::egui::Window::new("Polylines")
        .default_pos([10.0, 600.0])
        // `default_pos` is clamped so the window fits on screen, and on the
        // first frame egui assumes a window is its default size (340x420),
        // even one that opens collapsed. That reserves ~400px of height and
        // silently pins any y past `screen_height - 400` to that ceiling —
        // which is how two windows given different positions end up stacked.
        .constrain(false)
        .default_open(false)
        .show(egui_ctx, |ui| {
            let luma = background.r * 0.299 + background.g * 0.587 + background.b * 0.114;
            let default_color = if luma > 0.5 { BLACK } else { WHITE };
            if ui.button("Load Polyline File...").clicked() &&
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("Vectors", &["gpkg", "txt", "xyz", "poly", "dat", ""])
                    .add_filter("GeoPackage", &["gpkg"])
                    .add_filter("Salome ASCII", &["txt", "xyz", "poly", "dat", ""])
                    .pick_file()
                {
                    match polylines::load(path.to_str().unwrap_or(""), data_offset, default_color) {
                        Ok(pf) => { *poly_load_error = None; poly_files.push(pf); }
                        Err(e) => { *poly_load_error = Some(e); }
                }
                
            }
            if let Some(ref err) = *poly_load_error {
                ui.colored_label(egui_macroquad::egui::Color32::RED, err);
            }
            ui.separator();
            let mut to_remove: Option<usize> = None;
            // `(file, Some(new path))` is Save As; `(file, None)` is Save.
            let mut save_request: Option<(usize, Option<String>)> = None;
            // File → layer → line. A GeoPackage is a container of named feature
            // tables, so the panel is a tree even when a Salome file gives it
            // only one layer to show.
            for (fi, pf) in poly_files.iter_mut().enumerate() {
                let header = format!("{} ({} lines)", pf.file_name(), pf.line_count());
                ui.collapsing(header, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(pf.format.label());
                        if pf.dirty { ui.label("• unsaved"); }
                        if ui.small_button("Remove file").clicked() {
                            to_remove = Some(fi);
                            if selected_poly.map(|r| r.file) == Some(fi) { *selected_poly = None; }
                        }
                    });
                    ui.horizontal(|ui| {
                        // Save writes back in the format the file came from;
                        // Save As is where the format can change, and where
                        // the cost of changing it is stated first.
                        if ui.add_enabled(pf.dirty, egui_macroquad::egui::Button::new("Save"))
                            .clicked() {
                            save_request = Some((fi, None));
                        }
                        if ui.button("Save As…").clicked()
                            && let Some(path) = rfd::FileDialog::new()
                                .add_filter("GeoPackage", &["gpkg"])
                                .add_filter("Salome ASCII", &["txt", "xyz", "poly"])
                                .set_file_name(pf.file_name())
                                .save_file() {
                            save_request = Some((fi, Some(path.to_string_lossy().into_owned())));
                        }
                    });
                    for (li, layer) in pf.layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut layer.visible, "")
                                .on_hover_text("Show this layer");
                            ui.label(format!("{} ({})", layer.name, layer.polylines.len()));
                            // Detected from the file when there was one to
                            // detect; editable because a layer haiviz created
                            // has no way to know what the cloud is in.
                            ui.label("EPSG:");
                            ui.add(egui_macroquad::egui::DragValue::new(&mut layer.srs_id)
                                .range(0..=1_000_000))
                                .on_hover_text("Coordinate reference system written to GeoPackage");
                        });
                        ui.indent(("poly_layer", fi, li), |ui| {
                            for (pi, pl) in layer.polylines.iter_mut().enumerate() {
                                let at = polylines::PolyRef { file: fi, layer: li, line: pi };
                                ui.horizontal(|ui| {
                                    let is_sel = *selected_poly == Some(at);
                                    if ui.selectable_label(is_sel, &pl.name).clicked() {
                                        *selected_poly = if is_sel { None } else { Some(at) };
                                    }
                                    let mut ec = mq_to_egui(pl.color);
                                    if ui.color_edit_button_srgba(&mut ec).changed() {
                                        pl.color = egui_to_mq(ec);
                                    }
                                    ui.add(egui_macroquad::egui::Slider::new(&mut pl.radius, 0.001..=0.05).text("r (m)").logarithmic(true));
                                });
                            }
                        });
                    }
                });
            }
            if let Some(fi) = to_remove { poly_files.remove(fi); }

            if let Some((fi, target)) = save_request {
                *poly_save_status = save_polylines(&mut poly_files[fi], fi, target, pending_salome);
            }

            // Salome carries no attributes, no layers and no CRS, so a
            // GeoPackage saved as one loses things. Say what, and make the
            // operator agree, rather than just doing it.
            if let Some((fi, path, losses)) = pending_salome.clone() {
                ui.separator();
                ui.colored_label(
                    egui_macroquad::egui::Color32::from_rgb(255, 180, 60),
                    "Saving as Salome ASCII would lose:",
                );
                for l in &losses { ui.label(format!("  • {l}")); }
                ui.horizontal(|ui| {
                    if ui.button("Write anyway").clicked() {
                        let pf = &mut poly_files[fi];
                        pf.path = path.clone();
                        pf.format = polylines::VectorFormat::Salome;
                        pf.mtime = None;
                        *poly_save_status = Some(match pf.save() {
                            Ok(()) => format!("Saved {}", pf.file_name()),
                            Err(e) => format!("Save failed: {e}"),
                        });
                        *pending_salome = None;
                    }
                    if ui.button("Cancel").clicked() { *pending_salome = None; }
                });
            }

            if let Some(msg) = poly_save_status {
                ui.label(msg.as_str());
            }
        });
}

/// Perform a Save or Save As, returning the status line.
///
/// Save As decides the format from the chosen extension rather than from a
/// dropdown, which keeps it consistent with loading — where the format comes
/// from the file's magic bytes, not its name.
fn save_polylines(
    pf: &mut polylines::PolylineFile,
    fi: usize,
    target: Option<String>,
    pending_salome: &mut Option<(usize, String, Vec<String>)>,
) -> Option<String> {
    use polylines::VectorFormat;
    match target {
        None => {
            // Auto-save and manual save share this check: we always rewrite the
            // whole file, so an unnoticed external edit would be destroyed
            // rather than merely conflicted with.
            if pf.changed_on_disk() {
                return Some(format!(
                    "{} changed on disk since it was loaded. Saving would overwrite                      those changes — use Save As to write somewhere else.",
                    pf.file_name(),
                ));
            }
            Some(match pf.save() {
                Ok(()) => format!("Saved {}", pf.file_name()),
                Err(e) => format!("Save failed: {e}"),
            })
        }
        Some(path) => {
            let ext = std::path::Path::new(&path)
                .extension().and_then(|e| e.to_str()).unwrap_or("")
                .to_ascii_lowercase();
            let format = if ext == "gpkg" { VectorFormat::GeoPackage } else { VectorFormat::Salome };
            if format == VectorFormat::Salome {
                let losses = pf.salome_losses();
                if !losses.is_empty() {
                    *pending_salome = Some((fi, path, losses));
                    return None;
                }
            }
            pf.path = path;
            pf.format = format;
            // A new destination has no history to conflict with.
            pf.mtime = None;
            Some(match pf.save() {
                Ok(()) => format!("Saved {}", pf.file_name()),
                Err(e) => format!("Save failed: {e}"),
            })
        }
    }
}

/// Ground control point overlay: load text tables, review/override the detected
/// column mapping, and control how the markers are drawn in both views.
fn gcp_window(
    egui_ctx: &egui_macroquad::egui::Context,
    gcp_files: &mut Vec<gcp::GcpFile>,
    selected_gcp: &mut Option<(usize, usize)>,
    gcp_load_error: &mut Option<String>,
    background: Color,
) {
    use egui_macroquad::egui;
    egui::Window::new("Ground Control Points")
        .default_pos([10.0, 700.0])
        // See the note in `polylines_window`.
        .constrain(false)
        .default_open(false)
        .show(egui_ctx, |ui| {
            let luma = background.r * 0.299 + background.g * 0.587 + background.b * 0.114;
            let default_color = if luma > 0.5 {
                Color::new(0.85, 0.1, 0.1, 1.0)
            } else {
                Color::new(1.0, 0.3, 0.3, 1.0)
            };
            if ui.button("Load GCP File...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("GCP table", &["txt", "csv", "asc", "pts", "xyz", "dat", "tsv", ""])
                    .add_filter("All files", &["*"])
                    .pick_file()
            {
                match gcp::load(path.to_str().unwrap_or(""), default_color) {
                    Ok(f) => { *gcp_load_error = None; gcp_files.push(f); }
                    Err(e) => { *gcp_load_error = Some(e); }
                }
            }
            if let Some(ref err) = *gcp_load_error {
                ui.colored_label(egui::Color32::RED, err);
            }
            ui.separator();

            let mut to_remove: Option<usize> = None;
            for (fi, f) in gcp_files.iter_mut().enumerate() {
                let fname = std::path::Path::new(&f.path)
                    .file_name().and_then(|n| n.to_str()).unwrap_or(&f.path);
                ui.collapsing(format!("{} ({} points)", fname, f.points.len()), |ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut f.visible, "Show");
                        ui.checkbox(&mut f.show_labels, "Labels");
                        let mut ec = mq_to_egui(f.color);
                        if ui.color_edit_button_srgba(&mut ec).changed() {
                            f.color = egui_to_mq(ec);
                        }
                        if ui.small_button("Remove").clicked() {
                            to_remove = Some(fi);
                            if selected_gcp.map(|(s, _)| s) == Some(fi) { *selected_gcp = None; }
                        }
                    });
                    ui.add(egui::Slider::new(&mut f.size, 4.0..=24.0).text("marker (px)"));

                    // ---- Column mapping ----
                    let ncols = f.rows.iter().map(|r| r.len()).max().unwrap_or(3);
                    let col_name = |f: &gcp::GcpFile, c: usize| -> String {
                        match f.layout.headers.get(c) {
                            Some(h) if !h.is_empty() => format!("{}: {}", c + 1, h),
                            _ => format!("col {}", c + 1),
                        }
                    };
                    let mut remap = false;
                    ui.collapsing("Columns", |ui| {
                        ui.label(format!(
                            "separator: {}{}",
                            f.layout.sep.name(),
                            if f.layout.has_header { ", header row" } else { "" },
                        ));
                        egui::Grid::new(("gcp_cols", fi)).num_columns(2).show(ui, |ui| {
                            let mut label_sel: i32 = f.layout.label_col.map(|c| c as i32).unwrap_or(-1);
                            ui.label("label");
                            egui::ComboBox::from_id_salt(("gcp_label", fi))
                                .selected_text(if label_sel < 0 { "— none —".to_string() } else { col_name(f, label_sel as usize) })
                                .width(150.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut label_sel, -1, "— none —");
                                    for c in 0..ncols {
                                        ui.selectable_value(&mut label_sel, c as i32, col_name(f, c));
                                    }
                                });
                            let new_label = if label_sel < 0 { None } else { Some(label_sel as usize) };
                            if new_label != f.layout.label_col { f.layout.label_col = new_label; remap = true; }
                            ui.end_row();

                            for (name, id) in [("x", "gcp_x"), ("y", "gcp_y"), ("z", "gcp_z")] {
                                let cur = match name {
                                    "x" => f.layout.x_col,
                                    "y" => f.layout.y_col,
                                    _ => f.layout.z_col,
                                };
                                let mut sel = cur;
                                ui.label(name);
                                egui::ComboBox::from_id_salt((id, fi))
                                    .selected_text(col_name(f, cur))
                                    .width(150.0)
                                    .show_ui(ui, |ui| {
                                        for c in 0..ncols {
                                            ui.selectable_value(&mut sel, c, col_name(f, c));
                                        }
                                    });
                                if sel != cur {
                                    match name {
                                        "x" => f.layout.x_col = sel,
                                        "y" => f.layout.y_col = sel,
                                        _ => f.layout.z_col = sel,
                                    }
                                    remap = true;
                                }
                                ui.end_row();
                            }
                        });
                    });
                    if remap {
                        gcp::rebuild(f);
                        *selected_gcp = None;
                    }

                    // ---- Point list ----
                    egui::ScrollArea::vertical()
                        .max_height(180.0)
                        .id_salt(("gcp_pts", fi))
                        .show(ui, |ui| {
                            for (pi, p) in f.points.iter().enumerate() {
                                let is_sel = *selected_gcp == Some((fi, pi));
                                let text = format!(
                                    "{}  {:.3}  {:.3}  {:.3}",
                                    p.label, p.world[0], p.world[1], p.world[2],
                                );
                                if ui.selectable_label(is_sel, text).clicked() {
                                    *selected_gcp = if is_sel { None } else { Some((fi, pi)) };
                                }
                            }
                        });
                });
            }
            if let Some(fi) = to_remove {
                gcp_files.remove(fi);
                *selected_gcp = None;
            }
        });
}

/// Built-in editor for classification labels & colours (classes.txt).
/// Lists every class id up to the maximum present in the loaded data, plus any
/// ids already defined beyond that; missing entries are seeded from the default
/// palette/labels when the window is first opened. Returns `true` when a colour
/// was edited or the file was reloaded, i.e. when point colours need a rebuild.
/// Label-only edits update the defs in place and need no rebuild.
fn classes_window(
    egui_ctx: &egui_macroquad::egui::Context,
    class_defs: &mut Option<ClassDefs>,
    class_vis: &mut ClassVis,
    max_class: u8,
    save_status: &mut Option<String>,
) -> bool {
    use egui_macroquad::egui;
    let mut colors_changed = false;
    egui::Window::new("Classes")
        .default_pos([10.0, 500.0])
        // See the note in `polylines_window`.
        .constrain(false)
        .default_open(false)
        .show(egui_ctx, |ui| {
            let mut do_reload = false;
            {
                let defs = class_defs.get_or_insert_with(ClassDefs::new);
                let mut ids: Vec<u8> = (0..=max_class).collect();
                let mut extra: Vec<u8> = defs.keys().copied().filter(|id| *id > max_class).collect();
                extra.sort();
                ids.extend(extra);
                // All / None / Invert act on the listed ids only. An id that is
                // not listed cannot be switched off in this window, so leaving
                // it alone is what keeps "N classes hidden" a number the
                // operator can reconcile with what is on screen.
                ui.horizontal(|ui| {
                    ui.label("Show:");
                    if ui.button("All").clicked() {
                        for &id in &ids { class_vis.insert(id); }
                    }
                    if ui.button("None").clicked() {
                        for &id in &ids { class_vis.remove(id); }
                    }
                    if ui.button("Invert").clicked() {
                        for &id in &ids { let v = class_vis.contains(id); class_vis.set(id, !v); }
                    }
                });
                let mut isolate: Option<u8> = None;
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    for id in ids.iter().copied() {
                        let entry = defs.entry(id).or_insert_with(|| {
                            (default_class_label(id), classification_color(id))
                        });
                        ui.horizontal(|ui| {
                            let mut shown = class_vis.contains(id);
                            if ui.checkbox(&mut shown, "")
                                .on_hover_text("Show this class")
                                .changed() {
                                class_vis.set(id, shown);
                            }
                            ui.label(format!("{:3}", id));
                            let mut c32 = rgb_arr_to_egui(entry.1);
                            if ui.color_edit_button_srgba(&mut c32).changed() {
                                entry.1 = egui_to_rgb_arr(c32);
                                colors_changed = true;
                            }
                            ui.add(egui::TextEdit::singleline(&mut entry.0).desired_width(140.0));
                            if ui.small_button("only")
                                .on_hover_text("Isolate: show only this class")
                                .clicked() {
                                // Over the listed ids only, like All/None/Invert:
                                // ClassVis::single would also switch off the 200-odd
                                // ids this window never shows, and the "N classes
                                // hidden" indicator would stop meaning anything.
                                isolate = Some(id);
                            }
                        });
                    }
                });
                if let Some(keep) = isolate {
                    for &id in &ids { class_vis.set(id, id == keep); }
                }
                ui.separator();
                // Both of these are foot-guns, so they are said here rather
                // than only in the README.
                let hidden = ids.iter().filter(|&&id| !class_vis.contains(id)).count();
                if hidden > 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 180, 60),
                        format!("{hidden} class{} hidden. Hidden classes cannot be picked, \
                                 measured, sectioned, lassoed or fitted to either.",
                                if hidden == 1 { "" } else { "es" }),
                    );
                }
                ui.label("Visibility does not affect export — use Omit classes for that.");
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save classes.txt").clicked() {
                        *save_status = Some(match save_class_defs(defs) {
                            Ok(()) => format!("Saved {} classes to classes.txt", defs.len()),
                            Err(e) => format!("Save failed: {}", e),
                        });
                    }
                    if ui.button("Reload").clicked() {
                        do_reload = true;
                    }
                });
            }
            if do_reload {
                *class_defs = load_class_defs();
                *save_status = None;
                colors_changed = true;
            }
            if let Some(msg) = save_status {
                ui.label(msg.as_str());
            }
        });
    colors_changed
}

/// Push updated class definitions onto every loaded data source so subsequent
/// colour rebuilds and section extractions use them.
/// Combo box over the classification ids present in the data, plus the LAS
/// standard ids, with a numeric fallback for anything outside that range.
fn class_picker(
    ui: &mut egui_macroquad::egui::Ui,
    id: &str,
    value: &mut u8,
    max_class: u8,
    class_defs: &Option<ClassDefs>,
) {
    let label_for = |c: u8| -> String {
        if let Some(defs) = class_defs
            && let Some((label, _)) = defs.get(&c) {
                return format!("{c} — {label}");
        }
        format!("{c} — {}", default_class_label(c))
    };
    egui_macroquad::egui::ComboBox::from_id_salt(id)
        .selected_text(label_for(*value))
        .width(170.0)
        .show_ui(ui, |ui| {
            for c in 0..=max_class.max(18) {
                ui.selectable_value(value, c, label_for(c));
            }
        });
    ui.add(egui_macroquad::egui::DragValue::new(value).range(0..=255));
}

/// Controls for the plan-view lasso tools, shared by the three 3D panels.
/// Returns the flavour of lasso the operator asked to start, if any.
#[allow(clippy::too_many_arguments)]
fn ground_tool_controls(
    ui: &mut egui_macroquad::egui::Ui,
    app_mode: &mut AppMode,
    lasso_len: usize,
    kind: LassoKind,
    top_view: bool,
    target: &mut u8,
    from: &mut Option<u8>,
    offset: &mut f32,
    max_class: u8,
    class_defs: &Option<ClassDefs>,
    hidden_classes: usize,
    status: &Option<String>,
    apply: &mut bool,
    cancel: &mut bool,
) -> Option<LassoKind> {
    use egui_macroquad::egui;
    let drawing = *app_mode == AppMode::GroundLasso;
    let mut start = None;
    ui.separator();
    ui.label("Classify in plan view:");
    // A view setting that silently changes what a destructive tool does needs
    // to be visible from the tool, not only from the window that set it —
    // otherwise the first report is "why did my lasso catch nothing?".
    if hidden_classes > 0 {
        ui.colored_label(
            egui::Color32::from_rgb(255, 180, 60),
            format!("{hidden_classes} class{} hidden — the lasso ignores {}.",
                    if hidden_classes == 1 { "" } else { "es" },
                    if hidden_classes == 1 { "it" } else { "them" }),
        );
    }
    ui.horizontal(|ui| {
        ui.label("To:");
        class_picker(ui, "ground_to", target, max_class, class_defs);
    });
    ui.horizontal(|ui| {
        let mut restrict = from.is_some();
        if ui.checkbox(&mut restrict, "Only from:").changed() {
            *from = if restrict { Some(0) } else { None };
        }
        if let Some(f) = from.as_mut() {
            class_picker(ui, "ground_from", f, max_class, class_defs);
        }
    });
    // Only the above-ground tool has a ground to measure from, so the offset
    // greys out rather than disappearing — the panel keeps its shape when the
    // operator switches between the two.
    let offset_live = !drawing || kind == LassoKind::AboveGround;
    ui.add_enabled_ui(offset_live, |ui| {
        ui.horizontal(|ui| {
            ui.label("Height above ground:");
            ui.add(egui::DragValue::new(offset).speed(0.05).range(0.0..=50.0)
                .max_decimals(2).suffix(" m"));
        });
    });
    if drawing {
        ui.label(format!("{}: click to outline an area ({lasso_len} points so far)", kind.label()));
        if kind == LassoKind::Inside {
            ui.label("Everything inside the outline is reclassified, ground included.");
        }
        ui.horizontal(|ui| {
            if ui.add_enabled(lasso_len >= 3, egui::Button::new("Apply")).clicked() {
                *apply = true;
            }
            if ui.button("Cancel").clicked() {
                *cancel = true;
            }
        });
        ui.label("Enter applies, Backspace removes a point, Esc cancels.");
    } else {
        // Both tools unproject the outline onto a horizontal plane, so they are
        // offered only from straight above — anywhere else the polygon would
        // mean something other than what was drawn.
        ui.add_enabled_ui(top_view, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Above-ground lasso (G)").clicked() {
                    start = Some(LassoKind::AboveGround);
                }
                if ui.button("Inside lasso (V)").clicked() {
                    start = Some(LassoKind::Inside);
                }
            });
        });
        if !top_view {
            ui.label(TOP_VIEW_HINT);
        }
    }
    if let Some(msg) = status {
        ui.label(msg.as_str());
    }
    start
}

/// Export panel: pick which classes to drop, then write one source out.
/// Returns the index into `sources` the user asked to export, if any.
#[allow(clippy::too_many_arguments)]
fn export_window(
    egui_ctx: &egui_macroquad::egui::Context,
    open: &mut bool,
    filter: &mut crate::classify::ExportFilter,
    job: &mut Option<crate::export::ExportJob>,
    status: &mut Option<String>,
    sources: &[(String, bool)],
    max_class: u8,
    class_defs: &Option<ClassDefs>,
) -> Option<usize> {
    if !*open { return None; }
    let mut requested = None;
    let mut keep_open = true;
    egui_macroquad::egui::Window::new("Export")
        .open(&mut keep_open)
        .default_pos([420.0, 10.0])
        .show(egui_ctx, |ui| {
            ui.label("Writes LAS/LAZ with all manual classification applied.");
            ui.label("COPC input exports as plain LAZ — re-run copc-converter to");
            ui.label("rebuild the octree.");
            ui.separator();

            ui.label("Omit classes (noise filtering):");
            egui_macroquad::egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    for c in 0..=max_class.max(18) {
                        let label = if let Some(defs) = class_defs
                            && let Some((l, _)) = defs.get(&c) {
                                format!("{c} — {l}")
                        } else {
                            format!("{c} — {}", default_class_label(c))
                        };
                        let mut dropped = filter.drop_classes.contains(&c);
                        if ui.checkbox(&mut dropped, label).changed() {
                            if dropped { filter.drop_classes.insert(c); }
                            else { filter.drop_classes.remove(&c); }
                        }
                    }
                });
            if !filter.drop_classes.is_empty() {
                let mut ids: Vec<u8> = filter.drop_classes.iter().copied().collect();
                ids.sort();
                let names: Vec<String> = ids.iter().map(|c| c.to_string()).collect();
                ui.label(format!("Dropping: {}", names.join(", ")));
                if ui.button("Clear").clicked() { filter.drop_classes.clear(); }
            }

            ui.separator();
            if let Some(j) = job.as_ref()
                && j.finished.is_none() {
                    ui.label(format!("Exporting {}…", j.label));
                    ui.add(egui_macroquad::egui::ProgressBar::new(j.fraction()).show_percentage());
                    ui.label(format!("{} written, {} dropped", j.written, j.dropped));
                    return;
            }
            for (i, (path, is_copc)) in sources.iter().enumerate() {
                let name = std::path::Path::new(path)
                    .file_name().and_then(|s| s.to_str()).unwrap_or(path);
                ui.horizontal(|ui| {
                    if ui.button("Export…").clicked() { requested = Some(i); }
                    ui.label(if *is_copc { format!("{name} (COPC)") } else { name.to_string() });
                });
            }
            if sources.is_empty() {
                ui.label("Nothing loaded that can be exported.");
            }
            if let Some(msg) = status.as_ref() {
                ui.separator();
                ui.label(msg.as_str());
            }
        });
    if !keep_open { *open = false; }
    requested
}

/// Kick off an export of `sources[idx]`, taking that source's op list with it
/// so the written file matches what the viewer shows.
fn start_export(
    idx: usize,
    stages: &[Stage],
    copc_stages: &[CopcStage],
    filter: &crate::classify::ExportFilter,
    job: &mut Option<crate::export::ExportJob>,
    status: &mut Option<String>,
) {
    if job.as_ref().is_some_and(|j| j.finished.is_none()) {
        *status = Some("An export is already running.".to_string());
        return;
    }
    let (path, edits, is_copc) = if idx < stages.len() {
        (stages[idx].filepath.clone(), stages[idx].edits.clone(), false)
    } else if let Some(cs) = copc_stages.get(idx - stages.len()) {
        (cs.filepath.clone(), cs.edits.clone(), true)
    } else {
        return;
    };
    let out = crate::classify::default_export_path(&path);
    let out = rfd::FileDialog::new()
        .set_title("Export point cloud")
        .set_file_name(out.file_name().and_then(|s| s.to_str()).unwrap_or("export.laz"))
        .add_filter("LAZ", &["laz"])
        .add_filter("LAS", &["las"])
        .save_file();
    let Some(out) = out else { return };
    *status = None;
    *job = Some(crate::export::spawn(path, out, edits, filter.clone(), is_copc));
}

/// Files that can be exported, in the order the export panel lists them:
/// manually loaded stages first, then COPC stages.
fn export_source_list(stages: &[Stage], copc_stages: &[CopcStage]) -> Vec<(String, bool)> {
    stages.iter().map(|s| (s.filepath.clone(), false))
        .chain(copc_stages.iter().map(|cs| (cs.filepath.clone(), true)))
        .collect()
}

fn apply_class_defs(
    stages: &mut [Stage],
    copc_stages: &mut [CopcStage],
    tile_mgr: &mut Option<TileStageManager>,
    class_defs: &Option<ClassDefs>,
) {
    for s in stages.iter_mut() {
        s.class_defs = class_defs.clone();
    }
    for cs in copc_stages.iter_mut() {
        cs.class_defs = class_defs.clone();
    }
    if let Some(mgr) = tile_mgr.as_mut() {
        mgr.class_defs = class_defs.clone();
    }
}

fn mq_to_egui(c: Color) -> egui_macroquad::egui::Color32 {
    egui_macroquad::egui::Color32::from_rgba_unmultiplied(
        (c.r * 255.0) as u8, (c.g * 255.0) as u8,
        (c.b * 255.0) as u8, (c.a * 255.0) as u8,
    )
}

fn egui_to_mq(c: egui_macroquad::egui::Color32) -> Color {
    Color::new(c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, c.a() as f32 / 255.0)
}

fn rgb_arr_to_egui(c: [f32; 3]) -> egui_macroquad::egui::Color32 {
    egui_macroquad::egui::Color32::from_rgb(
        (c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8,
    )
}

fn egui_to_rgb_arr(c: egui_macroquad::egui::Color32) -> [f32; 3] {
    [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]
}

fn gen_icon() -> miniquad::conf::Icon {
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = || -> u8 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed & 0xFF) as u8
    };

    let mut small  = [0u8; 16 * 16 * 4];
    let mut medium = [0u8; 32 * 32 * 4];
    let mut big    = [0u8; 64 * 64 * 4];

    for pixel in small.chunks_mut(4)  { pixel[0]=rng(); pixel[1]=rng(); pixel[2]=rng(); pixel[3]=255; }
    for pixel in medium.chunks_mut(4) { pixel[0]=rng(); pixel[1]=rng(); pixel[2]=rng(); pixel[3]=255; }
    for pixel in big.chunks_mut(4)    { pixel[0]=rng(); pixel[1]=rng(); pixel[2]=rng(); pixel[3]=255; }

    miniquad::conf::Icon { small, medium, big }
}

fn conf() -> Conf {
 Conf {
   window_title: "HaiViz ".to_owned()+env!("CARGO_PKG_VERSION"), //this field is not optional!
   window_width: 1600,
   window_height: 900,
   fullscreen:false,   
   icon: Some(gen_icon()),
   ..Default::default()
 }
}

/// Axis-aligned bounding box of all currently-loaded data, expressed in viewer
/// space (world coordinates minus each source's viewer origin). Returns None when
/// nothing is loaded. Used by "Zoom All" so it frames the actual data wherever it
/// sits, instead of assuming it starts at the (possibly stale) viewer origin.
fn loaded_aabb_viewer(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
) -> Option<(Vec3, Vec3)> {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut any = false;
    for s in stages {
        for k in 0..3 {
            lo[k] = lo[k].min((s.cloud.min[k] - s.viewer_origin[k]) as f32);
            hi[k] = hi[k].max((s.cloud.max[k] - s.viewer_origin[k]) as f32);
        }
        any = true;
    }
    for c in copc_stages {
        for k in 0..3 {
            lo[k] = lo[k].min((c.global_min[k] - c.viewer_origin[k]) as f32);
            hi[k] = hi[k].max((c.global_max[k] - c.viewer_origin[k]) as f32);
        }
        any = true;
    }
    if let Some(m) = tile_mgr {
        // Tile positions are stored relative to `global_min`, so that is the
        // viewer origin and the lower bound in viewer space is 0.
        for k in 0..3 {
            lo[k] = lo[k].min(0.0);
            hi[k] = hi[k].max((m.global_max[k] - m.global_min[k]) as f32);
        }
        any = true;
    }
    if any { Some((Vec3::from(lo), Vec3::from(hi))) } else { None }
}

#[macroquad::main(conf)]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!(" *** HaiViz {} ***",env!("CARGO_PKG_VERSION"));
    let args: Vec<String> = env::args().collect();
    let input_path: Option<&str> = if args.len() >= 2 { Some(args[1].as_str()) } else { None };
    if input_path.is_none() {
        info!("Launched with no input — drag .laz / .copc.laz files onto the window to load.");
    }
    let is_tile_mode = input_path.map(|p| Path::new(p).is_dir()).unwrap_or(false);
    let is_copc = !is_tile_mode && input_path.map(|p| {
        las::Reader::from_path(p)
            .map(|r| r.header().copc_info_vlr().is_some())
            .unwrap_or(false)
    }).unwrap_or(false);
    if is_copc { info!("Detected COPC format"); }

    let mut class_defs = load_class_defs();
    if let Some(ref cd) = class_defs {
        info!("Custom classes: {:?}", cd.keys().collect::<Vec<_>>());
    }
    // Status line shown in the Classes editor after a save attempt.
    let mut classes_save_status: Option<String> = None;

    let mut color_mode = ColorMode::Height;
    let mut background: Color = BLACK;
    let mut app_mode = AppMode::View3D;

    // Non-COPC LAZ stages. First entry pins the shared viewer_origin used by
    // any subsequent stages added via drag-and-drop.
    let mut stages: Vec<Stage> = Vec::new();
    // Tile mode state
    let mut tile_mgr: Option<TileStageManager> = None;
    // COPC stages: one entry in single-file mode, one per COPC file in directory mode.
    let mut copc_stages: Vec<CopcStage> = Vec::new();

    // Default height range used for camera framing and section-corridor depth when
    // nothing is loaded yet. Overwritten by the first data source that loads.
    let mut height_range: f32 = 100.0;

    if let Some(input_path) = input_path {
        if is_tile_mode {
            let mut mgr = {
                let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
                TileStageManager::new(ctx, input_path, class_defs.clone())
            };

            // Also load any COPC files in the same directory.
            let copc_paths = scan_dir_for_copc(input_path);
            if !copc_paths.is_empty() {
                info!("Found {} COPC file(s) in directory", copc_paths.len());
                let mut unified_min = mgr.global_min;
                let mut unified_max = mgr.global_max;
                for path in &copc_paths {
                    if let Ok(r) = las::Reader::from_path(path) {
                        let b = r.header().bounds();
                        unified_min[0] = unified_min[0].min(b.min.x);
                        unified_min[1] = unified_min[1].min(b.min.y);
                        unified_min[2] = unified_min[2].min(b.min.z);
                        unified_max[0] = unified_max[0].max(b.max.x);
                        unified_max[1] = unified_max[1].max(b.max.y);
                        unified_max[2] = unified_max[2].max(b.max.z);
                    }
                }
                mgr.global_max = unified_max;
                mgr.global_min = unified_min;
                let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
                for path in &copc_paths {
                    let cs = CopcStage::new(ctx, path, Some(unified_min), color_mode, class_defs.clone());
                    copc_stages.push(cs);
                }
            }

            height_range = mgr.height_range();
            tile_mgr = Some(mgr);
        } else if is_copc {
            let cs = {
                let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
                CopcStage::new(ctx, input_path, None, color_mode, class_defs.clone())
            };
            height_range = cs.height_range();
            copc_stages.push(cs);
        } else {
            let s = {
                let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
                Stage::new(ctx, input_path, None, color_mode, class_defs.clone())
            };
            height_range = s.height_range();
            stages.push(s);
        }
    }

    // Editable height colour-bar endpoints, in metres relative to the dataset
    // minimum. Default to the full Z span; the stages were constructed with the
    // same defaults, so no rebuild is needed up front.
    let mut height_color_min: f32 = 0.0;
    let mut height_color_max: f32 = height_range;
    // Height band, in viewer space. Kept separately from `z_clip` so toggling
    // the filter off and on again restores the band the operator set up.
    let mut z_clip_enabled = false;
    let mut z_clip_lo: f32 = 0.0;
    let mut z_clip_hi: f32 = height_range;
    // Which classes are shown. A view setting like the height band: not saved
    // to classes.txt, and deliberately not applied to export — see the note in
    // the Classes window.
    let mut class_vis = ClassVis::ALL;

    let mut tile_rgb_overrides: Vec<Option<[f32; 3]>> = if let Some(ref mgr) = tile_mgr {
        vec![None; mgr.tiles.len()]
    } else {
        Vec::new()
    };

    let world_up = vec3(0.0, 0.0, 1.0);
    let mut yaw: f32 = std::f32::consts::FRAC_PI_2; //std::f32::consts::FRAC_PI_4;       // 45°
    let mut pitch: f32 = -std::f32::consts::FRAC_PI_2;    // -90° (true top-down)

    // Compute camera right/up from yaw+pitch, handling the top-down gimbal-lock case.
    // At pitch=-90°, front=(0,0,-1) is parallel to world_up so cross product is degenerate;
    // use the yaw direction as the horizontal reference instead.
    let recompute_right_up = |yaw: f32, pitch: f32, front: Vec3| -> (Vec3, Vec3) {
        let r = if pitch.abs() > std::f32::consts::FRAC_PI_2 - 0.01 {
            front.cross(vec3(yaw.cos(), yaw.sin(), 0.0)).normalize()
        } else {
            front.cross(world_up).normalize()
        };
        (r, r.cross(front).normalize())
    };

    let mut front = vec3(
        yaw.cos() * pitch.cos(),
        yaw.sin() * pitch.cos(),
        pitch.sin(),
    )
    .normalize();
    let (mut right, mut up) = recompute_right_up(yaw, pitch, front);

    // Compute initial camera position and ortho_scale to fit the data on screen.
    // Mutable: the first drag-and-drop load updates these if nothing was loaded at startup.
    let (mut data_extent_x, mut data_extent_y): (f32, f32) = if let Some(ref mgr) = tile_mgr {
        let mut gmax = mgr.global_max;
        let mut gmin = mgr.global_min;
        for cs in &copc_stages {
            gmin[0] = gmin[0].min(cs.global_min[0]); gmin[1] = gmin[1].min(cs.global_min[1]);
            gmax[0] = gmax[0].max(cs.global_max[0]); gmax[1] = gmax[1].max(cs.global_max[1]);
        }
        ((gmax[0] - gmin[0]) as f32, (gmax[1] - gmin[1]) as f32)
    } else if !copc_stages.is_empty() {
        let (mut gmin, mut gmax) = (copc_stages[0].global_min, copc_stages[0].global_max);
        for cs in &copc_stages[1..] {
            gmin[0] = gmin[0].min(cs.global_min[0]); gmin[1] = gmin[1].min(cs.global_min[1]);
            gmax[0] = gmax[0].max(cs.global_max[0]); gmax[1] = gmax[1].max(cs.global_max[1]);
        }
        ((gmax[0] - gmin[0]) as f32, (gmax[1] - gmin[1]) as f32)
    } else if let Some(stg) = stages.first() {
        (
            (stg.cloud.max[0] - stg.cloud.min[0]) as f32,
            (stg.cloud.max[1] - stg.cloud.min[1]) as f32,
        )
    } else {
        (100.0, 100.0)
    };
    let data_center = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5);
    let mut position = data_center - front * (data_extent_x.max(data_extent_y) * 2.0);
    let mut last_mouse_position: Vec2 = mouse_position().into();

    let mut orbit_pivot = data_center;
    let mut grabbed = false;
    set_cursor_grab(grabbed);
    show_mouse(!grabbed);
    let mut pivot_locked = false;
    let mut egui_wants_pointer_prev = false;
    // Whether egui had keyboard focus last frame — a class label being edited,
    // say. Without this, typing a name would trigger the single-letter view and
    // cross-section shortcuts.
    let mut egui_wants_keyboard_prev = false;

    let aspect = screen_width() / screen_height().max(1.0);
    let mut ortho_scale: f32 = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;

    // Section state (single-file only)
    let mut section_start: Option<(f32, f32)> = None;
    let mut section_end: Option<(f32, f32)> = None;
    let mut section_depth: f32 = 2.0;
    let mut section_data: Option<SectionData> = None;
    // Sections replaced since the last drain, waiting for their buffers to be
    // released. See `set_section`.
    let mut section_trash: Vec<SectionData> = Vec::new();
    // Cache of raw COPC section points (one Vec per stage) used to recolour on mode change.
    #[allow(clippy::type_complexity)]
    let mut copc_section_raw: Option<(Vec<Vec<(LasPoint, f32, f32)>>, f32, f32, f32)> = None;
    // The section view has its own colour mode. Changing it there must not
    // silently repaint the 3D view, and equally the 3D combo box must go on
    // reporting the mode the 3D view is actually drawn in.
    let mut section_color_mode = color_mode;
    // Set once the operator picks a mode in the section view. Until then a new
    // section opens in whatever the 3D view is using; afterwards it keeps the
    // section's own choice, so a workflow of cut → colour by classification →
    // back → cut again does not have to reselect it every time.
    let mut section_color_mode_pinned = false;
    let mut section_pan = glam::Vec2::ZERO;
    let mut section_zoom: f32 = 50.0;
    let mut sec_measure_a: Option<glam::Vec2> = None;
    let mut sec_measure_b: Option<glam::Vec2> = None;
    // Picked point in the section view: the LAS point plus its (along, height)
    // position in section space, used to mark it on screen.
    let mut sec_picked_point: Option<(LasPoint, glam::Vec2)> = None;

    // ---- Manual classification state ----
    // Lasso vertices in section space: (metres along the line, viewer-space Z).
    let mut lasso: Vec<glam::Vec2> = Vec::new();
    let mut classify_target: u8 = 7;
    let mut classify_from: Option<u8> = None;
    let mut classify_status: Option<String> = None;
    // Plan-view ground tool: lasso footprint in viewer-space XY, and how far
    // above the fitted ground a point must stand to be selected.
    let mut ground_lasso: Vec<glam::Vec2> = Vec::new();
    let mut ground_offset: f32 = 0.20;
    let mut ground_status: Option<String> = None;
    // Which flavour of plan-view lasso is being drawn.
    let mut lasso_kind = LassoKind::AboveGround;
    // Classes omitted from an export — the noise-filtering half of the workflow.
    let mut export_filter = crate::classify::ExportFilter::default();
    let mut export_job: Option<crate::export::ExportJob> = None;
    let mut export_status: Option<String> = None;
    let mut show_export_window = false;
    // Set once a corridor has been pinned, so leaving the section releases it.
    let mut corridor_pinned = false;
    let mut corridor_status: Option<String> = None;
    // Where to land once the corridor has streamed in: the lasso tool if the
    // operator asked for it, or plain section view after an undo.
    let mut corridor_load_to_lasso = false;
    let mut corridor_total_points: i64 = 0;

    // Point pick state
    // (stage_idx, point_idx) — picked point in the multi-Stage list.
    let mut picked_point: Option<(usize, usize)> = None;
    let mut picked_tile_point: Option<(usize, usize)> = None;
    let mut picked_copc_point: Option<LasPoint> = None;

    // Tile mode: selected tile for UI
    let mut selected_tile: Option<usize> = None;

    // Measure mode state (viewer-space positions)
    let mut measure_a: Option<glam::Vec3> = None;
    let mut measure_b: Option<glam::Vec3> = None;

    let mut point_size: f32 = 2.0;

    // Compute viewer-space offset (world coords of the viewer origin).
    // Mutable: updated when the first file is loaded via drag-and-drop.
    let mut data_offset: [f64; 3] = if let Some(ref mgr) = tile_mgr {
        mgr.global_min
    } else if !copc_stages.is_empty() {
        copc_stages[0].viewer_origin
    } else if let Some(stg) = stages.first() {
        stg.viewer_origin
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut poly_files: Vec<polylines::PolylineFile> = Vec::new();
    let mut selected_poly: Option<polylines::PolyRef> = None;
    let mut poly_save_status: Option<String> = None;
    let mut digitise = vectorise::Digitise::default();
    // Set by the panel or by `P`, acted on once below rather than in each of
    // the three per-mode blocks.
    let mut digitise_request: Option<vectorise::Request> = None;
    // Whether the COPC neighbourhood pins are currently held for digitising,
    // so leaving the mode releases them and the LOD is free to evict again.
    let mut digitise_pinned = false;
    let mut pending_salome: Option<(usize, String, Vec<String>)> = None;
    let mut poly_load_error: Option<String> = None;

    let mut gcp_files: Vec<gcp::GcpFile> = Vec::new();
    let mut selected_gcp: Option<(usize, usize)> = None; // (file_idx, point_idx)
    let mut gcp_load_error: Option<String> = None;

    let mut edl_enabled = false;
    let mut edl_strength: f32 = 8.0;
    let mut edl_renderer = {
        let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
        EdlRenderer::new(ctx, screen_width() as u32, screen_height() as u32)
    };

    // Pinned viewer origin for drag-and-drop. Seeded from whatever was loaded at
    // startup; if nothing was loaded and the user drops a file, it's set from
    // that file's bounds. Never reassigned for the rest of the session so all
    // stages share one coordinate space.
    let mut pinned_viewer_origin: Option<[f64; 3]> = if !copc_stages.is_empty() {
        Some(copc_stages[0].viewer_origin)
    } else if let Some(ref mgr) = tile_mgr {
        Some(mgr.global_min)
    } else if let Some(s) = stages.first() {
        Some(s.viewer_origin)
    } else {
        None
    };
    // Stages requested for unload this frame, applied after the egui block to
    // avoid borrow conflicts.
    let mut copc_unload_request: Option<usize> = None;
    let mut stage_unload_request: Option<usize> = None;
    // File paths queued for loading. Populated by the drag-and-drop handler
    // and by the "Open file…" button; drained at the top of each frame.
    let mut pending_loads: Vec<String> = Vec::new();

    // Windows-only: number of entries already taken from miniquad's drop list
    // (the list only grows on Windows, so we track how many we've consumed).
    #[cfg(target_os = "windows")]
    let mut last_dropped_count: usize = 0;

    loop {
        //println!("{} {}",yaw,pitch);
        //println!("{}",app_mode.label());
        let delta = get_frame_time();

        // ---- Collect drag-and-dropped files into the load queue ----
        // On X11/XWayland/wasm miniquad fires files_dropped_event, so this works.
        // On Wayland use the "Open file…" button.
        for df in get_dropped_files() {
            if let Some(path) = df.path {
                pending_loads.push(path.to_string_lossy().to_string());
            }
        }
        // On Windows, miniquad 0.4 records dropped files (DragAcceptFiles is enabled)
        // but never calls files_dropped_event nor clears its list, so get_dropped_files
        // stays empty. Poll miniquad's global drop list directly and take only the
        // entries appended since last frame.
        #[cfg(target_os = "windows")]
        {
            let count = miniquad::window::dropped_file_count();
            for i in last_dropped_count..count {
                if let Some(path) = miniquad::window::dropped_file_path(i) {
                    pending_loads.push(path.to_string_lossy().to_string());
                }
            }
            last_dropped_count = count;
        }

        // ---- Drain the pending-load queue ----
        for path_str in pending_loads.drain(..) {
            let header = match las::Reader::from_path(&path_str) {
                Ok(r) => r.header().clone(),
                Err(e) => { error!("Cannot open file {}: {}", path_str, e); continue; }
            };
            let is_copc = header.copc_info_vlr().is_some();
            // A drop into a previously-empty viewer should also re-fit the camera;
            // otherwise the default startup framing (sized for the 100×100 placeholder
            // extents) leaves the new data way off-screen until the user clicks Zoom All.
            let was_empty = pinned_viewer_origin.is_none();
            // Seed the session's viewer origin from the first load; reuse
            // afterwards so every stage lives in the same coordinate space.
            let origin = pinned_viewer_origin.unwrap_or_else(|| {
                let b = header.bounds();
                let o = [b.min.x, b.min.y, b.min.z];
                info!("Pinning viewer origin to first loaded file: {:?}", o);
                let dx = (b.max.x - b.min.x) as f32;
                let dy = (b.max.y - b.min.y) as f32;
                let dz = (b.max.z - b.min.z) as f32;
                if data_extent_x <= 100.5 && data_extent_y <= 100.5 {
                    data_extent_x = dx.max(1.0);
                    data_extent_y = dy.max(1.0);
                }
                if height_range <= 100.5 { height_range = dz.max(1.0); }
                data_offset = o;
                o
            });
            pinned_viewer_origin = Some(origin);
            // First load into an empty viewer: reset the colour bar to span the new
            // data. Subsequent drops keep whatever endpoints the user has set.
            if was_empty {
                height_color_min = 0.0;
                height_color_max = height_range;
                z_clip_lo = 0.0;
                z_clip_hi = height_range;
                z_clip_enabled = false;
            }
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            if is_copc {
                let mut cs = CopcStage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                cs.color_h_min = height_color_min;
                cs.color_h_max = height_color_max;
                cs.rebuild_colors(ctx, color_mode);
                info!("Loaded COPC file: {}", path_str);
                copc_stages.push(cs);
            } else {
                let mut s = Stage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                s.color_h_min = height_color_min;
                s.color_h_max = height_color_max;
                s.rebuild_colors(ctx, color_mode);
                info!("Loaded LAZ file: {}", path_str);
                stages.push(s);
            }
            if was_empty {
                let aspect = screen_width() / screen_height().max(1.0);
                ortho_scale = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;
                position = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5)
                    - front * (data_extent_x.max(data_extent_y) * 2.0);
                orbit_pivot = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5);
            }
        }

        // ---- Apply any pending unload requests from last frame's UI ----
        if let Some(idx) = copc_unload_request.take()
            && idx < copc_stages.len()
        {
            let cs = copc_stages.remove(idx);
            info!("Unloading COPC file: {}", cs.filepath);
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            cs.unload(ctx);
            set_section(&mut section_data, &mut section_trash, None);
            copc_section_raw = None;
        }
        if let Some(idx) = stage_unload_request.take()
            && idx < stages.len()
        {
            let s = stages.remove(idx);
            info!("Unloading LAZ file: {}", s.filepath);
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            s.unload(ctx);
            set_section(&mut section_data, &mut section_trash, None);
            // Picked point may reference an unloaded stage.
            if let Some((si, _)) = picked_point && si >= stages.len() {
                picked_point = None;
            }
        }

        // If everything has been unloaded, forget the pinned coordinate frame and
        // reset the placeholder extents. The next load then re-pins the viewer origin
        // to the new data's location and re-fits the camera (the `was_empty` path),
        // so loading clouds in a different place frames correctly.
        if stages.is_empty() && copc_stages.is_empty() && tile_mgr.is_none() {
            pinned_viewer_origin = None;
            data_extent_x = 100.0;
            data_extent_y = 100.0;
            height_range = 100.0;
        }

        // Height band in force this frame. Pushed unconditionally so a file
        // loaded mid-session picks it up without any special casing.
        let z_clip = if z_clip_enabled {
            crate::raw_miniquad::ZClip { min: z_clip_lo, max: z_clip_hi }
        } else {
            crate::raw_miniquad::ZClip::OPEN
        };
        apply_z_clip(&mut stages, &mut copc_stages, &mut tile_mgr, z_clip);
        // Polylines are stored in world f64 and drawn from a viewer-space
        // cache. The origin is re-pinned whenever everything is unloaded and
        // new data arrives somewhere else, so the caches are checked against
        // it here rather than assumed to be current. A no-op — one float
        // comparison per file — unless the origin actually moved.
        for pf in poly_files.iter_mut() { pf.refresh_view(data_offset); }
        // Same treatment for the visible class set. Unlike the height band this
        // is baked into the colour buffers, so a change — including a file
        // arriving unfiltered while a filter is on — has to force a rebuild.
        let class_vis_changed = apply_class_vis(&mut stages, &mut copc_stages, &mut tile_mgr, class_vis);
        // Rebuilt here rather than in each of the three per-mode blocks below,
        // and unconditionally rather than only in Classification mode: the
        // hidden-class sentinel rides in the colour buffer whatever the colour
        // mode is, so every mode has to be repainted. One toggle therefore
        // costs one colour rebuild — the same cost as changing colour mode,
        // which is already accepted as interactive.
        if class_vis_changed {
            let mut gl = unsafe { get_internal_gl() };
            gl.flush();
            for st in stages.iter_mut() { st.rebuild_colors(&mut *gl.quad_context, color_mode); }
            for cs in copc_stages.iter_mut() { cs.rebuild_colors(&mut *gl.quad_context, color_mode); }
            if let Some(mgr) = tile_mgr.as_mut() { mgr.rebuild_colors(&mut *gl.quad_context, color_mode); }
        }

        if !section_trash.is_empty() {
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            for sec in section_trash.drain(..) {
                sec.unload(ctx);
            }
        }

        // Highest classification id present in the loaded data; the class editor
        // shows one row per id up to this value.
        let max_class: u8 = stages.iter().map(|s| s.cloud.max_class)
            .chain(copc_stages.iter().map(|cs| cs.max_class))
            .chain(tile_mgr.iter().map(|m| m.max_class()))
            .max().unwrap_or(0);

        // ---- Export progress ----
        if let Some(job) = export_job.as_mut() {
            job.poll();
            if let Some(result) = job.finished.take() {
                export_status = Some(match result {
                    Ok((path, written, dropped)) => format!(
                        "Wrote {} — {} points, {} dropped",
                        path.file_name().and_then(|s| s.to_str()).unwrap_or("file"),
                        written, dropped,
                    ),
                    Err(e) => format!("Export failed: {e}"),
                });
                export_job = None;
            }
        }

        if is_key_pressed(KeyCode::Escape) {
            if app_mode == AppMode::SectionCorridorLoad {
                // Abandon the force-load and go back to browsing the slice that
                // is already on screen.
                for cs in copc_stages.iter_mut() { cs.clear_pins(); }
                corridor_pinned = false;
                corridor_status = None;
                app_mode = AppMode::SectionView;
            } else if app_mode == AppMode::GroundLasso {
                if ground_lasso.is_empty() {
                    app_mode = AppMode::View3D;
                } else {
                    ground_lasso.clear();
                }
            } else if app_mode == AppMode::Digitise {
                // Esc drops the line in progress; a second Esc leaves the mode.
                // Same shape as the lasso, so the two do not need learning
                // separately.
                if digitise.is_drawing() { digitise.clear(); } else { app_mode = AppMode::View3D; }
            } else if app_mode == AppMode::SectionLasso {
                if lasso.is_empty() {
                    app_mode = AppMode::SectionView;
                } else {
                    lasso.clear();
                }
            } else if app_mode == AppMode::SectionPointPick {
                app_mode = AppMode::SectionView;
            } else if app_mode != AppMode::View3D {
                // Any exit from the section releases the corridor pins, so the
                // LOD is free to evict those nodes again.
                if corridor_pinned {
                    for cs in copc_stages.iter_mut() { cs.clear_pins(); }
                    corridor_pinned = false;
                }
                corridor_status = None;
                lasso.clear();
                app_mode = AppMode::View3D;
                set_section(&mut section_data, &mut section_trash, None);
            } else {
                break;
            }
        }

        // ---- Section 2D view ----
        if matches!(app_mode, AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso | AppMode::SectionCorridorLoad) {
            // Navigation
            if is_key_down(KeyCode::W) || is_key_down(KeyCode::Up) {
                section_pan.y += MOVE_SPEED * section_zoom * 0.02;
            }
            if is_key_down(KeyCode::S) || is_key_down(KeyCode::Down) {
                section_pan.y -= MOVE_SPEED * section_zoom * 0.02;
            }
            if is_key_down(KeyCode::A) || is_key_down(KeyCode::Left) {
                section_pan.x -= MOVE_SPEED * section_zoom * 0.02;
            }
            if is_key_down(KeyCode::D) || is_key_down(KeyCode::Right) {
                section_pan.x += MOVE_SPEED * section_zoom * 0.02;
            }

            let sec_mouse_pos: Vec2 = mouse_position().into();
            let sec_mouse_delta = sec_mouse_pos - last_mouse_position;
            last_mouse_position = sec_mouse_pos;
            if is_mouse_button_down(MouseButton::Right) {
                let units_per_pixel = section_zoom * 2.0 / screen_height();
                section_pan.x -= sec_mouse_delta.x * units_per_pixel;
                section_pan.y += sec_mouse_delta.y * units_per_pixel;
            }

            let (_mwx, mwy) = mouse_wheel();
            if mwy.abs() > 0.001 {
                let factor = if mwy > 0.0 { 0.9 } else { 1.1 };
                section_zoom *= factor;
                section_zoom = section_zoom.clamp(0.1, 10000.0);
            }

            // ---- Streaming the full-resolution corridor ----
            if app_mode == AppMode::SectionCorridorLoad {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                for cs in copc_stages.iter_mut() {
                    cs.service_pins(&mut *gl.quad_context, section_color_mode);
                }
                let pending: usize = copc_stages.iter().map(|cs| cs.pins_pending()).sum();
                let total: usize = copc_stages.iter().map(|cs| cs.pinned_count()).sum();
                if pending == 0 {
                    if let (Some(start), Some(end)) = (section_start, section_end) {
                        let mut all_verts = Vec::new();
                        let mut per_stage_raw: Vec<Vec<(LasPoint, f32, f32)>> = Vec::new();
                        let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
                        let mut length = 0.0_f32;
                        let height_range = copc_stages[0].height_range();
                        for cs in &copc_stages {
                            let (verts, raw, len, hmin, hmax) =
                                cs.extract_section_verts(start, end, section_depth, section_color_mode);
                            all_verts.extend(verts);
                            per_stage_raw.push(raw);
                            h_min = h_min.min(hmin); h_max = h_max.max(hmax);
                            length = length.max(len);
                        }
                        let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
                        let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
                        info!("COPC Section re-cut at full resolution: {} points", all_verts.len());
                        copc_section_raw = Some((per_stage_raw, length, h_min, h_max));
                        let sec = make_section_data(
                            &mut *gl.quad_context, all_verts, length, h_min, h_max, height_range);
                        set_section(&mut section_data, &mut section_trash, Some(sec));
                        sec_picked_point = None;
                    }
                    corridor_status = None;
                    app_mode = if corridor_load_to_lasso {
                        AppMode::SectionLasso
                    } else {
                        AppMode::SectionView
                    };
                } else {
                    let done = total.saturating_sub(pending);
                    corridor_status = Some(format!(
                        "Loading corridor at full resolution: {done}/{total} nodes, {:.1}M points",
                        corridor_total_points as f64 / 1e6,
                    ));
                }
            }

            clear_background(background);

            if let Some(ref sec) = section_data {
                let aspect = screen_width() / screen_height();
                let half_h = section_zoom;
                let half_w = half_h * aspect;
                let section_mvp = glam::Mat4::orthographic_rh_gl(
                    section_pan.x - half_w, section_pan.x + half_w,
                    section_pan.y - half_h, section_pan.y + half_h,
                    -1.0, 1.0,
                );

                {
                    let mut gl = unsafe { get_internal_gl() };
                    gl.flush();
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                    if let Some(stg) = stages.first() {
                        stg.draw_section(&mut *gl.quad_context, sec, &section_mvp, point_size);
                    } else if let Some(ref mgr) = tile_mgr {
                        mgr.draw_section(&mut *gl.quad_context, sec, &section_mvp, point_size);
                    } else if let Some(cs) = copc_stages.first() {
                        cs.draw_section(&mut *gl.quad_context, sec, &section_mvp, point_size);
                    }
                    gl.quad_context.end_render_pass();
                }

                set_default_camera();

                let world_to_screen_x = |wx: f32| -> f32 {
                    (wx - (section_pan.x - half_w)) / (2.0 * half_w) * screen_width()
                };
                let world_to_screen_y = |wy: f32| -> f32 {
                    (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * screen_height()
                };

                let ground_sy = world_to_screen_y(0.0);
                if ground_sy > 0.0 && ground_sy < screen_height() {
                    draw_line(0.0, ground_sy, screen_width(), ground_sy, 1.0, Color::new(0.4, 0.4, 0.4, 0.5));
                    draw_text("0m", 5.0, ground_sy - 3.0, 16.0, GRAY);
                }

                let sx0 = world_to_screen_x(0.0);
                let sx1 = world_to_screen_x(sec.length);
                draw_line(sx0, 0.0, sx0, screen_height(), 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_line(sx1, 0.0, sx1, screen_height(), 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_text("start", sx0 + 3.0, 30.0, 16.0, YELLOW);
                draw_text(&format!("{:.1}m", sec.length), sx1 + 3.0, 30.0, 16.0, YELLOW);

                if !poly_files.is_empty() &&
                    let (Some(start), Some(end)) = (section_start, section_end) {
                        polylines::draw_section(
                            &poly_files, start, end,
                            section_depth,
                            section_pan, section_zoom,
                            screen_width(), screen_height(),
                        );

                }

                if !gcp_files.is_empty() &&
                    let (Some(start), Some(end)) = (section_start, section_end) {
                        gcp::draw_section(
                            &gcp_files, data_offset, start, end,
                            section_depth, sec.length,
                            section_pan, section_zoom,
                            selected_gcp,
                        );
                }

                // ---- Measure overlays ----
                let marker_sz = 8.0;
                if let Some(a) = sec_measure_a {
                    let sx = world_to_screen_x(a.x);
                    let sy = world_to_screen_y(a.y);
                    draw_line(sx - marker_sz, sy, sx + marker_sz, sy, 1.5, YELLOW);
                    draw_line(sx, sy - marker_sz, sx, sy + marker_sz, 1.5, YELLOW);
                    draw_circle_lines(sx, sy, 5.0, 1.0, YELLOW);
                    draw_text("A", sx + 8.0, sy - 6.0, 16.0, YELLOW);
                }
                if let Some(b) = sec_measure_b {
                    let sx = world_to_screen_x(b.x);
                    let sy = world_to_screen_y(b.y);
                    draw_line(sx - marker_sz, sy, sx + marker_sz, sy, 1.5, ORANGE);
                    draw_line(sx, sy - marker_sz, sx, sy + marker_sz, 1.5, ORANGE);
                    draw_circle_lines(sx, sy, 5.0, 1.0, ORANGE);
                    draw_text("B", sx + 8.0, sy - 6.0, 16.0, ORANGE);
                }
                if let (Some(a), Some(b)) = (sec_measure_a, sec_measure_b) {
                    draw_line(
                        world_to_screen_x(a.x), world_to_screen_y(a.y),
                        world_to_screen_x(b.x), world_to_screen_y(b.y),
                        1.5, WHITE,
                    );
                }
                // Picked point marker
                if let Some((_, spos)) = &sec_picked_point {
                    let sx = world_to_screen_x(spos.x);
                    let sy = world_to_screen_y(spos.y);
                    draw_circle_lines(sx, sy, 6.0, 1.5, SKYBLUE);
                    draw_line(sx - marker_sz, sy, sx + marker_sz, sy, 1.0, SKYBLUE);
                    draw_line(sx, sy - marker_sz, sx, sy + marker_sz, 1.0, SKYBLUE);
                }
                // ---- Classification lasso ----
                if app_mode == AppMode::SectionLasso || !lasso.is_empty() {
                    let lasso_col = class_color(classify_target, &class_defs);
                    let col = Color::new(lasso_col[0], lasso_col[1], lasso_col[2], 1.0);
                    let pts: Vec<(f32, f32)> = lasso.iter()
                        .map(|v| (world_to_screen_x(v.x), world_to_screen_y(v.y)))
                        .collect();
                    for w in pts.windows(2) {
                        draw_line(w[0].0, w[0].1, w[1].0, w[1].1, 2.0, col);
                    }
                    // Rubber-band from the last vertex to the cursor, and a
                    // dashed closing edge, so the polygon that will actually be
                    // applied is visible before committing to it.
                    if let Some(&last) = pts.last() {
                        let (mx, my) = mouse_position();
                        if app_mode == AppMode::SectionLasso {
                            draw_line(last.0, last.1, mx, my, 1.5, col);
                        }
                        if pts.len() >= 2 {
                            let first = pts[0];
                            let end = if app_mode == AppMode::SectionLasso { (mx, my) } else { last };
                            draw_line(end.0, end.1, first.0, first.1, 1.0,
                                Color::new(col.r, col.g, col.b, 0.4));
                        }
                    }
                    for (sx, sy) in &pts {
                        draw_circle_lines(*sx, *sy, 4.0, 1.5, col);
                    }
                }

                // Crosshair at mouse when in measure or pick mode
                if matches!(app_mode, AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso) {
                    let (mx, my) = mouse_position();
                    let ch = 12.0;
                    draw_line(mx - ch, my, mx + ch, my, 1.0, Color::new(1.0, 1.0, 1.0, 0.7));
                    draw_line(mx, my - ch, mx, my + ch, 1.0, Color::new(1.0, 1.0, 1.0, 0.7));
                }
            }

            // Corridor force-load banner. This is the one moment where the
            // viewer deliberately ignores the LOD point budget, and the
            // operator is waiting on it, so say what is happening.
            if let Some(msg) = &corridor_status {
                let font_size = 20.0;
                let dims = measure_text(msg, None, font_size as u16, 1.0);
                let (bx, by) = ((screen_width() - dims.width) * 0.5, screen_height() * 0.5);
                draw_rectangle(bx - 14.0, by - dims.height - 10.0,
                    dims.width + 28.0, dims.height + 22.0, Color::new(0.0, 0.0, 0.0, 0.72));
                draw_text(msg, bx, by, font_size, YELLOW);
                let hint = "Esc to cancel";
                let hd = measure_text(hint, None, 16, 1.0);
                draw_text(hint, (screen_width() - hd.width) * 0.5, by + 22.0, 16.0, GRAY);
            }

            let prev_color_mode = section_color_mode;
            let mut height_color_changed = false;
            let mut z_clip_changed = false;
            let mut egui_wants_sec_pointer = false;
            let mut egui_wants_sec_keyboard = false;
            let mut class_colors_changed = false;
            // Deferred so the work happens outside the egui closure, which
            // already borrows most of the app state.
            let mut do_apply_lasso = false;
            let mut start_lasso = false;
            let mut do_undo_op = false;
            let mut do_save_edits = false;
            let mut leave_corridor = false;
            let mut export_request: Option<usize> = None;
            // Tile mode streams whole tiles through a manager that owns its own
            // point storage; wiring edits through it is a separate job, so the
            // classify controls are hidden rather than silently doing nothing.
            let classify_supported = !stages.is_empty() || !copc_stages.is_empty();
            let edits_count: usize = stages.iter().map(|s| s.edits.ops.len())
                .chain(copc_stages.iter().map(|cs| cs.edits.ops.len()))
                .max().unwrap_or(0);
            let edits_dirty = stages.iter().any(|s| s.edits.dirty)
                || copc_stages.iter().any(|cs| cs.edits.dirty);
            let export_sources = export_source_list(&stages, &copc_stages);
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_sec_pointer = egui_ctx.is_pointer_over_area();
                egui_wants_sec_keyboard = egui_ctx.wants_keyboard_input();
                class_colors_changed = classes_window(egui_ctx, &mut class_defs, &mut class_vis, max_class, &mut classes_save_status);
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        if ui.button("Back to 3D").clicked() {
                            app_mode = AppMode::View3D;
                            set_section(&mut section_data, &mut section_trash, None);
                            sec_measure_a = None;
                            sec_measure_b = None;
                            sec_picked_point = None;
                            lasso.clear();
                            leave_corridor = true;
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            egui_macroquad::egui::ComboBox::from_id_salt("color_mode")
                                .selected_text(section_color_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in &ColorMode::ALL {
                                        ui.selectable_value(&mut section_color_mode, *mode, mode.label());
                                    }
                                });
                        });
                        height_color_changed = height_color_controls(ui, section_color_mode, &mut height_color_min, &mut height_color_max, data_offset[2]);
                        z_clip_changed = z_clip_controls(
                            ui, &mut z_clip_enabled, &mut z_clip_lo, &mut z_clip_hi,
                            data_offset[2], height_range,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        if let Some(ref sec) = section_data {
                            ui.label(format!(
                                "Section: {:.1}m | {} points | h: {:.1}..{:.1}m",
                                sec.length, sec.num_points, sec.height_min, sec.height_max
                            ));
                        }
                        ui.separator();
                        let pick_label = if app_mode == AppMode::SectionPointPick {
                            "Click a point... (Esc to cancel)"
                        } else {
                            "Point Info"
                        };
                        if ui.button(pick_label).clicked() {
                            if app_mode == AppMode::SectionPointPick {
                                app_mode = AppMode::SectionView;
                            } else {
                                app_mode = AppMode::SectionPointPick;
                                sec_picked_point = None;
                            }
                        }
                        let measuring = matches!(app_mode, AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond);
                        if measuring {
                            if ui.button("Stop measuring").clicked() {
                                app_mode = AppMode::SectionView;
                                sec_measure_a = None;
                                sec_measure_b = None;
                            }
                            if app_mode == AppMode::SectionMeasureFirst {
                                ui.label("Click point A");
                            } else {
                                ui.label("Click point B");
                            }
                        } else if ui.button("Measure").clicked() {
                            app_mode = AppMode::SectionMeasureFirst;
                            sec_measure_a = None;
                            sec_measure_b = None;
                        }
                        if let (Some(a), Some(b)) = (sec_measure_a, sec_measure_b) {
                            let dh = (b.x - a.x).abs();
                            let dv = b.y - a.y;
                            let slant = (dh * dh + dv * dv).sqrt();
                            let angle_deg = dv.atan2(dh).to_degrees();
                            let slope_pct = if dh > 0.0001 { dv / dh * 100.0 } else { f32::INFINITY };
                            ui.separator();
                            ui.label("Measurement:");
                            ui.label(format!("  Δ height:  {:+.3} m", dv));
                            ui.label(format!("  Horiz:     {:.3} m", dh));
                            ui.label(format!("  Slant:     {:.3} m", slant));
                            ui.label(format!("  Angle:     {:.2}°", angle_deg));
                            if slope_pct.is_finite() {
                                ui.label(format!("  Slope:     {:.1}%", slope_pct));
                            } else {
                                ui.label("  Slope:     vertical");
                            }
                        }
                        if let Some((ref p, _)) = sec_picked_point {
                            ui.separator();
                            point_info_ui(ui, "Point Info:", p, &class_defs);
                        }

                        // ---- Manual classification ----
                        ui.separator();
                        if classify_supported {
                            ui.label("Classify:");
                            ui.horizontal(|ui| {
                                ui.label("To:");
                                class_picker(ui, "classify_to", &mut classify_target, max_class, &class_defs);
                            });
                            ui.horizontal(|ui| {
                                let mut restrict = classify_from.is_some();
                                if ui.checkbox(&mut restrict, "Only from:").changed() {
                                    classify_from = if restrict { Some(0) } else { None };
                                }
                                if let Some(ref mut from) = classify_from {
                                    class_picker(ui, "classify_from", from, max_class, &class_defs);
                                }
                            });
                            if app_mode == AppMode::SectionLasso {
                                ui.label(format!("Click to add points ({} so far)", lasso.len()));
                                ui.horizontal(|ui| {
                                    if ui.add_enabled(lasso.len() >= 3,
                                        egui_macroquad::egui::Button::new("Apply")).clicked() {
                                        do_apply_lasso = true;
                                    }
                                    if ui.add_enabled(!lasso.is_empty(),
                                        egui_macroquad::egui::Button::new("Undo point")).clicked() {
                                        lasso.pop();
                                    }
                                    if ui.button("Cancel").clicked() {
                                        lasso.clear();
                                        app_mode = AppMode::SectionView;
                                    }
                                });
                                ui.label("Enter applies, Backspace removes a point, Esc cancels.");
                            } else if app_mode == AppMode::SectionCorridorLoad {
                                ui.label("Loading full-resolution corridor…");
                            } else if ui.button("Lasso classify (O)").clicked() {
                                lasso.clear();
                                start_lasso = true;
                            }
                            ui.horizontal(|ui| {
                                if ui.add_enabled(edits_count > 0,
                                    egui_macroquad::egui::Button::new("Undo last op")).clicked() {
                                    do_undo_op = true;
                                }
                                if ui.add_enabled(edits_dirty,
                                    egui_macroquad::egui::Button::new("Save edits")).clicked() {
                                    do_save_edits = true;
                                }
                            });
                            ui.label(format!(
                                "{} op{}{}",
                                edits_count,
                                if edits_count == 1 { "" } else { "s" },
                                if edits_dirty { " (unsaved)" } else { "" },
                            ));
                            if let Some(msg) = &classify_status {
                                ui.label(msg.as_str());
                            }
                        } else {
                            ui.label("Classification editing is not available in tile mode.");
                        }
                    });
                export_request = export_window(
                    egui_ctx, &mut show_export_window, &mut export_filter,
                    &mut export_job, &mut export_status, &export_sources,
                    max_class, &class_defs,
                );
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, data_offset, background);
                gcp_window(egui_ctx, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
            });

            // A narrowed band should drive the height ramp too: colouring
            // against the full dataset range wastes most of the gradient on
            // levels that are no longer drawn. Route it through the existing
            // height-colour path rather than adding a parallel one.
            if z_clip_changed {
                let (new_min, new_max) = if z_clip_enabled {
                    (z_clip_lo, z_clip_hi)
                } else {
                    (0.0, height_range)
                };
                // A band that maps onto the ramp already in force costs nothing
                // to "apply", and re-uploading every colour to arrive at the
                // same bytes is the most expensive thing this view can do.
                if (new_min, new_max) != (height_color_min, height_color_max) {
                    height_color_min = new_min;
                    height_color_max = new_max;
                    height_color_changed = true;
                }
            }
            egui_macroquad::draw();

            // ---- Lasso keyboard shortcuts ----
            if !egui_wants_sec_keyboard {
                if app_mode == AppMode::SectionLasso {
                    if is_key_pressed(KeyCode::Backspace) { lasso.pop(); }
                    if is_key_pressed(KeyCode::Enter) && lasso.len() >= 3 { do_apply_lasso = true; }
                } else if classify_supported
                    && app_mode != AppMode::SectionCorridorLoad
                    && is_key_pressed(KeyCode::O)
                {
                    // Same entry point as the button, so `O` also triggers the
                    // corridor force-load on a COPC cloud rather than dropping
                    // straight into a lasso over the thinned LOD sample.
                    lasso.clear();
                    start_lasso = true;
                }
            }

            // ---- Starting the lasso pulls in the full-resolution corridor ----
            // A section cut for browsing shows the resident LOD, which is a
            // thinned sample. Classifying against that would mark the sampled
            // points and silently leave the dense remainder alone, so the
            // corridor is force-loaded here — once per section, not on every
            // cut — and the slice re-cut from it before any lasso is drawn.
            if start_lasso {
                if copc_stages.is_empty() || corridor_pinned {
                    app_mode = AppMode::SectionLasso;
                } else if let (Some(start), Some(end)) = (section_start, section_end) {
                    let mut nodes = 0usize;
                    let mut pts: i64 = 0;
                    for cs in copc_stages.iter_mut() {
                        let (n, p) = cs.pin_corridor(start, end, section_depth);
                        nodes += n;
                        pts += p;
                    }
                    if pts > CORRIDOR_POINT_LIMIT {
                        for cs in copc_stages.iter_mut() { cs.clear_pins(); }
                        classify_status = Some(format!(
                            "Corridor holds {:.1}M points (limit {:.0}M) — shorten the \
                             section line or reduce the depth before classifying.",
                            pts as f64 / 1e6, CORRIDOR_POINT_LIMIT as f64 / 1e6,
                        ));
                    } else {
                        info!(
                            "Section corridor: {} nodes, {} points across {} file(s)",
                            nodes, pts, copc_stages.len(),
                        );
                        corridor_total_points = pts;
                        corridor_pinned = true;
                        corridor_load_to_lasso = true;
                        classify_status = None;
                        app_mode = AppMode::SectionCorridorLoad;
                    }
                }
            }

            // ---- Apply a classification op ----
            if do_apply_lasso && lasso.len() >= 3
                && op_from_mask(classify_from, class_vis).is_empty() {
                    classify_status = Some(empty_from_mask_reason(classify_from));
                    lasso.clear();
                    app_mode = AppMode::SectionView;
            } else if do_apply_lasso && lasso.len() >= 3
                && let (Some(start), Some(end)) = (section_start, section_end) {
                    // The op is stored in absolute world coordinates so it stays
                    // meaningful independently of the viewer origin, and of which
                    // nodes happened to be resident when it was drawn.
                    let op = crate::classify::ClassOp::corridor(
                        0,
                        (start.0 as f64 + data_offset[0], start.1 as f64 + data_offset[1]),
                        (end.0 as f64 + data_offset[0], end.1 as f64 + data_offset[1]),
                        section_depth as f64,
                        lasso.iter().map(|v| (v.x as f64, v.y as f64 + data_offset[2])).collect(),
                        classify_target,
                        op_from_mask(classify_from, class_vis),
                    );

                    let mut changed = 0usize;
                    let mut touched = 0usize;
                    {
                        let mut gl = unsafe { get_internal_gl() };
                        gl.flush();
                        for st in stages.iter_mut() {
                            if !op.intersects_aabb(&st.cloud.min, &st.cloud.max) { continue; }
                            let from = st.edits.ops.len();
                            st.edits.push(op.clone());
                            changed += st.apply_edits(&mut *gl.quad_context, color_mode, from);
                            touched += 1;
                        }
                        for cs in copc_stages.iter_mut() {
                            if !op.intersects_aabb(&cs.global_min, &cs.global_max) { continue; }
                            let from = cs.edits.ops.len();
                            cs.edits.push(op.clone());
                            changed += cs.apply_edits_to_loaded(&mut *gl.quad_context, color_mode, from);
                            touched += 1;
                        }
                    }

                    // The section keeps its own copies of the COPC points, for
                    // recolouring and Point Info; they have to move too or the
                    // slice would disagree with the 3D view.
                    if let Some((ref mut per_stage_raw, _, _, _)) = copc_section_raw {
                        for raw in per_stage_raw.iter_mut() {
                            for (p, _, _) in raw.iter_mut() {
                                op.apply_point(p);
                            }
                        }
                    }

                    {
                        let mut gl = unsafe { get_internal_gl() };
                        gl.flush();
                        if !stages.is_empty() {
                            let sec = build_section_from_stages(
                                &mut *gl.quad_context, &stages, start, end, section_depth, section_color_mode,
                            );
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                        } else if let Some((ref per_stage_raw, length, h_min, h_max)) = copc_section_raw {
                            let sec = build_section_from_raw_stages(
                                &mut *gl.quad_context, &copc_stages, per_stage_raw,
                                length, h_min, h_max, section_color_mode,
                            );
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                        }
                    }

                    // Persist immediately. The sidecar is a few hundred bytes and
                    // an unsaved session's worth of manual classification is not
                    // something to leave riding on a clean shutdown.
                    let mut save_err = None;
                    for st in stages.iter_mut().filter(|s| s.edits.dirty) {
                        let path = st.filepath.clone();
                        if let Err(e) = st.edits.save(&path) { save_err = Some(e.to_string()); }
                    }
                    for cs in copc_stages.iter_mut().filter(|c| c.edits.dirty) {
                        let path = cs.filepath.clone();
                        if let Err(e) = cs.edits.save(&path) { save_err = Some(e.to_string()); }
                    }

                    classify_status = Some(match (save_err, touched) {
                        (Some(e), _) => format!("{changed} points reclassified — SAVE FAILED: {e}"),
                        (None, 0) => "Lasso did not overlap any loaded file".to_string(),
                        (None, _) => format!("{changed} points → class {classify_target}"),
                    });
                    lasso.clear();
                    app_mode = AppMode::SectionView;
            }

            // ---- Undo the last op ----
            // An op does not record the classes it overwrote, so undoing means
            // re-deriving from the source: resident points are re-read, and
            // streamed COPC nodes are dropped so they arrive again with the
            // shortened op list applied.
            if do_undo_op {
                // An op is only recorded against the files it overlaps, so
                // "the last op" is the highest sequence number anywhere, and it
                // is removed from exactly the files that carry it.
                let last_seq = stages.iter().filter_map(|s| s.edits.ops.last().map(|o| o.seq))
                    .chain(copc_stages.iter().filter_map(|c| c.edits.ops.last().map(|o| o.seq)))
                    .max();
                let mut stages_touched = false;
                let mut copc_touched = false;
                if let Some(seq) = last_seq {
                    let mut gl = unsafe { get_internal_gl() };
                    gl.flush();
                    for st in stages.iter_mut() {
                        if st.edits.ops.last().map(|o| o.seq) != Some(seq) { continue; }
                        st.edits.undo();
                        st.reload_points(&mut *gl.quad_context, color_mode);
                        let path = st.filepath.clone();
                        let _ = st.edits.save(&path);
                        stages_touched = true;
                    }
                    for cs in copc_stages.iter_mut() {
                        if cs.edits.ops.last().map(|o| o.seq) != Some(seq) { continue; }
                        cs.edits.undo();
                        cs.invalidate_loaded(&mut *gl.quad_context);
                        let path = cs.filepath.clone();
                        let _ = cs.edits.save(&path);
                        copc_touched = true;
                    }
                }

                classify_status = Some(if last_seq.is_some() {
                    "Last op undone".to_string()
                } else {
                    "Nothing to undo".to_string()
                });

                if stages_touched && let (Some(a), Some(b)) = (section_start, section_end) {
                    let mut gl = unsafe { get_internal_gl() };
                    gl.flush();
                    let sec = build_section_from_stages(
                        &mut *gl.quad_context, &stages, a, b, section_depth, section_color_mode,
                    );
                    set_section(&mut section_data, &mut section_trash, Some(sec));
                } else if copc_touched {
                    // The corridor has to stream in again before the slice can
                    // be rebuilt; go back through the loading state.
                    copc_section_raw = None;
                    corridor_load_to_lasso = false;
                    app_mode = AppMode::SectionCorridorLoad;
                }
            }

            if do_save_edits {
                let mut err = None;
                for st in stages.iter_mut() {
                    let path = st.filepath.clone();
                    if let Err(e) = st.edits.save(&path) { err = Some(e.to_string()); }
                }
                for cs in copc_stages.iter_mut() {
                    let path = cs.filepath.clone();
                    if let Err(e) = cs.edits.save(&path) { err = Some(e.to_string()); }
                }
                classify_status = Some(match err {
                    Some(e) => format!("Save failed: {e}"),
                    None => "Edits saved".to_string(),
                });
            }

            if leave_corridor {
                for cs in copc_stages.iter_mut() { cs.clear_pins(); }
                corridor_pinned = false;
            }

            if let Some(idx) = export_request {
                start_export(
                    idx, &stages, &copc_stages, &export_filter,
                    &mut export_job, &mut export_status,
                );
            }

            // ---- Section measure click handling ----
            if !egui_wants_sec_pointer && is_mouse_button_pressed(MouseButton::Left) {
                let aspect = screen_width() / screen_height();
                let half_h = section_zoom;
                let half_w = half_h * aspect;
                let (mx, my) = mouse_position();
                let wx = (mx / screen_width()) * 2.0 * half_w + (section_pan.x - half_w);
                let wy = (1.0 - my / screen_height()) * 2.0 * half_h + (section_pan.y - half_h);
                let pos = glam::Vec2::new(wx, wy);
                match app_mode {
                    AppMode::SectionLasso => {
                        lasso.push(pos);
                    }
                    AppMode::SectionMeasureFirst => {
                        sec_measure_a = Some(pos);
                        sec_measure_b = None;
                        app_mode = AppMode::SectionMeasureSecond;
                    }
                    AppMode::SectionMeasureSecond => {
                        sec_measure_b = Some(pos);
                    }
                    AppMode::SectionPointPick => {
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            sec_picked_point = pick_section_point(
                                &stages, &copc_section_raw, &tile_mgr,
                                start, end, section_depth, z_clip, class_vis, wx, wy,
                            );
                        }
                    }
                    _ => {}
                }
            }

            if class_colors_changed {
                apply_class_defs(&mut stages, &mut copc_stages, &mut tile_mgr, &class_defs);
                // Recolour the 3D sources too, so the view is up to date when
                // the user returns from the section to the 3D view.
                if color_mode == ColorMode::Classification {
                    let gl = unsafe { get_internal_gl() };
                    for s in stages.iter_mut() { s.rebuild_colors(&mut *gl.quad_context, color_mode); }
                    for cs in copc_stages.iter_mut() { cs.rebuild_colors(&mut *gl.quad_context, color_mode); }
                    if let Some(mgr) = tile_mgr.as_mut() { mgr.rebuild_colors(&mut *gl.quad_context, color_mode); }
                }
            }
            // The COPC section keeps a cache of the points it cut, filtered by
            // the band that was in force at the time. Widening the band cannot
            // be served from that cache, so re-cut from the pinned corridor —
            // which is still resident, so this stays a memory operation.
            if (z_clip_changed || class_vis_changed) && !copc_stages.is_empty()
                && let (Some(start), Some(end)) = (section_start, section_end) {
                    let mut per_stage_raw: Vec<Vec<(LasPoint, f32, f32)>> = Vec::new();
                    let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
                    let mut length = 0.0_f32;
                    for cs in &copc_stages {
                        let (_verts, raw, len, hmin, hmax) =
                            cs.extract_section_verts(start, end, section_depth, section_color_mode);
                        per_stage_raw.push(raw);
                        h_min = h_min.min(hmin); h_max = h_max.max(hmax);
                        length = length.max(len);
                    }
                    let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
                    let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
                    copc_section_raw = Some((per_stage_raw, length, h_min, h_max));
                    sec_picked_point = None;
            }

            if section_color_mode != prev_color_mode {
                section_color_mode_pinned = true;
            }

            let class_rebuild = (class_colors_changed && section_color_mode == ColorMode::Classification)
                || class_vis_changed;
            // Storing the ramp bounds is cheap; re-uploading every colour is
            // not. Only Height mode reads the ramp, so in any other mode the
            // new bounds are recorded and the upload waits until the view is
            // switched back to Height — which is what the height filter needs,
            // since it drives these bounds on every slider frame.
            if height_color_changed {
                apply_height_color(&mut stages, &mut copc_stages, &mut tile_mgr, height_color_min, height_color_max);
            }
            let height_rebuild = height_color_changed && section_color_mode == ColorMode::Height;
            if (section_color_mode != prev_color_mode || height_rebuild || class_rebuild) &&
                let (Some(start), Some(end)) = (section_start, section_end) {
                    let gl = unsafe { get_internal_gl() };
                    if !stages.is_empty() {
                        let sec = build_section_from_stages(
                            &mut *gl.quad_context, &stages, start, end, section_depth, section_color_mode,
                        );
                        set_section(&mut section_data, &mut section_trash, Some(sec));
                    } else if !copc_stages.is_empty() {
                        if let Some((ref per_stage_raw, length, h_min, h_max)) = copc_section_raw {
                            let sec = build_section_from_raw_stages(
                                &mut *gl.quad_context, &copc_stages, per_stage_raw,
                                length, h_min, h_max, section_color_mode,
                            );
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                        }
                    } else if let Some(ref mgr) = tile_mgr {
                        let sec = mgr.extract_section(
                            &mut *gl.quad_context, start, end, section_depth, section_color_mode,
                        );
                        set_section(&mut section_data, &mut section_trash, Some(sec));
                    }
            }

            next_frame().await;
            continue;
        }

        // ---- 3D view modes (shared navigation) ----
        let mut start_ground_lasso: Option<LassoKind> = None;
        let mut apply_ground_lasso = false;
        let mut cancel_ground_lasso = false;
        // `C` mirrors the Cross Section button, including its toggle-to-cancel
        // behaviour, so the two stay interchangeable.
        if is_key_pressed(KeyCode::C) && !egui_wants_keyboard_prev {
            match app_mode {
                AppMode::View3D | AppMode::PointPick => {
                    app_mode = AppMode::SectionPickStart;
                    section_start = None;
                    section_end = None;
                    picked_point = None;
                    picked_tile_point = None;
                    picked_copc_point = None;
                }
                AppMode::SectionPickStart | AppMode::SectionPickEnd | AppMode::SectionAdjustDepth => {
                    app_mode = AppMode::View3D;
                    section_start = None;
                    section_end = None;
                }
                _ => {}
            }
        }

        // ---- Plan-view ground tool ----
        // Handled once here rather than in each render block's click `match`,
        // since it behaves identically in single-file, COPC and tile modes.
        // `G` fits the ground and takes what stands above it; `V` takes the
        // whole column. Both toggle off the same way.
        for (key, kind) in [
            (KeyCode::G, LassoKind::AboveGround),
            (KeyCode::V, LassoKind::Inside),
        ] {
            if !is_key_pressed(key) || egui_wants_keyboard_prev { continue; }
            match app_mode {
                AppMode::View3D | AppMode::PointPick => {
                    if is_top_view(pitch) {
                        start_ground_lasso = Some(kind);
                    } else {
                        ground_status = Some(TOP_VIEW_HINT.to_string());
                    }
                }
                AppMode::GroundLasso => {
                    ground_lasso.clear();
                    app_mode = AppMode::View3D;
                }
                _ => {}
            }
        }
        // ---- Digitising ----
        // `P` starts and cancels, mirroring how `G` and `V` toggle the lassos.
        // Unlike them it does not require a top view: see `AppMode::Digitise`.
        if app_mode == AppMode::Digitise && !egui_wants_keyboard_prev {
            if is_key_pressed(KeyCode::N) {
                digitise.snap_mode = digitise.snap_mode.next();
                digitise.status = Some(format!("Snap: {}", digitise.snap_mode.label()));
            }
            // Modifier + wheel, so the plain wheel keeps zooming. A deliberate
            // divergence from `SectionAdjustDepth`, which takes the plain
            // wheel — this is a tool the operator sits in for minutes at a
            // time, and losing zoom for that whole time would be worse.
            let wheel = mouse_wheel().1;
            if wheel != 0.0 && (is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl)) {
                let step = if wheel > 0.0 { 1.25 } else { 1.0 / 1.25 };
                digitise.radius = (digitise.radius * step)
                    .clamp(vectorise::MIN_SNAP_RADIUS, vectorise::MAX_SNAP_RADIUS);
                digitise.status = Some(format!("Snap radius {:.2} m", digitise.radius));
            }
        }

        if is_key_pressed(KeyCode::P) && !egui_wants_keyboard_prev {
            match app_mode {
                AppMode::View3D | AppMode::PointPick => digitise_request = Some(vectorise::Request::Start),
                AppMode::Digitise => digitise_request = Some(vectorise::Request::Cancel),
                _ => {}
            }
        }

        if is_key_pressed(KeyCode::Tab) {
            grabbed = !grabbed;
            set_cursor_grab(grabbed);
            show_mouse(!grabbed);
        }

        if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
            position += up * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
            position -= up * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
            position -= right * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
            position += right * MOVE_SPEED;
        }

        let (_mwx, mwy) = mouse_wheel();
        if app_mode == AppMode::SectionAdjustDepth {
            if mwy.abs() > 0.001 {
                let factor = if mwy > 0.0 { 1.2 } else { 0.8 };
                section_depth *= factor;
                section_depth = section_depth.clamp(0.1, 500.0);
            }
        } else if mwy.abs() > 0.001 {
            let zoom_factor = if mwy > 0.0 { 0.9 } else { 1.1 };
            ortho_scale *= zoom_factor;
            ortho_scale = ortho_scale.clamp(0.5, 10000.0);
        }

        let mouse_pos: Vec2 = mouse_position().into();
        let mouse_delta = mouse_pos - last_mouse_position;
        last_mouse_position = mouse_pos;

        let mid_z = height_range * 0.5;

        // Orbit pivot: screen-centre ray intersected with the mid_z plane.
        if !pivot_locked {
            if front.z.abs() > 0.01 {
                orbit_pivot = position + front * ((mid_z - position.z) / front.z);
            } else {
                // Looking horizontally, the screen-centre ray never meets the
                // mid-height plane. Dividing by `front.z` here yields an
                // infinity, and `front.z * inf` is NaN — which lands straight
                // in the camera's `target` and blanks the view. Keep the pivot
                // the same distance ahead of the camera instead, so it stays
                // centred and still follows panning.
                let dist = (orbit_pivot - position).dot(front).abs().max(1.0);
                orbit_pivot = position + front * dist;
            }
        }
        
        let mode_uses_click = matches!(app_mode,
            AppMode::SectionPickStart | AppMode::SectionPickEnd |
            AppMode::SectionAdjustDepth | AppMode::GroundLasso | AppMode::Digitise |
            AppMode::PointPick | AppMode::MeasureFirst | AppMode::MeasureSecond
        );
        pivot_locked = false;

        if (grabbed || is_mouse_button_down(MouseButton::Left)) && !mode_uses_click && !egui_wants_pointer_prev {

            pivot_locked = true;

            let old_front = front;
            yaw += mouse_delta.x * delta * -LOOK_SPEED;
            pitch += mouse_delta.y * delta * -LOOK_SPEED;
            pitch = pitch.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

            front = vec3(
                yaw.cos() * pitch.cos(),
                yaw.sin() * pitch.cos(),
                pitch.sin(),
            )
            .normalize();


            // Orbit: keep pivot at screen center after rotation
            position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);

            //front = (orbit_pivot-position).normalize();
            (right, up) = recompute_right_up(yaw, pitch, front);
            
            //println!("{}",front);
            
        }

        if is_mouse_button_down(MouseButton::Right) && !egui_wants_pointer_prev {
            // 1:1 with the cursor — the point under the pointer stays under it.
            // Any multiplier here makes the cloud outrun the mouse and shoot
            // off screen on a short drag.
            let units_per_pixel = ortho_scale * 2.0 / screen_height();
            position -= right * mouse_delta.x * units_per_pixel;
            position += up * mouse_delta.y * units_per_pixel;
        }

        clear_background(background);

        let camera = Camera3D {
            position,
            up,
            target: orbit_pivot,
            projection: Projection::Orthographics,
            fovy: ortho_scale * 2.0,
            z_far: 100000.0,
            z_near: -100000.0,
            ..Default::default()
        };
        set_camera(&camera);
        let view_proj: glam::Mat4 = camera.matrix();
        //println!("{}",view_proj);

        let (mx, my) = macroquad::input::mouse_position();
        let cursor_world = screen_to_world_xy(mx, my, &view_proj, mid_z);

        if app_mode == AppMode::GroundLasso {
            if !egui_wants_pointer_prev && is_mouse_button_pressed(MouseButton::Left)
                && let Some((wx, wy)) = cursor_world {
                    ground_lasso.push(glam::Vec2::new(wx, wy));
            }
            if !egui_wants_keyboard_prev {
                if is_key_pressed(KeyCode::Backspace) { ground_lasso.pop(); }
                if is_key_pressed(KeyCode::Enter) && ground_lasso.len() >= 3 {
                    apply_ground_lasso = true;
                }
            }
        }

        // Digitising input, handled once here rather than in each render
        // block's click match, exactly as the plan-view lasso above is.
        if app_mode == AppMode::Digitise {
            if !egui_wants_pointer_prev && is_mouse_button_pressed(MouseButton::Left) {
                let free = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
                if free {
                    // Free placement: the cursor ray meets a horizontal plane
                    // through the previous vertex. The only well-defined answer
                    // when the operator has said they do not want the data to
                    // decide — and it is what makes drawing across a gap in the
                    // cloud possible at all.
                    let z_view = digitise.last()
                        .map(|p| (p[2] - data_offset[2]) as f32)
                        .unwrap_or(mid_z);
                    if let Some((wx, wy)) = screen_to_world_xy(mx, my, &view_proj, z_view) {
                        digitise.push([
                            wx as f64 + data_offset[0],
                            wy as f64 + data_offset[1],
                            z_view as f64 + data_offset[2],
                        ]);
                        digitise.status = None;
                    }
                } else {
                    // Whatever the marker is showing is what gets placed —
                    // including the refusals, which is why a coarse or empty
                    // snap reports rather than silently falling back to
                    // something the operator did not ask for.
                    let hit = resolve_snap(
                        &stages, &copc_stages, &tile_mgr, &poly_files,
                        &view_proj, mx, my, data_offset, &digitise,
                    );
                    if hit.is_placeable() {
                        digitise.push(hit.world);
                        digitise.status = None;
                    } else {
                        digitise.status = hit.note.clone();
                    }
                }
            }
            if !egui_wants_keyboard_prev {
                if is_key_pressed(KeyCode::Backspace) { digitise.pop(); }
                if is_key_pressed(KeyCode::Enter) {
                    digitise_request = Some(vectorise::Request::Finish);
                }
            }
        }

        // Where the next vertex would land. Resolved once per frame rather
        // than per render block, and only while digitising — it is a pick over
        // the resident points, not something to spend on every frame of
        // ordinary navigation.
        let have_cloud = !stages.is_empty() || !copc_stages.is_empty() || tile_mgr.is_some();
        digitise.hover = if app_mode == AppMode::Digitise && !egui_wants_pointer_prev {
            Some(resolve_snap(
                &stages, &copc_stages, &tile_mgr, &poly_files,
                &view_proj, mx, my, data_offset, &digitise,
            ))
        } else {
            None
        };
        // The rubber band follows a placeable snap only, so it never draws to
        // somewhere a click would refuse to go.
        let digitise_cursor = digitise.hover.as_ref()
            .filter(|h| h.is_placeable())
            .map(|h| h.world);

        // ---- COPC neighbourhood pinning ----
        // The spacing guard alone would only ever say "no". This is the other
        // half: force-load the octree nodes around the cursor so there is
        // full-resolution data to snap to in the first place. Skipped while a
        // section corridor owns the pins — the two use the same set, and the
        // corridor is the one the operator is waiting on.
        if app_mode == AppMode::Digitise && !copc_stages.is_empty() && !corridor_pinned {
            let centre = digitise.hover.as_ref()
                .map(|h| ((h.world[0] - data_offset[0]) as f32, (h.world[1] - data_offset[1]) as f32))
                .or(cursor_world);
            if let Some((cx, cy)) = centre {
                let half = (4.0 * digitise.radius as f64).max(SNAP_NEIGHBOURHOOD_M) / 2.0;
                let mut refused = 0i64;
                for cs in copc_stages.iter_mut() {
                    let (n, pts) = cs.pin_neighbourhood(cx, cy, half);
                    if n == 0 && pts > 0 { refused = refused.max(pts); }
                }
                digitise_pinned = true;
                if refused > 0 {
                    digitise.status = Some(format!(
                        "Neighbourhood holds {:.1}M points (limit {:.0}M) — not pinned; \
                         snapping uses whatever the LOD loaded.",
                        refused as f64 / 1e6,
                        crate::raw_miniquad::SNAP_PIN_POINT_LIMIT as f64 / 1e6,
                    ));
                }
            }
        } else if digitise_pinned {
            for cs in copc_stages.iter_mut() { cs.clear_pins(); }
            digitise_pinned = false;
        }

        // Measure markers and the section preview are drawn per-render-block,
        // after the cloud has been composited. See `draw_3d_overlays`.

        // ---- Single-file (multi-Stage) mode rendering ----
        if !stages.is_empty() {
            // Render points across all stages (with optional EDL post-process)
            let drawn = {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    screen_width() as u32,
                    screen_height() as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                }
                let drawn: i32 = stages.iter()
                    .map(|s| s.draw_visible(&mut *gl.quad_context, &view_proj, point_size))
                    .sum();
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength);
                } else {
                    gl.quad_context.end_render_pass();
                }
                drawn
            };

            draw_3d_overlays(
                app_mode, measure_a, measure_b, section_start, section_end,
                cursor_world, section_depth, mid_z, height_range, ortho_scale,
                &ground_lasso, class_color_mq(classify_target, &class_defs),
                &digitise, data_offset, digitise_cursor,
            );

            set_default_camera();

            // View cube
            let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive, !egui_wants_keyboard_prev) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let mut height_color_changed = false;
            let mut z_clip_changed = false;
            let overrides_before: Vec<Option<[f32; 3]>> =
                stages.iter().map(|s| s.rgb_override).collect();
            let mut new_overrides = overrides_before.clone();
            let visible_before: Vec<bool> = stages.iter().map(|s| s.visible).collect();
            let mut new_visible = visible_before.clone();
            let mut egui_wants_pointer = false;
            let mut egui_wants_keyboard = false;
            let mut do_zoom_all = false;
            let mut class_colors_changed = false;
            let mut export_request: Option<usize> = None;
            let export_sources = export_source_list(&stages, &copc_stages);
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                class_colors_changed = classes_window(egui_ctx, &mut class_defs, &mut class_vis, max_class, &mut classes_save_status);
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso => "Back to 3D",
                        };
                        if ui.button(section_label).clicked() {
                            match app_mode {
                                AppMode::View3D | AppMode::PointPick => {
                                    app_mode = AppMode::SectionPickStart;
                                    section_start = None;
                                    section_end = None;
                                    picked_point = None;
                                }
                                _ => {
                                    app_mode = AppMode::View3D;
                                    set_section(&mut section_data, &mut section_trash, None);
                                }
                            }
                        }

                        let pick_label = if app_mode == AppMode::PointPick {
                            "Click a point... (Esc to cancel)"
                        } else {
                            "Point Info"
                        };
                        if ui.button(pick_label).clicked() {
                            if app_mode == AppMode::PointPick {
                                app_mode = AppMode::View3D;
                            } else {
                                app_mode = AppMode::PointPick;
                                picked_point = None;
                            }
                        }
                        let measure_label = match app_mode {
                            AppMode::MeasureFirst => "Click point A... (Esc to cancel)",
                            AppMode::MeasureSecond => "Click point B... (Esc to cancel)",
                            _ => "Measure",
                        };
                        if ui.button(measure_label).clicked() {
                            if matches!(app_mode, AppMode::MeasureFirst | AppMode::MeasureSecond) {
                                app_mode = AppMode::View3D;
                            } else {
                                app_mode = AppMode::MeasureFirst;
                                measure_a = None;
                                measure_b = None;
                            }
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            egui_macroquad::egui::ComboBox::from_id_salt("color_mode")
                                .selected_text(color_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in &ColorMode::ALL {
                                        ui.selectable_value(&mut color_mode, *mode, mode.label());
                                    }
                                });
                        });
                        height_color_changed = height_color_controls(ui, color_mode, &mut height_color_min, &mut height_color_max, data_offset[2]);
                        z_clip_changed = z_clip_controls(
                            ui, &mut z_clip_enabled, &mut z_clip_lo, &mut z_clip_hi,
                            data_offset[2], height_range,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Export…").clicked() { show_export_window = !show_export_window; }
                            if ui.button("Zoom All").clicked() { do_zoom_all = true; }
                            ui.label("Pt size:");
                            if ui.button("-").clicked() { point_size = (point_size - 1.0).max(1.0); }
                            ui.label(format!("{:.0}", point_size));
                            if ui.button("+").clicked() { point_size = (point_size + 1.0).min(10.0); }
                        });
                        ui.horizontal(|ui| {
                            let edl_label = if edl_enabled { "EDL: ON" } else { "EDL: OFF" };
                            if ui.button(edl_label).clicked() { edl_enabled = !edl_enabled; }
                            if edl_enabled {
                                ui.label("Strength:");
                                ui.add(egui_macroquad::egui::Slider::new(&mut edl_strength, 1.0..=30.0).step_by(1.0));
                            }
                        });
                        if let Some(kind) = ground_tool_controls(
                            ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                            is_top_view(pitch),
                            &mut classify_target, &mut classify_from, &mut ground_offset,
                            max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                            &mut apply_ground_lasso, &mut cancel_ground_lasso,
                        ) { start_ground_lasso = Some(kind); }
                        if let Some(r) = vectorise::controls(
                            ui, &mut digitise, &poly_files,
                            app_mode == AppMode::Digitise, have_cloud,
                        ) { digitise_request = Some(r); }
                        if color_mode == ColorMode::Rgb {
                            for (i, ov) in new_overrides.iter_mut().enumerate() {
                                let label = if stages.len() > 1 {
                                    format!("Cloud {} override:", i + 1)
                                } else {
                                    "Override color:".to_string()
                                };
                                ui.horizontal(|ui| {
                                    ui.label(&label);
                                    let mut enabled = ov.is_some();
                                    if ui.checkbox(&mut enabled, "").changed() {
                                        *ov = if enabled { Some([1.0, 1.0, 1.0]) } else { None };
                                    }
                                    if let Some(c) = ov {
                                        let mut c32 = rgb_arr_to_egui(*c);
                                        if ui.color_edit_button_srgba(&mut c32).changed() {
                                            *c = egui_to_rgb_arr(c32);
                                        }
                                    }
                                });
                            }
                        }
                        let total_points: i32 = stages.iter().map(|s| s.total_points).sum();
                        let total_chunks: usize = stages.iter().map(|s| s.chunks.len()).sum();
                        ui.label(format!(
                            "{}/{} points | {} chunks | FPS: {:.0} | zoom: {:.1}",
                            drawn, total_points, total_chunks,
                            1.0 / get_frame_time(), ortho_scale
                        ));
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Loaded files:");
                            if ui.button("Open file…").clicked()
                                && let Some(paths) = rfd::FileDialog::new()
                                    .add_filter("LAS / LAZ", &["laz", "las"])
                                    .pick_files()
                            {
                                for p in paths {
                                    pending_loads.push(p.to_string_lossy().to_string());
                                }
                            }
                        });
                        for (i, s) in stages.iter().enumerate() {
                            let name = std::path::Path::new(&s.filepath)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| s.filepath.clone());
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut new_visible[i], "")
                                    .on_hover_text("Show / hide this cloud");
                                ui.label(name);
                                if ui.button("Unload").clicked() {
                                    stage_unload_request = Some(i);
                                }
                            });
                        }
                        if app_mode == AppMode::SectionAdjustDepth {
                            ui.label(format!("Section depth: {:.1}m", section_depth));
                        }
                        if let (Some(a), Some(b)) = (measure_a, measure_b) {
                            // All stages share `data_offset` as the viewer origin,
                            // so measure positions translate back to world coords with it.
                            let off = glam::Vec3::new(
                                data_offset[0] as f32, data_offset[1] as f32, data_offset[2] as f32,
                            );
                            let wa = a + off;
                            let wb = b + off;
                            let dx = wb.x - wa.x;
                            let dy = wb.y - wa.y;
                            let dz = wb.z - wa.z;
                            let dist3d = (dx*dx + dy*dy + dz*dz).sqrt();
                            let dist2d = (dx*dx + dy*dy).sqrt();
                            ui.separator();
                            ui.label("Measurement:");
                            ui.label(format!("  A: ({:.3}, {:.3}, {:.3})", wa.x, wa.y, wa.z));
                            ui.label(format!("  B: ({:.3}, {:.3}, {:.3})", wb.x, wb.y, wb.z));
                            ui.label(format!("  dX: {:.3}  dY: {:.3}  dZ: {:.3}", dx, dy, dz));
                            ui.label(format!("  3D dist: {:.3}m", dist3d));
                            ui.label(format!("  Horiz:   {:.3}m  ΔZ: {:.3}m", dist2d, dz));
                        } else if measure_a.is_some() {
                            ui.separator();
                            ui.label("A set — click point B");
                        }
                        if let Some((si, idx)) = picked_point
                            && let Some(s) = stages.get(si)
                            && let Some(p) = s.cloud.points.get(idx)
                        {
                            ui.separator();
                            point_info_ui(ui, "Point Info:", p, &class_defs);
                        }
                    });
                export_request = export_window(
                    egui_ctx, &mut show_export_window, &mut export_filter,
                    &mut export_job, &mut export_status, &export_sources,
                    max_class, &class_defs,
                );
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, data_offset, background);
                gcp_window(egui_ctx, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
            });

            // A narrowed band should drive the height ramp too: colouring
            // against the full dataset range wastes most of the gradient on
            // levels that are no longer drawn. Route it through the existing
            // height-colour path rather than adding a parallel one.
            if z_clip_changed {
                let (new_min, new_max) = if z_clip_enabled {
                    (z_clip_lo, z_clip_hi)
                } else {
                    (0.0, height_range)
                };
                // A band that maps onto the ramp already in force costs nothing
                // to "apply", and re-uploading every colour to arrive at the
                // same bytes is the most expensive thing this view can do.
                if (new_min, new_max) != (height_color_min, height_color_max) {
                    height_color_min = new_min;
                    height_color_max = new_max;
                    height_color_changed = true;
                }
            }

            // ---- GCP overlay ----
            // Screen-space markers under the default camera, so they keep one
            // size at any zoom and are never swallowed by the cloud. Drawn
            // before egui so the panels still sit on top.
            if !gcp_files.is_empty() {
                gcp::draw_3d(&gcp_files, data_offset, &view_proj, selected_gcp);
            }
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;
            egui_wants_keyboard_prev = egui_wants_keyboard;

            if let Some(idx) = export_request {
                start_export(
                    idx, &stages, &copc_stages, &export_filter,
                    &mut export_job, &mut export_status,
                );
            }

            if do_zoom_all
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr)
            {
                let aspect = screen_width() / screen_height().max(1.0);
                let center = (lo + hi) * 0.5;
                let ext = hi - lo;
                ortho_scale = (ext.y * 0.5).max(ext.x * 0.5 / aspect) * 1.05;
                position = center - front * (ext.x.max(ext.y) * 2.0);
                orbit_pivot = center;
            }

            // Helper closure: find nearest point across all stages, returning
            // (stage_idx, point_idx) plus the viewer-space position.
            let pick_across = |view_proj: &glam::Mat4, mx: f32, my: f32| -> Option<(usize, usize)> {
                let mut best: Option<(usize, usize, f32)> = None;
                for (si, s) in stages.iter().enumerate() {
                    if !s.visible { continue; }
                    if let Some(idx) = s.find_nearest_point(view_proj, mx, my, screen_width(), screen_height()) {
                        let pos = s.positions[idx];
                        let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        let ndc_x = mx / screen_width() * 2.0 - 1.0;
                        let ndc_y = 1.0 - my / screen_height() * 2.0;
                        let d = (p.x / p.w - ndc_x).powi(2) + (p.y / p.w - ndc_y).powi(2);
                        if best.map_or(true, |(_, _, bd)| d < bd) {
                            best = Some((si, idx, d));
                        }
                    }
                }
                best.map(|(si, idx, _)| (si, idx))
            };

            // Handle clicks
            if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
                match app_mode {
                    AppMode::PointPick => {
                        picked_point = pick_across(&view_proj, mx, my);
                    }
                    AppMode::SectionPickStart => {
                        if let Some(world) = cursor_world {
                            section_start = Some(world);
                            app_mode = AppMode::SectionPickEnd;
                        }
                    }
                    AppMode::SectionPickEnd => {
                        if let Some(world) = cursor_world {
                            section_end = Some(world);
                            app_mode = AppMode::SectionAdjustDepth;
                        }
                    }
                    AppMode::SectionAdjustDepth => {
                        // Combine verts from all loaded stages into one SectionData.
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            let gl = unsafe { get_internal_gl() };
                            let sec = build_section_from_stages(
                                &mut *gl.quad_context, &stages, start, end, section_depth, color_mode,
                            );
                            section_pan = glam::Vec2::new(sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                            section_zoom = (sec.length / 2.0).max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                            sec_picked_point = None;
                            if !section_color_mode_pinned { section_color_mode = color_mode; }
                            app_mode = AppMode::SectionView;
                        }
                    }
                    AppMode::MeasureFirst => {
                        if let Some((si, idx)) = pick_across(&view_proj, mx, my) {
                            measure_a = Some(glam::Vec3::from(stages[si].positions[idx]));
                            measure_b = None;
                            app_mode = AppMode::MeasureSecond;
                        }
                    }
                    AppMode::MeasureSecond => {
                        if let Some((si, idx)) = pick_across(&view_proj, mx, my) {
                            measure_b = Some(glam::Vec3::from(stages[si].positions[idx]));
                        }
                    }
                    AppMode::View3D => {
                        selected_poly = polylines::pick(&poly_files, &view_proj, mx, my, screen_width(), screen_height(), 12.0);
                    }
                    _ => {}
                }
            }

            if class_colors_changed {
                apply_class_defs(&mut stages, &mut copc_stages, &mut tile_mgr, &class_defs);
            }
            let class_rebuild = (class_colors_changed && color_mode == ColorMode::Classification)
                || class_vis_changed;
            // Storing the ramp bounds is cheap; re-uploading every colour is
            // not. Only Height mode reads the ramp, so in any other mode the
            // new bounds are recorded and the upload waits until the view is
            // switched back to Height — which is what the height filter needs,
            // since it drives these bounds on every slider frame.
            if height_color_changed {
                apply_height_color(&mut stages, &mut copc_stages, &mut tile_mgr, height_color_min, height_color_max);
            }
            let height_rebuild = height_color_changed && color_mode == ColorMode::Height;
            if color_mode != prev_color_mode || height_rebuild || class_rebuild {
                let gl = unsafe { get_internal_gl() };
                if height_rebuild || class_rebuild {
                    for cs in copc_stages.iter_mut() {
                        cs.rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                    if let Some(mgr) = tile_mgr.as_mut() {
                        mgr.rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                }
                for s in stages.iter_mut() {
                    s.rebuild_colors(&mut *gl.quad_context, color_mode);
                }
            }
            for (i, ov) in new_overrides.iter().enumerate() {
                if *ov != overrides_before[i] {
                    stages[i].rgb_override = *ov;
                    let gl = unsafe { get_internal_gl() };
                    stages[i].rebuild_colors(&mut *gl.quad_context, color_mode);
                }
            }
            for (i, vis) in new_visible.iter().enumerate() {
                if *vis != visible_before[i] {
                    stages[i].visible = *vis;
                }
            }
        }

        // ---- COPC mode rendering ----
        if !copc_stages.is_empty() {
            // Update LOD: load/unload nodes for current view (non-blocking).
            {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                let mut loaded_so_far: i32 = 0;
                for cs in copc_stages.iter_mut() {
                    cs.update_view(&mut *gl.quad_context, &view_proj, ortho_scale, screen_height(), color_mode, loaded_so_far);
                    loaded_so_far += cs.total_loaded_points();
                }
            }

            let copc_overrides_before: Vec<Option<[f32; 3]>> = copc_stages.iter().map(|cs| cs.rgb_override).collect();
            let copc_visible_before: Vec<bool> = copc_stages.iter().map(|cs| cs.visible).collect();
            let mut new_copc_visible = copc_visible_before.clone();
            let mut new_copc_overrides = copc_overrides_before.clone();

            // Render visible chunks (with optional EDL).
            let drawn = {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    screen_width() as u32,
                    screen_height() as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                }
                let drawn: i32 = copc_stages.iter().map(|cs| cs.draw_visible(&mut *gl.quad_context, &view_proj, point_size)).sum();
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength);
                } else {
                    gl.quad_context.end_render_pass();
                }
                drawn
            };

            draw_3d_overlays(
                app_mode, measure_a, measure_b, section_start, section_end,
                cursor_world, section_depth, mid_z, height_range, ortho_scale,
                &ground_lasso, class_color_mq(classify_target, &class_defs),
                &digitise, data_offset, digitise_cursor,
            );

            set_default_camera();

            // View cube
            let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive, !egui_wants_keyboard_prev) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let mut height_color_changed = false;
            let mut z_clip_changed = false;
            let mut egui_wants_pointer = false;
            let mut egui_wants_keyboard = false;
            let mut do_zoom_all = false;
            let mut class_colors_changed = false;
            let mut leave_corridor = false;
            let mut export_request: Option<usize> = None;
            let export_sources = export_source_list(&stages, &copc_stages);
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                class_colors_changed = classes_window(egui_ctx, &mut class_defs, &mut class_vis, max_class, &mut classes_save_status);
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso => "Back to 3D",
                        };
                        if ui.button(section_label).clicked() {
                            match app_mode {
                                AppMode::View3D | AppMode::PointPick => {
                                    app_mode = AppMode::SectionPickStart;
                                    section_start = None;
                                    section_end = None;
                                    picked_point = None;
                                }
                                _ => {
                                    app_mode = AppMode::View3D;
                                    set_section(&mut section_data, &mut section_trash, None);
                                    leave_corridor = true;
                                    corridor_status = None;
                                }
                            }
                        }
                        let pick_label = if app_mode == AppMode::PointPick {
                            "Click a point... (Esc to cancel)"
                        } else { "Point Info" };
                        if ui.button(pick_label).clicked() {
                            if app_mode == AppMode::PointPick { app_mode = AppMode::View3D; }
                            else { app_mode = AppMode::PointPick; picked_copc_point = None; }
                        }
                        let measure_label = match app_mode {
                            AppMode::MeasureFirst => "Click point A... (Esc to cancel)",
                            AppMode::MeasureSecond => "Click point B... (Esc to cancel)",
                            _ => "Measure",
                        };
                        if ui.button(measure_label).clicked() {
                            if matches!(app_mode, AppMode::MeasureFirst | AppMode::MeasureSecond) {
                                app_mode = AppMode::View3D;
                            } else {
                                app_mode = AppMode::MeasureFirst;
                                measure_a = None; measure_b = None;
                            }
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            egui_macroquad::egui::ComboBox::from_id_salt("color_mode")
                                .selected_text(color_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in &ColorMode::ALL {
                                        ui.selectable_value(&mut color_mode, *mode, mode.label());
                                    }
                                });
                        });
                        height_color_changed = height_color_controls(ui, color_mode, &mut height_color_min, &mut height_color_max, data_offset[2]);
                        z_clip_changed = z_clip_controls(
                            ui, &mut z_clip_enabled, &mut z_clip_lo, &mut z_clip_hi,
                            data_offset[2], height_range,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Export…").clicked() { show_export_window = !show_export_window; }
                            if ui.button("Zoom All").clicked() { do_zoom_all = true; }
                            ui.label("Pt size:");
                            if ui.button("-").clicked() { point_size = (point_size - 1.0).max(1.0); }
                            ui.label(format!("{:.0}", point_size));
                            if ui.button("+").clicked() { point_size = (point_size + 1.0).min(10.0); }
                        });
                        ui.horizontal(|ui| {
                            let edl_label = if edl_enabled { "EDL: ON" } else { "EDL: OFF" };
                            if ui.button(edl_label).clicked() { edl_enabled = !edl_enabled; }
                            if edl_enabled {
                                ui.label("Strength:");
                                ui.add(egui_macroquad::egui::Slider::new(&mut edl_strength, 1.0..=30.0).step_by(1.0));
                            }
                        });
                        if let Some(kind) = ground_tool_controls(
                            ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                            is_top_view(pitch),
                            &mut classify_target, &mut classify_from, &mut ground_offset,
                            max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                            &mut apply_ground_lasso, &mut cancel_ground_lasso,
                        ) { start_ground_lasso = Some(kind); }
                        if let Some(r) = vectorise::controls(
                            ui, &mut digitise, &poly_files,
                            app_mode == AppMode::Digitise, have_cloud,
                        ) { digitise_request = Some(r); }
                        if color_mode == ColorMode::Rgb {
                            for (i, ov) in new_copc_overrides.iter_mut().enumerate() {
                                let label = if copc_stages.len() > 1 {
                                    format!("Cloud {} override:", i + 1)
                                } else {
                                    "Override color:".to_string()
                                };
                                ui.horizontal(|ui| {
                                    ui.label(&label);
                                    let mut enabled = ov.is_some();
                                    if ui.checkbox(&mut enabled, "").changed() {
                                        *ov = if enabled { Some([1.0, 1.0, 1.0]) } else { None };
                                    }
                                    if let Some(c) = ov {
                                        let mut c32 = rgb_arr_to_egui(*c);
                                        if ui.color_edit_button_srgba(&mut c32).changed() {
                                            *c = egui_to_rgb_arr(c32);
                                        }
                                    }
                                });
                            }
                        }
                        let mpp = ortho_scale * 2.0 / screen_height().max(1.0);
                        let spacing0 = copc_stages[0].spacing;
                        let target_depth = if mpp > 0.0 && spacing0 > 0.0 {
                            ((spacing0 as f32 / mpp).log2().ceil() as i32).max(0)
                        } else { 0 };
                        if app_mode == AppMode::SectionAdjustDepth {
                            ui.label(format!("Section depth: {:.1}m", section_depth));
                        }
                        let total_pts: i32 = copc_stages.iter().map(|cs| cs.total_loaded_points()).sum();
                        let total_nodes: usize = copc_stages.iter().map(|cs| cs.loaded_node_count()).sum();
                        let total_inflight: usize = copc_stages.iter().map(|cs| cs.pending_count()).sum();
                        ui.label(format!(
                            "COPC | {}/{} pts | {} nodes | LOD {} | {} in-flight | FPS: {:.0}",
                            drawn, total_pts, total_nodes, target_depth, total_inflight,
                            1.0 / get_frame_time(),
                        ));
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Loaded COPC files:");
                            if ui.button("Open file…").clicked()
                                && let Some(paths) = rfd::FileDialog::new()
                                    .add_filter("LAS / LAZ", &["laz", "las"])
                                    .pick_files()
                            {
                                for p in paths {
                                    pending_loads.push(p.to_string_lossy().to_string());
                                }
                            }
                        });
                        for (i, cs) in copc_stages.iter().enumerate() {
                            let name = std::path::Path::new(&cs.filepath)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| cs.filepath.clone());
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut new_copc_visible[i], "")
                                    .on_hover_text("Show / hide this cloud");
                                ui.label(name);
                                if ui.button("Unload").clicked() {
                                    copc_unload_request = Some(i);
                                }
                            });
                        }
                        if let (Some(a), Some(b)) = (measure_a, measure_b) {
                            let vo = copc_stages[0].viewer_origin;
                            let off = glam::Vec3::new(vo[0] as f32, vo[1] as f32, vo[2] as f32);
                            let wa = a + off; let wb = b + off;
                            let dx = wb.x - wa.x; let dy = wb.y - wa.y; let dz = wb.z - wa.z;
                            let dist3d = (dx*dx + dy*dy + dz*dz).sqrt();
                            let dist2d = (dx*dx + dy*dy).sqrt();
                            ui.separator();
                            ui.label("Measurement:");
                            ui.label(format!("  A: ({:.3}, {:.3}, {:.3})", wa.x, wa.y, wa.z));
                            ui.label(format!("  B: ({:.3}, {:.3}, {:.3})", wb.x, wb.y, wb.z));
                            ui.label(format!("  dX: {:.3}  dY: {:.3}  dZ: {:.3}", dx, dy, dz));
                            ui.label(format!("  3D dist: {:.3}m", dist3d));
                            ui.label(format!("  Horiz:   {:.3}m  ΔZ: {:.3}m", dist2d, dz));
                        } else if measure_a.is_some() {
                            ui.separator();
                            ui.label("A set — click point B");
                        }
                        if let Some(ref p) = picked_copc_point {
                            ui.separator();
                            point_info_ui(ui, "Point Info:", p, &class_defs);
                        }
                    });
                export_request = export_window(
                    egui_ctx, &mut show_export_window, &mut export_filter,
                    &mut export_job, &mut export_status, &export_sources,
                    max_class, &class_defs,
                );
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, data_offset, background);
                gcp_window(egui_ctx, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
            });

            // A narrowed band should drive the height ramp too: colouring
            // against the full dataset range wastes most of the gradient on
            // levels that are no longer drawn. Route it through the existing
            // height-colour path rather than adding a parallel one.
            if z_clip_changed {
                let (new_min, new_max) = if z_clip_enabled {
                    (z_clip_lo, z_clip_hi)
                } else {
                    (0.0, height_range)
                };
                // A band that maps onto the ramp already in force costs nothing
                // to "apply", and re-uploading every colour to arrive at the
                // same bytes is the most expensive thing this view can do.
                if (new_min, new_max) != (height_color_min, height_color_max) {
                    height_color_min = new_min;
                    height_color_max = new_max;
                    height_color_changed = true;
                }
            }

            // ---- GCP overlay ----
            // Screen-space markers under the default camera, so they keep one
            // size at any zoom and are never swallowed by the cloud. Drawn
            // before egui so the panels still sit on top.
            if !gcp_files.is_empty() {
                gcp::draw_3d(&gcp_files, data_offset, &view_proj, selected_gcp);
            }
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;
            egui_wants_keyboard_prev = egui_wants_keyboard;

            if leave_corridor {
                for cs in copc_stages.iter_mut() { cs.clear_pins(); }
                corridor_pinned = false;
            }

            if let Some(idx) = export_request {
                start_export(
                    idx, &stages, &copc_stages, &export_filter,
                    &mut export_job, &mut export_status,
                );
            }

            if do_zoom_all
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr)
            {
                let aspect = screen_width() / screen_height().max(1.0);
                let center = (lo + hi) * 0.5;
                let ext = hi - lo;
                ortho_scale = (ext.y * 0.5).max(ext.x * 0.5 / aspect) * 1.05;
                position = center - front * (ext.x.max(ext.y) * 2.0);
                orbit_pivot = center;
            }

            // Mouse clicks
            if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
                match app_mode {
                    AppMode::PointPick => {
                        let mut best_dist = f32::MAX;
                        for cs in copc_stages.iter().filter(|cs| cs.visible) {
                            if let Some((p, _vpos, dist)) = cs.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) 
                                && dist < best_dist { best_dist = dist; picked_copc_point = Some(p.clone()); 
                            }
                        }
                    }
                    AppMode::SectionPickStart => {
                        if let Some(world) = cursor_world {
                            section_start = Some(world);
                            app_mode = AppMode::SectionPickEnd;
                        }
                    }
                    AppMode::SectionPickEnd => {
                        if let Some(world) = cursor_world {
                            section_end = Some(world);
                            app_mode = AppMode::SectionAdjustDepth;
                        }
                    }
                    AppMode::SectionAdjustDepth => {
                        // Cut from the resident LOD, as before. Full-resolution
                        // force-loading is deferred until the classification
                        // tool is actually started: browsing sections is the
                        // common case and has to stay instant.
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            let gl = unsafe { get_internal_gl() };
                            let mut all_verts = Vec::new();
                            let mut per_stage_raw: Vec<Vec<(LasPoint, f32, f32)>> = Vec::new();
                            let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
                            let mut length = 0.0_f32;
                            let height_range = copc_stages[0].height_range();
                            for cs in &copc_stages {
                                // Hidden clouds still push an (empty) entry so
                                // `per_stage_raw` stays indexed by stage.
                                if !cs.visible { per_stage_raw.push(Vec::new()); continue; }
                                let (verts, raw, len, hmin, hmax) =
                                    cs.extract_section_verts(start, end, section_depth, color_mode);
                                all_verts.extend(verts);
                                per_stage_raw.push(raw);
                                h_min = h_min.min(hmin); h_max = h_max.max(hmax);
                                length = length.max(len);
                            }
                            let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
                            let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
                            info!("COPC Section: {:.1}m long, {} points", length, all_verts.len());
                            copc_section_raw = Some((per_stage_raw, length, h_min, h_max));
                            let sec = make_section_data(
                                &mut *gl.quad_context, all_verts, length, h_min, h_max, height_range);
                            section_pan = glam::Vec2::new(
                                sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                            section_zoom = (sec.length / 2.0)
                                .max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                            sec_picked_point = None;
                            corridor_status = None;
                            if !section_color_mode_pinned { section_color_mode = color_mode; }
                            app_mode = AppMode::SectionView;
                        }
                    }
                    AppMode::MeasureFirst => {
                        let mut best_dist = f32::MAX;
                        for cs in copc_stages.iter().filter(|cs| cs.visible) {
                            if let Some((_p, vpos, dist)) = cs.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) 
                                && dist < best_dist { best_dist = dist; measure_a = Some(vpos); measure_b = None; app_mode = AppMode::MeasureSecond; 
                            }
                        }
                    }
                    AppMode::MeasureSecond => {
                        let mut best_dist = f32::MAX;
                        for cs in copc_stages.iter().filter(|cs| cs.visible) {
                            if let Some((_p, vpos, dist)) = cs.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) 
                                && dist < best_dist { best_dist = dist; measure_b = Some(vpos); 
                            }
                        }
                    }
                    AppMode::View3D => {
                        selected_poly = polylines::pick(&poly_files, &view_proj, mx, my, screen_width(), screen_height(), 12.0);
                    }
                    _ => {}
                }
            }

            if class_colors_changed {
                apply_class_defs(&mut stages, &mut copc_stages, &mut tile_mgr, &class_defs);
            }
            let class_rebuild = (class_colors_changed && color_mode == ColorMode::Classification)
                || class_vis_changed;
            // Storing the ramp bounds is cheap; re-uploading every colour is
            // not. Only Height mode reads the ramp, so in any other mode the
            // new bounds are recorded and the upload waits until the view is
            // switched back to Height — which is what the height filter needs,
            // since it drives these bounds on every slider frame.
            if height_color_changed {
                apply_height_color(&mut stages, &mut copc_stages, &mut tile_mgr, height_color_min, height_color_max);
            }
            let height_rebuild = height_color_changed && color_mode == ColorMode::Height;
            if color_mode != prev_color_mode || height_rebuild || class_rebuild {
                let gl = unsafe { get_internal_gl() };
                if height_rebuild || class_rebuild {
                    for s in stages.iter_mut() {
                        s.rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                    if let Some(mgr) = tile_mgr.as_mut() {
                        mgr.rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                }
                for cs in copc_stages.iter_mut() {
                    cs.rebuild_colors(&mut *gl.quad_context, color_mode);
                }
            }
            if new_copc_overrides != copc_overrides_before {
                let gl = unsafe { get_internal_gl() };
                for (i, new_ov) in new_copc_overrides.into_iter().enumerate() {
                    if new_ov != copc_overrides_before[i] {
                        copc_stages[i].rgb_override = new_ov;
                        copc_stages[i].rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                }
            }
            for (i, vis) in new_copc_visible.iter().enumerate() {
                if *vis != copc_visible_before[i] {
                    copc_stages[i].visible = *vis;
                }
            }
        }

        // ---- Tile mode rendering ----
        if copc_stages.is_empty() && let Some(ref mgr) = tile_mgr {
            set_camera(&camera);

            // Render loaded tile points (with optional EDL post-process)
            let (drawn, total) = {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    screen_width() as u32,
                    screen_height() as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                }
                let result = mgr.draw_visible(&mut *gl.quad_context, &view_proj, point_size);
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength);
                } else {
                    gl.quad_context.end_render_pass();
                }
                result
            };

            // Tile footprints, after the composite for the same reason as the
            // overlays: a footprint over loaded points would otherwise be
            // erased by the EDL blit.
            for i in 0..mgr.tiles.len() {
                let corners = mgr.tile_corners(i);
                let is_loaded = mgr.loaded[i].is_some();
                let is_selected = selected_tile == Some(i);
                let col = if is_selected { YELLOW } else if is_loaded { GREEN } else { RED };
                let thickness = if is_selected { 2.0 } else { 1.0 };
                for j in 0..4 {
                    let k = (j + 1) % 4;
                    // draw_line_3d doesn't support thickness, use it anyway
                    draw_line_3d(
                        vec3(corners[j].0, corners[j].1, 0.),
                        vec3(corners[k].0, corners[k].1, 0.),
                        col,
                    );
                    if thickness > 1.0 {
                        // Draw a slightly offset line for thicker appearance
                        draw_line_3d(
                            vec3(corners[j].0, corners[j].1, 0.1),
                            vec3(corners[k].0, corners[k].1, 0.1),
                            col,
                        );
                    }
                }
            }

            draw_3d_overlays(
                app_mode, measure_a, measure_b, section_start, section_end,
                cursor_world, section_depth, mid_z, height_range, ortho_scale,
                &ground_lasso, class_color_mq(classify_target, &class_defs),
                &digitise, data_offset, digitise_cursor,
            );

            set_default_camera();

            // Draw tile labels (2D overlay)
            for i in 0..mgr.tiles.len() {
                let center = mgr.tile_center(i);
                let (sx, sy) = world_to_screen(vec3(center.0, center.1, 0.0), &view_proj);
                if sx > 0.0 && sx < screen_width() && sy > 0.0 && sy < screen_height() {
                    let is_loaded = mgr.loaded[i].is_some();
                    let label_color = if is_loaded { GREEN } else { Color::new(1.0, 0.4, 0.4, 0.8) };
                    let name = &mgr.tiles[i].filename;
                    let font_size = 12.0;
                    let text_w = name.len() as f32 * font_size * 0.35;
                    draw_text(name, sx - text_w / 2.0, sy, font_size, label_color);
                }
            }

            // View cube
            let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive, !egui_wants_keyboard_prev) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let mut height_color_changed = false;
            let mut z_clip_changed = false;
            let mut egui_wants_pointer = false;
            let mut egui_wants_keyboard = false;
            let mut request_load: Option<usize> = None;
            let mut request_unload: Option<usize> = None;
            let mut do_zoom_all = false;
            let mut do_zoom_selected = false;
            let mut tile_override_changed: Option<usize> = None;
            let mut class_colors_changed = false;
            let mut export_request: Option<usize> = None;
            let export_sources = export_source_list(&stages, &copc_stages);
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                class_colors_changed = classes_window(egui_ctx, &mut class_defs, &mut class_vis, max_class, &mut classes_save_status);
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        // Cross Section button
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso => "Back to 3D",
                        };
                        if ui.button(section_label).clicked() {
                            match app_mode {
                                AppMode::View3D | AppMode::PointPick => {
                                    app_mode = AppMode::SectionPickStart;
                                    section_start = None;
                                    section_end = None;
                                    picked_tile_point = None;
                                }
                                _ => {
                                    app_mode = AppMode::View3D;
                                    set_section(&mut section_data, &mut section_trash, None);
                                }
                            }
                        }

                        // Point pick button
                        let pick_label = if app_mode == AppMode::PointPick {
                            "Click a point... (Esc to cancel)"
                        } else {
                            "Point Info"
                        };
                        if ui.button(pick_label).clicked() {
                            if app_mode == AppMode::PointPick {
                                app_mode = AppMode::View3D;
                            } else {
                                app_mode = AppMode::PointPick;
                                picked_tile_point = None;
                            }
                        }
                        let measure_label = match app_mode {
                            AppMode::MeasureFirst => "Click point A... (Esc to cancel)",
                            AppMode::MeasureSecond => "Click point B... (Esc to cancel)",
                            _ => "Measure",
                        };
                        if ui.button(measure_label).clicked() {
                            if matches!(app_mode, AppMode::MeasureFirst | AppMode::MeasureSecond) {
                                app_mode = AppMode::View3D;
                            } else {
                                app_mode = AppMode::MeasureFirst;
                                measure_a = None;
                                measure_b = None;
                            }
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            egui_macroquad::egui::ComboBox::from_id_salt("color_mode")
                                .selected_text(color_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in &ColorMode::ALL {
                                        ui.selectable_value(&mut color_mode, *mode, mode.label());
                                    }
                                });
                        });
                        height_color_changed = height_color_controls(ui, color_mode, &mut height_color_min, &mut height_color_max, data_offset[2]);
                        z_clip_changed = z_clip_controls(
                            ui, &mut z_clip_enabled, &mut z_clip_lo, &mut z_clip_hi,
                            data_offset[2], height_range,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Zoom All").clicked() { do_zoom_all = true; }
                            if selected_tile.is_some() && ui.button("Zoom on Object").clicked() {
                                do_zoom_selected = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Pt size:");
                            if ui.button("-").clicked() { point_size = (point_size - 1.0).max(1.0); }
                            ui.label(format!("{:.0}", point_size));
                            if ui.button("+").clicked() { point_size = (point_size + 1.0).min(10.0); }
                        });
                        ui.horizontal(|ui| {
                            let edl_label = if edl_enabled { "EDL: ON" } else { "EDL: OFF" };
                            if ui.button(edl_label).clicked() { edl_enabled = !edl_enabled; }
                            if edl_enabled {
                                ui.label("Strength:");
                                ui.add(egui_macroquad::egui::Slider::new(&mut edl_strength, 1.0..=30.0).step_by(1.0));
                            }
                        });
                        if let Some(kind) = ground_tool_controls(
                            ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                            is_top_view(pitch),
                            &mut classify_target, &mut classify_from, &mut ground_offset,
                            max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                            &mut apply_ground_lasso, &mut cancel_ground_lasso,
                        ) { start_ground_lasso = Some(kind); }
                        if let Some(r) = vectorise::controls(
                            ui, &mut digitise, &poly_files,
                            app_mode == AppMode::Digitise, have_cloud,
                        ) { digitise_request = Some(r); }
                        ui.label(format!(
                            "{}/{} points | {} chunks | {}/{} tiles | FPS: {:.0}",
                            drawn, total, mgr.total_chunks(),
                            mgr.loaded_count(), mgr.tiles.len(),
                            1.0 / get_frame_time()
                        ));

                        if let (Some(a), Some(b)) = (measure_a, measure_b) {
                            let off = glam::Vec3::new(
                                mgr.global_min[0] as f32, mgr.global_min[1] as f32, mgr.global_min[2] as f32,
                            );
                            let wa = a + off;
                            let wb = b + off;
                            let dx = wb.x - wa.x;
                            let dy = wb.y - wa.y;
                            let dz = wb.z - wa.z;
                            let dist3d = (dx*dx + dy*dy + dz*dz).sqrt();
                            let dist2d = (dx*dx + dy*dy).sqrt();
                            ui.separator();
                            ui.label("Measurement:");
                            ui.label(format!("  A: ({:.3}, {:.3}, {:.3})", wa.x, wa.y, wa.z));
                            ui.label(format!("  B: ({:.3}, {:.3}, {:.3})", wb.x, wb.y, wb.z));
                            ui.label(format!("  dX: {:.3}  dY: {:.3}  dZ: {:.3}", dx, dy, dz));
                            ui.label(format!("  3D dist: {:.3}m", dist3d));
                            ui.label(format!("  Horiz:   {:.3}m  ΔZ: {:.3}m", dist2d, dz));
                        } else if measure_a.is_some() {
                            ui.separator();
                            ui.label("A set — click point B");
                        }

                        // Selected tile info
                        if let Some(sel) = selected_tile {
                            ui.separator();
                            ui.label(format!("Selected: {}", mgr.tiles[sel].filename));
                            let is_loaded = mgr.loaded[sel].is_some();
                            if !is_loaded {
                                if ui.button("Load").clicked() {
                                    request_load = Some(sel);
                                }
                            } 
                            else if ui.button("Unload").clicked() {
                                    request_unload = Some(sel);
                            }
                            if let Some(ref lt) = mgr.loaded[sel] {
                                ui.label(format!("  {} points loaded", lt.total_points));
                                if color_mode == ColorMode::Rgb {
                                    ui.horizontal(|ui| {
                                        ui.label("Override color:");
                                        let mut enabled = tile_rgb_overrides[sel].is_some();
                                        if ui.checkbox(&mut enabled, "").changed() {
                                            tile_rgb_overrides[sel] = if enabled { Some([1.0, 1.0, 1.0]) } else { None };
                                            tile_override_changed = Some(sel);
                                        }
                                        if let Some(ref mut c) = tile_rgb_overrides[sel] {
                                            let mut c32 = rgb_arr_to_egui(*c);
                                            if ui.color_edit_button_srgba(&mut c32).changed() {
                                                *c = egui_to_rgb_arr(c32);
                                                tile_override_changed = Some(sel);
                                            }
                                        }
                                    });
                                }
                            }
                        }

                        // Picked point info
                        if let Some((tile_idx, point_idx)) = picked_tile_point &&
                            let Some(ref lt) = mgr.loaded[tile_idx] {
                                ui.separator();
                                let p = &lt.cloud.points[point_idx];
                                point_info_ui(ui, &format!("Point Info ({})", mgr.tiles[tile_idx].filename), p, &class_defs);
                            
                        }
                    });
                export_request = export_window(
                    egui_ctx, &mut show_export_window, &mut export_filter,
                    &mut export_job, &mut export_status, &export_sources,
                    max_class, &class_defs,
                );
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, data_offset, background);
                gcp_window(egui_ctx, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
            });

            // A narrowed band should drive the height ramp too: colouring
            // against the full dataset range wastes most of the gradient on
            // levels that are no longer drawn. Route it through the existing
            // height-colour path rather than adding a parallel one.
            if z_clip_changed {
                let (new_min, new_max) = if z_clip_enabled {
                    (z_clip_lo, z_clip_hi)
                } else {
                    (0.0, height_range)
                };
                // A band that maps onto the ramp already in force costs nothing
                // to "apply", and re-uploading every colour to arrive at the
                // same bytes is the most expensive thing this view can do.
                if (new_min, new_max) != (height_color_min, height_color_max) {
                    height_color_min = new_min;
                    height_color_max = new_max;
                    height_color_changed = true;
                }
            }

            // ---- GCP overlay ----
            // Screen-space markers under the default camera, so they keep one
            // size at any zoom and are never swallowed by the cloud. Drawn
            // before egui so the panels still sit on top.
            if !gcp_files.is_empty() {
                gcp::draw_3d(&gcp_files, data_offset, &view_proj, selected_gcp);
            }
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;
            egui_wants_keyboard_prev = egui_wants_keyboard;

            if let Some(idx) = export_request {
                start_export(
                    idx, &stages, &copc_stages, &export_filter,
                    &mut export_job, &mut export_status,
                );
            }

            if do_zoom_all
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr)
            {
                let aspect = screen_width() / screen_height().max(1.0);
                let center = (lo + hi) * 0.5;
                let ext = hi - lo;
                ortho_scale = (ext.y * 0.5).max(ext.x * 0.5 / aspect) * 1.05;
                position = center - front * (ext.x.max(ext.y) * 2.0);
                orbit_pivot = center;
            }
            if do_zoom_selected &&
                let Some(sel) = selected_tile {
                    let t = &mgr.tiles[sel];
                    let ext_x = (t.xmax - t.xmin) as f32;
                    let ext_y = (t.ymax - t.ymin) as f32;
                    let cx = ((t.xmin + t.xmax) / 2.0 - mgr.global_min[0]) as f32;
                    let cy = ((t.ymin + t.ymax) / 2.0 - mgr.global_min[1]) as f32;
                    let aspect = screen_width() / screen_height().max(1.0);
                    ortho_scale = (ext_y * 0.5).max(ext_x * 0.5 / aspect) * 1.2;
                    position = vec3(cx, cy, height_range * 0.5)
                        - front * (ext_x.max(ext_y) * 2.0);
                
            }

            // Handle clicks
            if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
                match app_mode {
                    AppMode::PointPick => {
                        picked_tile_point = mgr.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height());
                    }
                    AppMode::SectionPickStart => {
                        if let Some(world) = cursor_world {
                            section_start = Some(world);
                            app_mode = AppMode::SectionPickEnd;
                        }
                    }
                    AppMode::SectionPickEnd => {
                        if let Some(world) = cursor_world {
                            section_end = Some(world);
                            app_mode = AppMode::SectionAdjustDepth;
                        }
                    }
                    AppMode::SectionAdjustDepth => {
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            let gl = unsafe { get_internal_gl() };
                            let sec = mgr.extract_section(
                                &mut *gl.quad_context, start, end, section_depth, color_mode,
                            );
                            section_pan = glam::Vec2::new(sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                            section_zoom = (sec.length / 2.0).max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                            set_section(&mut section_data, &mut section_trash, Some(sec));
                            sec_picked_point = None;
                            if !section_color_mode_pinned { section_color_mode = color_mode; }
                            app_mode = AppMode::SectionView;
                        }
                    }
                    AppMode::MeasureFirst => {
                        if let Some((tile_idx, pt_idx)) = mgr.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) {
                            measure_a = Some(glam::Vec3::from(mgr.loaded[tile_idx].as_ref().unwrap().positions[pt_idx]));
                            measure_b = None;
                            app_mode = AppMode::MeasureSecond;
                        }
                    }
                    AppMode::MeasureSecond => {
                        if let Some((tile_idx, pt_idx)) = mgr.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) {
                            measure_b = Some(glam::Vec3::from(mgr.loaded[tile_idx].as_ref().unwrap().positions[pt_idx]));
                        }
                    }
                    AppMode::View3D => {
                        if let Some((wx, wy)) = screen_to_world_xy(mx, my, &view_proj, 0.0) {
                            selected_tile = mgr.tile_at(wx, wy);
                        }
                    }
                    _ => {}
                }
            }

            // Handle load/unload (from buttons or keyboard)
            if let Some(sel) = selected_tile {
                if is_key_pressed(KeyCode::Enter) && mgr.loaded[sel].is_none() {
                    request_load = Some(sel);
                }
                if is_key_pressed(KeyCode::Delete) && mgr.loaded[sel].is_some() {
                    request_unload = Some(sel);
                }
            }
            if let Some(idx) = request_load &&
                let Some(ref mut mgr) = tile_mgr {
                    let gl = unsafe { get_internal_gl() };
                    mgr.load_tile(&mut *gl.quad_context, idx, color_mode, tile_rgb_overrides[idx]);
                
            }
            if let Some(idx) = request_unload &&
                let Some(ref mut mgr) = tile_mgr {
                    let gl = unsafe { get_internal_gl() };
                    mgr.unload_tile(&mut *gl.quad_context, idx);
                
            }

            if class_colors_changed {
                apply_class_defs(&mut stages, &mut copc_stages, &mut tile_mgr, &class_defs);
            }
            let class_rebuild = (class_colors_changed && color_mode == ColorMode::Classification)
                || class_vis_changed;
            // Storing the ramp bounds is cheap; re-uploading every colour is
            // not. Only Height mode reads the ramp, so in any other mode the
            // new bounds are recorded and the upload waits until the view is
            // switched back to Height — which is what the height filter needs,
            // since it drives these bounds on every slider frame.
            if height_color_changed {
                apply_height_color(&mut stages, &mut copc_stages, &mut tile_mgr, height_color_min, height_color_max);
            }
            let height_rebuild = height_color_changed && color_mode == ColorMode::Height;
            if color_mode != prev_color_mode || height_rebuild || class_rebuild {
                let gl = unsafe { get_internal_gl() };
                if height_rebuild || class_rebuild {
                    for s in stages.iter_mut() {
                        s.rebuild_colors(&mut *gl.quad_context, color_mode);
                    }
                }
                if let Some(ref mut mgr) = tile_mgr {
                    mgr.rebuild_colors(&mut *gl.quad_context, color_mode);
                }
            }
            if let Some(idx) = tile_override_changed &&
                let Some(ref mut mgr) = tile_mgr {
                    if let Some(ref mut loaded) = mgr.loaded[idx] {
                        loaded.rgb_override = tile_rgb_overrides[idx];
                    }
                    let gl = unsafe { get_internal_gl() };
                    mgr.rebuild_tile_colors(&mut *gl.quad_context, idx, color_mode);

            }
        }

        // ---- Polyline 3D overlay (both modes) ----
        if !poly_files.is_empty() {
            set_camera(&camera);
            polylines::draw_3d(&poly_files);
            set_default_camera();
        }

        // ---- Empty state: no data loaded ----
        // Show a minimal Controls window so the user knows how to load files.
        if stages.is_empty() && copc_stages.is_empty() && tile_mgr.is_none() {
            egui_macroquad::ui(|egui_ctx| {
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        ui.label("No point cloud loaded.");
                        if ui.button("Open file…").clicked()
                            && let Some(paths) = rfd::FileDialog::new()
                                .add_filter("LAS / LAZ", &["laz", "las"])
                                .pick_files()
                        {
                            for p in paths {
                                pending_loads.push(p.to_string_lossy().to_string());
                            }
                        }
                        ui.label("…or drag a .laz / .copc.laz file onto the window (Windows / X11).");
                    });
            });
            egui_macroquad::draw();
        }

        // ---- Act on a digitising request ----
        // One place, whether it came from `P`, from Enter, or from the panel
        // in any of the three per-mode blocks.
        if let Some(req) = digitise_request.take() {
            match req {
                vectorise::Request::Start => {
                    if digitise.target.is_none() {
                        // A layer has a destination from the moment it exists,
                        // because layers auto-save — there is no later "save
                        // this somewhere" step to hang a file dialog on.
                        match cloud_source_path(&stages, &copc_stages, &tile_mgr) {
                            Some(cloud) => {
                                let srs = poly_files.iter()
                                    .flat_map(|f| &f.layers)
                                    .map(|l| l.srs_id)
                                    .find(|s| *s != 0)
                                    .unwrap_or(0);
                                let t = vectorise::ensure_target(&mut poly_files, &cloud, srs);
                                digitise.target = Some(t);
                                digitise.status = Some(format!(
                                    "New lines go to {}",
                                    vectorise::short_path(&poly_files[t.0].path),
                                ));
                            }
                            None => digitise.status =
                                Some("Load a point cloud first.".to_string()),
                        }
                    }
                    if digitise.target.is_some() {
                        digitise.clear();
                        app_mode = AppMode::Digitise;
                    }
                }
                vectorise::Request::Cancel => {
                    digitise.clear();
                    app_mode = AppMode::View3D;
                }
                vectorise::Request::UndoVertex => { digitise.pop(); }
                vectorise::Request::Finish => {
                    if let Some(fi) = digitise.finish(&mut poly_files) {
                        // Granularity is the completed line, not the vertex:
                        // a whole-file rewrite per click would be waste, and a
                        // half-drawn line is not a meaningful thing to persist.
                        // A crash therefore costs at most the line in progress.
                        let pf = &mut poly_files[fi];
                        digitise.status = Some(if pf.changed_on_disk() {
                            format!(
                                "{} changed on disk since it was loaded — the line is \
                                 held in memory, not written. Use Save As to keep it.",
                                pf.file_name(),
                            )
                        } else {
                            match pf.save() {
                                Ok(()) => format!(
                                    "{} lines saved to {}",
                                    pf.line_count(), pf.file_name(),
                                ),
                                Err(e) => format!("Save failed: {e}"),
                            }
                        });
                    }
                }
            }
        }

        // ---- Plan-view ground tool: start / cancel / apply ----
        if let Some(kind) = start_ground_lasso {
            ground_lasso.clear();
            ground_status = None;
            lasso_kind = kind;
            // The camera is left exactly where it is: the tools are only
            // offered from the top view, so there is nothing to correct for.
            app_mode = AppMode::GroundLasso;
        }
        if cancel_ground_lasso {
            ground_lasso.clear();
            app_mode = AppMode::View3D;
        }

        if apply_ground_lasso && ground_lasso.len() >= 3
            && op_from_mask(classify_from, class_vis).is_empty() {
            ground_status = Some(empty_from_mask_reason(classify_from));
            ground_lasso.clear();
            app_mode = AppMode::View3D;
        } else if apply_ground_lasso && ground_lasso.len() >= 3 {
            // Footprint and band in absolute world coordinates, so the op means
            // the same thing at export as it does on screen.
            let polygon: Vec<(f64, f64)> = ground_lasso.iter()
                .map(|v| (v.x as f64 + data_offset[0], v.y as f64 + data_offset[1]))
                .collect();
            let (z_lo, z_hi) = if z_clip_enabled {
                (z_clip_lo as f64 + data_offset[2], z_clip_hi as f64 + data_offset[2])
            } else {
                // No band set: record the data's own extent rather than an
                // unbounded range, so the op stays a closed description.
                let mut lo = f64::MAX;
                let mut hi = f64::MIN;
                for st in &stages { lo = lo.min(st.cloud.min[2]); hi = hi.max(st.cloud.max[2]); }
                for cs in &copc_stages { lo = lo.min(cs.global_min[2]); hi = hi.max(cs.global_max[2]); }
                if let Some(mgr) = tile_mgr.as_ref() { lo = lo.min(mgr.global_min[2]); hi = hi.max(mgr.global_max[2]); }
                if lo > hi { (f64::MIN, f64::MAX) } else { (lo, hi) }
            };

            let (mut bx0, mut by0) = (f64::MAX, f64::MAX);
            let (mut bx1, mut by1) = (f64::MIN, f64::MIN);
            for &(x, y) in &polygon {
                bx0 = bx0.min(x); bx1 = bx1.max(x);
                by0 = by0.min(y); by1 = by1.max(y);
            }

            // The inside lasso is a plain prism: no samples to gather and no
            // plane to fit, so it cannot fail the way the ground tool can.
            let region = match lasso_kind {
                LassoKind::Inside => Some((
                    crate::classify::ClassRegion::Volume {
                        polygon: polygon.clone(),
                        z_min: z_lo,
                        z_max: z_hi,
                    },
                    None,
                )),
                LassoKind::AboveGround => {
                    let mut samples: Vec<(f64, f64, f64)> = Vec::new();
                    for st in &stages { st.collect_points_in_bbox((bx0, by0), (bx1, by1), z_lo, z_hi, &mut samples); }
                    for cs in &copc_stages { cs.collect_points_in_bbox((bx0, by0), (bx1, by1), z_lo, z_hi, &mut samples); }
                    if let Some(mgr) = tile_mgr.as_ref() { mgr.collect_points_in_bbox((bx0, by0), (bx1, by1), z_lo, z_hi, &mut samples); }

                    match crate::classify::fit_ground_plane(&samples, &polygon, z_lo, z_hi) {
                        None => {
                            ground_status = Some(format!(
                                "Not enough ground in view to fit a plane ({} points found). \
                                 Zoom in so more detail is loaded, or outline a larger area.",
                                samples.len(),
                            ));
                            None
                        }
                        Some(fit) => Some((
                            crate::classify::ClassRegion::AboveGround {
                                polygon: polygon.clone(),
                                origin: fit.origin,
                                plane: fit.plane,
                                offset: ground_offset as f64,
                                z_min: z_lo,
                                z_max: z_hi,
                            },
                            Some(fit),
                        )),
                    }
                }
            };

            if let Some((region, fit)) = region {
                let op = crate::classify::ClassOp::new(
                    0,
                    region,
                    classify_target,
                    op_from_mask(classify_from, class_vis),
                );

                let mut changed = 0usize;
                let mut touched = 0usize;
                {
                    let mut gl = unsafe { get_internal_gl() };
                    gl.flush();
                    for st in stages.iter_mut() {
                        if !op.intersects_aabb(&st.cloud.min, &st.cloud.max) { continue; }
                        let from = st.edits.ops.len();
                        st.edits.push(op.clone());
                        changed += st.apply_edits(&mut *gl.quad_context, color_mode, from);
                        touched += 1;
                    }
                    for cs in copc_stages.iter_mut() {
                        if !op.intersects_aabb(&cs.global_min, &cs.global_max) { continue; }
                        let from = cs.edits.ops.len();
                        cs.edits.push(op.clone());
                        changed += cs.apply_edits_to_loaded(&mut *gl.quad_context, color_mode, from);
                        touched += 1;
                    }
                }

                let mut save_err = None;
                for st in stages.iter_mut().filter(|s| s.edits.dirty) {
                    let path = st.filepath.clone();
                    if let Err(e) = st.edits.save(&path) { save_err = Some(e.to_string()); }
                }
                for cs in copc_stages.iter_mut().filter(|c| c.edits.dirty) {
                    let path = cs.filepath.clone();
                    if let Err(e) = cs.edits.save(&path) { save_err = Some(e.to_string()); }
                }

                match &fit {
                    Some(fit) => {
                        let centre = (
                            (bx0 + bx1) * 0.5,
                            (by0 + by1) * 0.5,
                        );
                        info!(
                            "Ground fit: {} cells from {} points, {:.2}m at footprint centre, \
                             slope {:.2}°, rms {:.3}m; {} points reclassified",
                            fit.cells_used, fit.points_considered,
                            fit.height_at(centre.0, centre.1), fit.slope_degrees(), fit.rms, changed,
                        );
                    }
                    None => info!(
                        "Inside lasso: {:.2}m band, {} points reclassified",
                        z_hi - z_lo, changed,
                    ),
                }
                ground_status = Some(match (save_err, touched) {
                    (Some(e), _) => format!("{changed} points reclassified — SAVE FAILED: {e}"),
                    (None, 0) => "Footprint did not overlap any loaded file".to_string(),
                    (None, _) => match &fit {
                        Some(fit) => format!(
                            "{changed} points → class {classify_target}  (ground rms {:.2} m, slope {:.1}°)",
                            fit.rms, fit.slope_degrees(),
                        ),
                        None => format!("{changed} points → class {classify_target}"),
                    },
                });
                ground_lasso.clear();
                app_mode = AppMode::View3D;
            }
        }

        next_frame().await
    }
}
