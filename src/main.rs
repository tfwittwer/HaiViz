use std::env;
use std::path::{Path, PathBuf};
//use rfd;
use log::{info,error};
//use console::style;
use crate::glam::vec3;

use macroquad::prelude::*;

use crate::raw_miniquad::{ClassDefs, ClassVis, CopcStage, EdlRenderer, HeatmapRenderer, SectionData, Stage, TileStageManager, build_section_from_raw_stages, build_section_from_stages, classification_color, default_class_label, load_class_defs, load_class_defs_from, make_section_data, save_class_defs_to, scan_dir_for_copc, DEFAULT_CLASS_FILE};
use crate::classify::{ClassMask, effective_class};
use las::Point as LasPoint;

const MOVE_SPEED: f32 = 0.5;
/// Orbit rotation, in radians per logical pixel of mouse travel.
///
/// Per pixel, not per second: a mouse delta is already the movement since the
/// last frame. Scaling it by the frame time as well makes the same hand
/// movement rotate further the slower the frame — a tile load or a heavy EDL
/// frame then turns a small drag into a jump that flips the view.
const LOOK_SPEED: f32 = 0.008;

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
mod vector_edit;
mod edge_follow;
mod gcp;
mod classify;
mod export;
mod gltf_load;
mod tiles3d;
mod splats;
mod layers;

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
            FaceDef { corners: [3, 2, 1, 0], label: "Bot",    normal: [ 0.0, 0.0,-1.0], snap_yaw: -PI/2.0, snap_pitch:  PI/2.0, color: Color::new(0.2, 0.3, 0.5, 0.85) },
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
    /// Drawing a lasso on screen, from any camera angle, to reclassify what
    /// projects inside it. What "inside" means is `LassoKind`, kept separate so
    /// both flavours share one drawing mode.
    GroundLasso,
    /// Drawing a polyline onto the cloud, vertex by vertex, from any camera
    /// angle: a digitised vertex comes from a real point and means the same
    /// thing from anywhere.
    Digitise,
    /// Editing existing polylines: dragging vertices, pulling new ones out of
    /// segments, deleting. From any camera angle, for the same reason as
    /// digitising — a dragged vertex is snapped onto real data.
    EditLines,
    /// Digitising in the section view, in (along, height).
    SectionDigitise,
}

/// Where to land once a force-loaded section corridor has streamed in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CorridorThen {
    /// Plain section view — after an undo, which has to re-stream the slice.
    View,
    Lasso,
    Digitise,
}

impl CorridorThen {
    fn mode(self) -> AppMode {
        match self {
            CorridorThen::View => AppMode::SectionView,
            CorridorThen::Lasso => AppMode::SectionLasso,
            CorridorThen::Digitise => AppMode::SectionDigitise,
        }
    }

    /// What the corridor is being loaded for, for the refusal message.
    fn purpose(self) -> &'static str {
        match self {
            CorridorThen::View | CorridorThen::Lasso => "classifying",
            CorridorThen::Digitise => "digitising",
        }
    }
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
/// This no longer decides whether a lasso is allowed — outlines work from any
/// angle — only how one is recorded. From straight above, a screen outline and
/// a plan footprint select exactly the same points, and the plan form is what
/// older haiviz builds can read, so that is the one written. It has to be
/// *exactly* straight above for that to hold: at one degree of tilt, a point
/// 20 m up is a third of a metre from where a plan footprint would put it. The
/// snaps (`T`, or the cube's top face) land on zero, so this is a tolerance for
/// floating point only.
const TOP_VIEW_TOL: f32 = std::f32::consts::PI / 1800.0;

/// True when the camera is looking straight down. Yaw does not matter: the
/// view direction is vertical whatever it is.
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

/// What a queued path turned out to be, once read far enough to know its
/// world box.
enum LoadSource {
    Cloud { copc: bool },
    Layer(layers::Prepared),
}

/// Unproject screen coordinates to world XY plane at given Z height.
/// The part of the window the 3D and section views render into.
///
/// The panel is docked to the left edge, and the views render to the right of
/// it rather than underneath it — so the data is never hidden behind the panel
/// and Zoom All frames into what can actually be seen. Everything that converts
/// between window pixels and normalised device coordinates goes through this:
/// a projection that still assumed the whole window would put every pick a
/// panel-width out.
#[derive(Clone, Copy)]
struct ViewRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl ViewRect {
    fn aspect(&self) -> f32 { self.w / self.h.max(1.0) }

    /// Normalised device coordinates for a window pixel.
    fn ndc(&self, sx: f32, sy: f32) -> (f32, f32) {
        ((sx - self.x) / self.w.max(1.0) * 2.0 - 1.0,
         1.0 - (sy - self.y) / self.h.max(1.0) * 2.0)
    }

    /// Window pixel for a normalised device position.
    fn to_screen(self, nx: f32, ny: f32) -> (f32, f32) {
        (self.x + (nx + 1.0) * 0.5 * self.w, self.y + (1.0 - ny) * 0.5 * self.h)
    }

    /// As GL wants it: origin at the bottom-left, in **physical** pixels.
    ///
    /// `screen_width`, `screen_height` and `mouse_position` are all logical —
    /// macroquad divides them by the DPI scale — but `apply_viewport` addresses
    /// the framebuffer, which is not. Mixing the two puts the rendered view a
    /// scale factor away from where every pick thinks it is, which shows up as
    /// a snap marker that trails the cursor by a growing offset.
    fn gl(&self) -> (i32, i32, i32, i32) {
        let dpi = miniquad::window::dpi_scale();
        (
            (self.x * dpi).round() as i32,
            ((screen_height() - self.y - self.h) * dpi).round() as i32,
            (self.w.max(1.0) * dpi).round() as i32,
            (self.h.max(1.0) * dpi).round() as i32,
        )
    }
}

fn screen_to_world_xy(
    sx: f32, sy: f32, view_proj: &glam::Mat4, z_plane: f32, view: ViewRect,
) -> Option<(f32, f32)> {
    let inv = view_proj.inverse();
    let (ndc_x, ndc_y) = view.ndc(sx, sy);

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
fn cursor_ray(view_proj: &glam::Mat4, mx: f32, my: f32, view: ViewRect) -> ([f32; 3], [f32; 3]) {
    let (o, d) = cursor_ray_f64(view_proj, mx, my, view);
    (o.as_vec3().to_array(), d.as_vec3().to_array())
}

/// [`cursor_ray`] in double precision.
///
/// The near plane sits 100 km behind the camera. At that magnitude an f32
/// coordinate cannot resolve a centimetre, and neither can the matrix inverse
/// that finds it, so the unprojection is done in f64 and callers that need
/// precision move the origin up to the data before narrowing it.
fn cursor_ray_f64(view_proj: &glam::Mat4, mx: f32, my: f32, view: ViewRect) -> (glam::DVec3, glam::DVec3) {
    let inv = view_proj.as_dmat4().inverse();
    let (ndc_x, ndc_y) = view.ndc(mx, my);
    let unproject = |z: f64| {
        let p = inv * glam::DVec4::new(ndc_x as f64, ndc_y as f64, z, 1.0);
        p.truncate() / p.w
    };
    let near = unproject(-1.0);
    (near, (unproject(1.0) - near).normalize_or_zero())
}

/// How far along a ray it enters an axis-aligned box, or `None` when it
/// misses. Negative when the origin is already inside.
fn ray_box_entry(o: glam::DVec3, d: glam::DVec3, lo: glam::DVec3, hi: glam::DVec3) -> Option<f64> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for i in 0..3 {
        if d[i].abs() < 1e-12 {
            if o[i] < lo[i] || o[i] > hi[i] { return None; }
            continue;
        }
        let (a, b) = ((lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t0 <= t1).then_some(t0)
}

/// Screen right and screen up, as unit vectors in viewer space.
fn view_axes(view_proj: &glam::Mat4) -> (Vec3, Vec3) {
    let inv = view_proj.inverse();
    let at = |x: f32, y: f32| {
        let p = inv * glam::Vec4::new(x, y, 0.0, 1.0);
        glam::Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
    };
    let c = at(0.0, 0.0);
    ((at(1.0, 0.0) - c).normalize_or_zero(), (at(0.0, 1.0) - c).normalize_or_zero())
}

/// The view plane a screen lasso is recorded in: through `origin`, along the
/// camera's screen axes, made exactly orthonormal in double precision — the
/// axes come out of an f32 matrix inverse, and the op keeps them for good.
fn view_plane(origin: [f64; 3], right: Vec3, up: Vec3) -> crate::classify::ViewPlane {
    let unit = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        v.map(|c| c / l)
    };
    let r = unit([right.x as f64, right.y as f64, right.z as f64]);
    let u0 = [up.x as f64, up.y as f64, up.z as f64];
    let k = u0[0] * r[0] + u0[1] * r[1] + u0[2] * r[2];
    let u = unit([u0[0] - k * r[0], u0[1] - k * r[1], u0[2] - k * r[2]]);
    crate::classify::ViewPlane { origin, right: r, up: u }
}

/// The classification lasso, in screen space.
///
/// Its vertices live on a view plane in the world, so projecting them each
/// frame keeps the outline exactly where it was drawn through a pan or a zoom —
/// and drawn in screen space, nothing in the cloud can hide it.
fn draw_screen_lasso(
    points: &[Vec3],
    drawing: bool,
    view_proj: &glam::Mat4,
    view: ViewRect,
    cursor: (f32, f32),
    color: Color,
) {
    if points.is_empty() { return; }
    let pts: Vec<(f32, f32)> = points.iter().map(|p| world_to_screen(*p, view_proj, view)).collect();
    for w in pts.windows(2) {
        draw_line(w[0].0, w[0].1, w[1].0, w[1].1, 2.0, color);
    }
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    if drawing {
        draw_line(last.0, last.1, cursor.0, cursor.1, 1.5, color);
    }
    // The closing edge, faint, so the polygon that will be applied is visible
    // before committing to it.
    if pts.len() >= 2 {
        let end = if drawing { cursor } else { last };
        draw_line(end.0, end.1, first.0, first.1, 1.0, Color::new(color.r, color.g, color.b, 0.4));
    }
    for (x, y) in &pts {
        draw_circle_lines(*x, *y, 4.0, 1.5, color);
    }
}

/// Resolve the vertex under the cursor for the active snap mode.
///
/// The percentile and same-Z modes need an XY to work from, and it comes from
/// the ray hit rather than from a fixed plane: that is what makes them mean
/// "within 0.2 m of the surface I am pointing at" from an oblique view as well
/// as from directly above.
///
/// `prev` is the vertex a same-Z snap holds the height of: the last one placed
/// while digitising, or where a dragged vertex started. `exclude` is the part
/// of a line being dragged, which the existing-polyline snap must not find.
#[allow(clippy::too_many_arguments)]
fn resolve_snap(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    poly_files: &[polylines::PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    view: ViewRect,
    data_offset: [f64; 3],
    dig: &vectorise::Digitise,
    prev: Option<[f64; 3]>,
    exclude: Option<polylines::SnapExclude>,
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

    let (ray_o, ray_d) = cursor_ray_f64(view_proj, mx, my, view);
    let r = dig.radius;

    // Start the ray where it enters the loaded data, not at the near plane
    // 100 km back. Every point a snap can find is still in front of it, and
    // the distances along and across the ray become small numbers that f32 —
    // which the grids work in — resolves to well under a millimetre. From the
    // near plane, the tube test itself had centimetres of noise in it.
    // Clouds only: snapping never lands on a mesh or a splat.
    let start = loaded_aabb_viewer(stages, copc_stages, tile_mgr, None)
        .and_then(|(lo, hi)| {
            let pad = glam::DVec3::splat(r as f64 + 1.0);
            ray_box_entry(ray_o, ray_d, lo.as_dvec3() - pad, hi.as_dvec3() + pad)
        })
        .filter(|t| *t > 0.0)
        .map_or(ray_o, |t| ray_o + ray_d * t);
    let (origin, dir) = (start.as_vec3().to_array(), ray_d.as_vec3().to_array());

    // One screen pixel in metres, under the orthographic camera: the first
    // row of the matrix scales world metres along screen-right into NDC.
    let pixel = {
        let row = glam::Vec3::new(view_proj.x_axis.x, view_proj.y_axis.x, view_proj.z_axis.x);
        2.0 / (row.length() * view.w.max(1.0)).max(1e-9)
    };

    // Where on the surface the cursor is pointing. Every mode except
    // `Existing` starts from this. The thinnest tube that finds anything, from
    // about a pixel up to the snap radius — not the frontmost point in the
    // whole radius, which on a slanted wall is a radius off to one side. See
    // `spatial::surface_hit`.
    let anchor: Option<[f32; 3]> = crate::spatial::surface_hit(pixel, r, |tube| {
        let mut best: Option<(f32, [f32; 3])> = None;
        for st in stages.iter().filter(|s| s.visible) {
            if let Some(h) = st.snap_nearest_ray(origin, dir, tube)
                && best.is_none_or(|(t, _)| h.t < t) { best = Some((h.t, h.pos)); }
        }
        for cs in copc_stages.iter().filter(|c| c.visible) {
            if let Some((h, _)) = cs.snap_nearest_ray(origin, dir, tube)
                && best.is_none_or(|(t, _)| h.t < t) { best = Some((h.t, h.pos)); }
        }
        if let Some(mgr) = tile_mgr
            && let Some((_, h)) = mgr.snap_nearest_ray(origin, dir, tube)
            && best.is_none_or(|(t, _)| h.t < t) { best = Some((h.t, h.pos)); }
        best.map(|(_, pos)| pos)
    });

    if dig.snap_mode == SnapMode::Existing {
        // Against the ray, not against a point on the surface: the cursor may
        // be over a gap in the cloud, and tying into an existing line is
        // exactly what one does there.
        // From the near plane, in f64: a polyline need not lie inside the
        // cloud's box, so the shifted start could be past one.
        let o = (ray_o + glam::DVec3::from_array(data_offset)).to_array();
        let d = ray_d.to_array();
        return match polylines::nearest_on_lines_to_ray(poly_files, o, d, r as f64, exclude) {
            Some((w, _)) => SnapHit::good(w),
            None => empty("No polyline within the snap radius."),
        };
    }

    let Some(anchor) = anchor else {
        return empty("Nothing within the snap radius. Widen it with Ctrl + wheel, \
                      or Shift-click to place a vertex freely.");
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
                        "Point spacing here is {spacing:.2} m, wider than the {r:.2} m \
                         snap radius — a percentile would be taken over whatever the \
                         LOD happened to load. Zoom in, or widen the radius.",
                    )),
                };
            }
        }
    }

    match dig.snap_mode {
        SnapMode::Nearest => SnapHit::good(to_world(anchor)),
        SnapMode::SameZ => match prev {
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

/// Resolve the vertex under the cursor in the section view, at section-space
/// `(along, height)`.
///
/// Queries the section's own points, which were cut with the height band and
/// hidden classes already applied. The vertex lands on the section plane — see
/// the note on the section view in `vectorise.rs` — except when tying into an
/// existing line, which lands on that line.
///
/// There is no COPC spacing guard here: digitising in a section force-loads
/// the corridor first, exactly as the section lasso does, so the slice is at
/// full resolution by the time a vertex can be placed.
#[allow(clippy::too_many_arguments)]
fn resolve_section_snap(
    sec: &SectionData,
    poly_files: &[polylines::PolylineFile],
    start: (f32, f32),
    end: (f32, f32),
    depth: f32,
    along: f32,
    height: f32,
    data_offset: [f64; 3],
    dig: &vectorise::Digitise,
) -> vectorise::SnapHit {
    use vectorise::{SnapHit, SnapMode, SnapQuality};
    let empty = |note: &str| SnapHit {
        world: [0.0; 3], quality: SnapQuality::Empty, note: Some(note.to_string()),
    };
    let (dx, dy) = (end.0 - start.0, end.1 - start.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 { return empty("The section line has no length."); }
    let dir = (dx / len, dy / len);
    let r = dig.radius;
    let on_plane = |a: f32, h: f32| vectorise::section_to_world(start, dir, a, h, data_offset);

    if dig.snap_mode == SnapMode::Existing {
        return match polylines::nearest_in_section(poly_files, start, end, depth, along, height, r) {
            Some(w) => SnapHit::good(w),
            None => empty("No polyline within the snap radius."),
        };
    }

    let (grid, pts) = sec.snap_index();
    let nothing = "Nothing within the snap radius. Widen it with Ctrl + wheel, \
                   or Shift-click to place a vertex freely.";
    match dig.snap_mode {
        SnapMode::Nearest => match vectorise::section_nearest(grid, pts, along, height, r) {
            Some((a, h)) => SnapHit::good(on_plane(a, h)),
            None => empty(nothing),
        },
        // As in 3D: the position comes from a real point near the cursor, the
        // height from the vertex already committed to.
        SnapMode::SameZ => match vectorise::section_nearest(grid, pts, along, height, r) {
            Some((a, h)) => {
                let w = on_plane(a, h);
                SnapHit::good(match dig.last() { Some(prev) => [w[0], w[1], prev[2]], None => w })
            }
            None => empty(nothing),
        },
        SnapMode::LowZ | SnapMode::HighZ => {
            let pct = if dig.snap_mode == SnapMode::LowZ { dig.low_pct } else { dig.high_pct };
            match vectorise::section_percentile(grid, pts, along, r, pct) {
                Some((a, h)) => SnapHit::good(on_plane(a, h)),
                None => empty(nothing),
            }
        }
        SnapMode::Existing => unreachable!("handled above"),
    }
}

/// Re-drape a line onto the cloud with the current snap mode and radius.
///
/// Vertex by vertex through the same backends, filters and COPC spacing guard
/// as a snap, so a re-draped vertex gets the height a click at its plan position
/// would have. `vector_edit::drape` says why the plan position is kept.
#[allow(clippy::too_many_arguments)]
fn redrape_line(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    poly_files: &[polylines::PolylineFile],
    line: polylines::PolyRef,
    data_offset: [f64; 3],
    dig: &vectorise::Digitise,
) -> Option<vector_edit::Drape> {
    let pl = poly_files.get(line.file)?.get(line.line_ref())?;
    let mut zs: Vec<(f32, [f32; 3])> = Vec::new();
    Some(vector_edit::drape(pl.points(), |p| {
        height_at(stages, copc_stages, tile_mgr, p, data_offset, dig.snap_mode, dig, &mut zs)
    }))
}

/// The height `mode` gives a vertex at `p`'s plan position, taking `p`'s own
/// height as the reference for Nearest. Shared by re-drape and by edge
/// following, which both have a plan position and need the cloud's height
/// there, through the same filters and COPC spacing guard as a snap.
///
/// `zs` is a scratch buffer, so a whole line costs one allocation.
#[allow(clippy::too_many_arguments)]
fn height_at(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    p: [f64; 3],
    data_offset: [f64; 3],
    mode: vectorise::SnapMode,
    dig: &vectorise::Digitise,
    zs: &mut Vec<(f32, [f32; 3])>,
) -> Result<f64, vector_edit::DrapeMiss> {
    use vector_edit::DrapeMiss;
    use vectorise::SnapMode;
    let r = dig.radius;
    let v = [
        (p[0] - data_offset[0]) as f32,
        (p[1] - data_offset[1]) as f32,
        (p[2] - data_offset[2]) as f32,
    ];
    if matches!(mode, SnapMode::LowZ | SnapMode::HighZ)
        && copc_stages.iter().filter(|c| c.visible)
            .any(|c| c.lod_spacing_near(v[0], v[1], r).is_some_and(|s| s > r as f64)) {
        return Err(DrapeMiss::Coarse);
    }
    zs.clear();
    for st in stages.iter().filter(|s| s.visible) { st.snap_z_candidates(v[0], v[1], r, zs); }
    for cs in copc_stages.iter().filter(|c| c.visible) { cs.snap_z_candidates(v[0], v[1], r, zs); }
    if let Some(mgr) = tile_mgr { mgr.snap_z_candidates(v[0], v[1], r, zs); }
    vector_edit::drape_z(mode, v, zs, dig.low_pct, dig.high_pct)
        .map(|z| z as f64 + data_offset[2])
        .ok_or(DrapeMiss::Empty)
}

/// Heights for a traced route, at full resolution.
///
/// The route's interior vertices come off the raster, whose heights are only
/// good enough to draw; each is re-snapped here as a click would be. The ends
/// are real already — the last vertex, and the snapped target. If any interior
/// vertex has no answer the whole route is refused: a trace that fell back to
/// interpolated heights part of the way along would be exactly the silent
/// degradation the snap refuses.
fn drape_route(
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    route: &[[f64; 3]],
    data_offset: [f64; 3],
    dig: &vectorise::Digitise,
) -> Result<Vec<[f64; 3]>, String> {
    use vector_edit::DrapeMiss;
    use vectorise::SnapMode;
    // Same Z holds the height already committed to, and the route carries it.
    if dig.snap_mode == SnapMode::SameZ { return Ok(route.to_vec()); }
    // Tying into a line means something at the end of the route; along it, the
    // nearest real point is the honest height.
    let mode = if dig.snap_mode == SnapMode::Existing { SnapMode::Nearest } else { dig.snap_mode };
    let mut zs = Vec::new();
    let (mut empty, mut coarse) = (0usize, 0usize);
    let last = route.len().saturating_sub(1);
    let draped: Vec<[f64; 3]> = route.iter().enumerate().map(|(i, &p)| {
        if i == 0 || i == last { return p; }
        match height_at(stages, copc_stages, tile_mgr, p, data_offset, mode, dig, &mut zs) {
            Ok(z) => [p[0], p[1], z],
            Err(DrapeMiss::Empty) => { empty += 1; p }
            Err(DrapeMiss::Coarse) => { coarse += 1; p }
        }
    }).collect();
    match (empty, coarse) {
        (0, 0) => Ok(draped),
        (n, 0) => Err(format!(
            "{n} traced vertices have nothing within the snap radius — widen it, \
             press E for a straight segment, or Shift-click.",
        )),
        _ => Err(format!(
            "The loaded data is sparser than the snap radius at {} traced vertices — \
             zoom in, or press E for a straight segment.",
            empty + coarse,
        )),
    }
}

/// Where a traced vertex's *display* height comes from, for the snap mode in
/// force. The committed height is re-snapped by `drape_route`.
fn edge_z_source(dig: &vectorise::Digitise) -> edge_follow::ZSource {
    use edge_follow::ZSource;
    use vectorise::SnapMode;
    match dig.snap_mode {
        SnapMode::Nearest | SnapMode::LowZ => ZSource::Lowest,
        SnapMode::HighZ => ZSource::Highest,
        SnapMode::SameZ => dig.last().map_or(ZSource::Interpolated, |p| ZSource::Fixed(p[2])),
        SnapMode::Existing => ZSource::Interpolated,
    }
}

/// A cheap fingerprint of the height band and the visible classes, folded into
/// the edge-following data signature so that hiding a class to steer the
/// tracer rebuilds its raster.
fn filter_signature(z_clip: crate::raw_miniquad::ZClip, class_vis: ClassVis) -> usize {
    let mut h = (z_clip.min.to_bits() as usize).wrapping_mul(0x9E37_79B9) ^ z_clip.max.to_bits() as usize;
    for id in 0..=255u8 {
        if class_vis.contains(id) { h = h.rotate_left(5) ^ (id as usize + 1); }
    }
    h
}

/// Auto-save a vector file after a committed change.
///
/// Digitising and editing share this, and the granularity: one write per
/// completed change. A file changed on disk since it was loaded is left alone
/// and the change held in memory — every save rewrites the whole file, so an
/// external edit would be destroyed rather than merely conflicted with.
fn autosave_vectors(pf: &mut polylines::PolylineFile) -> Result<(), String> {
    if pf.changed_on_disk() {
        return Err(format!(
            "{} changed on disk since it was loaded — the change is held in memory, \
             not written. Use Save As to keep it.",
            pf.file_name(),
        ));
    }
    pf.save().map_err(|e| format!("Save failed: {e}"))
}

/// A status line for an edit and the auto-save that followed it.
fn saved_status(what: String, saved: Result<(), String>) -> String {
    match saved {
        Ok(()) => format!("{what} Saved."),
        Err(e) => format!("{what} {e}"),
    }
}

/// Give digitising a layer to put lines in — the sidecar beside the cloud,
/// loaded if an earlier session left one, created if not. False, with the
/// reason in the status line, when there is nowhere for a line to go.
fn ensure_digitise_target(
    digitise: &mut vectorise::Digitise,
    poly_files: &mut Vec<polylines::PolylineFile>,
    stages: &[Stage],
    copc_stages: &[CopcStage],
    tile_mgr: &Option<TileStageManager>,
    data_offset: [f64; 3],
) -> bool {
    if digitise.target.is_some() { return true; }
    // A layer has a destination from the moment it exists, because layers
    // auto-save — there is no later "save this somewhere" step to hang a file
    // dialog on.
    match cloud_source_path(stages, copc_stages, tile_mgr) {
        Some(cloud) => {
            let srs = poly_files.iter()
                .flat_map(|f| &f.layers)
                .map(|l| l.srs_id)
                .find(|s| *s != 0)
                .unwrap_or(0);
            match vectorise::ensure_target(
                poly_files, &cloud, srs, digitise.width_px, data_offset, digitise.color,
            ) {
                Ok(t) => {
                    let pf = &poly_files[t.0];
                    digitise.target = Some(t);
                    digitise.status = Some(match pf.line_count() {
                        0 => format!("New lines go to {}", pf.file_name()),
                        n => format!("New lines are added to {} ({n} already there)", pf.file_name()),
                    });
                    true
                }
                Err(e) => {
                    digitise.status = Some(e);
                    false
                }
            }
        }
        None => {
            digitise.status = Some("Load a point cloud first.".to_string());
            false
        }
    }
}

/// Commit the line in progress and auto-save it — from the 3D view or the
/// section view alike.
///
/// Granularity is the completed line, not the vertex: a whole-file rewrite per
/// click would be waste, and a half-drawn line is not a meaningful thing to
/// persist. A crash therefore costs at most the line in progress.
fn finish_digitised_line(
    digitise: &mut vectorise::Digitise,
    editor: &mut vector_edit::Editor,
    poly_files: &mut [polylines::PolylineFile],
    closed: bool,
) {
    let Some(fi) = digitise.finish(poly_files, closed) else { return };
    // On the same undo order as edits, so Ctrl+Z straight after finishing a
    // line takes the line back.
    editor.record(fi);
    let pf = &mut poly_files[fi];
    digitise.status = Some(match autosave_vectors(pf) {
        Ok(()) => format!("{} lines saved to {}", pf.line_count(), pf.file_name()),
        Err(e) => e,
    });
}

/// Everything vector-shaped drawn under the 3D camera: the loaded lines, the
/// line being digitised with its traced route, and a vertex drag in progress.
/// Before `set_default_camera()`, so all of it is depth-tested against the
/// cloud and still composites beneath the panel.
#[allow(clippy::too_many_arguments)]
fn draw_vector_overlays_3d(
    poly_files: &[polylines::PolylineFile],
    selected: Option<polylines::PolyRef>,
    digitise: &vectorise::Digitise,
    editor: &vector_edit::Editor,
    app_mode: AppMode,
    data_offset: [f64; 3],
    view_dir: Vec3,
    metres_per_pixel: f32,
    snap_target: Option<[f64; 3]>,
    edge_route: Option<&[[f64; 3]]>,
) {
    if !poly_files.is_empty() {
        polylines::draw_3d(poly_files, view_dir, metres_per_pixel, selected);
    }
    if app_mode == AppMode::Digitise || digitise.is_drawing() {
        vectorise::draw_overlay(
            digitise, data_offset, view_dir, metres_per_pixel,
            snap_target, app_mode == AppMode::Digitise, edge_route,
        );
    }
    if app_mode == AppMode::EditLines && editor.drag.as_ref().is_some_and(|d| d.moving) {
        vector_edit::draw_drag_preview(
            editor, poly_files, data_offset, view_dir, metres_per_pixel, snap_target,
        );
        vectorise::draw_snap_marker(digitise, data_offset);
    }
}

/// The edge-following heatmap, draped on the ground under the trace window.
///
/// Through its own pipeline rather than macroquad's: with no depth test, since
/// it lies among the ground points and they would hide it, and from one mesh
/// uploaded per trace rather than a quarter of a million vertices re-sent every
/// frame. Whatever macroquad has queued is flushed first, so the lines drawn so
/// far show through it.
fn draw_edge_heatmap(
    renderer: &mut HeatmapRenderer,
    digitise: &vectorise::Digitise,
    app_mode: AppMode,
    view_proj: &glam::Mat4,
    view: ViewRect,
) {
    let ef = &digitise.edge;
    let wanted = app_mode == AppMode::Digitise && ef.enabled && ef.show_heatmap;
    match ef.trace.as_ref().filter(|_| wanted) {
        Some(trace) => {
            let mut gl = unsafe { get_internal_gl() };
            gl.flush();
            let layer = ef.heat_layer.map_or(0, |c| c as u8 + 1);
            renderer.sync(
                &mut *gl.quad_context,
                ef.raster_gen,
                (ef.field_gen, layer),
                (trace.raster.nx, trace.raster.ny),
                || trace.heat_geometry(),
                || trace.heat_colours(ef.heat_layer),
            );
            renderer.draw(&mut *gl.quad_context, view_proj, view.gl());
        }
        None if renderer.is_loaded() => {
            let gl = unsafe { get_internal_gl() };
            renderer.release(&mut *gl.quad_context);
        }
        None => {}
    }
}

/// The screen-space half of the vector overlays, under the default camera and
/// before egui: the cursor readout and the editing handles.
#[allow(clippy::too_many_arguments)]
fn draw_vector_overlays_2d(
    poly_files: &[polylines::PolylineFile],
    selected: Option<polylines::PolyRef>,
    digitise: &vectorise::Digitise,
    editor: &vector_edit::Editor,
    app_mode: AppMode,
    view_proj: &glam::Mat4,
    view: ViewRect,
    mx: f32,
    my: f32,
    ui_has_pointer: bool,
) {
    match app_mode {
        AppMode::Digitise if !ui_has_pointer => vectorise::draw_cursor_readout(digitise, mx, my),
        AppMode::EditLines => {
            vector_edit::draw_handles(editor, poly_files, selected, view_proj, (view.x, view.y, view.w, view.h));
            if editor.drag.as_ref().is_some_and(|d| d.moving) {
                vectorise::draw_cursor_readout(digitise, mx, my);
            }
        }
        _ => {}
    }
}

/// The Digitising section of the panel — drawing new lines, and editing the
/// ones already there. Shared by the three per-mode blocks.
#[allow(clippy::too_many_arguments)]
fn vector_tools_section(
    ui: &mut egui_macroquad::egui::Ui,
    digitise: &mut vectorise::Digitise,
    editor: &vector_edit::Editor,
    poly_files: &[polylines::PolylineFile],
    selected: Option<polylines::PolyRef>,
    app_mode: AppMode,
    have_cloud: bool,
    digitise_request: &mut Option<vectorise::Request>,
    edit_request: &mut Option<vector_edit::Request>,
) {
    panel_section(ui, "Digitising", false, |ui| {
        if let Some(r) = vectorise::controls(
            ui, digitise, poly_files, app_mode == AppMode::Digitise, have_cloud, true,
        ) { *digitise_request = Some(r); }
        if let Some(r) = vector_edit::controls(
            ui, editor, poly_files, selected, app_mode == AppMode::EditLines, digitise.snap_mode,
        ) { *edit_request = Some(r); }
    });
}

/// Project a world position to screen coordinates.
fn world_to_screen(pos: glam::Vec3, view_proj: &glam::Mat4, view: ViewRect) -> (f32, f32) {
    let clip = *view_proj * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
    view.to_screen(clip.x / clip.w, clip.y / clip.w)
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
/// Draw the meshes and splats over the cloud just rendered — once a frame,
/// whichever render block gets there first.
#[allow(clippy::too_many_arguments)]
fn draw_layers(
    layers: &layers::Layers,
    drawn: &mut bool,
    view_proj: &glam::Mat4,
    front: Vec3,
    right: Vec3,
    up: Vec3,
    view: ViewRect,
    z_clip: crate::raw_miniquad::ZClip,
) {
    if *drawn || layers.is_empty() {
        return;
    }
    *drawn = true;
    let mut gl = unsafe { get_internal_gl() };
    gl.flush();
    layers.draw(&mut *gl.quad_context, view_proj, front, right, up, view.gl(), z_clip);
}

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
) {
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

/// One collapsible section of the docked panel.
///
/// haiviz used to scatter five floating windows across the viewport; they
/// overlapped each other, covered the data, and had to be re-found after every
/// resize. One docked panel with collapsible sections keeps all of it in one
/// place and leaves the view unobstructed — and docking **left** specifically,
/// because the view cube lives in the top-right corner.
fn panel_section<R>(
    ui: &mut egui_macroquad::egui::Ui,
    title: &str,
    default_open: bool,
    add: impl FnOnce(&mut egui_macroquad::egui::Ui) -> R,
) -> Option<R> {
    panel_section_forced(ui, title, default_open, None, add)
}

/// As [`panel_section`], but a `force_open` of `Some(_)` overrides the stored
/// state for this frame — and egui stores the new value, so passing it on a
/// single frame opens the section and leaves the header free to toggle it
/// afterwards. That is what lets a button elsewhere in the panel reveal a
/// section without taking ownership of it.
fn panel_section_forced<R>(
    ui: &mut egui_macroquad::egui::Ui,
    title: &str,
    default_open: bool,
    force_open: Option<bool>,
    add: impl FnOnce(&mut egui_macroquad::egui::Ui) -> R,
) -> Option<R> {
    egui_macroquad::egui::CollapsingHeader::new(title)
        .default_open(default_open)
        .open(force_open)
        .show(ui, add)
        .body_returned
}

#[allow(clippy::too_many_arguments)]
fn polylines_section(
    ui: &mut egui_macroquad::egui::Ui,
    poly_files: &mut Vec<polylines::PolylineFile>,
    selected_poly: &mut Option<polylines::PolyRef>,
    poly_load_error: &mut Option<String>,
    poly_save_status: &mut Option<String>,
    // A Salome save that would drop something, waiting for the operator to
    // say yes: (file index, target path, what would be lost).
    pending_salome: &mut Option<(usize, String, Vec<String>)>,
    // Unloading is deferred to the top of the next frame, where everything
    // that addresses a file by index can be fixed up in one place.
    remove_request: &mut Option<usize>,
    data_offset: [f64; 3],
    background: Color,
) {
    panel_section(ui, "Polylines", false, |ui| {
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
                            *remove_request = Some(fi);
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
                            ui.add(egui_macroquad::egui::DragValue::new(&mut layer.width_px)
                                .speed(0.2).range(0.5..=12.0).max_decimals(1).suffix(" px"))
                                .on_hover_text("Drawn width in the 3D view");
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
                    "{} changed on disk since it was loaded. Saving would overwrite \
                     those changes — use Save As to write somewhere else.",
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
fn gcp_section(
    ui: &mut egui_macroquad::egui::Ui,
    gcp_files: &mut Vec<gcp::GcpFile>,
    selected_gcp: &mut Option<(usize, usize)>,
    gcp_load_error: &mut Option<String>,
    background: Color,
) {
    use egui_macroquad::egui;
    panel_section(ui, "Ground Control Points", false, |ui| {
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

/// The file dialog for class definitions, opening beside the current file so a
/// set kept with a project is one click away.
fn class_file_dialog(current: Option<&Path>) -> rfd::FileDialog {
    let mut dialog = rfd::FileDialog::new()
        .add_filter("Class definitions", &["txt"])
        .add_filter("All files", &["*"]);
    match current {
        Some(p) => {
            let p = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            if let Some(dir) = p.parent() { dialog = dialog.set_directory(dir); }
            if let Some(name) = p.file_name() { dialog = dialog.set_file_name(name.to_string_lossy()); }
        }
        None => dialog = dialog.set_file_name(DEFAULT_CLASS_FILE),
    }
    dialog
}

/// Built-in editor for classification labels & colours.
/// Lists every class id up to the maximum present in the loaded data, plus any
/// ids already defined beyond that; missing entries are seeded from the default
/// palette/labels when the window is first opened. Returns `true` when a colour
/// was edited or a file was loaded, i.e. when point colours need a rebuild.
/// Label-only edits update the defs in place and need no rebuild.
///
/// `class_file` is where the definitions came from or were last saved to —
/// what **Save** and **Reload** act on. `None` until one is loaded or saved,
/// in which case **Save** writes `classes.txt` in the launch directory, as it
/// always has.
fn classes_section(
    ui: &mut egui_macroquad::egui::Ui,
    class_defs: &mut Option<ClassDefs>,
    class_vis: &mut ClassVis,
    max_class: u8,
    class_file: &mut Option<PathBuf>,
    save_status: &mut Option<String>,
) -> bool {
    use egui_macroquad::egui;
    let mut colors_changed = false;
    panel_section(ui, "Classes", false, |ui| {
            // Acted on after the buttons, outside the borrow of the definitions.
            let (mut save, mut load): (Option<PathBuf>, Option<PathBuf>) = (None, None);
            {
                let defs = class_defs.get_or_insert_with(ClassDefs::new);
                let mut ids: Vec<u8> = (0..=max_class).collect();
                let mut extra: Vec<u8> = defs.keys().copied().filter(|id| *id > max_class).collect();
                extra.sort();
                ids.extend(extra);
                // All / None / Invert act on the listed ids only. An id that is
                // not listed cannot be switched off here, so leaving
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
                                // ids this panel never shows, and the "N classes
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
                ui.label(match class_file.as_ref() {
                    Some(p) => format!("File: {}", p.display()),
                    None => format!("File: none — Save writes {DEFAULT_CLASS_FILE} where haiviz was started"),
                });
                ui.horizontal(|ui| {
                    let target = class_file.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_CLASS_FILE));
                    if ui.button("Save")
                        .on_hover_text(format!("Write to {}", target.display()))
                        .clicked() {
                        save = Some(target);
                    }
                    if ui.button("Save As…").clicked()
                        && let Some(p) = class_file_dialog(class_file.as_deref()).save_file() {
                        save = Some(p);
                    }
                    if ui.button("Load…")
                        .on_hover_text("Replace these definitions with a file's")
                        .clicked()
                        && let Some(p) = class_file_dialog(class_file.as_deref()).pick_file() {
                        load = Some(p);
                    }
                    if ui.add_enabled(class_file.is_some(), egui::Button::new("Reload"))
                        .on_hover_text("Re-read the file, discarding unsaved edits")
                        .clicked() {
                        load = class_file.clone();
                    }
                });
                if let Some(path) = save {
                    *save_status = Some(match save_class_defs_to(defs, &path) {
                        Ok(()) => {
                            let msg = format!("Saved {} classes to {}", defs.len(), path.display());
                            *class_file = Some(path);
                            msg
                        }
                        Err(e) => format!("Save failed: {e}"),
                    });
                }
            }
            if let Some(path) = load {
                // A file that cannot be read leaves the current definitions as
                // they are, rather than falling back to the built-in defaults.
                *save_status = Some(match load_class_defs_from(&path) {
                    Ok((defs, skipped)) => {
                        let mut msg = format!("Loaded {} classes from {}", defs.len(), path.display());
                        if skipped > 0 {
                            msg.push_str(&format!(" — {skipped} unreadable line{} skipped", if skipped == 1 { "" } else { "s" }));
                        }
                        *class_defs = Some(defs);
                        *class_file = Some(path);
                        colors_changed = true;
                        msg
                    }
                    Err(e) => format!("Load failed: {e}"),
                });
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

/// The seq for a new classification op: past every op in every loaded file.
///
/// Each file's sidecar numbers its own ops, so on their own the counters say
/// nothing about order across files — a file loaded with 25 ops would outrank
/// the op just drawn on a file with 3, and `undo_last_class_op` would take
/// back the wrong one. Drawing from one counter makes the newest op the
/// highest seq everywhere, and gives an op that spans files the same seq in
/// each, so it is undone from all of them together.
fn next_class_seq(stages: &[Stage], copc_stages: &[CopcStage]) -> u64 {
    stages.iter().map(|s| s.edits.next_seq())
        .chain(copc_stages.iter().map(|c| c.edits.next_seq()))
        .max()
        .unwrap_or(0)
}

/// Takes back the most recent classification op, from whichever view it was
/// drawn in. An op does not record the classes it overwrote, so undoing means
/// re-deriving from the source: resident points are re-read, and streamed COPC
/// nodes are dropped so they arrive again with the shortened op list applied.
///
/// An op is only recorded against the files it overlaps, so "the last op" is
/// the highest sequence number anywhere, and it is removed from exactly the
/// files that carry it. Returns whether there was one, then whether in-memory
/// files and COPC files were touched — the section view rebuilds its slice
/// from those.
fn undo_last_class_op(
    stages: &mut [Stage],
    copc_stages: &mut [CopcStage],
    color_mode: ColorMode,
) -> (bool, bool, bool) {
    let last_seq = stages.iter().filter_map(|s| s.edits.ops.last().map(|o| o.seq))
        .chain(copc_stages.iter().filter_map(|c| c.edits.ops.last().map(|o| o.seq)))
        .max();
    let Some(seq) = last_seq else { return (false, false, false) };
    let mut stages_touched = false;
    let mut copc_touched = false;
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
    (true, stages_touched, copc_touched)
}

/// Controls for the plan-view lasso tools, shared by the three 3D panels.
/// Returns the flavour of lasso the operator asked to start, if any.
#[allow(clippy::too_many_arguments)]
fn ground_tool_controls(
    ui: &mut egui_macroquad::egui::Ui,
    app_mode: &mut AppMode,
    lasso_len: usize,
    kind: LassoKind,
    target: &mut u8,
    from: &mut Option<u8>,
    offset: &mut f32,
    max_class: u8,
    class_defs: &Option<ClassDefs>,
    hidden_classes: usize,
    status: &Option<String>,
    // Ops in the most-edited file, which is what the undo button can reach.
    op_count: usize,
    undo: &mut bool,
    apply: &mut bool,
    cancel: &mut bool,
) -> Option<LassoKind> {
    use egui_macroquad::egui;
    let drawing = *app_mode == AppMode::GroundLasso;
    let mut start = None;
    ui.separator();
    ui.label("Classify by outline:");
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
        ui.horizontal(|ui| {
            if ui.button("Above-ground lasso (G)").clicked() {
                start = Some(LassoKind::AboveGround);
            }
            if ui.button("Inside lasso (V)").clicked() {
                start = Some(LassoKind::Inside);
            }
        });
        // From any angle — but an outline selects through the whole depth of
        // the view, which is worth saying before someone expects it to take
        // only the surface they can see.
        ui.label("An outline selects everything behind it, through the whole");
        ui.label("depth of the view, within the height filter.");
        // The same op list the section view's button works on, so a lasso
        // drawn in either view can be taken back from either.
        if ui.add_enabled(op_count > 0, egui::Button::new("Undo last op"))
            .on_hover_text("Take back the most recent classification, from this view or a section")
            .clicked() {
            *undo = true;
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
fn export_section(
    ui: &mut egui_macroquad::egui::Ui,
    open: &mut bool,
    filter: &mut crate::classify::ExportFilter,
    job: &mut Option<crate::export::ExportJob>,
    status: &mut Option<String>,
    sources: &[(String, bool)],
    max_class: u8,
    class_defs: &Option<ClassDefs>,
) -> Option<usize> {
    let mut requested = None;
    // The Export button elsewhere in the panel reveals this section once; the
    // header owns it from then on.
    let force = if *open { Some(true) } else { None };
    *open = false;
    panel_section_forced(ui, "Export", false, force, |ui| {
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
    layers: Option<(Vec3, Vec3)>,
) -> Option<(Vec3, Vec3)> {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut any = false;
    // Meshes and splats, already in viewer space.
    if let Some((l, h)) = layers {
        lo = l.to_array();
        hi = h.to_array();
        any = true;
    }
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
    // A tileset or splat file on the command line goes through the same
    // queue as a dropped one, in the first frame; the startup modes below
    // are all point-cloud modes.
    let startup_layer: Option<String> = input_path
        .filter(|p| !matches!(layers::classify(Path::new(p)), layers::LoadKind::PointCloud))
        .map(str::to_string);
    let input_path = if startup_layer.is_some() { None } else { input_path };
    if input_path.is_none() && startup_layer.is_none() {
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
    // The file the definitions came from, for Save and Reload.
    let mut class_file: Option<PathBuf> = class_defs.as_ref().map(|_| PathBuf::from(DEFAULT_CLASS_FILE));
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

    let mut yaw: f32 = std::f32::consts::FRAC_PI_2; //std::f32::consts::FRAC_PI_4;       // 45°
    let mut pitch: f32 = -std::f32::consts::FRAC_PI_2;    // -90° (true top-down)

    // Camera right/up from yaw. Right is horizontal and perpendicular to the
    // yaw heading at every pitch, so it is taken from yaw directly rather than
    // from `front × Z`, which degenerates looking straight down or up.
    // The old special case for that swapped in a reference whose sign was
    // wrong looking up: orbiting towards the bottom view spun the picture 180°
    // in the last half degree.
    let recompute_right_up = |yaw: f32, _pitch: f32, front: Vec3| -> (Vec3, Vec3) {
        let r = vec3(yaw.sin(), -yaw.cos(), 0.0);
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
    // A drag that began over the UI stays the UI's until the button comes up.
    //
    // The per-frame flag alone is not enough: drag a slider out over the
    // viewport and the pointer is no longer over an egui area, so the guard
    // releases mid-gesture and the camera starts orbiting under the cursor.
    // Latching on the press is what makes the grab last as long as the drag.
    let mut drag_owned_by_ui = false;
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
    let mut ground_lasso: Vec<glam::Vec3> = Vec::new();
    // The camera the lasso is being drawn through — screen axes, and whether it
    // is exactly top-down — taken at the first vertex. The camera cannot rotate
    // while the lasso tool is up, but the op must describe the view the outline
    // was drawn in, not whatever the view is when Enter is pressed.
    let mut ground_lasso_view: Option<((Vec3, Vec3), bool)> = None;
    let mut ground_offset: f32 = 0.20;
    let mut ground_status: Option<String> = None;
    // Which flavour of plan-view lasso is being drawn.
    let mut lasso_kind = LassoKind::AboveGround;
    // Classes omitted from an export — the noise-filtering half of the workflow.
    let mut export_filter = crate::classify::ExportFilter::default();
    let mut export_job: Option<crate::export::ExportJob> = None;
    let mut export_status: Option<String> = None;
    // Set by the Export button to reveal the Export section once.
    let mut reveal_export = false;
    // Set once a corridor has been pinned, so leaving the section releases it.
    let mut corridor_pinned = false;
    let mut corridor_status: Option<String> = None;
    // Where to land once the corridor has streamed in: the lasso tool if the
    // operator asked for it, or plain section view after an undo.
    let mut corridor_then = CorridorThen::View;
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
    // Width the docked panel actually took, in window pixels. Measured inside
    // the egui pass, which runs after the view is rendered, so it lands one
    // frame later — invisible except while dragging the panel edge.
    let mut panel_px: f32 = 0.0;
    let mut digitise = vectorise::Digitise::default();
    // Set by the panel or by `P`, acted on once below rather than in each of
    // the three per-mode blocks.
    let mut digitise_request: Option<vectorise::Request> = None;
    // Vertex editing, whose requests are likewise acted on in one place.
    let mut editor = vector_edit::Editor::default();
    let mut edit_request: Option<vector_edit::Request> = None;
    // A polyline file the panel asked to unload, applied at the top of the
    // next frame like the cloud unloads.
    let mut poly_remove_request: Option<usize> = None;
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
    let mut heatmap_renderer = {
        let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
        HeatmapRenderer::new(ctx)
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
    let mut pending_loads: Vec<String> = startup_layer.into_iter().collect();
    // 3D Tiles tilesets and Gaussian splats, drawn with the clouds.
    let mut layers = layers::Layers::default();
    // Set once the layers have been drawn this frame. With both LAZ and COPC
    // files loaded two render blocks run, and the layers belong to the first.
    let mut layers_drawn;

    // Windows-only: number of entries already taken from miniquad's drop list
    // (the list only grows on Windows, so we track how many we've consumed).
    #[cfg(target_os = "windows")]
    let mut last_dropped_count: usize = 0;

    // egui-macroquad divides every wheel delta by 90 before handing it to egui
    // as lines, and miniquad's X11 backend reports one per notch — so at egui's
    // default speed a notch scrolls the panel less than a point. Undo the
    // division to get egui's native 40 points a notch.
    egui_macroquad::cfg(|ctx| ctx.options_mut(|o| o.line_scroll_speed = 40.0 * 90.0));

    loop {
        //println!("{} {}",yaw,pitch);
        //println!("{}",app_mode.label());

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
            // What the file is decides its loader; its world box pins the
            // frame if it is the first thing loaded.
            let (source, lo, hi) = match layers::classify(Path::new(&path_str)) {
                layers::LoadKind::PointCloud => {
                    let header = match las::Reader::from_path(&path_str) {
                        Ok(r) => r.header().clone(),
                        Err(e) => { error!("Cannot open file {}: {}", path_str, e); continue; }
                    };
                    let b = header.bounds();
                    (LoadSource::Cloud { copc: header.copc_info_vlr().is_some() }, [b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z])
                }
                kind => match layers::prepare(kind) {
                    Ok(p) => {
                        let (lo, hi) = p.world_bounds();
                        (LoadSource::Layer(p), lo, hi)
                    }
                    Err(e) => {
                        error!("Cannot load {}: {}", path_str, e);
                        layers.status = Some(e);
                        continue;
                    }
                },
            };
            // A drop into a previously-empty viewer should also re-fit the camera;
            // otherwise the default startup framing (sized for the 100×100 placeholder
            // extents) leaves the new data way off-screen until the user clicks Zoom All.
            let was_empty = pinned_viewer_origin.is_none();
            // Seed the session's viewer origin from the first load; reuse
            // afterwards so every stage lives in the same coordinate space.
            let origin = pinned_viewer_origin.unwrap_or_else(|| {
                let o = lo;
                info!("Pinning viewer origin to first loaded file: {:?}", o);
                let dx = (hi[0] - lo[0]) as f32;
                let dy = (hi[1] - lo[1]) as f32;
                let dz = (hi[2] - lo[2]) as f32;
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
            let is_copc = match source {
                LoadSource::Layer(layer) => {
                    layers.attach(ctx, layer, origin, front);
                    info!("Loaded {}", path_str);
                    None
                }
                LoadSource::Cloud { copc } => Some(copc),
            };
            if is_copc == Some(true) {
                let mut cs = CopcStage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                cs.color_h_min = height_color_min;
                cs.color_h_max = height_color_max;
                cs.rebuild_colors(ctx, color_mode);
                info!("Loaded COPC file: {}", path_str);
                copc_stages.push(cs);
            } else if is_copc == Some(false) {
                let mut s = Stage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                s.color_h_min = height_color_min;
                s.color_h_max = height_color_max;
                s.rebuild_colors(ctx, color_mode);
                info!("Loaded LAZ file: {}", path_str);
                stages.push(s);
            }
            if was_empty {
                // Framed into the space beside the panel. This runs before `view_rect` is
                // built for the frame, so the width is taken directly.
                let aspect = (screen_width() - panel_px).max(32.0) / screen_height().max(1.0);
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
        if let Some(fi) = poly_remove_request.take()
            && fi < poly_files.len()
        {
            poly_files.remove(fi);
            // Everything that addresses a file by index shifts down past it.
            digitise.file_removed(fi);
            editor.file_removed(fi);
            selected_poly = match selected_poly {
                Some(r) if r.file == fi => None,
                Some(r) if r.file > fi => Some(polylines::PolyRef { file: r.file - 1, ..r }),
                s => s,
            };
            pending_salome = match pending_salome.take() {
                Some((f, _, _)) if f == fi => None,
                Some((f, p, l)) if f > fi => Some((f - 1, p, l)),
                s => s,
            };
        }
        {
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            layers.apply_unload(ctx);
        }
        layers_drawn = false;

        // If everything has been unloaded, forget the pinned coordinate frame and
        // reset the placeholder extents. The next load then re-pins the viewer origin
        // to the new data's location and re-fits the camera (the `was_empty` path),
        // so loading clouds in a different place frames correctly.
        if stages.is_empty() && copc_stages.is_empty() && tile_mgr.is_none() && layers.is_empty() {
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

        // The views render beside the panel, not under it. Computed once here
        // because the section view and the 3D view both need it and the
        // section view runs first.
        let view_rect = ViewRect {
            x: panel_px.clamp(0.0, (screen_width() - 32.0).max(0.0)),
            y: 0.0,
            w: (screen_width() - panel_px).max(32.0),
            h: screen_height().max(1.0),
        };
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
            } else if app_mode == AppMode::EditLines {
                // One layer at a time: a drag in progress, then the selected
                // vertex, then the mode itself.
                if editor.drag.take().is_some() {
                    digitise.hover = None;
                } else if editor.vertex.is_some() {
                    editor.vertex = None;
                } else {
                    editor.hover = None;
                    app_mode = AppMode::View3D;
                }
            } else if app_mode == AppMode::SectionDigitise {
                if digitise.is_drawing() { digitise.clear(); } else { app_mode = AppMode::SectionView; }
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
        if matches!(app_mode, AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso | AppMode::SectionCorridorLoad | AppMode::SectionDigitise) {
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
                let units_per_pixel = section_zoom * 2.0 / view_rect.h.max(1.0);
                section_pan.x -= sec_mouse_delta.x * units_per_pixel;
                section_pan.y += sec_mouse_delta.y * units_per_pixel;
            }

            // Over the panel the wheel scrolls it, and must not zoom as well.
            let (_mwx, mwy) = if egui_wants_pointer_prev { (0.0, 0.0) } else { mouse_wheel() };
            let ctrl_down = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
            if mwy.abs() > 0.001 && ctrl_down && app_mode == AppMode::SectionDigitise {
                // The snap radius, as in the 3D view; the plain wheel keeps zooming.
                let step = if mwy > 0.0 { 1.25 } else { 1.0 / 1.25 };
                digitise.radius = (digitise.radius * step)
                    .clamp(vectorise::MIN_SNAP_RADIUS, vectorise::MAX_SNAP_RADIUS);
                digitise.status = Some(format!("Snap radius {:.2} m", digitise.radius));
            } else if mwy.abs() > 0.001 {
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
                    app_mode = corridor_then.mode();
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
                let aspect = view_rect.aspect();
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
                    let vp = view_rect.gl();
                    gl.quad_context.apply_viewport(vp.0, vp.1, vp.2, vp.3);
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
                    view_rect.x + (wx - (section_pan.x - half_w)) / (2.0 * half_w) * view_rect.w
                };
                let world_to_screen_y = |wy: f32| -> f32 {
                    view_rect.y + (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * view_rect.h
                };

                let ground_sy = world_to_screen_y(0.0);
                if ground_sy > view_rect.y && ground_sy < view_rect.y + view_rect.h {
                    draw_line(view_rect.x, ground_sy, view_rect.x + view_rect.w, ground_sy, 1.0, Color::new(0.4, 0.4, 0.4, 0.5));
                    draw_text("0m", 5.0, ground_sy - 3.0, 16.0, GRAY);
                }

                let sx0 = world_to_screen_x(0.0);
                let sx1 = world_to_screen_x(sec.length);
                draw_line(sx0, view_rect.y, sx0, view_rect.y + view_rect.h, 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_line(sx1, view_rect.y, sx1, view_rect.y + view_rect.h, 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_text("start", sx0 + 3.0, 30.0, 16.0, YELLOW);
                draw_text(&format!("{:.1}m", sec.length), sx1 + 3.0, 30.0, 16.0, YELLOW);

                if !poly_files.is_empty() &&
                    let (Some(start), Some(end)) = (section_start, section_end) {
                        polylines::draw_section(
                            &poly_files, start, end,
                            section_depth,
                            section_pan, section_zoom,
                            (view_rect.x, view_rect.y, view_rect.w, view_rect.h),
                        );

                }

                if !gcp_files.is_empty() &&
                    let (Some(start), Some(end)) = (section_start, section_end) {
                        gcp::draw_section(
                            &gcp_files, data_offset, start, end,
                            section_depth, sec.length,
                            section_pan, section_zoom,
                            selected_gcp,
                            (view_rect.x, view_rect.y, view_rect.w, view_rect.h),
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

                // ---- Digitising in the section ----
                if let (Some(start), Some(end)) = (section_start, section_end)
                    && (app_mode == AppMode::SectionDigitise || digitise.is_drawing()) {
                    let (mx, my) = mouse_position();
                    let in_view = mx >= view_rect.x;
                    // Resolved here rather than with the click, because the
                    // marker is drawn now; the click further down places
                    // whatever this found.
                    digitise.hover = if app_mode == AppMode::SectionDigitise && in_view {
                        let (ndc_x, ndc_y) = view_rect.ndc(mx, my);
                        Some(resolve_section_snap(
                            sec, &poly_files, start, end, section_depth,
                            section_pan.x + ndc_x * half_w, section_pan.y + ndc_y * half_h,
                            data_offset, &digitise,
                        ))
                    } else {
                        None
                    };
                    let (dx, dy) = (end.0 - start.0, end.1 - start.1);
                    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
                    vectorise::draw_section_overlay(
                        &digitise, start, (dx / len, dy / len), data_offset,
                        |a, h| (world_to_screen_x(a), world_to_screen_y(h)),
                        view_rect.h / (2.0 * half_h),
                    );
                    if app_mode == AppMode::SectionDigitise && in_view {
                        vectorise::draw_cursor_readout(&digitise, mx, my);
                    }
                }
            }

            // Corridor force-load banner. This is the one moment where the
            // viewer deliberately ignores the LOD point budget, and the
            // operator is waiting on it, so say what is happening.
            if let Some(msg) = &corridor_status {
                let font_size = 20.0;
                let dims = measure_text(msg, None, font_size as u16, 1.0);
                // Centred in the view, not the window — half of it would sit
                // under the panel otherwise.
                let (bx, by) = (
                    view_rect.x + (view_rect.w - dims.width) * 0.5,
                    view_rect.y + view_rect.h * 0.5,
                );
                draw_rectangle(bx - 14.0, by - dims.height - 10.0,
                    dims.width + 28.0, dims.height + 22.0, Color::new(0.0, 0.0, 0.0, 0.72));
                draw_text(msg, bx, by, font_size, YELLOW);
                let hint = "Esc to cancel";
                let hd = measure_text(hint, None, 16, 1.0);
                draw_text(hint, view_rect.x + (view_rect.w - hd.width) * 0.5, by + 22.0, 16.0, GRAY);
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
            let mut start_section_digitise = false;
            let mut sec_digitise_request: Option<vectorise::Request> = None;
            let have_cloud = !stages.is_empty() || !copc_stages.is_empty() || tile_mgr.is_some();
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
                egui_macroquad::egui::SidePanel::left("haiviz_panel")
                    .resizable(true)
                    .default_width(340.0)
                    .show(egui_ctx, |ui| {
                    egui_macroquad::egui::ScrollArea::vertical().show(ui, |ui| {
                panel_section(ui, "Controls", true, |ui| {
                        if ui.button("Back to 3D").clicked() {
                            app_mode = AppMode::View3D;
                            set_section(&mut section_data, &mut section_trash, None);
                            sec_measure_a = None;
                            sec_measure_b = None;
                            sec_picked_point = None;
                            lasso.clear();
                            digitise.clear();
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
                class_colors_changed = classes_section(ui, &mut class_defs, &mut class_vis, max_class, &mut class_file, &mut classes_save_status);
                panel_section(ui, "Classification", false, |ui| {
                    export_request = export_section(
                        ui, &mut reveal_export, &mut export_filter,
                        &mut export_job, &mut export_status, &export_sources,
                        max_class, &class_defs,
                    );
                });
                panel_section(ui, "Digitising", false, |ui| {
                    if let Some(r) = vectorise::controls(
                        ui, &mut digitise, &poly_files,
                        app_mode == AppMode::SectionDigitise, have_cloud, false,
                    ) { sec_digitise_request = Some(r); }
                    ui.label("In a section, vertices land on the section plane.");
                    ui.label("Editing existing lines is in the 3D view.");
                });
                polylines_section(ui, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, &mut poly_remove_request, data_offset, background);
                gcp_section(ui, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
                    });
                });
                // Read after the panel is laid out, not before.
                // `is_pointer_over_area` treats a SidePanel as background and falls
                // back to egui's unused_rect, which still covers the whole screen
                // until a panel has shrunk it — so asking first says the pointer is
                // over the viewport even when it is on a button, and the click both
                // presses the button and drops a vertex.
                egui_wants_sec_pointer = egui_ctx.is_pointer_over_area() || egui_ctx.wants_pointer_input();
                egui_wants_sec_keyboard = egui_ctx.wants_keyboard_input();
                // What the panel actually took, so the view renders beside it.
                panel_px = egui_ctx.available_rect().min.x * egui_ctx.pixels_per_point();
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
            // The section view never reaches the 3D path that records this, and
            // its wheel handling reads it before the panel is laid out.
            egui_wants_pointer_prev = egui_wants_sec_pointer;

            // ---- Lasso keyboard shortcuts ----
            if !egui_wants_sec_keyboard {
                if app_mode == AppMode::SectionLasso {
                    if is_key_pressed(KeyCode::Backspace) { lasso.pop(); }
                    if is_key_pressed(KeyCode::Enter) && lasso.len() >= 3 { do_apply_lasso = true; }
                } else if app_mode == AppMode::SectionDigitise {
                    // The same keys as digitising in 3D.
                    let shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
                    if is_key_pressed(KeyCode::N) {
                        digitise.snap_mode = digitise.snap_mode.next();
                        digitise.status = Some(format!("Snap: {}", digitise.snap_mode.label()));
                    }
                    if is_key_pressed(KeyCode::Backspace) { digitise.pop(); }
                    if is_key_pressed(KeyCode::Enter) {
                        sec_digitise_request = Some(if shift {
                            vectorise::Request::FinishRing
                        } else {
                            vectorise::Request::Finish
                        });
                    }
                    if is_key_pressed(KeyCode::P) {
                        sec_digitise_request = Some(vectorise::Request::Cancel);
                    }
                } else if app_mode == AppMode::SectionView && is_key_pressed(KeyCode::P) {
                    sec_digitise_request = Some(vectorise::Request::Start);
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

            // ---- Digitising requests from the section view ----
            match sec_digitise_request.take() {
                Some(vectorise::Request::Start)
                    if app_mode != AppMode::SectionCorridorLoad
                        && ensure_digitise_target(&mut digitise, &mut poly_files, &stages, &copc_stages, &tile_mgr, data_offset) => {
                    digitise.clear();
                    lasso.clear();
                    start_section_digitise = true;
                }
                Some(vectorise::Request::Start) => {}
                Some(vectorise::Request::Cancel) => {
                    digitise.clear();
                    app_mode = AppMode::SectionView;
                }
                Some(vectorise::Request::Finish) =>
                    finish_digitised_line(&mut digitise, &mut editor, &mut poly_files, false),
                Some(vectorise::Request::FinishRing) =>
                    finish_digitised_line(&mut digitise, &mut editor, &mut poly_files, true),
                Some(vectorise::Request::UndoVertex) => { digitise.pop(); }
                None => {}
            }

            // ---- Starting the lasso pulls in the full-resolution corridor ----
            // Digitising in the section needs the same, for the same reason: a
            // snap over the thinned LOD sample would be quietly wrong.
            // A section cut for browsing shows the resident LOD, which is a
            // thinned sample. Classifying against that would mark the sampled
            // points and silently leave the dense remainder alone, so the
            // corridor is force-loaded here — once per section, not on every
            // cut — and the slice re-cut from it before any lasso is drawn.
            let corridor_for = if start_lasso {
                Some(CorridorThen::Lasso)
            } else if start_section_digitise {
                Some(CorridorThen::Digitise)
            } else {
                None
            };
            if let Some(then) = corridor_for {
                if copc_stages.is_empty() || corridor_pinned {
                    app_mode = then.mode();
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
                        let msg = format!(
                            "Corridor holds {:.1}M points (limit {:.0}M) — shorten the \
                             section line or reduce the depth before {}.",
                            pts as f64 / 1e6, CORRIDOR_POINT_LIMIT as f64 / 1e6, then.purpose(),
                        );
                        if then == CorridorThen::Digitise {
                            digitise.status = Some(msg);
                        } else {
                            classify_status = Some(msg);
                        }
                    } else {
                        info!(
                            "Section corridor: {} nodes, {} points across {} file(s)",
                            nodes, pts, copc_stages.len(),
                        );
                        corridor_total_points = pts;
                        corridor_pinned = true;
                        corridor_then = then;
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

                    // One seq for every file the op lands in; see `next_class_seq`.
                    let seq = next_class_seq(&stages, &copc_stages);
                    let mut changed = 0usize;
                    let mut touched = 0usize;
                    {
                        let mut gl = unsafe { get_internal_gl() };
                        gl.flush();
                        for st in stages.iter_mut() {
                            if !op.intersects_aabb(&st.cloud.min, &st.cloud.max) { continue; }
                            let from = st.edits.ops.len();
                            st.edits.push_at(op.clone(), seq);
                            changed += st.apply_edits(&mut *gl.quad_context, color_mode, from);
                            touched += 1;
                        }
                        for cs in copc_stages.iter_mut() {
                            if !op.intersects_aabb(&cs.global_min, &cs.global_max) { continue; }
                            let from = cs.edits.ops.len();
                            cs.edits.push_at(op.clone(), seq);
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
            if do_undo_op {
                let (undone, stages_touched, copc_touched) =
                    undo_last_class_op(&mut stages, &mut copc_stages, color_mode);

                classify_status = Some(if undone {
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
                    corridor_then = CorridorThen::View;
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
                let half_h = section_zoom;
                let half_w = half_h * view_rect.aspect();
                let (mx, my) = mouse_position();
                let (ndc_x, ndc_y) = view_rect.ndc(mx, my);
                let wx = section_pan.x + ndc_x * half_w;
                let wy = section_pan.y + ndc_y * half_h;
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
                    AppMode::SectionDigitise => {
                        let free = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
                        if free {
                            // Exactly where clicked, on the plane — the section's
                            // counterpart of the horizontal plane Shift uses in 3D.
                            if let (Some(start), Some(end)) = (section_start, section_end) {
                                let (dx, dy) = (end.0 - start.0, end.1 - start.1);
                                let len = (dx * dx + dy * dy).sqrt();
                                if len > 0.001 {
                                    digitise.push(vectorise::section_to_world(
                                        start, (dx / len, dy / len), wx, wy, data_offset,
                                    ));
                                    digitise.status = None;
                                }
                            }
                        } else if let Some(hit) = digitise.hover.clone() {
                            if hit.is_placeable() {
                                digitise.push(hit.world);
                                digitise.status = None;
                            } else {
                                digitise.status = hit.note;
                            }
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
        let mut undo_class_op = false;
        // Counted as the section view counts it: the ops in the most-edited file.
        let class_op_count: usize = stages.iter().map(|s| s.edits.ops.len())
            .chain(copc_stages.iter().map(|cs| cs.edits.ops.len()))
            .max().unwrap_or(0);
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
                AppMode::View3D | AppMode::PointPick => start_ground_lasso = Some(kind),
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
        let ctrl_down = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
        // Editing snaps a dragged vertex with the same settings, so the same
        // keys adjust them there.
        let snap_tool = matches!(app_mode, AppMode::Digitise | AppMode::EditLines);
        if snap_tool && !egui_wants_keyboard_prev {
            if app_mode == AppMode::Digitise && is_key_pressed(KeyCode::E) {
                digitise.edge.enabled = !digitise.edge.enabled;
                digitise.status = Some(format!(
                    "Edge following {}", if digitise.edge.enabled { "on" } else { "off" },
                ));
            }
            if app_mode == AppMode::Digitise && digitise.edge.enabled && is_key_pressed(KeyCode::H) {
                digitise.edge.show_heatmap = !digitise.edge.show_heatmap;
            }
            if is_key_pressed(KeyCode::N) {
                digitise.snap_mode = digitise.snap_mode.next();
                digitise.status = Some(format!("Snap: {}", digitise.snap_mode.label()));
            }
            // Modifier + wheel, so the plain wheel keeps zooming. A deliberate
            // divergence from `SectionAdjustDepth`, which takes the plain
            // wheel — this is a tool the operator sits in for minutes at a
            // time, and losing zoom for that whole time would be worse.
            let wheel = if egui_wants_pointer_prev { 0.0 } else { mouse_wheel().1 };
            if wheel != 0.0 && (is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl)) {
                let step = if wheel > 0.0 { 1.25 } else { 1.0 / 1.25 };
                digitise.radius = (digitise.radius * step)
                    .clamp(vectorise::MIN_SNAP_RADIUS, vectorise::MAX_SNAP_RADIUS);
                digitise.status = Some(format!("Snap radius {:.2} m", digitise.radius));
            }
        }

        if is_key_pressed(KeyCode::P) && !egui_wants_keyboard_prev {
            match app_mode {
                AppMode::View3D | AppMode::PointPick | AppMode::EditLines =>
                    digitise_request = Some(vectorise::Request::Start),
                AppMode::Digitise => digitise_request = Some(vectorise::Request::Cancel),
                _ => {}
            }
        }
        // `M` does the same for editing. From digitising it switches straight
        // over, but only between lines: a half-drawn line is not something to
        // throw away on a stray key.
        if is_key_pressed(KeyCode::M) && !egui_wants_keyboard_prev {
            match app_mode {
                AppMode::View3D | AppMode::PointPick => edit_request = Some(vector_edit::Request::Start),
                AppMode::Digitise if !digitise.is_drawing() => edit_request = Some(vector_edit::Request::Start),
                AppMode::EditLines => edit_request = Some(vector_edit::Request::Stop),
                _ => {}
            }
        }
        // Ctrl+Z takes back the last committed change — a finished line, a
        // moved vertex. Mid-line it takes back a vertex instead, like Backspace.
        if ctrl_down && is_key_pressed(KeyCode::Z) && !egui_wants_keyboard_prev {
            match app_mode {
                AppMode::Digitise if digitise.is_drawing() => { digitise.pop(); }
                AppMode::Digitise | AppMode::EditLines => edit_request = Some(vector_edit::Request::Undo),
                _ => {}
            }
        }

        // Not while a text field has focus, where Tab moves to the next widget,
        // and not with Alt held, in case the window manager lets an Alt+Tab
        // through. Either would leave the view orbiting under a bare pointer.
        let alt_down = is_key_down(KeyCode::LeftAlt) || is_key_down(KeyCode::RightAlt);
        if is_key_pressed(KeyCode::Tab) && !egui_wants_keyboard_prev && !alt_down {
            grabbed = !grabbed;
            set_cursor_grab(grabbed);
            show_mouse(!grabbed);
        }

        // Camera movement, suppressed while a text field has focus. `W`/`A`/
        // `S`/`D` are obviously ambiguous with typing, but so are the arrows:
        // they move the caret in a line-name or code field, and the view has
        // no business following.
        if !egui_wants_keyboard_prev {
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
        }

        // Over the panel the wheel scrolls it, and must not zoom as well.
        let (_mwx, mwy) = if egui_wants_pointer_prev { (0.0, 0.0) } else { mouse_wheel() };
        if app_mode == AppMode::SectionAdjustDepth {
            if mwy.abs() > 0.001 {
                let factor = if mwy > 0.0 { 1.2 } else { 0.8 };
                section_depth *= factor;
                section_depth = section_depth.clamp(0.1, 500.0);
            }
        } else if mwy.abs() > 0.001 && !(snap_tool && ctrl_down) {
            // Ctrl + wheel is the snap radius while a snapping tool is active,
            // and must not zoom as well.
            let zoom_factor = if mwy > 0.0 { 0.9 } else { 1.1 };
            ortho_scale *= zoom_factor;
            ortho_scale = ortho_scale.clamp(0.5, 10000.0);
        }

        let mouse_pos: Vec2 = mouse_position().into();
        let mouse_delta = mouse_pos - last_mouse_position;
        last_mouse_position = mouse_pos;

        let any_button_down = is_mouse_button_down(MouseButton::Left)
            || is_mouse_button_down(MouseButton::Right)
            || is_mouse_button_down(MouseButton::Middle);
        if !any_button_down {
            drag_owned_by_ui = false;
        } else if (is_mouse_button_pressed(MouseButton::Left)
            || is_mouse_button_pressed(MouseButton::Right)
            || is_mouse_button_pressed(MouseButton::Middle))
            && egui_wants_pointer_prev {
            drag_owned_by_ui = true;
        }
        let ui_has_pointer = egui_wants_pointer_prev || drag_owned_by_ui;

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
            AppMode::SectionAdjustDepth | AppMode::GroundLasso | AppMode::Digitise | AppMode::EditLines |
            AppMode::PointPick | AppMode::MeasureFirst | AppMode::MeasureSecond
        );
        pivot_locked = false;

        if (grabbed || is_mouse_button_down(MouseButton::Left)) && !mode_uses_click && !ui_has_pointer {

            pivot_locked = true;

            let old_front = front;
            yaw += mouse_delta.x * -LOOK_SPEED;
            pitch += mouse_delta.y * -LOOK_SPEED;
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

        if is_mouse_button_down(MouseButton::Right) && !ui_has_pointer {
            // 1:1 with the cursor — the point under the pointer stays under it.
            // Any multiplier here makes the cloud outrun the mouse and shoot
            // off screen on a short drag.
            let units_per_pixel = ortho_scale * 2.0 / view_rect.h.max(1.0);
            position -= right * mouse_delta.x * units_per_pixel;
            position += up * mouse_delta.y * units_per_pixel;
        }

        clear_background(background);

        let camera = Camera3D {
            position,
            up,
            // Along `front`, not at the pivot. The pivot is where the centre
            // ray meets the mid-height plane, and that can lie *behind* the
            // camera — pan a side view down below mid-height, tilt to look
            // down — and a look-at target behind the camera reverses the view.
            target: position + front,
            projection: Projection::Orthographics,
            fovy: ortho_scale * 2.0,
            // Both are needed: `aspect` shapes the projection for the narrower
            // rect, `viewport` puts macroquad's own drawing inside it.
            aspect: Some(view_rect.aspect()),
            viewport: Some(view_rect.gl()),
            z_far: 100000.0,
            z_near: -100000.0,
            ..Default::default()
        };
        set_camera(&camera);
        let view_proj: glam::Mat4 = camera.matrix();
        //println!("{}",view_proj);
        if !layers.is_empty() {
            // Tile selection is in physical pixels, the ones the screen-space
            // error is actually seen in.
            let mpp_phys = ortho_scale * 2.0 / (view_rect.gl().3.max(1) as f32);
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            layers.update(ctx, &view_proj, front, mpp_phys);
        }

        let (mx, my) = macroquad::input::mouse_position();
        let cursor_world = screen_to_world_xy(mx, my, &view_proj, mid_z, view_rect);
        // World units one screen pixel spans, under the orthographic camera.
        // Turns a pixel width into the world-space geometry that carries it.
        let metres_per_pixel = ortho_scale * 2.0 / view_rect.h.max(1.0);

        if app_mode == AppMode::GroundLasso {
            if !ui_has_pointer && is_mouse_button_pressed(MouseButton::Left) {
                // On the view plane through the orbit pivot. Any depth along
                // the line of sight would select the same prism; this one keeps
                // the vertex under the cursor through a pan or a zoom.
                let (o, d) = cursor_ray(&view_proj, mx, my, view_rect);
                let (o, d) = (Vec3::from(o), Vec3::from(d));
                if ground_lasso.is_empty() {
                    ground_lasso_view = Some((view_axes(&view_proj), is_top_view(pitch)));
                }
                ground_lasso.push(o + d * (orbit_pivot - o).dot(d));
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
            if !ui_has_pointer && is_mouse_button_pressed(MouseButton::Left) {
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
                    if let Some((wx, wy)) = screen_to_world_xy(mx, my, &view_proj, z_view, view_rect) {
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
                        &view_proj, mx, my, view_rect, data_offset, &digitise, digitise.last(), None,
                    );
                    if hit.is_placeable() {
                        // With edge following on, the click commits the traced
                        // route to the snap rather than a straight segment —
                        // from the trace grown at this same anchor, so what is
                        // placed is what the ghost path showed.
                        let traced = digitise.edge.trace.as_ref()
                            .filter(|t| digitise.edge.enabled && Some(t.anchor) == digitise.last())
                            .and_then(|t| t.route(hit.world, edge_z_source(&digitise)))
                            .filter(|route| route.len() > 2);
                        match traced {
                            Some(route) => match drape_route(
                                &stages, &copc_stages, &tile_mgr, &route, data_offset, &digitise,
                            ) {
                                Ok(route) => {
                                    digitise.push_click(&route[1..]);
                                    digitise.status = None;
                                }
                                Err(e) => digitise.status = Some(e),
                            },
                            None => {
                                digitise.push(hit.world);
                                digitise.status = None;
                            }
                        }
                    } else {
                        digitise.status = hit.note.clone();
                    }
                }
            }
            if !egui_wants_keyboard_prev {
                if is_key_pressed(KeyCode::Backspace) { digitise.pop(); }
                if is_key_pressed(KeyCode::Enter) {
                    // Shift+Enter closes the line into a ring, for outlines.
                    let shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
                    digitise_request = Some(if shift {
                        vectorise::Request::FinishRing
                    } else {
                        vectorise::Request::Finish
                    });
                }
            }
        }

        // Where the next vertex would land. Resolved once per frame rather
        // than per render block, and only while digitising — it is a pick over
        // the resident points, not something to spend on every frame of
        // ordinary navigation.
        let have_cloud = !stages.is_empty() || !copc_stages.is_empty() || tile_mgr.is_some();
        digitise.hover = if app_mode == AppMode::Digitise && !ui_has_pointer {
            Some(resolve_snap(
                &stages, &copc_stages, &tile_mgr, &poly_files,
                &view_proj, mx, my, view_rect, data_offset, &digitise, digitise.last(), None,
            ))
        } else {
            None
        };
        // ---- Editing ----
        // Handled once here too. A press on a handle only becomes a change
        // when the button comes up after a drag, so a drag is one undo step
        // and one save however many frames it took.
        if app_mode == AppMode::EditLines {
            if let Some(drag) = editor.drag.as_mut() { drag.track(mx, my); }
            if let Some((handle, origin)) = editor.drag.as_ref()
                .filter(|d| d.moving)
                .map(|d| (d.handle, d.origin)) {
                // Over the panel there is nothing to place onto, and releasing
                // there cancels rather than committing wherever the pointer
                // last was in the view.
                digitise.hover = if mx < view_rect.x {
                    None
                } else if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
                    // Free placement, at the height the vertex had when grabbed.
                    let z_view = (origin[2] - data_offset[2]) as f32;
                    screen_to_world_xy(mx, my, &view_proj, z_view, view_rect).map(|(wx, wy)| {
                        vectorise::SnapHit::good([
                            wx as f64 + data_offset[0], wy as f64 + data_offset[1], origin[2],
                        ])
                    })
                } else {
                    Some(resolve_snap(
                        &stages, &copc_stages, &tile_mgr, &poly_files, &view_proj, mx, my,
                        view_rect, data_offset, &digitise, Some(origin), Some(handle.exclude()),
                    ))
                };
            }

            if editor.drag.is_some() && !is_mouse_button_down(MouseButton::Left) {
                if let Some(drag) = editor.drag.take() {
                    let line = drag.handle.line();
                    if !drag.moving {
                        // A click: select, change nothing.
                        selected_poly = Some(line);
                        editor.vertex = match drag.handle {
                            vector_edit::Handle::Vertex { idx, .. } => Some((line, idx)),
                            vector_edit::Handle::Segment { .. } => None,
                        };
                        editor.status = None;
                    } else {
                        editor.status = Some(match digitise.hover.take() {
                            Some(hit) if hit.is_placeable() => {
                                let what = match drag.handle {
                                    vector_edit::Handle::Vertex { idx, .. } => format!("Vertex {} moved.", idx + 1),
                                    vector_edit::Handle::Segment { i, .. } => format!("Vertex {} added.", i + 2),
                                };
                                match editor.commit_drag(&mut poly_files, drag.handle, hit.world) {
                                    Some(fi) => {
                                        selected_poly = Some(line);
                                        saved_status(what, autosave_vectors(&mut poly_files[fi]))
                                    }
                                    None => "That line is no longer loaded.".to_string(),
                                }
                            }
                            // Never somewhere else instead: a refused snap on
                            // release leaves the line exactly as it was.
                            Some(hit) => format!("Not moved: {}", hit.note.unwrap_or_default()),
                            None => "Not moved: released outside the view.".to_string(),
                        });
                    }
                }
            } else if editor.drag.is_none() {
                editor.hover = if ui_has_pointer {
                    None
                } else {
                    vector_edit::hit_test(
                        &poly_files, &view_proj,
                        mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h,
                    )
                };
                if !ui_has_pointer && is_mouse_button_pressed(MouseButton::Left) {
                    match editor.hover.and_then(|h| h.world(&poly_files).map(|w| (h, w))) {
                        Some((handle, origin)) => {
                            editor.drag = Some(vector_edit::Drag {
                                handle, press: (mx, my), moving: false, origin,
                            });
                        }
                        None => {
                            selected_poly = None;
                            editor.vertex = None;
                        }
                    }
                }
            }

            if !egui_wants_keyboard_prev {
                let delete = is_key_pressed(KeyCode::Delete);
                if (delete || is_key_pressed(KeyCode::Backspace)) && editor.vertex.is_some() {
                    edit_request = Some(vector_edit::Request::DeleteVertex);
                } else if delete && selected_poly.is_some() {
                    edit_request = Some(vector_edit::Request::DeleteLine);
                }
            }
        }

        // The rubber band follows a placeable snap only, so it never draws to
        // somewhere a click would refuse to go.
        let digitise_cursor = digitise.hover.as_ref()
            .filter(|h| h.is_placeable())
            .map(|h| h.world);

        // ---- Edge following ----
        // Grown once per anchor; after that the route to the cursor is a walk
        // back along the parent pointers. 3D view only (decision 8), and only
        // from the second vertex on — the first has nothing to trace from.
        let edge_route: Option<Vec<[f64; 3]>> = match digitise.last() {
            Some(anchor) if app_mode == AppMode::Digitise && digitise.edge.enabled => {
                let (lo, hi) = digitise.edge.window_box(anchor, data_offset);
                // Anything that changes which points the window holds: COPC
                // nodes arriving, a tile loading, the height band, and the
                // visible classes — hiding a class to steer the tracer has to
                // reach the raster.
                let resident = stages.iter().filter(|s| s.visible).map(|s| s.resident_in_box(lo, hi)).sum::<usize>()
                    + copc_stages.iter().filter(|c| c.visible).map(|c| c.resident_in_box(lo, hi)).sum::<usize>()
                    + tile_mgr.as_ref().map_or(0, |m| m.resident_in_box(lo, hi));
                let data_sig = resident ^ filter_signature(z_clip, class_vis);
                digitise.edge.update(anchor, data_offset, data_sig, get_time(), |(lo, hi)| {
                    let mut samples = Vec::new();
                    for st in stages.iter().filter(|s| s.visible) { st.edge_samples(lo, hi, &mut samples); }
                    for cs in copc_stages.iter().filter(|c| c.visible) { cs.edge_samples(lo, hi, &mut samples); }
                    if let Some(mgr) = &tile_mgr { mgr.edge_samples(lo, hi, &mut samples); }
                    samples
                });
                let z = edge_z_source(&digitise);
                digitise.edge.probe = digitise_cursor.and_then(|target| digitise.edge.trace.as_ref()?.probe(target));
                digitise_cursor.and_then(|target| digitise.edge.trace.as_ref()?.route(target, z))
            }
            _ => {
                // Nothing to trace from: let the raster go rather than hold a
                // few megabytes for a line that is not being drawn.
                if !(digitise.edge.enabled && digitise.is_drawing()) { digitise.edge.clear(); }
                None
            }
        };

        // ---- COPC neighbourhood pinning ----
        // The spacing guard alone would only ever say "no". This is the other
        // half: force-load the octree nodes around the cursor so there is
        // full-resolution data to snap to in the first place. Skipped while a
        // section corridor owns the pins — the two use the same set, and the
        // corridor is the one the operator is waiting on.
        if snap_tool && !copc_stages.is_empty() && !corridor_pinned {
            let centre = digitise.hover.as_ref()
                .map(|h| ((h.world[0] - data_offset[0]) as f32, (h.world[1] - data_offset[1]) as f32))
                .or(cursor_world);
            if let Some((cx, cy)) = centre {
                let mut side = (4.0 * digitise.radius as f64).max(SNAP_NEIGHBOURHOOD_M);
                // While tracing, the raster window wants full-resolution data
                // too, not only the snap radius under the cursor.
                if app_mode == AppMode::Digitise && digitise.edge.enabled {
                    side = side.max(digitise.edge.params.window_m as f64);
                }
                let half = side / 2.0;
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
        // Also the block for meshes and splats loaded on their own: it is the
        // one with no cloud of its own to require.
        if !stages.is_empty() || (copc_stages.is_empty() && tile_mgr.is_none() && !layers.is_empty()) {
            // Render points across all stages (with optional EDL post-process)
            let drawn = {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                // Sized to the view rather than the window, so the EDL target
                // is exactly the region it will be composited into.
                let vp = view_rect.gl();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    vp.2.max(1) as u32,
                    vp.3.max(1) as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                    gl.quad_context.apply_viewport(vp.0, vp.1, vp.2, vp.3);
                }
                let drawn: i32 = stages.iter()
                    .map(|s| s.draw_visible(&mut *gl.quad_context, &view_proj, point_size, false))
                    .sum();
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength, vp);
                    // Or every overlay below draws over the cloud.
                    edl_renderer.restore_depth(&mut *gl.quad_context, vp, |ctx| {
                        for s in &stages { s.draw_visible(ctx, &view_proj, point_size, true); }
                    });
                } else {
                    gl.quad_context.end_render_pass();
                }
                drawn
            };
            // Meshes and splats over the cloud, depth-tested against it, and
            // under the overlays.
            draw_layers(&layers, &mut layers_drawn, &view_proj, front, right, up, view_rect, z_clip);

            draw_3d_overlays(
                app_mode, measure_a, measure_b, section_start, section_end,
                cursor_world, section_depth, mid_z, height_range, ortho_scale,
            );
            // Vector overlays are real 3D geometry, so they are occluded by the
            // points in front of them — which is the only way to judge where a
            // line actually sits. Drawn here, under the 3D camera and before
            // the panel composites.
            draw_vector_overlays_3d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                data_offset, front, metres_per_pixel, digitise_cursor, edge_route.as_deref(),
            );
            draw_edge_heatmap(&mut heatmap_renderer, &digitise, app_mode, &view_proj, view_rect);

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
                egui_macroquad::egui::SidePanel::left("haiviz_panel")
                    .resizable(true)
                    .default_width(340.0)
                    .show(egui_ctx, |ui| {
                    egui_macroquad::egui::ScrollArea::vertical().show(ui, |ui| {
                panel_section(ui, "Controls", true, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise | AppMode::EditLines => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso | AppMode::SectionDigitise => "Back to 3D",
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
                            if ui.button("Export…").clicked() { reveal_export = true; }
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
                                    .add_filter("All supported", &["laz", "las", "json", "ply"])
                                    .add_filter("LAS / LAZ", &["laz", "las"])
                                    .add_filter("3D Tiles tileset", &["json"])
                                    .add_filter("Gaussian splats", &["ply"])
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
                class_colors_changed = classes_section(ui, &mut class_defs, &mut class_vis, max_class, &mut class_file, &mut classes_save_status);
                // Classification: the lasso tools and the export that consumes
                // them, together because they are one job.
                panel_section(ui, "Meshes & splats", true, |ui| layers.ui(ui, &mut pending_loads));
                panel_section(ui, "Classification", false, |ui| {
                    if let Some(kind) = ground_tool_controls(
                        ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                        &mut classify_target, &mut classify_from, &mut ground_offset,
                        max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                        class_op_count, &mut undo_class_op,
                        &mut apply_ground_lasso, &mut cancel_ground_lasso,
                    ) { start_ground_lasso = Some(kind); }
                    export_request = export_section(
                        ui, &mut reveal_export, &mut export_filter,
                        &mut export_job, &mut export_status, &export_sources,
                        max_class, &class_defs,
                    );
                });
                vector_tools_section(
                    ui, &mut digitise, &editor, &poly_files, selected_poly, app_mode, have_cloud,
                    &mut digitise_request, &mut edit_request,
                );
                polylines_section(ui, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, &mut poly_remove_request, data_offset, background);
                gcp_section(ui, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
                    });
                });
                // Read after the panel is laid out, not before.
                // `is_pointer_over_area` treats a SidePanel as background and falls
                // back to egui's unused_rect, which still covers the whole screen
                // until a panel has shrunk it — so asking first says the pointer is
                // over the viewport even when it is on a button, and the click both
                // presses the button and drops a vertex.
                egui_wants_pointer = egui_ctx.is_pointer_over_area() || egui_ctx.wants_pointer_input();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                // What the panel actually took, so the view renders beside it.
                panel_px = egui_ctx.available_rect().min.x * egui_ctx.pixels_per_point();
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
                gcp::draw_3d(
                    &gcp_files, data_offset, &view_proj, selected_gcp,
                    (view_rect.x, view_rect.y, view_rect.w, view_rect.h),
                );
            }
            draw_screen_lasso(
                &ground_lasso, app_mode == AppMode::GroundLasso, &view_proj, view_rect,
                (mx, my), class_color_mq(classify_target, &class_defs),
            );
            draw_vector_overlays_2d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                &view_proj, view_rect, mx, my, ui_has_pointer,
            );
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
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr, layers.aabb_viewer(data_offset))
            {
                let aspect = view_rect.aspect();
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
                    if let Some(idx) = s.find_nearest_point(view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) {
                        let pos = s.positions[idx];
                        let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        let (ndc_x, ndc_y) = view_rect.ndc(mx, my);
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
                        selected_poly = polylines::pick(&poly_files, &view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h, 12.0);
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
                // Sized to the view rather than the window, so the EDL target
                // is exactly the region it will be composited into.
                let vp = view_rect.gl();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    vp.2.max(1) as u32,
                    vp.3.max(1) as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                    gl.quad_context.apply_viewport(vp.0, vp.1, vp.2, vp.3);
                }
                let drawn: i32 = copc_stages.iter().map(|cs| cs.draw_visible(&mut *gl.quad_context, &view_proj, point_size, false)).sum();
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength, vp);
                    // Or every overlay below draws over the cloud.
                    edl_renderer.restore_depth(&mut *gl.quad_context, vp, |ctx| {
                        for cs in &copc_stages { cs.draw_visible(ctx, &view_proj, point_size, true); }
                    });
                } else {
                    gl.quad_context.end_render_pass();
                }
                drawn
            };
            // Meshes and splats over the cloud, depth-tested against it, and
            // under the overlays.
            draw_layers(&layers, &mut layers_drawn, &view_proj, front, right, up, view_rect, z_clip);

            draw_3d_overlays(
                app_mode, measure_a, measure_b, section_start, section_end,
                cursor_world, section_depth, mid_z, height_range, ortho_scale,
            );
            // Vector overlays are real 3D geometry, so they are occluded by the
            // points in front of them — which is the only way to judge where a
            // line actually sits. Drawn here, under the 3D camera and before
            // the panel composites.
            draw_vector_overlays_3d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                data_offset, front, metres_per_pixel, digitise_cursor, edge_route.as_deref(),
            );
            draw_edge_heatmap(&mut heatmap_renderer, &digitise, app_mode, &view_proj, view_rect);

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
                egui_macroquad::egui::SidePanel::left("haiviz_panel")
                    .resizable(true)
                    .default_width(340.0)
                    .show(egui_ctx, |ui| {
                    egui_macroquad::egui::ScrollArea::vertical().show(ui, |ui| {
                panel_section(ui, "Controls", true, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise | AppMode::EditLines => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso | AppMode::SectionDigitise => "Back to 3D",
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
                            if ui.button("Export…").clicked() { reveal_export = true; }
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
                        let mpp = ortho_scale * 2.0 / view_rect.h.max(1.0);
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
                                    .add_filter("All supported", &["laz", "las", "json", "ply"])
                                    .add_filter("LAS / LAZ", &["laz", "las"])
                                    .add_filter("3D Tiles tileset", &["json"])
                                    .add_filter("Gaussian splats", &["ply"])
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
                class_colors_changed = classes_section(ui, &mut class_defs, &mut class_vis, max_class, &mut class_file, &mut classes_save_status);
                // Classification: the lasso tools and the export that consumes
                // them, together because they are one job.
                panel_section(ui, "Meshes & splats", true, |ui| layers.ui(ui, &mut pending_loads));
                panel_section(ui, "Classification", false, |ui| {
                    if let Some(kind) = ground_tool_controls(
                        ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                        &mut classify_target, &mut classify_from, &mut ground_offset,
                        max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                        class_op_count, &mut undo_class_op,
                        &mut apply_ground_lasso, &mut cancel_ground_lasso,
                    ) { start_ground_lasso = Some(kind); }
                    export_request = export_section(
                        ui, &mut reveal_export, &mut export_filter,
                        &mut export_job, &mut export_status, &export_sources,
                        max_class, &class_defs,
                    );
                });
                vector_tools_section(
                    ui, &mut digitise, &editor, &poly_files, selected_poly, app_mode, have_cloud,
                    &mut digitise_request, &mut edit_request,
                );
                polylines_section(ui, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, &mut poly_remove_request, data_offset, background);
                gcp_section(ui, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
                    });
                });
                // Read after the panel is laid out, not before.
                // `is_pointer_over_area` treats a SidePanel as background and falls
                // back to egui's unused_rect, which still covers the whole screen
                // until a panel has shrunk it — so asking first says the pointer is
                // over the viewport even when it is on a button, and the click both
                // presses the button and drops a vertex.
                egui_wants_pointer = egui_ctx.is_pointer_over_area() || egui_ctx.wants_pointer_input();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                // What the panel actually took, so the view renders beside it.
                panel_px = egui_ctx.available_rect().min.x * egui_ctx.pixels_per_point();
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
                gcp::draw_3d(
                    &gcp_files, data_offset, &view_proj, selected_gcp,
                    (view_rect.x, view_rect.y, view_rect.w, view_rect.h),
                );
            }
            draw_screen_lasso(
                &ground_lasso, app_mode == AppMode::GroundLasso, &view_proj, view_rect,
                (mx, my), class_color_mq(classify_target, &class_defs),
            );
            draw_vector_overlays_2d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                &view_proj, view_rect, mx, my, ui_has_pointer,
            );
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
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr, layers.aabb_viewer(data_offset))
            {
                let aspect = view_rect.aspect();
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
                            if let Some((p, _vpos, dist)) = cs.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) 
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
                            if let Some((_p, vpos, dist)) = cs.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) 
                                && dist < best_dist { best_dist = dist; measure_a = Some(vpos); measure_b = None; app_mode = AppMode::MeasureSecond; 
                            }
                        }
                    }
                    AppMode::MeasureSecond => {
                        let mut best_dist = f32::MAX;
                        for cs in copc_stages.iter().filter(|cs| cs.visible) {
                            if let Some((_p, vpos, dist)) = cs.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) 
                                && dist < best_dist { best_dist = dist; measure_b = Some(vpos); 
                            }
                        }
                    }
                    AppMode::View3D => {
                        selected_poly = polylines::pick(&poly_files, &view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h, 12.0);
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
                // Sized to the view rather than the window, so the EDL target
                // is exactly the region it will be composited into.
                let vp = view_rect.gl();
                edl_renderer.resize_if_needed(
                    &mut *gl.quad_context,
                    vp.2.max(1) as u32,
                    vp.3.max(1) as u32,
                );
                if edl_enabled {
                    edl_renderer.begin(&mut *gl.quad_context);
                } else {
                    gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
                    gl.quad_context.apply_viewport(vp.0, vp.1, vp.2, vp.3);
                }
                let result = mgr.draw_visible(&mut *gl.quad_context, &view_proj, point_size, false);
                if edl_enabled {
                    edl_renderer.apply(&mut *gl.quad_context, edl_strength, vp);
                    // Or every overlay below draws over the cloud.
                    edl_renderer.restore_depth(&mut *gl.quad_context, vp, |ctx| {
                        mgr.draw_visible(ctx, &view_proj, point_size, true);
                    });
                } else {
                    gl.quad_context.end_render_pass();
                }
                result
            };
            draw_layers(&layers, &mut layers_drawn, &view_proj, front, right, up, view_rect, z_clip);

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
            );
            // Vector overlays are real 3D geometry, so they are occluded by the
            // points in front of them — which is the only way to judge where a
            // line actually sits. Drawn here, under the 3D camera and before
            // the panel composites.
            draw_vector_overlays_3d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                data_offset, front, metres_per_pixel, digitise_cursor, edge_route.as_deref(),
            );
            draw_edge_heatmap(&mut heatmap_renderer, &digitise, app_mode, &view_proj, view_rect);

            set_default_camera();

            // Draw tile labels (2D overlay)
            for i in 0..mgr.tiles.len() {
                let center = mgr.tile_center(i);
                let (sx, sy) = world_to_screen(vec3(center.0, center.1, 0.0), &view_proj, view_rect);
                if sx > view_rect.x && sx < view_rect.x + view_rect.w
                    && sy > view_rect.y && sy < view_rect.y + view_rect.h {
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
                egui_macroquad::egui::SidePanel::left("haiviz_panel")
                    .resizable(true)
                    .default_width(340.0)
                    .show(egui_ctx, |ui| {
                    egui_macroquad::egui::ScrollArea::vertical().show(ui, |ui| {
                panel_section(ui, "Controls", true, |ui| {
                        // Cross Section button
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst
                                | AppMode::MeasureSecond | AppMode::GroundLasso
                                | AppMode::Digitise | AppMode::EditLines => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionCorridorLoad => "Loading corridor...",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond | AppMode::SectionPointPick | AppMode::SectionLasso | AppMode::SectionDigitise => "Back to 3D",
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
                class_colors_changed = classes_section(ui, &mut class_defs, &mut class_vis, max_class, &mut class_file, &mut classes_save_status);
                // Classification: the lasso tools and the export that consumes
                // them, together because they are one job.
                panel_section(ui, "Meshes & splats", true, |ui| layers.ui(ui, &mut pending_loads));
                panel_section(ui, "Classification", false, |ui| {
                    if let Some(kind) = ground_tool_controls(
                        ui, &mut app_mode, ground_lasso.len(), lasso_kind,
                        &mut classify_target, &mut classify_from, &mut ground_offset,
                        max_class, &class_defs, class_vis.inverted().count() as usize, &ground_status,
                        class_op_count, &mut undo_class_op,
                        &mut apply_ground_lasso, &mut cancel_ground_lasso,
                    ) { start_ground_lasso = Some(kind); }
                    export_request = export_section(
                        ui, &mut reveal_export, &mut export_filter,
                        &mut export_job, &mut export_status, &export_sources,
                        max_class, &class_defs,
                    );
                });
                vector_tools_section(
                    ui, &mut digitise, &editor, &poly_files, selected_poly, app_mode, have_cloud,
                    &mut digitise_request, &mut edit_request,
                );
                polylines_section(ui, &mut poly_files, &mut selected_poly, &mut poly_load_error, &mut poly_save_status, &mut pending_salome, &mut poly_remove_request, data_offset, background);
                gcp_section(ui, &mut gcp_files, &mut selected_gcp, &mut gcp_load_error, background);
                    });
                });
                // Read after the panel is laid out, not before.
                // `is_pointer_over_area` treats a SidePanel as background and falls
                // back to egui's unused_rect, which still covers the whole screen
                // until a panel has shrunk it — so asking first says the pointer is
                // over the viewport even when it is on a button, and the click both
                // presses the button and drops a vertex.
                egui_wants_pointer = egui_ctx.is_pointer_over_area() || egui_ctx.wants_pointer_input();
                egui_wants_keyboard = egui_ctx.wants_keyboard_input();
                // What the panel actually took, so the view renders beside it.
                panel_px = egui_ctx.available_rect().min.x * egui_ctx.pixels_per_point();
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
                gcp::draw_3d(
                    &gcp_files, data_offset, &view_proj, selected_gcp,
                    (view_rect.x, view_rect.y, view_rect.w, view_rect.h),
                );
            }
            draw_screen_lasso(
                &ground_lasso, app_mode == AppMode::GroundLasso, &view_proj, view_rect,
                (mx, my), class_color_mq(classify_target, &class_defs),
            );
            draw_vector_overlays_2d(
                &poly_files, selected_poly, &digitise, &editor, app_mode,
                &view_proj, view_rect, mx, my, ui_has_pointer,
            );
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
                && let Some((lo, hi)) = loaded_aabb_viewer(&stages, &copc_stages, &tile_mgr, layers.aabb_viewer(data_offset))
            {
                let aspect = view_rect.aspect();
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
                    let aspect = view_rect.aspect();
                    ortho_scale = (ext_y * 0.5).max(ext_x * 0.5 / aspect) * 1.2;
                    position = vec3(cx, cy, height_range * 0.5)
                        - front * (ext_x.max(ext_y) * 2.0);
                
            }

            // Handle clicks
            if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
                match app_mode {
                    AppMode::PointPick => {
                        picked_tile_point = mgr.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h);
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
                        if let Some((tile_idx, pt_idx)) = mgr.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) {
                            measure_a = Some(glam::Vec3::from(mgr.loaded[tile_idx].as_ref().unwrap().positions[pt_idx]));
                            measure_b = None;
                            app_mode = AppMode::MeasureSecond;
                        }
                    }
                    AppMode::MeasureSecond => {
                        if let Some((tile_idx, pt_idx)) = mgr.find_nearest_point(&view_proj, mx - view_rect.x, my - view_rect.y, view_rect.w, view_rect.h) {
                            measure_b = Some(glam::Vec3::from(mgr.loaded[tile_idx].as_ref().unwrap().positions[pt_idx]));
                        }
                    }
                    AppMode::View3D => {
                        if let Some((wx, wy)) = screen_to_world_xy(mx, my, &view_proj, 0.0, view_rect) {
                            selected_tile = mgr.tile_at(wx, wy);
                        }
                    }
                    _ => {}
                }
            }

            // Handle load/unload (from buttons or keyboard)
            // Only while browsing: in the vector tools Enter finishes a line and
            // Delete removes a vertex, and neither should load or drop a tile.
            if let Some(sel) = selected_tile
                && app_mode == AppMode::View3D && !egui_wants_keyboard_prev {
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

        // ---- Empty state: no data loaded ----
        // Show a minimal Controls window so the user knows how to load files.
        if stages.is_empty() && copc_stages.is_empty() && tile_mgr.is_none() && layers.is_empty() {
            egui_macroquad::ui(|egui_ctx| {
                egui_macroquad::egui::SidePanel::left("haiviz_panel")
                    .resizable(true)
                    .default_width(340.0)
                    .show(egui_ctx, |ui| {
                    egui_macroquad::egui::ScrollArea::vertical().show(ui, |ui| {
                panel_section(ui, "Controls", true, |ui| {
                        ui.label("No point cloud loaded.");
                        if ui.button("Open file…").clicked()
                            && let Some(paths) = rfd::FileDialog::new()
                                .add_filter("All supported", &["laz", "las", "json", "ply"])
                                .add_filter("LAS / LAZ", &["laz", "las"])
                                .add_filter("3D Tiles tileset", &["json"])
                                .add_filter("Gaussian splats", &["ply"])
                                .pick_files()
                        {
                            for p in paths {
                                pending_loads.push(p.to_string_lossy().to_string());
                            }
                        }
                        ui.label("…or drag a .laz / .copc.laz file, a 3D Tiles tileset.json or a Gaussian-splat .ply onto the window (Windows / X11).");
                        if let Some(e) = &layers.status {
                            ui.colored_label(egui_macroquad::egui::Color32::LIGHT_RED, e);
                        }
                        // Nothing is loaded, so there is nothing to classify,
                        // digitise or export; the panel stays a single group.
                });
                    });
                });
                // Read after the panel is laid out, not before.
                // `is_pointer_over_area` treats a SidePanel as background and falls
                // back to egui's unused_rect, which still covers the whole screen
                // until a panel has shrunk it — so asking first says the pointer is
                // over the viewport even when it is on a button, and the click both
                // presses the button and drops a vertex.
                // What the panel actually took, so the view renders beside it.
                panel_px = egui_ctx.available_rect().min.x * egui_ctx.pixels_per_point();
            });
            egui_macroquad::draw();
        }

        // ---- Act on a digitising request ----
        // One place, whether it came from `P`, from Enter, or from the panel
        // in any of the three per-mode blocks.
        if let Some(req) = digitise_request.take() {
            match req {
                vectorise::Request::Start => {
                    if ensure_digitise_target(&mut digitise, &mut poly_files, &stages, &copc_stages, &tile_mgr, data_offset) {
                        digitise.clear();
                        editor.drag = None;
                        editor.hover = None;
                        app_mode = AppMode::Digitise;
                    }
                }
                vectorise::Request::Cancel => {
                    digitise.clear();
                    app_mode = AppMode::View3D;
                }
                vectorise::Request::UndoVertex => { digitise.pop(); }
                vectorise::Request::Finish =>
                    finish_digitised_line(&mut digitise, &mut editor, &mut poly_files, false),
                vectorise::Request::FinishRing =>
                    finish_digitised_line(&mut digitise, &mut editor, &mut poly_files, true),
            }
        }

        // ---- Act on an editing request ----
        // Likewise one place, whether it came from a key or from the panel.
        if let Some(req) = edit_request.take() {
            use vector_edit::Request as Edit;
            match req {
                Edit::Start => {
                    if digitise.is_drawing() {
                        editor.status = Some("Finish or cancel the line being drawn first.".to_string());
                    } else {
                        editor.status = None;
                        app_mode = AppMode::EditLines;
                    }
                }
                Edit::Stop => {
                    editor.drag = None;
                    editor.hover = None;
                    app_mode = AppMode::View3D;
                }
                Edit::DeleteVertex => {
                    if let Some((line, idx)) = editor.vertex.filter(|(l, _)| Some(*l) == selected_poly) {
                        editor.status = Some(match editor.delete_vertex(&mut poly_files, line, idx) {
                            Ok(fi) => saved_status(
                                format!("Vertex {} deleted.", idx + 1),
                                autosave_vectors(&mut poly_files[fi]),
                            ),
                            Err(e) => e,
                        });
                    }
                }
                Edit::DeleteLine => {
                    if let Some(line) = selected_poly {
                        let name = poly_files.get(line.file)
                            .and_then(|f| f.get(line.line_ref()))
                            .map(|l| l.name.clone())
                            .unwrap_or_default();
                        if let Some(fi) = editor.delete_line(&mut poly_files, line) {
                            selected_poly = None;
                            editor.status = Some(saved_status(
                                format!("Deleted {name} — Ctrl+Z brings it back."),
                                autosave_vectors(&mut poly_files[fi]),
                            ));
                        }
                    }
                }
                Edit::ToggleRing => {
                    if let Some(line) = selected_poly {
                        editor.status = Some(match editor.toggle_ring(&mut poly_files, line) {
                            Ok(fi) => {
                                let closed = poly_files[fi].get(line.line_ref()).is_some_and(|l| l.closed);
                                let what = if closed { "Closed into a ring." } else { "Ring opened." };
                                saved_status(what.to_string(), autosave_vectors(&mut poly_files[fi]))
                            }
                            Err(e) => e,
                        });
                    }
                }
                Edit::Redrape => {
                    if let Some(line) = selected_poly
                        && let Some(d) = redrape_line(
                            &stages, &copc_stages, &tile_mgr, &poly_files, line, data_offset, &digitise,
                        ) {
                        let summary = d.summary(digitise.snap_mode);
                        editor.status = Some(if d.moved == 0 {
                            summary
                        } else {
                            match editor.replace_points(&mut poly_files, line, d.points) {
                                Some(fi) => saved_status(summary, autosave_vectors(&mut poly_files[fi])),
                                None => "That line is no longer loaded.".to_string(),
                            }
                        });
                    }
                }
                Edit::Undo => {
                    editor.status = Some(match editor.undo(&mut poly_files) {
                        Some(fi) => saved_status("Undone.".to_string(), autosave_vectors(&mut poly_files[fi])),
                        None => "Nothing to undo.".to_string(),
                    });
                }
            }
        }
        // An undo, a delete or an unload can take a selected line or vertex
        // away; drop the selection rather than let it address something else.
        editor.validate(&poly_files, &mut selected_poly);

        // ---- Plan-view ground tool: start / cancel / apply ----
        if let Some(kind) = start_ground_lasso {
            ground_lasso.clear();
            ground_status = None;
            lasso_kind = kind;
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
            // Outline and band in absolute world coordinates, so the op means
            // the same thing at export as it does on screen.
            let world = |p: Vec3| [
                p.x as f64 + data_offset[0], p.y as f64 + data_offset[1], p.z as f64 + data_offset[2],
            ];
            let ((right, up), top) = ground_lasso_view
                .unwrap_or_else(|| (view_axes(&view_proj), is_top_view(pitch)));
            let (polygon, view): (Vec<(f64, f64)>, Option<crate::classify::ViewPlane>) = if top {
                // From straight above, a screen outline *is* a plan footprint,
                // and written as one it stays readable by older builds.
                (ground_lasso.iter().map(|p| { let w = world(*p); (w[0], w[1]) }).collect(), None)
            } else {
                let plane = view_plane(world(ground_lasso[0]), right, up);
                let outline = ground_lasso.iter()
                    .map(|p| { let w = world(*p); plane.project(w[0], w[1], w[2]) })
                    .collect();
                (outline, Some(plane))
            };
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

            // What the outline can reach inside the band: its own bounds from
            // above, its lines of sight cut by the band from anywhere else.
            let ((bx0, by0), (bx1, by1)) =
                crate::classify::outline_xy_bounds(&polygon, view.as_ref(), z_lo, z_hi);

            // The inside lasso is a plain prism: no samples to gather and no
            // plane to fit, so it cannot fail the way the ground tool can.
            let region = match lasso_kind {
                LassoKind::Inside => Some((
                    crate::classify::ClassRegion::Volume {
                        polygon: polygon.clone(),
                        view,
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

                    match crate::classify::fit_ground_plane_in(&samples, &polygon, view.as_ref(), z_lo, z_hi) {
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
                                view,
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

                // One seq for every file the op lands in; see `next_class_seq`.
                let seq = next_class_seq(&stages, &copc_stages);
                let mut changed = 0usize;
                let mut touched = 0usize;
                {
                    let mut gl = unsafe { get_internal_gl() };
                    gl.flush();
                    for st in stages.iter_mut() {
                        if !op.intersects_aabb(&st.cloud.min, &st.cloud.max) { continue; }
                        let from = st.edits.ops.len();
                        st.edits.push_at(op.clone(), seq);
                        changed += st.apply_edits(&mut *gl.quad_context, color_mode, from);
                        touched += 1;
                    }
                    for cs in copc_stages.iter_mut() {
                        if !op.intersects_aabb(&cs.global_min, &cs.global_max) { continue; }
                        let from = cs.edits.ops.len();
                        cs.edits.push_at(op.clone(), seq);
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
                        // At the fit's own origin: a screen outline's bounds can
                        // be unlimited, so their centre is not a place.
                        info!(
                            "Ground fit: {} cells from {} points, {:.2}m at its centre, \
                             slope {:.2}°, rms {:.3}m; {} points reclassified{}",
                            fit.cells_used, fit.points_considered,
                            fit.height_at(fit.origin.0, fit.origin.1), fit.slope_degrees(), fit.rms, changed,
                            if view.is_some() { " (screen outline)" } else { "" },
                        );
                    }
                    None => info!(
                        "Inside lasso: {:.2}m band, {} points reclassified",
                        z_hi - z_lo, changed,
                    ),
                }
                ground_status = Some(match (save_err, touched) {
                    (Some(e), _) => format!("{changed} points reclassified — SAVE FAILED: {e}"),
                    (None, 0) => "The outline did not reach any loaded file".to_string(),
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
        if undo_class_op {
            // Nothing to rebuild here: re-read files redraw as they are, and
            // dropped COPC nodes stream back in on their own.
            let (undone, _, _) = undo_last_class_op(&mut stages, &mut copc_stages, color_mode);
            ground_status = Some(if undone { "Last op undone" } else { "Nothing to undo" }.to_string());
        }

        next_frame().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{ClassOp, ClassRegion};

    /// Draw a rectangular lasso on screen through the same chain the tool uses
    /// — cursor ray, vertex on the view plane through the pivot, `view_axes`,
    /// `view_plane` — and check that the op selects exactly what projects
    /// inside the rectangle. `plan` records it as a plan footprint instead, as
    /// the tool does from straight above.
    fn outline_selects_what_is_inside_on_screen(eye: Vec3, up: Vec3, plan: bool) {
        let view = ViewRect { x: 100.0, y: 0.0, w: 800.0, h: 600.0 };
        let pivot = Vec3::ZERO;
        let view_proj = glam::Mat4::orthographic_rh_gl(-20.0, 20.0, -15.0, 15.0, -1.0e5, 1.0e5)
            * glam::Mat4::look_at_rh(eye, pivot, up);
        let (x0, y0, x1, y1) = (400.0, 200.0, 650.0, 420.0);
        let lasso: Vec<Vec3> = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].iter().map(|&(mx, my)| {
            let (o, d) = cursor_ray(&view_proj, mx, my, view);
            let (o, d) = (Vec3::from(o), Vec3::from(d));
            o + d * (pivot - o).dot(d)
        }).collect();

        // A survey-sized viewer origin, so the f32 → f64 step is exercised.
        let offset = [208_000.0, 373_000.0, 5.0];
        let world = |p: Vec3| [p.x as f64 + offset[0], p.y as f64 + offset[1], p.z as f64 + offset[2]];
        let (polygon, plane) = if plan {
            (lasso.iter().map(|p| { let w = world(*p); (w[0], w[1]) }).collect(), None)
        } else {
            let (right, screen_up) = view_axes(&view_proj);
            let plane = view_plane(world(lasso[0]), right, screen_up);
            let outline = lasso.iter().map(|p| { let w = world(*p); plane.project(w[0], w[1], w[2]) }).collect();
            (outline, Some(plane))
        };
        let op = ClassOp::new(
            0,
            ClassRegion::Volume { polygon, view: plane, z_min: -1.0e3, z_max: 1.0e3 },
            7,
            ClassMask::ALL,
        );

        let (mut inside, mut outside) = (0, 0);
        for i in -20..=20 {
            for j in -20..=20 {
                for k in -4..=4 {
                    let p = Vec3::new(i as f32, j as f32, k as f32 * 2.5);
                    let (sx, sy) = world_to_screen(p, &view_proj, view);
                    // Points within half a pixel of an edge could honestly land
                    // either side after f32 projection; judge only the clear ones.
                    let m = 0.5;
                    let clearly_in = sx > x0 + m && sx < x1 - m && sy > y0 + m && sy < y1 - m;
                    let clearly_out = sx < x0 - m || sx > x1 + m || sy < y0 - m || sy > y1 + m;
                    let w = world(p);
                    let selected = op.contains(w[0], w[1], w[2]);
                    if clearly_in {
                        assert!(selected, "{p} is inside the outline on screen at ({sx}, {sy}) but not selected");
                        inside += 1;
                    }
                    if clearly_out {
                        assert!(!selected, "{p} is outside the outline on screen at ({sx}, {sy}) but selected");
                        outside += 1;
                    }
                }
            }
        }
        assert!(inside > 50 && outside > 50, "the probes must land on both sides: {inside} in, {outside} out");
    }

    #[test]
    fn a_screen_lasso_selects_what_it_encloses_from_any_angle() {
        // Oblique, looking north.
        outline_selects_what_is_inside_on_screen(Vec3::new(0.0, -50.0, 50.0), Vec3::Z, false);
        // Steep, from an arbitrary yaw.
        outline_selects_what_is_inside_on_screen(Vec3::new(30.0, 20.0, 40.0), Vec3::Z, false);
        // Level, from the side.
        outline_selects_what_is_inside_on_screen(Vec3::new(-50.0, 0.0, 0.0), Vec3::Z, false);
        // Straight down, recorded either way.
        outline_selects_what_is_inside_on_screen(Vec3::new(0.0, 0.0, 50.0), Vec3::Y, false);
        outline_selects_what_is_inside_on_screen(Vec3::new(0.0, 0.0, 50.0), Vec3::Y, true);
    }
}
