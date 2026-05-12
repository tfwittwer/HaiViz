use std::env;
use std::path::Path;
//use rfd;
use log::{info,error};
//use console::style;
use crate::glam::vec3;

use macroquad::prelude::*;

use crate::raw_miniquad::{CopcStage, EdlRenderer, SectionData, Stage, TileStageManager, build_section_from_raw_stages, load_class_defs, make_section_data, scan_dir_for_copc};
use las::Point as LasPoint;

const MOVE_SPEED: f32 = 0.5;
const LOOK_SPEED: f32 = 0.5;

mod raw_miniquad;
mod polylines;

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

    pub fn draw_and_interact(front: Vec3, right: Vec3, up: Vec3, interactive: bool) -> Option<(f32, f32)> {
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

        if interactive && is_key_pressed(KeyCode::T) {
            return Some((std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2));
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

fn polylines_window(
    egui_ctx: &egui_macroquad::egui::Context,
    poly_files: &mut Vec<polylines::PolylineFile>,
    selected_poly: &mut Option<(usize, usize)>,
    poly_load_error: &mut Option<String>,
    data_offset: [f64; 3],
    background: Color,
) {
    egui_macroquad::egui::Window::new("Polylines")
        .default_pos([10.0, 300.0])
        .default_open(false)
        .show(egui_ctx, |ui| {
            let luma = background.r * 0.299 + background.g * 0.587 + background.b * 0.114;
            let default_color = if luma > 0.5 { BLACK } else { WHITE };
            if ui.button("Load Polyline File...").clicked() &&
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("Polyline", &["txt", "xyz", "poly", "dat", ""])
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
            for (fi, pf) in poly_files.iter_mut().enumerate() {
                let fname = std::path::Path::new(&pf.path)
                    .file_name().and_then(|n| n.to_str()).unwrap_or(&pf.path);
                ui.collapsing(format!("{} ({} lines)", fname, pf.polylines.len()), |ui| {
                    if ui.small_button("Remove file").clicked() {
                        to_remove = Some(fi);
                        if selected_poly.map(|(f, _)| f) == Some(fi) { *selected_poly = None; }
                    }
                    for (pi, pl) in pf.polylines.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let is_sel = *selected_poly == Some((fi, pi));
                            if ui.selectable_label(is_sel, &pl.name).clicked() {
                                *selected_poly = if is_sel { None } else { Some((fi, pi)) };
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
            if let Some(fi) = to_remove { poly_files.remove(fi); }
        });
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

    let class_defs = load_class_defs();
    if let Some(ref cd) = class_defs {
        info!("Custom classes: {:?}", cd.keys().collect::<Vec<_>>());
    }

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

    let aspect = screen_width() / screen_height().max(1.0);
    let mut ortho_scale: f32 = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;

    // Section state (single-file only)
    let mut section_start: Option<(f32, f32)> = None;
    let mut section_end: Option<(f32, f32)> = None;
    let mut section_depth: f32 = 2.0;
    let mut section_data: Option<SectionData> = None;
    // Cache of raw COPC section points (one Vec per stage) used to recolour on mode change.
    #[allow(clippy::type_complexity)]
    let mut copc_section_raw: Option<(Vec<Vec<(LasPoint, f32, f32)>>, f32, f32, f32)> = None;
    let mut section_pan = glam::Vec2::ZERO;
    let mut section_zoom: f32 = 50.0;
    let mut sec_measure_a: Option<glam::Vec2> = None;
    let mut sec_measure_b: Option<glam::Vec2> = None;

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
    let mut selected_poly: Option<(usize, usize)> = None; // (file_idx, poly_idx)
    let mut poly_load_error: Option<String> = None;

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

    loop {
        //println!("{} {}",yaw,pitch);
        //println!("{}",app_mode.label());
        let delta = get_frame_time();

        // ---- Collect drag-and-dropped files into the load queue ----
        // (DnD only works on X11/XWayland; on Wayland use the "Open file…" button.)
        for df in get_dropped_files() {
            if let Some(path) = df.path {
                pending_loads.push(path.to_string_lossy().to_string());
            }
        }

        // ---- Drain the pending-load queue ----
        for path_str in pending_loads.drain(..) {
            let header = match las::Reader::from_path(&path_str) {
                Ok(r) => r.header().clone(),
                Err(e) => { error!("Cannot open file {}: {}", path_str, e); continue; }
            };
            let is_copc = header.copc_info_vlr().is_some();
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
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            if is_copc {
                let cs = CopcStage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                info!("Loaded COPC file: {}", path_str);
                copc_stages.push(cs);
            } else {
                let s = Stage::new(ctx, &path_str, Some(origin), color_mode, class_defs.clone());
                info!("Loaded LAZ file: {}", path_str);
                stages.push(s);
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
            section_data = None;
            copc_section_raw = None;
        }
        if let Some(idx) = stage_unload_request.take()
            && idx < stages.len()
        {
            let s = stages.remove(idx);
            info!("Unloading LAZ file: {}", s.filepath);
            let InternalGlContext { quad_context: ctx, .. } = unsafe { get_internal_gl() };
            s.unload(ctx);
            section_data = None;
            // Picked point may reference an unloaded stage.
            if let Some((si, _)) = picked_point && si >= stages.len() {
                picked_point = None;
            }
        }

        if is_key_pressed(KeyCode::Escape) {
            if app_mode != AppMode::View3D {
                app_mode = AppMode::View3D;
                section_data = None;
            } else {
                break;
            }
        }

        // ---- Section 2D view ----
        if matches!(app_mode, AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond) {
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
                // Crosshair at mouse when in measure mode
                if matches!(app_mode, AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond) {
                    let (mx, my) = mouse_position();
                    let ch = 12.0;
                    draw_line(mx - ch, my, mx + ch, my, 1.0, Color::new(1.0, 1.0, 1.0, 0.7));
                    draw_line(mx, my - ch, mx, my + ch, 1.0, Color::new(1.0, 1.0, 1.0, 0.7));
                }
            }

            let prev_color_mode = color_mode;
            let mut egui_wants_sec_pointer = false;
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_sec_pointer = egui_ctx.is_pointer_over_area();
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        if ui.button("Back to 3D").clicked() {
                            app_mode = AppMode::View3D;
                            section_data = None;
                            sec_measure_a = None;
                            sec_measure_b = None;
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
                    });
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, data_offset, background);
            });
            egui_macroquad::draw();

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
                    AppMode::SectionMeasureFirst => {
                        sec_measure_a = Some(pos);
                        sec_measure_b = None;
                        app_mode = AppMode::SectionMeasureSecond;
                    }
                    AppMode::SectionMeasureSecond => {
                        sec_measure_b = Some(pos);
                    }
                    _ => {}
                }
            }

            if color_mode != prev_color_mode &&
                let (Some(start), Some(end)) = (section_start, section_end) {
                    let gl = unsafe { get_internal_gl() };
                    if let Some(stg) = stages.first() {
                        section_data = Some(stg.extract_section(
                            &mut *gl.quad_context, start, end, section_depth, color_mode,
                        ));
                    } else if !copc_stages.is_empty() {
                        if let Some((ref per_stage_raw, length, h_min, h_max)) = copc_section_raw {
                            section_data = Some(build_section_from_raw_stages(
                                &mut *gl.quad_context, &copc_stages, per_stage_raw,
                                length, h_min, h_max, color_mode,
                            ));
                        }
                    } else if let Some(ref mgr) = tile_mgr {
                        section_data = Some(mgr.extract_section(
                            &mut *gl.quad_context, start, end, section_depth, color_mode,
                        ));
                    }
            }

            next_frame().await;
            continue;
        }

        // ---- 3D view modes (shared navigation) ----
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

        // Orbit pivot: screen-center ray intersected with mid_z plane
        /*let orbit_pivot = if front.z.abs() > 0.01 {
            //println!("orbit {} {}",position + front * ((mid_z - position.z) / front.z),mid_z);
            position + front * ((mid_z - position.z) / front.z)
        } else {
            //println!("data_center {}",data_center);
            data_center
        };*/
        if !pivot_locked {
            orbit_pivot = position + front * ((mid_z - position.z) / front.z);
            //println!("not locked");
        }
        
        let mode_uses_click = matches!(app_mode,
            AppMode::SectionPickStart | AppMode::SectionPickEnd |
            AppMode::SectionAdjustDepth |
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
            let units_per_pixel = ortho_scale * 2.0 / screen_height();
            position -= right * mouse_delta.x * units_per_pixel * 10.0;
            position += up * mouse_delta.y * units_per_pixel * 10.0;
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

        // ---- Measure overlays ----
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

        // ---- Single-file (multi-Stage) mode rendering ----
        if !stages.is_empty() {
            // Section preview
            match app_mode {
                AppMode::SectionPickEnd => {
                    if let (Some(start), Some((wx, wy))) = (section_start, cursor_world) {
                        draw_line_3d(
                            vec3(start.0, start.1, mid_z),
                            vec3(wx, wy, mid_z),
                            YELLOW,
                        );
                    }
                }
                AppMode::SectionAdjustDepth => {
                    if let (Some(start), Some(end)) = (section_start, section_end) {
                        draw_section_corridor(start, end, section_depth, height_range);
                    }
                }
                _ => {}
            }

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

            set_default_camera();

            // View cube
            let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let overrides_before: Vec<Option<[f32; 3]>> =
                stages.iter().map(|s| s.rgb_override).collect();
            let mut new_overrides = overrides_before.clone();
            let mut egui_wants_pointer = false;
            let mut do_zoom_all = false;
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst | AppMode::MeasureSecond => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond => "Back to 3D",
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
                                    section_data = None;
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
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        ui.horizontal(|ui| {
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
                                if ui.button("Unload").clicked() {
                                    stage_unload_request = Some(i);
                                }
                                ui.label(name);
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
                            ui.label("Point Info:");
                            ui.label(format!("  X: {:.3}  Y: {:.3}  Z: {:.3}", p.x, p.y, p.z));
                            ui.label(format!("  Intensity: {}", p.intensity));
                            ui.label(format!("  Classification: {}", class_label(p, &class_defs)));
                            ui.label(format!("  Return: {}/{}", p.return_number, p.number_of_returns));
                            if let Some(ref color) = p.color {
                                ui.label(format!("  RGB: {} {} {}", color.red, color.green, color.blue));
                            }
                            if let Some(time) = p.gps_time {
                                ui.label(format!("  GPS time: {:.6}", time));
                            }
                            ui.label(format!("  Point Source Id: {}", p.point_source_id));
                        }
                    });
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, data_offset, background);
            });
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;

            if do_zoom_all {
                let aspect = screen_width() / screen_height().max(1.0);
                ortho_scale = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;
                position = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5)
                    - front * (data_extent_x.max(data_extent_y) * 2.0);
            }

            // Helper closure: find nearest point across all stages, returning
            // (stage_idx, point_idx) plus the viewer-space position.
            let pick_across = |view_proj: &glam::Mat4, mx: f32, my: f32| -> Option<(usize, usize)> {
                let mut best: Option<(usize, usize, f32)> = None;
                for (si, s) in stages.iter().enumerate() {
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
                        // Section currently extracts only from the first stage.
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            let gl = unsafe { get_internal_gl() };
                            let sec = stages[0].extract_section(
                                &mut *gl.quad_context, start, end, section_depth, color_mode,
                            );
                            section_pan = glam::Vec2::new(sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                            section_zoom = (sec.length / 2.0).max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                            section_data = Some(sec);
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

            if color_mode != prev_color_mode {
                let gl = unsafe { get_internal_gl() };
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
            let mut new_copc_overrides = copc_overrides_before.clone();

            // Section corridor preview in 3D.
            match app_mode {
                AppMode::SectionPickEnd => {
                    if let (Some(start), Some((wx, wy))) = (section_start, cursor_world) {
                        draw_line_3d(vec3(start.0, start.1, mid_z), vec3(wx, wy, mid_z), YELLOW);
                    }
                }
                AppMode::SectionAdjustDepth => {
                    if let (Some(start), Some(end)) = (section_start, section_end) {
                        //println!("here!");
                        //set_camera(&camera);
                        draw_section_corridor(start, end, section_depth, height_range);
                    }
                }
                _ => {}
            }

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

            set_default_camera();

            // View cube
            let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let mut egui_wants_pointer = false;
            let mut do_zoom_all = false;
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst | AppMode::MeasureSecond => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond => "Back to 3D",
                        };
                        if ui.button(section_label).clicked() {
                            match app_mode {
                                AppMode::View3D | AppMode::PointPick => {
                                    app_mode = AppMode::SectionPickStart;
                                    section_start = None;
                                    section_end = None;
                                    picked_point = None;
                                }
                                _ => { app_mode = AppMode::View3D; section_data = None; }
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
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            let mut bg_egui = mq_to_egui(background);
                            if ui.color_edit_button_srgba(&mut bg_egui).changed() {
                                background = egui_to_mq(bg_egui);
                            }
                        });
                        ui.horizontal(|ui| {
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
                                if ui.button("Unload").clicked() {
                                    copc_unload_request = Some(i);
                                }
                                ui.label(name);
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
                            ui.label("Point Info:");
                            ui.label(format!("  X: {:.3}  Y: {:.3}  Z: {:.3}", p.x, p.y, p.z));
                            ui.label(format!("  Intensity: {}", p.intensity));
                            ui.label(format!("  Classification: {}", class_label(p, &class_defs)));
                            ui.label(format!("  Return: {}/{}", p.return_number, p.number_of_returns));
                            if let Some(ref color) = p.color {
                                ui.label(format!("  RGB: {} {} {}", color.red, color.green, color.blue));
                            }
                            if let Some(time) = p.gps_time {
                                ui.label(format!("  GPS time: {:.6}", time));
                            }
                            ui.label(format!("  Point Source Id: {}", p.point_source_id));
                        }
                    });
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, data_offset, background);
            });
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;

            if do_zoom_all {
                let aspect = screen_width() / screen_height().max(1.0);
                ortho_scale = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;
                position = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5)
                    - front * (data_extent_x.max(data_extent_y) * 2.0);
            }

            // Mouse clicks
            if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
                match app_mode {
                    AppMode::PointPick => {
                        let mut best_dist = f32::MAX;
                        for cs in &copc_stages {
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
                        if let (Some(start), Some(end)) = (section_start, section_end) {
                            let gl = unsafe { get_internal_gl() };
                            // Combine verts from all COPC stages into one SectionData.
                            let mut all_verts = Vec::new();
                            let mut per_stage_raw: Vec<Vec<(LasPoint, f32, f32)>> = Vec::new();
                            let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
                            let mut length = 0.0_f32;
                            let height_range = copc_stages[0].height_range();
                            for cs in &copc_stages {
                                let (verts, raw, len, hmin, hmax) = cs.extract_section_verts(start, end, section_depth, color_mode);
                                all_verts.extend(verts);
                                per_stage_raw.push(raw);
                                h_min = h_min.min(hmin); h_max = h_max.max(hmax);
                                length = length.max(len);
                            }
                            let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
                            let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
                            info!("COPC Section: {:.1}m long, {} points", length, all_verts.len());
                            copc_section_raw = Some((per_stage_raw, length, h_min, h_max));
                            let sec = make_section_data(&mut *gl.quad_context, all_verts, length, h_min, h_max, height_range);
                            section_pan = glam::Vec2::new(sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                            section_zoom = (sec.length / 2.0).max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                            section_data = Some(sec);
                            app_mode = AppMode::SectionView;
                        }
                    }
                    AppMode::MeasureFirst => {
                        let mut best_dist = f32::MAX;
                        for cs in &copc_stages {
                            if let Some((_p, vpos, dist)) = cs.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height()) 
                                && dist < best_dist { best_dist = dist; measure_a = Some(vpos); measure_b = None; app_mode = AppMode::MeasureSecond; 
                            }
                        }
                    }
                    AppMode::MeasureSecond => {
                        let mut best_dist = f32::MAX;
                        for cs in &copc_stages {
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

            if color_mode != prev_color_mode {
                let gl = unsafe { get_internal_gl() };
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
        }

        // ---- Tile mode rendering ----
        if copc_stages.is_empty() && let Some(ref mgr) = tile_mgr {
            //println!("tile mode");
            // Section preview (in 3D)
            match app_mode {
                AppMode::SectionPickEnd => {
                    if let (Some(start), Some((wx, wy))) = (section_start, cursor_world) {
                        draw_line_3d(
                            vec3(start.0, start.1, mid_z),
                            vec3(wx, wy, mid_z),
                            YELLOW,
                        );
                    }
                }
                AppMode::SectionAdjustDepth => {
                    if let (Some(start), Some(end)) = (section_start, section_end) {
                        //println!("here!");
                        //set_camera(&camera);
                        draw_section_corridor(start, end, section_depth, height_range);
                    }
                }
                _ => {}
            }

            // Draw tile boundaries at z=0
            set_camera(&camera);
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
            if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive) {
                let old_front = front;
                yaw = new_yaw;
                pitch = new_pitch;
                front = vec3(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).normalize();
                (right, up) = recompute_right_up(yaw, pitch, front);
                position = orbit_pivot + front * (position - orbit_pivot).dot(old_front);
            }

            // egui UI
            let prev_color_mode = color_mode;
            let mut egui_wants_pointer = false;
            let mut request_load: Option<usize> = None;
            let mut request_unload: Option<usize> = None;
            let mut do_zoom_all = false;
            let mut do_zoom_selected = false;
            let mut tile_override_changed: Option<usize> = None;
            egui_macroquad::ui(|egui_ctx| {
                egui_wants_pointer = egui_ctx.is_pointer_over_area();
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        // Cross Section button
                        let section_label = match app_mode {
                            AppMode::View3D | AppMode::PointPick | AppMode::MeasureFirst | AppMode::MeasureSecond => "Cross Section",
                            AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                            AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                            AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                            AppMode::SectionView | AppMode::SectionMeasureFirst | AppMode::SectionMeasureSecond => "Back to 3D",
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
                                    section_data = None;
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
                                ui.label(format!("Point Info ({})", mgr.tiles[tile_idx].filename));
                                ui.label(format!("  X: {:.3}  Y: {:.3}  Z: {:.3}", p.x, p.y, p.z));
                                ui.label(format!("  Intensity: {}", p.intensity));
                                ui.label(format!("  Classification: {}", class_label(p, &class_defs)));
                                ui.label(format!("  Return: {}/{}", p.return_number, p.number_of_returns));
                                if let Some(ref color) = p.color {
                                    ui.label(format!("  RGB: {} {} {}", color.red, color.green, color.blue));
                                }
                                if let Some(time) = p.gps_time {
                                    ui.label(format!("  GPS time: {:.6}", time));
                                }
                                ui.label(format!("  Intensity: {}", p.point_source_id));
                            
                        }
                    });
                polylines_window(egui_ctx, &mut poly_files, &mut selected_poly, &mut poly_load_error, data_offset, background);
            });
            egui_macroquad::draw();
            egui_wants_pointer_prev = egui_wants_pointer;

            if do_zoom_all {
                let aspect = screen_width() / screen_height().max(1.0);
                ortho_scale = (data_extent_y * 0.5).max(data_extent_x * 0.5 / aspect) * 1.05;
                position = vec3(data_extent_x * 0.5, data_extent_y * 0.5, height_range * 0.5)
                    - front * (data_extent_x.max(data_extent_y) * 2.0);
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
                            section_data = Some(sec);
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

            if color_mode != prev_color_mode &&
                let Some(ref mut mgr) = tile_mgr {
                    let gl = unsafe { get_internal_gl() };
                    mgr.rebuild_colors(&mut *gl.quad_context, color_mode);

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
                        ui.label("…or drag a .laz / .copc.laz file onto the window (X11 only).");
                    });
            });
            egui_macroquad::draw();
        }

        next_frame().await
    }
}
