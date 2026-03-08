use std::env;

use console::style;
use crate::glam::vec3;

use macroquad::prelude::*;

//use raw_miniquad::{SectionData};
use crate::raw_miniquad::raw_miniquad::{SectionData,Stage};

const MOVE_SPEED: f32 = 0.5;
const LOOK_SPEED: f32 = 0.5;

mod raw_miniquad;

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
            FaceDef { corners: [4, 5, 6, 7], label: "Top",    normal: [ 0.0, 0.0, 1.0], snap_yaw: PI/2.0, snap_pitch: -1.5,     color: Color::new(0.3, 0.5, 0.8, 0.85) },
            FaceDef { corners: [3, 2, 1, 0], label: "Bot",    normal: [ 0.0, 0.0,-1.0], snap_yaw: PI/2.0, snap_pitch:  1.5,     color: Color::new(0.2, 0.3, 0.5, 0.85) },
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
            return Some((PI / 2.0, -1.5));
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
}

impl ColorMode {
    fn label(&self) -> &'static str {
        match self {
            ColorMode::Height => "Height",
            ColorMode::Intensity => "Intensity",
            ColorMode::Classification => "Classification",
            ColorMode::Rgb => "RGB",
        }
    }

    const ALL: [ColorMode; 4] = [
        ColorMode::Height,
        ColorMode::Intensity,
        ColorMode::Classification,
        ColorMode::Rgb,
    ];
}

#[derive(Clone, Copy, PartialEq)]
pub enum Background {
    Black,
    White,
}

impl Background {
    fn label(&self) -> &'static str {
        match self {
            Background::Black => "Black",
            Background::White => "White",
        }
    }

    fn color(&self) -> macroquad::color::Color {
        match self {
            Background::Black => BLACK,
            Background::White => WHITE,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum AppMode {
    View3D,
    PointPick,
    SectionPickStart,
    SectionPickEnd,
    SectionAdjustDepth,
    SectionView,
}

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

/// Draw the section corridor as 3D wireframe box.
fn draw_section_corridor(
    start: (f32, f32),
    end: (f32, f32),
    depth: f32,
    height_range: f32,
) {
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
    // Bottom rectangle
    for i in 0..4 {
        draw_line_3d(b[i], b[(i + 1) % 4], col);
    }
    // Top rectangle
    for i in 0..4 {
        draw_line_3d(t[i], t[(i + 1) % 4], col);
    }
    // Verticals
    for i in 0..4 {
        draw_line_3d(b[i], t[i], col);
    }
    // Center line
    draw_line_3d(
        vec3(start.0, start.1, height_range * 0.5),
        vec3(end.0, end.1, height_range * 0.5),
        RED,
    );
}

#[macroquad::main("HaiViz")]
async fn main() {
    if env::args().count() != 2 {
        println!("{}", style("Usage: haiviz pointcloud.laz").red());
        return;
    }
    let args: Vec<String> = env::args().collect();
    let cloudfile = &args[1];

    let mut color_mode = ColorMode::Height;
    let mut background = Background::Black;
    let mut app_mode = AppMode::View3D;

    let mut stage = {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        Stage::new(ctx, cloudfile, color_mode)
    };

    let height_range = stage.height_range();

    let world_up = vec3(0.0, 0.0, 1.0);
    let mut yaw: f32 = 1.18;
    let mut pitch: f32 = 0.0;

    let mut front = vec3(
        yaw.cos() * pitch.cos(),
        yaw.sin() * pitch.cos(),
        pitch.sin(),
    )
    .normalize();
    let mut right = front.cross(world_up).normalize();
    let mut up = right.cross(front).normalize();
    let mut position = vec3(0.0, 0.0, 1.0);
    let mut last_mouse_position: Vec2 = mouse_position().into();

    let mut grabbed = false;
    set_cursor_grab(grabbed);
    show_mouse(!grabbed);

    let mut ortho_scale: f32 = 50.0;

    // Section state
    let mut section_start: Option<(f32, f32)> = None;
    let mut section_end: Option<(f32, f32)> = None;
    let mut section_depth: f32 = 2.0;
    let mut section_data: Option<SectionData> = None;
    // Section 2D view state
    let mut section_pan = glam::Vec2::ZERO;
    let mut section_zoom: f32 = 50.0;

    // Point pick state
    let mut picked_point: Option<usize> = None;

    loop {
        let delta = get_frame_time();

        if is_key_pressed(KeyCode::Escape) {
            if app_mode != AppMode::View3D {
                app_mode = AppMode::View3D;
                section_data = None;
            } else {
                break;
            }
        }

        if app_mode == AppMode::SectionView {
            // ---- Section 2D view ----
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

            // Middle mouse drag to pan section view
            let sec_mouse_pos: Vec2 = mouse_position().into();
            let sec_mouse_delta = sec_mouse_pos - last_mouse_position;
            last_mouse_position = sec_mouse_pos;
            if is_mouse_button_down(MouseButton::Middle) {
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

            clear_background(background.color());

            // Render section points
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
                    stage.draw_section(&mut *gl.quad_context, sec, &section_mvp);
                    gl.quad_context.end_render_pass();
                }

                set_default_camera();

                // Draw reference axes in screen space
                // Map section world coords to screen coords
                let world_to_screen_x = |wx: f32| -> f32 {
                    (wx - (section_pan.x - half_w)) / (2.0 * half_w) * screen_width()
                };
                let world_to_screen_y = |wy: f32| -> f32 {
                    (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * screen_height()
                };

                // Ground line (Y=0)
                let ground_sy = world_to_screen_y(0.0);
                if ground_sy > 0.0 && ground_sy < screen_height() {
                    draw_line(0.0, ground_sy, screen_width(), ground_sy, 1.0, Color::new(0.4, 0.4, 0.4, 0.5));
                    draw_text("0m", 5.0, ground_sy - 3.0, 16.0, GRAY);
                }

                // Section start/end markers
                let sx0 = world_to_screen_x(0.0);
                let sx1 = world_to_screen_x(sec.length);
                draw_line(sx0, 0.0, sx0, screen_height(), 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_line(sx1, 0.0, sx1, screen_height(), 1.0, Color::new(0.4, 0.4, 0.0, 0.5));
                draw_text("start", sx0 + 3.0, 30.0, 16.0, YELLOW);
                draw_text(&format!("{:.1}m", sec.length), sx1 + 3.0, 30.0, 16.0, YELLOW);
            }

            // egui
            let prev_color_mode = color_mode;
            egui_macroquad::ui(|egui_ctx| {
                egui_macroquad::egui::Window::new("Controls")
                    .default_pos([10.0, 10.0])
                    .show(egui_ctx, |ui| {
                        if ui.button("Back to 3D").clicked() {
                            app_mode = AppMode::View3D;
                            section_data = None;
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
                            egui_macroquad::egui::ComboBox::from_id_salt("background")
                                .selected_text(background.label())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut background, Background::Black, "Black");
                                    ui.selectable_value(&mut background, Background::White, "White");
                                });
                        });
                        if let Some(ref sec) = section_data {
                            ui.label(format!(
                                "Section: {:.1}m | {} points | h: {:.1}..{:.1}m",
                                sec.length, sec.num_points, sec.height_min, sec.height_max
                            ));
                        }
                    });
            });
            egui_macroquad::draw();

            if color_mode != prev_color_mode {
                // Rebuild section with new colors
                if let (Some(start), Some(end)) = (section_start, section_end) {
                    let gl = unsafe { get_internal_gl() };
                    section_data = Some(stage.extract_section(
                        &mut *gl.quad_context, start, end, section_depth, color_mode,
                    ));
                }
            }

            next_frame().await;
            continue;
        }

        // ---- 3D view modes ----
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

        // Scroll: zoom in 3D mode, adjust depth in SectionAdjustDepth
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

        if grabbed || is_mouse_button_down(MouseButton::Right) {
            yaw += mouse_delta.x * delta * LOOK_SPEED;
            pitch += mouse_delta.y * delta * -LOOK_SPEED;
            pitch = pitch.clamp(-1.5, 1.5);

            front = vec3(
                yaw.cos() * pitch.cos(),
                yaw.sin() * pitch.cos(),
                pitch.sin(),
            )
            .normalize();

            right = front.cross(world_up).normalize();
            up = right.cross(front).normalize();
        }

        // Middle mouse drag to pan
        if is_mouse_button_down(MouseButton::Middle) {
            let units_per_pixel = ortho_scale * 2.0 / screen_height();
            position -= right * mouse_delta.x * units_per_pixel;
            position += up * mouse_delta.y * units_per_pixel;
        }

        clear_background(background.color());

        let camera = Camera3D {
            position,
            up,
            target: position + front,
            projection: Projection::Orthographics,
            fovy: ortho_scale * 2.0,
            z_far: 100000.0,
            z_near: -100000.0,
            ..Default::default()
        };
        set_camera(&camera);
        let view_proj: glam::Mat4 = camera.matrix();

        //draw_cube_wires(vec3(90., 9., 17.), vec3(2., 2., 2.), GREEN);
        //draw_cube_wires(vec3(0., 1., 6.), vec3(2., 2., 2.), BLUE);
        //draw_cube_wires(vec3(16., 13., 8.), vec3(2., 2., 2.), RED);

        // Section preview drawing (in 3D, behind points)
        let mid_z = height_range * 0.5;
        let (mx, my) = macroquad::input::mouse_position();
        let cursor_world = screen_to_world_xy(mx, my, &view_proj, mid_z);

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

        // Render point cloud
        let drawn = {
            let mut gl = unsafe { get_internal_gl() };
            gl.flush();

            gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
            let drawn = stage.draw_visible(&mut *gl.quad_context, &view_proj);
            gl.quad_context.end_render_pass();
            drawn
        };

        set_default_camera();

        // View cube (only interactive in View3D)
        let vc_interactive = app_mode == AppMode::View3D || app_mode == AppMode::PointPick;
        if let Some((new_yaw, new_pitch)) = view_cube::draw_and_interact(front, right, up, vc_interactive) {
            yaw = new_yaw;
            pitch = new_pitch;
            front = vec3(
                yaw.cos() * pitch.cos(),
                yaw.sin() * pitch.cos(),
                pitch.sin(),
            )
            .normalize();
            right = front.cross(world_up).normalize();
            up = right.cross(front).normalize();
        }

        // egui UI
        let prev_color_mode = color_mode;
        let mut egui_wants_pointer = false;
        egui_macroquad::ui(|egui_ctx| {
            egui_wants_pointer = egui_ctx.is_pointer_over_area();
            egui_macroquad::egui::Window::new("Controls")
                .default_pos([10.0, 10.0])
                .show(egui_ctx, |ui| {
                    let section_label = match app_mode {
                        AppMode::View3D | AppMode::PointPick => "Cross Section",
                        AppMode::SectionPickStart => "Click start point... (Esc to cancel)",
                        AppMode::SectionPickEnd => "Click end point... (Esc to cancel)",
                        AppMode::SectionAdjustDepth => "Scroll depth, click to confirm",
                        AppMode::SectionView => "Back to 3D",
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
                        egui_macroquad::egui::ComboBox::from_id_salt("background")
                            .selected_text(background.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut background, Background::Black, "Black");
                                ui.selectable_value(&mut background, Background::White, "White");
                            });
                    });
                    ui.label(format!(
                        "{}/{} points | {} chunks | FPS: {:.0} | zoom: {:.1}",
                        drawn, stage.total_points, stage.chunks.len(),
                        1.0 / get_frame_time(), ortho_scale
                    ));
                    if app_mode == AppMode::SectionAdjustDepth {
                        ui.label(format!("Section depth: {:.1}m", section_depth));
                    }
                    if let Some(idx) = picked_point {
                        ui.separator();
                        let p = &stage.cloud.points[idx];
                        ui.label("Point Info:");
                        ui.label(format!("  X: {:.3}  Y: {:.3}  Z: {:.3}", p.x, p.y, p.z));
                        ui.label(format!("  Intensity: {}", p.intensity));
                        let class_id: u8 = p.classification.into();
                        ui.label(format!("  Classification: {} ({:?})", class_id, p.classification));
                        ui.label(format!("  Return: {}/{}", p.return_number, p.number_of_returns));
                        if let Some(ref color) = p.color {
                            ui.label(format!("  RGB: {} {} {}", color.red, color.green, color.blue));
                        }
                        if let Some(time) = p.gps_time {
                            ui.label(format!("  GPS time: {:.6}", time));
                        }
                    }
                });
        });
        egui_macroquad::draw();

        // Handle mode-specific clicks (only if egui didn't consume the click)
        if !egui_wants_pointer && is_mouse_button_pressed(MouseButton::Left) {
            match app_mode {
                AppMode::PointPick => {
                    picked_point = stage.find_nearest_point(&view_proj, mx, my, screen_width(), screen_height());
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
                        let sec = stage.extract_section(
                            &mut *gl.quad_context, start, end, section_depth, color_mode,
                        );
                        // Set initial section view to fit the data
                        section_pan = glam::Vec2::new(sec.length / 2.0, (sec.height_min + sec.height_max) / 2.0);
                        section_zoom = (sec.length / 2.0).max((sec.height_max - sec.height_min) / 2.0).max(1.0) * 1.2;
                        section_data = Some(sec);
                        app_mode = AppMode::SectionView;
                    }
                }
                _ => {}
            }
        }

        if color_mode != prev_color_mode {
            let gl = unsafe { get_internal_gl() };
            stage.rebuild_colors(&mut *gl.quad_context, color_mode);
        }

        next_frame().await
    }
}

