use std::env;

use console::style;
use crate::glam::vec3;

use macroquad::prelude::*;

const MOVE_SPEED: f32 = 0.5;
const LOOK_SPEED: f32 = 0.5;

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

#[macroquad::main("Hai Performance Point Cloud Viewer")]
async fn main() {
    if env::args().count() != 2 {
        println!("{}", style("Usage: haiviz pointcloud.laz").red());
        return;
    }
    let args: Vec<String> = env::args().collect();
    let cloudfile = &args[1];

    let mut color_mode = ColorMode::Height;
    let mut background = Background::Black;

    let mut stage = {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        raw_miniquad::Stage::new(ctx, cloudfile, color_mode)
    };

    let world_up = vec3(0.0, 1.0, 0.0);
    let mut yaw: f32 = 1.18;
    let mut pitch: f32 = 0.0;

    let mut front = vec3(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos(),
    )
    .normalize();
    let mut right = front.cross(world_up).normalize();
    let mut up = right.cross(front).normalize();
    let mut position = vec3(0.0, 1.0, 0.0);
    let mut last_mouse_position: Vec2 = mouse_position().into();

    let mut grabbed = false;
    set_cursor_grab(grabbed);
    show_mouse(!grabbed);

    // Orthographic zoom: visible half-height in world units
    let mut ortho_scale: f32 = 50.0;

    loop {
        let delta = get_frame_time();

        if is_key_pressed(KeyCode::Escape) {
            break;
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

        let (_mouse_wheel_x, mouse_wheel_y) = mouse_wheel();
        if mouse_wheel_y.abs() > 0.001 {
            let zoom_factor = if mouse_wheel_y > 0.0 { 0.9 } else { 1.1 };
            ortho_scale *= zoom_factor;
            ortho_scale = ortho_scale.clamp(0.5, 10000.0);
        }

        let mouse_position: Vec2 = mouse_position().into();
        let mouse_delta = mouse_position - last_mouse_position;
        last_mouse_position = mouse_position;

        if grabbed || is_mouse_button_down(MouseButton::Right) {
            yaw += mouse_delta.x * delta * LOOK_SPEED;
            pitch += mouse_delta.y * delta * -LOOK_SPEED;
            pitch = pitch.clamp(-1.5, 1.5);

            front = vec3(
                yaw.cos() * pitch.cos(),
                pitch.sin(),
                yaw.sin() * pitch.cos(),
            )
            .normalize();

            right = front.cross(world_up).normalize();
            up = right.cross(front).normalize();
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

        draw_cube_wires(vec3(90., 9., 17.), vec3(2., 2., 2.), GREEN);
        draw_cube_wires(vec3(0., 1., 6.), vec3(2., 2., 2.), BLUE);
        draw_cube_wires(vec3(16., 13., 8.), vec3(2., 2., 2.), RED);

        let drawn = {
            let mut gl = unsafe { get_internal_gl() };
            gl.flush();

            gl.quad_context.begin_default_pass(miniquad::PassAction::Nothing);
            let view_proj: glam::Mat4 = camera.matrix();
            let drawn = stage.draw_visible(&mut *gl.quad_context, &view_proj);
            gl.quad_context.end_render_pass();
            drawn
        };

        set_default_camera();

        // egui UI
        let prev_color_mode = color_mode;
        egui_macroquad::ui(|egui_ctx| {
            egui_macroquad::egui::Window::new("Controls")
                .default_pos([10.0, 10.0])
                .show(egui_ctx, |ui| {
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
                });
        });
        egui_macroquad::draw();

        if color_mode != prev_color_mode {
            let gl = unsafe { get_internal_gl() };
            stage.rebuild_colors(&mut *gl.quad_context, color_mode);
        }

        next_frame().await
    }
}

mod raw_miniquad {
    use std::collections::HashMap;

    use las::{Point, Reader};
    use miniquad::*;

    use crate::ColorMode;

    const CHUNK_SIZE: f32 = 50.0;

    /// Point cloud backed by las::Point for full attribute access.
    pub struct VizPointcloud {
        pub points: Vec<Point>,
        pub min: [f64; 3],
        pub max: [f64; 3],
        pub intensity_min: u16,
        pub intensity_max: u16,
    }

    impl VizPointcloud {
        pub fn load(filename: &str) -> VizPointcloud {
            let mut reader = Reader::from_path(filename).unwrap();
            let n = reader.header().number_of_points() as usize;

            let mut points = Vec::with_capacity(n);
            reader.read_all_points_into(&mut points).unwrap();

            let mut min = [f64::MAX; 3];
            let mut max = [f64::MIN; 3];
            let mut intensity_min = u16::MAX;
            let mut intensity_max = u16::MIN;

            for point in &points {
                min[0] = min[0].min(point.x);
                min[1] = min[1].min(point.y);
                min[2] = min[2].min(point.z);
                max[0] = max[0].max(point.x);
                max[1] = max[1].max(point.y);
                max[2] = max[2].max(point.z);
                intensity_min = intensity_min.min(point.intensity);
                intensity_max = intensity_max.max(point.intensity);
            }

            println!("{} points loaded from {}", points.len(), filename);
            println!("Intensity range: {}..{}", intensity_min, intensity_max);
            VizPointcloud { points, min, max, intensity_min, intensity_max }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Vertex {
        pub pos: [f32; 3],
        pub color: [f32; 3],
    }

    pub struct Chunk {
        bindings: Bindings,
        num_points: i32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        point_indices: Vec<usize>,
    }

    pub struct Stage {
        pipeline: Pipeline,
        pub chunks: Vec<Chunk>,
        pub total_points: i32,
        pub cloud: VizPointcloud,
        positions: Vec<[f32; 3]>,
    }

    fn color_for_point(point: &Point, pos_y: f32, height_range: f32, intensity_min: u16, intensity_range: f32, mode: ColorMode) -> [f32; 3] {
        match mode {
            ColorMode::Height => {
                let h = if height_range > 0.0 { (pos_y / height_range).clamp(0.0, 1.0) } else { 0.5 };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMode::Intensity => {
                let i = if intensity_range > 0.0 {
                    ((point.intensity - intensity_min) as f32 / intensity_range).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                [i, i, i]
            }
            ColorMode::Classification => {
                classification_color(point.classification.into())
            }
            ColorMode::Rgb => {
                if let Some(color) = &point.color {
                    let r = color.red as f32 / 65535.0;
                    let g = color.green as f32 / 65535.0;
                    let b = color.blue as f32 / 65535.0;
                    [r, g, b]
                } else {
                    [0.5, 0.5, 0.5]
                }
            }
        }
    }

    fn classification_color(class: u8) -> [f32; 3] {
        match class {
            0 => [0.5, 0.5, 0.5],     // Created/Never classified
            1 => [0.6, 0.6, 0.6],     // Unclassified
            2 => [0.6, 0.4, 0.2],     // Ground
            3 => [0.0, 0.8, 0.0],     // Low Vegetation
            4 => [0.0, 0.6, 0.0],     // Medium Vegetation
            5 => [0.0, 0.4, 0.0],     // High Vegetation
            6 => [1.0, 0.0, 0.0],     // Building
            7 => [1.0, 0.5, 0.0],     // Low Point (noise)
            8 => [0.8, 0.8, 0.0],     // Model Key-point
            9 => [0.0, 0.0, 1.0],     // Water
            10 => [0.8, 0.8, 0.8],    // Rail
            11 => [0.3, 0.3, 0.3],    // Road Surface
            12 => [0.9, 0.9, 0.0],    // Overlap
            13 => [0.7, 0.7, 0.0],    // Wire - Guard
            14 => [0.9, 0.9, 0.0],    // Wire - Conductor
            15 => [0.4, 0.0, 0.6],    // Transmission Tower
            16 => [0.6, 0.0, 0.8],    // Wire - Connector
            17 => [0.2, 0.2, 0.6],    // Bridge Deck
            18 => [1.0, 0.0, 0.5],    // High Noise
            _ => {
                let h = (class as f32 * 0.618033988) % 1.0;
                hsv_to_rgb(h, 0.8, 0.9)
            }
        }
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
        let i = (h * 6.0).floor();
        let f = h * 6.0 - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);
        match (i as i32) % 6 {
            0 => [v, t, p],
            1 => [q, v, p],
            2 => [p, v, t],
            3 => [p, q, v],
            4 => [t, p, v],
            _ => [v, p, q],
        }
    }

    fn extract_frustum_planes(vp: &glam::Mat4) -> [[f32; 4]; 6] {
        let c = vp.to_cols_array_2d();
        let mut planes = [[0.0f32; 4]; 6];
        for i in 0..4 {
            planes[0][i] = c[i][3] + c[i][0];
            planes[1][i] = c[i][3] - c[i][0];
            planes[2][i] = c[i][3] + c[i][1];
            planes[3][i] = c[i][3] - c[i][1];
            planes[4][i] = c[i][3] + c[i][2];
            planes[5][i] = c[i][3] - c[i][2];
        }
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 0.0 {
                for v in plane.iter_mut() {
                    *v /= len;
                }
            }
        }
        planes
    }

    fn aabb_outside_frustum(planes: &[[f32; 4]; 6], amin: &[f32; 3], amax: &[f32; 3]) -> bool {
        for plane in planes {
            let px = if plane[0] >= 0.0 { amax[0] } else { amin[0] };
            let py = if plane[1] >= 0.0 { amax[1] } else { amin[1] };
            let pz = if plane[2] >= 0.0 { amax[2] } else { amin[2] };
            if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
                return true;
            }
        }
        false
    }

    impl Stage {
        pub fn new(ctx: &mut dyn RenderingBackend, filename: &str, color_mode: ColorMode) -> Stage {
            let cloud = VizPointcloud::load(filename);

            let height_range = (cloud.max[2] - cloud.min[2]) as f32;
            let intensity_range = (cloud.intensity_max - cloud.intensity_min) as f32;

            let positions: Vec<[f32; 3]> = cloud
                .points
                .iter()
                .map(|p| [
                    (p.x - cloud.min[0]) as f32,
                    (p.z - cloud.min[2]) as f32, // z -> y (up)
                    (p.y - cloud.min[1]) as f32,
                ])
                .collect();

            let vertices: Vec<Vertex> = cloud
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| Vertex {
                    pos: positions[i],
                    color: color_for_point(p, positions[i][1], height_range, cloud.intensity_min, intensity_range, color_mode),
                })
                .collect();

            let mut chunk_map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            for (i, pos) in positions.iter().enumerate() {
                let cx = (pos[0] / CHUNK_SIZE).floor() as i32;
                let cz = (pos[2] / CHUNK_SIZE).floor() as i32;
                chunk_map.entry((cx, cz)).or_default().push(i);
            }

            let shader = ctx
                .new_shader(
                    miniquad::ShaderSource::Glsl {
                        vertex: shader::VERTEX,
                        fragment: shader::FRAGMENT,
                    },
                    shader::meta(),
                )
                .unwrap();

            let pipeline = ctx.new_pipeline(
                &[BufferLayout::default()],
                &[
                    VertexAttribute::new("pos", VertexFormat::Float3),
                    VertexAttribute::new("color", VertexFormat::Float3),
                ],
                shader,
                PipelineParams {
                    primitive_type: PrimitiveType::Points,
                    depth_test: Comparison::LessOrEqual,
                    depth_write: true,
                    ..Default::default()
                },
            );

            let total_points = vertices.len() as i32;

            let chunks: Vec<Chunk> = chunk_map
                .into_iter()
                .map(|(_key, point_indices)| {
                    let chunk_verts: Vec<Vertex> =
                        point_indices.iter().map(|&i| vertices[i]).collect();

                    let mut aabb_min = [f32::MAX; 3];
                    let mut aabb_max = [f32::MIN; 3];
                    for v in &chunk_verts {
                        for j in 0..3 {
                            aabb_min[j] = aabb_min[j].min(v.pos[j]);
                            aabb_max[j] = aabb_max[j].max(v.pos[j]);
                        }
                    }

                    let n = chunk_verts.len() as i32;
                    let vertex_buffer = ctx.new_buffer(
                        BufferType::VertexBuffer,
                        BufferUsage::Dynamic,
                        BufferSource::slice(&chunk_verts),
                    );
                    let indices: Vec<u32> = (0..n as u32).collect();
                    let index_buffer = ctx.new_buffer(
                        BufferType::IndexBuffer,
                        BufferUsage::Immutable,
                        BufferSource::slice(&indices),
                    );

                    Chunk {
                        bindings: Bindings {
                            vertex_buffers: vec![vertex_buffer],
                            index_buffer,
                            images: vec![],
                        },
                        num_points: n,
                        aabb_min,
                        aabb_max,
                        point_indices,
                    }
                })
                .collect();

            println!("{} points in {} chunks ({}m cells), height range: {:.1}m",
                total_points, chunks.len(), CHUNK_SIZE, height_range);

            Stage {
                pipeline,
                chunks,
                total_points,
                cloud,
                positions,
            }
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let height_range = (self.cloud.max[2] - self.cloud.min[2]) as f32;
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;

            for chunk in &mut self.chunks {
                let chunk_verts: Vec<Vertex> = chunk
                    .point_indices
                    .iter()
                    .map(|&i| {
                        let p = &self.cloud.points[i];
                        Vertex {
                            pos: self.positions[i],
                            color: color_for_point(p, self.positions[i][1], height_range, self.cloud.intensity_min, intensity_range, color_mode),
                        }
                    })
                    .collect();

                ctx.buffer_update(
                    chunk.bindings.vertex_buffers[0],
                    BufferSource::slice(&chunk_verts),
                );
            }
        }

        pub fn draw_visible(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4) -> i32 {
            let planes = extract_frustum_planes(view_proj);
            let mut drawn = 0;

            ctx.apply_pipeline(&self.pipeline);
            for chunk in &self.chunks {
                if aabb_outside_frustum(&planes, &chunk.aabb_min, &chunk.aabb_max) {
                    continue;
                }
                ctx.apply_bindings(&chunk.bindings);
                ctx.apply_uniforms(miniquad::UniformsSource::table(
                    &shader::Uniforms {
                        mvp: *view_proj,
                    },
                ));
                ctx.draw(0, chunk.num_points, 1);
                drawn += chunk.num_points;
            }
            drawn
        }
    }

    pub mod shader {
        use miniquad::*;

        pub const VERTEX: &str = r#"#version 100
            attribute vec3 pos;
            attribute vec3 color;

            uniform mat4 mvp;

            varying lowp vec3 v_color;

            void main() {
                gl_Position = mvp * vec4(pos, 1.0);
                gl_PointSize = 2.0;
                v_color = color;
            }"#;

        pub const FRAGMENT: &str = r#"#version 100
            varying lowp vec3 v_color;

            void main() {
                gl_FragColor = vec4(v_color, 1.0);
            }"#;

        pub fn meta() -> ShaderMeta {
            ShaderMeta {
                images: vec![],
                uniforms: UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("mvp", UniformType::Mat4),
                    ],
                },
            }
        }

        #[repr(C)]
        pub struct Uniforms {
            pub mvp: glam::Mat4,
        }
    }
}
