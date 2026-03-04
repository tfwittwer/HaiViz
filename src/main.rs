use crate::glam::{vec3,Mat4};

use macroquad::prelude::*;

//use macroquad::prelude::Mat4;


const MOVE_SPEED: f32 = 0.1;
const LOOK_SPEED: f32 = 0.1;

#[macroquad::main("Raw miniquad")]
async fn main() {
    let stage = {
        let InternalGlContext {
            quad_context: ctx, ..
        } = unsafe { get_internal_gl() };

        raw_miniquad::Stage::new(ctx)
    };

    let mut x = 0.0;
    let mut switch = false;
    let bounds = 8.0;

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
    let mut up = right.cross(front).normalize();    let mut position = vec3(0.0, 1.0, 0.0);
    let mut last_mouse_position: Vec2 = mouse_position().into();

    let mut grabbed = false;
    set_cursor_grab(grabbed);
    show_mouse(!grabbed);

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
            position += front * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
            position -= front * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Left)  || is_key_down(KeyCode::A) {
            position -= right * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Right)  || is_key_down(KeyCode::D) {
            position += right * MOVE_SPEED;
        }

        let (_mouse_wheel_x, mouse_wheel_y) = mouse_wheel();
        if mouse_wheel_y.abs() > 0.001 {
            if mouse_wheel_y < 0.0 {
                position -= front * MOVE_SPEED;
                
            }
            else if mouse_wheel_y > 0.0 {
                position += front * MOVE_SPEED;
            }
        }

        let mouse_position: Vec2 = mouse_position().into();
        let mouse_delta = mouse_position - last_mouse_position;

        last_mouse_position = mouse_position;


        if grabbed || is_mouse_button_down(MouseButton::Right) {
            yaw += mouse_delta.x * delta * LOOK_SPEED;
            pitch += mouse_delta.y * delta * -LOOK_SPEED;

            pitch = if pitch > 1.5 { 1.5 } else { pitch };
            pitch = if pitch < -1.5 { -1.5 } else { pitch };

            front = vec3(
                yaw.cos() * pitch.cos(),
                pitch.sin(),
                yaw.sin() * pitch.cos(),
            )
            .normalize();

            right = front.cross(world_up).normalize();
            up = right.cross(front).normalize();

            x += if switch { 0.04 } else { -0.04 };
            if x >= bounds || x <= -bounds {
                switch = !switch;
            }
        }

        clear_background(LIGHTGRAY);

        let camera = Camera3D {
            position,
            up,
            target: position + front,
            //projection: Projection::Orthographics,
            z_far: 100000.0,
            ..Default::default()
        };
        set_camera(&camera);

        draw_cube_wires(vec3(0., 1., -6.), vec3(2., 2., 2.), GREEN);
        draw_cube_wires(vec3(0., 1., 6.), vec3(2., 2., 2.), BLUE);
        draw_cube_wires(vec3(16., 13., 8.), vec3(2., 2., 2.), RED);


        // Render some primitives in camera space

        //draw_line(-0.4, 0.4, -0.8, 0.9, 0.05, BLUE);
        //draw_rectangle(-0.3, 0.3, 0.2, 0.2, GREEN);
        //draw_circle(0., 0., 0.1, YELLOW);

        {
            let mut gl = unsafe { get_internal_gl() };

            // Ensure that macroquad's shapes are not going to be lost
            gl.flush();

            let t = get_time();

            gl.quad_context.apply_pipeline(&stage.pipeline);

            gl.quad_context
                .begin_default_pass(miniquad::PassAction::Nothing);
            gl.quad_context.apply_bindings(&stage.bindings);

            let t = t as f64 * 0.3;

            //let (view_mat, proj_mat) = (camera.viewport(),camera.projection);
            let view_mat = camera.matrix();
            let proj_mat = Mat4::perspective_rh_gl(60.0f32.to_radians(), screen_width() / screen_height(), 0.01, 10.0);
            let view_proj: glam::Mat4 = proj_mat*view_mat;

            gl.quad_context
                .apply_uniforms(miniquad::UniformsSource::table(
                    &raw_miniquad::shader::Uniforms {
                        offset: (t.sin() as f32 * 0.5, (t * 3.).cos() as f32 * 0.5),
                        mvp: view_proj,
                    },
                ));
            gl.quad_context.draw(0, 6, 1);
            gl.quad_context.end_render_pass();
        }

        // Back to screen space, render some text
        set_camera(&Camera2D {
            zoom: vec2(1., screen_width() / screen_height()),
            ..Default::default()
        });

        set_default_camera();
        draw_text("HELLO", 30.0, 200.0, 30.0, BLACK);

        next_frame().await
    }
}

mod raw_miniquad {
    use miniquad::*;

    #[repr(C)]
    struct Vec2 {
        x: f32,
        y: f32,
    }
    #[repr(C)]
    struct Vertex {
        pos: Vec2,
        uv: Vec2,
    }

    pub struct Stage {
        pub pipeline: Pipeline,
        pub bindings: Bindings,
    }

    impl Stage {                
        pub fn new(ctx: &mut dyn RenderingBackend) -> Stage {
            #[rustfmt::skip]
            let vertices: [Vertex; 4] = [
                Vertex { pos : Vec2 { x: -0.5, y: -0.5 }, uv: Vec2 { x: 0., y: 0. } },
                Vertex { pos : Vec2 { x:  0.5, y: -0.5 }, uv: Vec2 { x: 1., y: 0. } },
                Vertex { pos : Vec2 { x:  0.5, y:  0.5 }, uv: Vec2 { x: 1., y: 1. } },
                Vertex { pos : Vec2 { x: -0.5, y:  0.5 }, uv: Vec2 { x: 0., y: 1. } },
            ];
            let vertex_buffer = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&vertices),
            );

            let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
            let index_buffer = ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&indices[..]),
            );

            let pixels: [u8; 4 * 4 * 4] = [
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
                0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ];
            let texture = ctx.new_texture_from_rgba8(4, 4, &pixels);

            let bindings = Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer,
                images: vec![texture],
            };

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
                    VertexAttribute::new("pos", VertexFormat::Float2),
                    VertexAttribute::new("uv", VertexFormat::Float2),
                ],
                shader,
                Default::default(),
            );

            Stage { pipeline, bindings }
        }
    }

    pub mod shader {
        use miniquad::*;

        pub const VERTEX: &str = r#"#version 100
        attribute vec2 pos;
        attribute vec2 uv;

        uniform vec2 offset;
        uniform mat4 mvp;

        varying lowp vec2 texcoord;

        void main() {
            //gl_Position = mvp*vec4(pos, 1, 1);
            gl_Position = vec4(pos, 1, 1);
            texcoord = uv;
        }"#;

        pub const FRAGMENT: &str = r#"#version 100
            varying lowp vec2 texcoord;

            uniform sampler2D tex;

            void main() {
            gl_FragColor = texture2D(tex, texcoord);
        }"#;

        pub fn meta() -> ShaderMeta {
            ShaderMeta {
                images: vec!["tex".to_string()],
                uniforms: UniformBlockLayout {
                    uniforms: vec![UniformDesc::new("offset", UniformType::Float2)],
                },
            }
        }

        #[repr(C)]
        pub struct Uniforms {
            pub offset: (f32, f32),
            pub mvp: glam::Mat4,
            //pub projection: macroquad::prelude::Mat4,
            //pub model: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32),
            //pub projection: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32),
        }
    }
}