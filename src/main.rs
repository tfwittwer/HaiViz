use macroquad::prelude::*;
use haicloudrs::pointcloud::Pointcloud;
// use glam::vec3;
use std::env;
use console::style;

const MOVE_SPEED: f32 = 0.1;
const LOOK_SPEED: f32 = 0.1;

fn window_conf() -> Conf {
    Conf {
        window_title: "HaiViz ".to_owned() +env!("CARGO_PKG_VERSION"),
        window_width: 1920,
        window_height: 1080,
        fullscreen: false,
        ..Default::default()
    }
}

fn draw_cloud(cloud: &Pointcloud) {
    let mut i: usize=0;
    while i < cloud.points.len() {
        /*if i==0 {
            println!("{} {} {}",(pt.x-cloud.min.x) as f32, (pt.y-cloud.min.y) as f32, (pt.z-cloud.min.z) as f32);
        }*/
        draw_sphere(vec3((cloud.points[i].x-cloud.min.x) as f32, (cloud.points[i].z-cloud.min.z) as f32, (cloud.points[i].y-cloud.min.y) as f32), 0.15, None, BLUE);
        i += 500;
        /*if i>1000 {
            break;
        }*/
    }
}

async fn main_loop(cloud: &Pointcloud) {
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
    let mut up = right.cross(front).normalize();

    let mut position = vec3(0.0, 1.0, 0.0);
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

        // Going 3d!

        set_camera(&Camera3D {
            position,
            up,
            target: position + front,
            //projection: Projection::Orthographics,
            z_far: 100000.0,
            ..Default::default()
        });

        //draw_grid(20, 1., BLACK, GRAY);

        draw_cloud(cloud);

        /*draw_line_3d(
            vec3(x, 0.0, x),
            vec3(5.0, 5.0, 5.0),
            Color::new(1.0, 1.0, 0.0, 1.0),
        );*/

        draw_cube_wires(vec3(0., 1., -6.), vec3(2., 2., 2.), GREEN);
        draw_cube_wires(vec3(0., 1., 6.), vec3(2., 2., 2.), BLUE);
        draw_cube_wires(vec3(16., 13., 8.), vec3(2., 2., 2.), RED);



        // Back to screen space, render some text

        set_default_camera();
        draw_text("First Person Camera", 10.0, 20.0, 30.0, BLACK);

        draw_text(
            format!("X: {} Y: {}", mouse_position.x, mouse_position.y).as_str(),
            10.0,
            48.0 + 18.0,
            30.0,
            BLACK,
        );
        draw_text(
            format!("Press <TAB> to toggle mouse grab: {grabbed}").as_str(),
            10.0,
            48.0 + 42.0,
            30.0,
            BLACK,
        );
        next_frame().await
    }

}

#[macroquad::main(window_conf)]
async fn main() {
    if env::args().count() != 2    {
        println!("{}",style("Usage: haiviz pointcloud.laz").red());
        return;
    }
    let args: Vec<String> = env::args().collect();
    let cloudfile = &args[1];

    let mut cloud = Pointcloud::new();
    println!("Loading {}",cloudfile);
    cloud.load(cloudfile.to_owned(),true,None);
    println!("Starting loop");
    main_loop(&cloud).await
}
