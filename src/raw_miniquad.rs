pub mod raw_miniquad {
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

    pub struct SectionData {
        bindings: Bindings,
        pub num_points: i32,
        pub length: f32,
        pub height_min: f32,
        pub height_max: f32,
    }

    pub struct Stage {
        pipeline: Pipeline,
        pub chunks: Vec<Chunk>,
        pub total_points: i32,
        pub cloud: VizPointcloud,
        positions: Vec<[f32; 3]>,
    }

    fn color_for_point(point: &Point, pos_z: f32, height_range: f32, intensity_min: u16, intensity_range: f32, mode: ColorMode) -> [f32; 3] {
        match mode {
            ColorMode::Height => {
                let h = if height_range > 0.0 { (pos_z / height_range).clamp(0.0, 1.0) } else { 0.5 };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMode::Intensity => {
                // Logarithmic scaling with the same blue-red gradient as height
                let h = if intensity_range > 0.0 {
                    let lin = (point.intensity - intensity_min) as f32 / intensity_range;
                    ((1.0 + lin * 99.0).log10() / 2.0).clamp(0.0, 1.0) // log10(1..100) -> 0..2 -> 0..1
                } else {
                    0.5
                };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
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
            0 => [0.5, 0.5, 0.5],
            1 => [0.6, 0.6, 0.6],
            2 => [0.6, 0.4, 0.2],
            3 => [0.0, 0.8, 0.0],
            4 => [0.0, 0.6, 0.0],
            5 => [0.0, 0.4, 0.0],
            6 => [1.0, 0.0, 0.0],
            7 => [1.0, 0.5, 0.0],
            8 => [0.8, 0.8, 0.0],
            9 => [0.0, 0.0, 1.0],
            10 => [0.8, 0.8, 0.8],
            11 => [0.3, 0.3, 0.3],
            12 => [0.9, 0.9, 0.0],
            13 => [0.7, 0.7, 0.0],
            14 => [0.9, 0.9, 0.0],
            15 => [0.4, 0.0, 0.6],
            16 => [0.6, 0.0, 0.8],
            17 => [0.2, 0.2, 0.6],
            18 => [1.0, 0.0, 0.5],
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
                    (p.y - cloud.min[1]) as f32,
                    (p.z - cloud.min[2]) as f32,
                ])
                .collect();

            let vertices: Vec<Vertex> = cloud
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| Vertex {
                    pos: positions[i],
                    color: color_for_point(p, positions[i][2], height_range, cloud.intensity_min, intensity_range, color_mode),
                })
                .collect();

            let mut chunk_map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            for (i, pos) in positions.iter().enumerate() {
                let cx = (pos[0] / CHUNK_SIZE).floor() as i32;
                let cy = (pos[1] / CHUNK_SIZE).floor() as i32;
                chunk_map.entry((cx, cy)).or_default().push(i);
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

        pub fn find_nearest_point(
            &self,
            view_proj: &glam::Mat4,
            screen_x: f32,
            screen_y: f32,
            screen_w: f32,
            screen_h: f32,
        ) -> Option<usize> {
            let ndc_x = screen_x / screen_w * 2.0 - 1.0;
            let ndc_y = 1.0 - screen_y / screen_h * 2.0;

            let mut best_idx = None;
            let mut best_dist = f32::MAX;

            for (i, pos) in self.positions.iter().enumerate() {
                let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                let px = p.x / p.w;
                let py = p.y / p.w;
                let dist = (px - ndc_x) * (px - ndc_x) + (py - ndc_y) * (py - ndc_y);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }

            best_idx
        }

        pub fn height_range(&self) -> f32 {
            (self.cloud.max[2] - self.cloud.min[2]) as f32
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let height_range = self.height_range();
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;

            for chunk in &mut self.chunks {
                let chunk_verts: Vec<Vertex> = chunk
                    .point_indices
                    .iter()
                    .map(|&i| {
                        let p = &self.cloud.points[i];
                        Vertex {
                            pos: self.positions[i],
                            color: color_for_point(p, self.positions[i][2], height_range, self.cloud.intensity_min, intensity_range, color_mode),
                        }
                    })
                    .collect();

                ctx.buffer_update(
                    chunk.bindings.vertex_buffers[0],
                    BufferSource::slice(&chunk_verts),
                );
            }
        }

        pub fn extract_section(
            &self,
            ctx: &mut dyn RenderingBackend,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> SectionData {
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            let len = (dx * dx + dy * dy).sqrt();
            let dir = (dx / len, dy / len);
            let normal = (-dir.1, dir.0);

            let height_range = self.height_range();
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;

            let mut section_verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for (i, pos) in self.positions.iter().enumerate() {
                let rel_x = pos[0] - start.0;
                let rel_y = pos[1] - start.1;
                let along = rel_x * dir.0 + rel_y * dir.1;
                let perp = rel_x * normal.0 + rel_y * normal.1;

                if along >= 0.0 && along <= len && perp.abs() <= depth {
                    let p = &self.cloud.points[i];
                    h_min = h_min.min(pos[2]);
                    h_max = h_max.max(pos[2]);
                    section_verts.push(Vertex {
                        pos: [along, pos[2], 0.0],
                        color: color_for_point(p, pos[2], height_range, self.cloud.intensity_min, intensity_range, color_mode),
                    });
                }
            }

            let num_points = section_verts.len() as i32;
            println!("Section: {:.1}m long, {:.1}m depth, {} points, height {:.1}..{:.1}m",
                len, depth, num_points, h_min, h_max);

            // Create GPU buffers
            let vertex_buffer = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&section_verts),
            );
            let indices: Vec<u32> = (0..num_points as u32).collect();
            let index_buffer = ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&indices),
            );

            SectionData {
                bindings: Bindings {
                    vertex_buffers: vec![vertex_buffer],
                    index_buffer,
                    images: vec![],
                },
                num_points,
                length: len,
                height_min: h_min,
                height_max: h_max,
            }
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp },
            ));
            ctx.draw(0, section.num_points, 1);
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
