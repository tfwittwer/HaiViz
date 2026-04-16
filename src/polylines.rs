use macroquad::prelude::*;
use std::fs;

#[derive(Clone)]
pub struct Polyline {
    pub points: Vec<[f32; 3]>, // viewer-space (world - offset)
    pub color: Color,
    pub radius: f32, // cylinder radius in world units; controls drawn width in section view
    pub name: String,
}

pub struct PolylineFile {
    pub path: String,
    pub polylines: Vec<Polyline>,
}

/// Parse Salome Hydro Polyline ASCII format.
/// One "x y z" per line; blank lines separate polylines.
/// `offset` is subtracted so coordinates end up in viewer space.
pub fn load(path: &str, offset: [f64; 3], default_color: Color) -> Result<PolylineFile, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut polylines: Vec<Polyline> = Vec::new();
    let mut current: Vec<[f32; 3]> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            flush(&mut current, &mut polylines, default_color);
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 3 &&
            let (Ok(x), Ok(y), Ok(z)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
            ) {
                current.push([
                    (x - offset[0]) as f32,
                    (y - offset[1]) as f32,
                    (z - offset[2]) as f32,
                ]);
        }
        
    }
    flush(&mut current, &mut polylines, default_color);

    Ok(PolylineFile { path: path.to_string(), polylines })
}

fn flush(current: &mut Vec<[f32; 3]>, polylines: &mut Vec<Polyline>, color: Color) {
    if current.len() >= 2 {
        let name = format!("Polyline {}", polylines.len() + 1);
        polylines.push(Polyline {
            points: std::mem::take(current),
            color,
            radius: 0.01,
            name,
        });
    } else {
        current.clear();
    }
}

/// Draw all polylines as 3D lines (single-pixel width; thickness applies in section view).
pub fn draw_3d(files: &[PolylineFile]) {
    for pf in files {
        for pl in &pf.polylines {
            for i in 0..pl.points.len().saturating_sub(1) {
                draw_line_3d(
                    glam::Vec3::from(pl.points[i]),
                    glam::Vec3::from(pl.points[i + 1]),
                    pl.color,
                );
            }
        }
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
    sw: f32,
    sh: f32,
) {
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

    let to_sx = |wx: f32| (wx - (section_pan.x - half_w)) / (2.0 * half_w) * sw;
    let to_sy = |wy: f32| (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * sh;

    for pf in files {
        for pl in &pf.polylines {
            if pl.points.len() < 2 { continue; }

            // Compute (along, perp, height) for each point
            let proj: Vec<(f32, f32, f32)> = pl.points.iter().map(|p| {
                let rx = p[0] - section_start.0;
                let ry = p[1] - section_start.1;
                let along = rx * ux + ry * uy;
                let perp  = rx * px + ry * py;
                (along, perp, p[2])
            }).collect();

            // Convert world-unit radius to screen pixels (x and y scales are equal)
            let r_px = (pl.radius * sh / (2.0 * half_h)).max(0.5);

            for i in 0..proj.len() - 1 {
                let (a_along, a_perp, a_h) = proj[i];
                let (b_along, b_perp, b_h) = proj[i + 1];
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

/// Return (file_idx, poly_idx) of the polyline closest to (mx, my) in screen space,
/// within `threshold` pixels. Returns None if nothing is close enough.
pub fn pick(
    files: &[PolylineFile],
    view_proj: &glam::Mat4,
    mx: f32,
    my: f32,
    sw: f32,
    sh: f32,
    threshold: f32,
) -> Option<(usize, usize)> {
    let mut best = threshold;
    let mut result = None;

    for (fi, pf) in files.iter().enumerate() {
        for (pi, pl) in pf.polylines.iter().enumerate() {
            for i in 0..pl.points.len().saturating_sub(1) {
                let a = project(view_proj, pl.points[i], sw, sh);
                let b = project(view_proj, pl.points[i + 1], sw, sh);
                let d = seg_dist(mx, my, a, b);
                if d < best {
                    best = d;
                    result = Some((fi, pi));
                }
            }
        }
    }
    result
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
