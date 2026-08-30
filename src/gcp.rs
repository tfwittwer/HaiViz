//! Ground control point overlay.
//!
//! Loads GCPs from plain text tables (CSV, semicolon, tab or whitespace
//! separated), auto-detecting the separator, an optional header row and which
//! columns hold the label and the x/y/z coordinates. The detection is a guess
//! that the operator can override in the UI, so the raw rows are kept around
//! and the file can be re-mapped without re-reading it.

use macroquad::prelude::*;
use std::fs;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sep {
    Whitespace,
    Comma,
    Semicolon,
    Tab,
    Pipe,
}

impl Sep {
    pub const ALL: [Sep; 5] = [Sep::Comma, Sep::Semicolon, Sep::Tab, Sep::Pipe, Sep::Whitespace];

    pub fn name(self) -> &'static str {
        match self {
            Sep::Whitespace => "whitespace",
            Sep::Comma => "comma",
            Sep::Semicolon => "semicolon",
            Sep::Tab => "tab",
            Sep::Pipe => "pipe",
        }
    }

    fn split(self, line: &str) -> Vec<String> {
        let raw: Vec<&str> = match self {
            Sep::Whitespace => line.split_whitespace().collect(),
            Sep::Comma => line.split(',').collect(),
            Sep::Semicolon => line.split(';').collect(),
            Sep::Tab => line.split('\t').collect(),
            Sep::Pipe => line.split('|').collect(),
        };
        raw.iter().map(|f| clean(f)).collect()
    }
}

/// Trim whitespace and surrounding quotes from one field.
fn clean(f: &str) -> String {
    let t = f.trim();
    let t = t.strip_prefix('"').unwrap_or(t);
    let t = t.strip_suffix('"').unwrap_or(t);
    t.trim().to_string()
}

fn num(f: &str) -> Option<f64> {
    if f.is_empty() { return None; }
    f.parse::<f64>().ok()
}

/// Which column means what. Column indices refer to `GcpFile::rows`.
#[derive(Clone)]
pub struct Layout {
    pub sep: Sep,
    pub has_header: bool,
    pub label_col: Option<usize>,
    pub x_col: usize,
    pub y_col: usize,
    pub z_col: usize,
    /// Header names when the file has a header row, else "col 1", "col 2", …
    pub headers: Vec<String>,
}

#[derive(Clone)]
pub struct GcpPoint {
    pub label: String,
    /// Absolute coordinates as read from the file.
    pub world: [f64; 3],
}

pub struct GcpFile {
    pub path: String,
    /// Data rows as split by the detected separator, header excluded. Kept so
    /// the column mapping can be changed after loading.
    pub rows: Vec<Vec<String>>,
    pub layout: Layout,
    pub points: Vec<GcpPoint>,
    pub color: Color,
    pub visible: bool,
    pub show_labels: bool,
    /// Marker radius in screen pixels — markers keep their size at any zoom.
    pub size: f32,
}

fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('#') || t.starts_with("//") || t.starts_with(';') || t.starts_with('!')
}

/// Read a GCP file, auto-detecting separator and column meaning.
pub fn load(path: &str, color: Color) -> Result<GcpFile, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse(&text, color).map(|mut f| {
        f.path = path.to_string();
        f
    })
}

/// The body of [`load`], split out so it can be exercised without a file.
pub fn parse(text: &str, color: Color) -> Result<GcpFile, String> {
    let mut lines: Vec<&str> = Vec::new();
    // A "# Name X Y Z" banner is a header in all but syntax; remember the last
    // comment line ahead of the data so it can still name the columns.
    let mut comment_header: Option<&str> = None;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if is_comment(line) {
            if lines.is_empty() {
                comment_header = Some(line.trim().trim_start_matches(['#', '/', ';', '!']));
            }
            continue;
        }
        lines.push(line);
    }
    if lines.is_empty() {
        return Err("File contains no data lines".to_string());
    }

    let sep = detect_sep(&lines)?;
    let mut rows: Vec<Vec<String>> = lines.iter().map(|l| sep.split(l)).collect();

    // Rows with fewer fields than the bulk are usually trailing junk; drop them.
    let ncols = mode_count(&rows.iter().map(|r| r.len()).collect::<Vec<_>>());
    rows.retain(|r| r.len() >= ncols.min(3));
    if rows.is_empty() {
        return Err(format!("No usable rows after splitting on {}", sep.name()));
    }

    let drop_first = looks_like_header(&rows);
    let mut headers: Vec<String> = if drop_first { rows.remove(0) } else { Vec::new() };
    if rows.is_empty() {
        return Err("File has a header but no data rows".to_string());
    }
    if headers.is_empty()
        && let Some(c) = comment_header
    {
        let fields = sep.split(c);
        let numeric = fields.iter().filter(|f| num(f).is_some()).count();
        if fields.len() == ncols && numeric * 2 < ncols {
            headers = fields;
        }
    }
    let has_header = !headers.is_empty();
    if !has_header {
        headers = (1..=ncols).map(|i| format!("col {i}")).collect();
    }

    let (label_col, x_col, y_col, z_col) = detect_columns(&rows, &headers, has_header, ncols)?;

    let layout = Layout { sep, has_header, label_col, x_col, y_col, z_col, headers };
    let points = build_points(&rows, &layout);
    if points.is_empty() {
        return Err("No rows parsed as x/y/z coordinates".to_string());
    }

    Ok(GcpFile {
        path: String::new(),
        rows,
        layout,
        points,
        color,
        visible: true,
        show_labels: true,
        size: 9.0,
    })
}

/// Re-apply the (possibly edited) column mapping to the stored rows.
pub fn rebuild(file: &mut GcpFile) {
    file.points = build_points(&file.rows, &file.layout);
}

fn build_points(rows: &[Vec<String>], layout: &Layout) -> Vec<GcpPoint> {
    let mut out = Vec::new();
    for row in rows {
        let get = |i: usize| row.get(i).map(|s| s.as_str()).unwrap_or("");
        let (Some(x), Some(y), Some(z)) =
            (num(get(layout.x_col)), num(get(layout.y_col)), num(get(layout.z_col)))
        else {
            continue;
        };
        let label = match layout.label_col {
            Some(c) if !get(c).is_empty() => get(c).to_string(),
            _ => format!("GCP {}", out.len() + 1),
        };
        out.push(GcpPoint { label, world: [x, y, z] });
    }
    out
}

fn mode_count(counts: &[usize]) -> usize {
    let mut best = 0;
    let mut best_n = 0;
    for &c in counts {
        let n = counts.iter().filter(|&&o| o == c).count();
        if n > best_n || (n == best_n && c > best) {
            best_n = n;
            best = c;
        }
    }
    best
}

/// Pick the separator that splits the sample into a consistent number of
/// columns with the most numeric fields. Ties go to the first candidate, and
/// whitespace is tried last: it splits tab-separated data identically, but only
/// the real separator survives empty fields or labels containing spaces. Splitting "1, 2, 3" on whitespace also
/// yields three fields, so consistency alone is not enough to tell the
/// candidates apart — the numeric count is what breaks the tie.
fn detect_sep(lines: &[&str]) -> Result<Sep, String> {
    let sample: Vec<&str> = lines.iter().take(50).copied().collect();
    let mut best: Option<(Sep, f32)> = None;

    for sep in Sep::ALL {
        let split: Vec<Vec<String>> = sample.iter().map(|l| sep.split(l)).collect();
        let counts: Vec<usize> = split.iter().map(|r| r.len()).collect();
        let ncols = mode_count(&counts);
        if ncols < 3 {
            continue;
        }
        let consistent = counts.iter().filter(|&&c| c == ncols).count() as f32 / counts.len() as f32;
        if consistent < 0.6 {
            continue;
        }
        let numeric: usize = split
            .iter()
            .map(|r| r.iter().filter(|f| num(f).is_some()).count())
            .sum();
        let numeric = numeric as f32 / split.len() as f32;
        // Numeric fields dominate; consistency and column count only nudge.
        let score = numeric * 10.0 + consistent * 3.0 + (ncols as f32) * 0.01;
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((sep, score));
        }
    }

    best.map(|(s, _)| s)
        .ok_or_else(|| "Could not find a column separator (need at least 3 columns)".to_string())
}

/// A first row whose fields are mostly non-numeric, followed by rows that are
/// mostly numeric, is a header.
fn looks_like_header(rows: &[Vec<String>]) -> bool {
    if rows.len() < 2 {
        return false;
    }
    let numeric = |r: &Vec<String>| r.iter().filter(|f| num(f).is_some()).count();
    let first = numeric(&rows[0]);
    let rest: usize = rows[1..].iter().take(20).map(numeric).sum();
    let rest = rest as f32 / rows[1..].iter().take(20).count() as f32;
    (first as f32) < rest - 0.5 && rest >= 3.0
}

type ColMap = (Option<usize>, usize, usize, usize);

fn detect_columns(
    rows: &[Vec<String>],
    headers: &[String],
    has_header: bool,
    ncols: usize,
) -> Result<ColMap, String> {
    let ncols = ncols.max(3);
    let n = rows.len() as f32;
    // Per-column statistics over the data rows.
    let mut numeric_frac = vec![0.0_f32; ncols];
    let mut all_integral = vec![true; ncols];
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); ncols];
    for row in rows {
        for c in 0..ncols {
            if let Some(v) = row.get(c).and_then(|f| num(f)) {
                numeric_frac[c] += 1.0 / n;
                if v.fract().abs() > 1e-9 {
                    all_integral[c] = false;
                }
                values[c].push(v);
            }
        }
    }
    let is_numeric = |c: usize| numeric_frac[c] >= 0.9;

    let mut label_col: Option<usize> = None;
    let mut x_col: Option<usize> = None;
    let mut y_col: Option<usize> = None;
    let mut z_col: Option<usize> = None;

    if has_header {
        for (c, h) in headers.iter().enumerate().take(ncols) {
            let h = h.trim().trim_start_matches('#').to_ascii_lowercase();
            let key: String = h.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
            let m = |cands: &[&str]| cands.iter().any(|k| key == *k);
            if x_col.is_none() && m(&["x", "e", "east", "easting", "xm", "xcoord", "lon", "long", "longitude", "rechtswert"]) {
                x_col = Some(c);
            } else if y_col.is_none() && m(&["y", "n", "north", "northing", "ym", "ycoord", "lat", "latitude", "hochwert"]) {
                y_col = Some(c);
            } else if z_col.is_none() && m(&["z", "h", "zm", "zcoord", "height", "hoehe", "elev", "elevation", "alt", "altitude", "hgt"]) {
                z_col = Some(c);
            } else if label_col.is_none()
                && m(&["label", "name", "id", "pt", "pnt", "point", "pointid", "ptid", "no", "nr", "num", "number", "code", "punkt", "punktnr", "desc", "description"])
            {
                label_col = Some(c);
            }
        }
    }

    // Anything the header did not settle falls back to position and type.
    if label_col.is_none() {
        // A leading text column is a label. So is a leading column of unique
        // whole numbers next to columns that carry decimals — that is a point
        // number, not a coordinate.
        let taken = |c: usize| Some(c) == x_col || Some(c) == y_col || Some(c) == z_col;
        if let Some(c) = (0..ncols).find(|&c| !is_numeric(c) && !taken(c)) {
            label_col = Some(c);
        } else {
            let free: Vec<usize> = (0..ncols).filter(|&c| !taken(c)).collect();
            let decimals = (0..ncols).filter(|&c| is_numeric(c) && !all_integral[c]).count();
            if free.len() > 3 && decimals >= 3 {
                let c = free[0];
                let unique = {
                    let mut v: Vec<String> = values[c].iter().map(|v| format!("{v}")).collect();
                    v.sort();
                    v.dedup();
                    v.len() == values[c].len()
                };
                if all_integral[c] && unique {
                    label_col = Some(c);
                }
            }
        }
    }

    let free: Vec<usize> = (0..ncols)
        .filter(|&c| is_numeric(c) && Some(c) != label_col)
        .filter(|&c| Some(c) != x_col && Some(c) != y_col && Some(c) != z_col)
        .collect();
    let mut free = free.into_iter();
    for slot in [&mut x_col, &mut y_col, &mut z_col] {
        if slot.is_none() {
            *slot = free.next();
        }
    }

    match (x_col, y_col, z_col) {
        (Some(x), Some(y), Some(z)) => Ok((label_col, x, y, z)),
        _ => Err("Could not find three numeric coordinate columns".to_string()),
    }
}

// ---------------------------------------------------------------- rendering

fn project(vp: &glam::Mat4, p: glam::Vec3) -> Option<(f32, f32)> {
    let clip = *vp * glam::Vec4::new(p.x, p.y, p.z, 1.0);
    if clip.w <= 1e-6 {
        return None; // behind the camera
    }
    Some((
        (clip.x / clip.w + 1.0) / 2.0 * screen_width(),
        (1.0 - clip.y / clip.w) / 2.0 * screen_height(),
    ))
}

/// Screen-space marker: a dark halo under a bright ring and crosshair, so it
/// stays readable over any point colour or background.
fn draw_marker(sx: f32, sy: f32, r: f32, color: Color, selected: bool) {
    let halo = Color::new(0.0, 0.0, 0.0, 0.55 * color.a);
    let arm_in = r * 0.55;
    let arm_out = r * 1.9;
    for (col, w, extra) in [(halo, 4.0, 1.0), (color, 2.0, 0.0)] {
        draw_circle_lines(sx, sy, r + extra, w, col);
        draw_line(sx - arm_out - extra, sy, sx - arm_in, sy, w, col);
        draw_line(sx + arm_in, sy, sx + arm_out + extra, sy, w, col);
        draw_line(sx, sy - arm_out - extra, sx, sy - arm_in, w, col);
        draw_line(sx, sy + arm_in, sx, sy + arm_out + extra, w, col);
    }
    draw_circle(sx, sy, 1.5, color);
    if selected {
        let hl = Color::new(1.0, 1.0, 0.0, color.a);
        draw_circle_lines(sx, sy, r * 2.4, 2.0, hl);
        draw_circle_lines(sx, sy, r * 2.4 + 2.0, 1.0, halo);
    }
}

fn draw_label(text: &str, sx: f32, sy: f32, color: Color) {
    let shadow = Color::new(0.0, 0.0, 0.0, 0.75 * color.a);
    for dx in [-1.0_f32, 1.0] {
        for dy in [-1.0_f32, 1.0] {
            draw_text(text, sx + dx, sy + dy, 16.0, shadow);
        }
    }
    draw_text(text, sx, sy, 16.0, color);
}

/// Draw GCP markers over the 3D view. Call after `set_default_camera()`:
/// markers are drawn in screen space so they keep a constant size and are never
/// swallowed by the point cloud.
pub fn draw_3d(
    files: &[GcpFile],
    offset: [f64; 3],
    view_proj: &glam::Mat4,
    selected: Option<(usize, usize)>,
) {
    let (sw, sh) = (screen_width(), screen_height());
    for (fi, f) in files.iter().enumerate() {
        if !f.visible {
            continue;
        }
        for (pi, p) in f.points.iter().enumerate() {
            let v = glam::Vec3::new(
                (p.world[0] - offset[0]) as f32,
                (p.world[1] - offset[1]) as f32,
                (p.world[2] - offset[2]) as f32,
            );
            let Some((sx, sy)) = project(view_proj, v) else { continue };
            if sx < -50.0 || sy < -50.0 || sx > sw + 50.0 || sy > sh + 50.0 {
                continue;
            }
            let is_sel = selected == Some((fi, pi));
            draw_marker(sx, sy, f.size, f.color, is_sel);
            if f.show_labels {
                draw_label(&p.label, sx + f.size * 2.1, sy - f.size * 0.8, f.color);
            }
        }
    }
}

/// Draw GCP markers in the section view. Points within `depth` of the section
/// plane are drawn solid; the marker fades with distance from the plane, and
/// the label carries the perpendicular offset once it is more than a few
/// centimetres off.
#[allow(clippy::too_many_arguments)]
pub fn draw_section(
    files: &[GcpFile],
    offset: [f64; 3],
    section_start: (f32, f32),
    section_end: (f32, f32),
    depth: f32,
    length: f32,
    section_pan: glam::Vec2,
    section_zoom: f32,
    selected: Option<(usize, usize)>,
) {
    let (sw, sh) = (screen_width(), screen_height());
    let dx = section_end.0 - section_start.0;
    let dy = section_end.1 - section_start.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        return;
    }
    let (ux, uy) = (dx / len, dy / len);
    let (px, py) = (-uy, ux);

    let half_h = section_zoom;
    let half_w = half_h * (sw / sh);
    let to_sx = |wx: f32| (wx - (section_pan.x - half_w)) / (2.0 * half_w) * sw;
    let to_sy = |wy: f32| (1.0 - (wy - (section_pan.y - half_h)) / (2.0 * half_h)) * sh;

    for (fi, f) in files.iter().enumerate() {
        if !f.visible {
            continue;
        }
        for (pi, p) in f.points.iter().enumerate() {
            let vx = (p.world[0] - offset[0]) as f32;
            let vy = (p.world[1] - offset[1]) as f32;
            let vz = (p.world[2] - offset[2]) as f32;
            let (rx, ry) = (vx - section_start.0, vy - section_start.1);
            let along = rx * ux + ry * uy;
            let perp = rx * px + ry * py;
            if perp.abs() > depth || along < -depth || along > length + depth {
                continue;
            }
            let fade = 1.0 - 0.55 * (perp.abs() / depth.max(1e-6)).min(1.0);
            let col = Color::new(f.color.r, f.color.g, f.color.b, f.color.a * fade);
            let (sx, sy) = (to_sx(along), to_sy(vz));
            if sx < -50.0 || sy < -50.0 || sx > sw + 50.0 || sy > sh + 50.0 {
                continue;
            }
            let is_sel = selected == Some((fi, pi));
            draw_marker(sx, sy, f.size, col, is_sel);
            if f.show_labels {
                let text = if perp.abs() > 0.05 {
                    format!("{} ({:+.2}m)", p.label, perp)
                } else {
                    p.label.clone()
                };
                draw_label(&text, sx + f.size * 2.1, sy - f.size * 0.8, col);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(text: &str) -> GcpFile {
        parse(text, WHITE).expect("parse failed")
    }

    fn cols(f: &GcpFile) -> (Option<usize>, usize, usize, usize) {
        (f.layout.label_col, f.layout.x_col, f.layout.y_col, f.layout.z_col)
    }

    #[test]
    fn whitespace_xyz() {
        let f = p("123456.120 456789.450 12.340\n123457.000 456790.000 12.900\n");
        assert_eq!(f.layout.sep, Sep::Whitespace);
        assert_eq!(cols(&f), (None, 0, 1, 2));
        assert_eq!(f.points.len(), 2);
        assert_eq!(f.points[0].label, "GCP 1");
        assert_eq!(f.points[1].world, [123457.0, 456790.0, 12.9]);
    }

    #[test]
    fn whitespace_named_points() {
        let f = p("GCP01 123456.12 456789.45 12.34\nGCP02 123457.00 456790.00 12.90\n");
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
        assert_eq!(f.points[0].label, "GCP01");
    }

    #[test]
    fn csv_with_header() {
        let f = p("Name,X,Y,Z\nA1,10.5,20.5,3.5\nA2,11.5,21.5,4.5\n");
        assert_eq!(f.layout.sep, Sep::Comma);
        assert!(f.layout.has_header);
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
        assert_eq!(f.points.len(), 2);
    }

    #[test]
    fn header_names_out_of_order() {
        let f = p("Height;Northing;Easting;PointID\n12.3;456789.4;123456.1;P1\n13.3;456790.4;123457.1;P2\n");
        assert_eq!(f.layout.sep, Sep::Semicolon);
        assert_eq!(cols(&f), (Some(3), 2, 1, 0));
        assert_eq!(f.points[0].world, [123456.1, 456789.4, 12.3]);
    }

    #[test]
    fn csv_with_spaces_after_comma() {
        // Whitespace splitting also yields four fields here, so the separator
        // has to be chosen on how many of them are numbers.
        let f = p("1, 123456.12, 456789.45, 12.34\n2, 123457.00, 456790.00, 12.90\n");
        assert_eq!(f.layout.sep, Sep::Comma);
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
        assert_eq!(f.points[0].label, "1");
    }

    #[test]
    fn tab_separated_with_trailing_code() {
        let f = p("P1\t10.5\t20.5\t3.5\tROAD\nP2\t11.5\t21.5\t4.5\tKERB\n");
        assert_eq!(f.layout.sep, Sep::Tab);
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
    }

    #[test]
    fn numeric_id_column_is_not_a_coordinate() {
        let f = p("1 123456.12 456789.45 12.34\n2 123457.00 456790.00 12.90\n3 123458.00 456791.00 13.90\n");
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
        assert_eq!(f.points[2].world, [123458.0, 456791.0, 13.9]);
    }

    #[test]
    fn comment_line_names_the_columns() {
        let f = p("# id easting northing height\n1 123456.12 456789.45 12.34\n2 123457.00 456790.00 12.90\n");
        assert!(f.layout.has_header);
        assert_eq!(f.layout.headers[1], "easting");
        assert_eq!(cols(&f), (Some(0), 1, 2, 3));
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let f = p("// survey 2026-05\n\nA 1.0 2.0 3.0\n\n! note\nB 4.0 5.0 6.0\n");
        assert_eq!(f.points.len(), 2);
    }

    #[test]
    fn quoted_fields() {
        let f = p("\"Name\",\"X\",\"Y\",\"Z\"\n\"A 1\",10.5,20.5,3.5\n\"A 2\",11.5,21.5,4.5\n");
        assert_eq!(f.points[0].label, "A 1");
        assert_eq!(f.points[1].world, [11.5, 21.5, 4.5]);
    }

    #[test]
    fn remapping_columns_rebuilds_points() {
        let mut f = p("A 1.0 2.0 3.0\nB 4.0 5.0 6.0\n");
        f.layout.x_col = 3;
        f.layout.z_col = 1;
        rebuild(&mut f);
        assert_eq!(f.points[0].world, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn unparseable_file_is_an_error() {
        assert!(parse("just some prose\nwith no numbers at all\n", WHITE).is_err());
    }
}
