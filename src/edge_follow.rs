//! Edge following: letting the tool trace a real feature between two clicks.
//!
//! Livewire — "intelligent scissors" — adapted to point clouds. On each new
//! anchor (the last committed vertex) the resident points in a window around it
//! are rasterised into a world-XY grid, an edge-strength field is derived from
//! the raster, and Dijkstra runs once from the anchor over the whole grid. From
//! then on, the path to wherever the cursor is costs only a walk back along the
//! parent pointers — which is what lets the ghost path follow the mouse live.
//!
//! ## What counts as an edge
//!
//! A weighted sum of five channels, each normalised to 0..1:
//!
//! - **step** — the height difference of the lowest return between
//!   neighbouring cells: kerbs, breaklines, ditch edges, embankment tops;
//! - **height range** — the spread of Z within a cell: walls, building
//!   footprints, anything vertical;
//! - **intensity** — the difference in mean intensity between neighbours:
//!   painted road markings, probably the most valuable channel for road work;
//! - **class** — a change of majority class: vegetation against ground,
//!   building against ground;
//! - **occupancy** — the boundary between cells with points and cells without:
//!   the edge of the cloud, a roof outline.
//!
//! The weights are exposed and the field can be drawn over the cloud. That is
//! not a nicety: without seeing what the tracer sees, tuning the weights is
//! guesswork.
//!
//! Hidden classes and the height band are applied before a point reaches the
//! raster, so class visibility steers the tracer exactly as it steers a snap —
//! hide vegetation, and a kerb is no longer drowned out by the hedge behind it.
//!
//! ## What it never does
//!
//! Block. When the cursor is outside the window there is no trace and the
//! rubber band is the plain straight segment; Shift-click still places a vertex
//! freely.
//!
//! This module depends on nothing else in the crate, so the bench includes it
//! by path — the same arrangement as `spatial.rs`.

use macroquad::models::{Mesh, Vertex, draw_mesh};
use macroquad::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Largest raster side, in cells. 512² is 262 k nodes for Dijkstra, run once
/// per anchor and not at all per frame.
pub const MAX_CELLS: usize = 512;
pub const MIN_WINDOW_M: f32 = 5.0;
pub const MAX_WINDOW_M: f32 = 150.0;
/// Points the raster aims to put in an occupied cell. With fewer, the lowest
/// return in a cell is one point and the step channel is mostly noise; with
/// many more, a kerb blurs across a cell wider than the kerb.
const TARGET_POINTS_PER_CELL: f64 = 4.0;
/// Smallest cell whatever the density. Finer than this resolves nothing a snap
/// radius can place a vertex on.
const MIN_CELL_M: f32 = 0.03;
/// Coarsest raster, in cells per side, however sparse the data: below this a
/// window stops describing a feature at all.
const MIN_CELLS: f32 = 16.0;
/// Added to the edge value in the path cost. A cell with no edge still has a
/// finite price, and the ratio between the two extremes is roughly how far the
/// path will detour to stay on an edge.
const EDGE_EPS: f32 = 0.05;
/// Cost of a step whatever the edge strength, so that among equally good
/// edges the path does not wander.
const LENGTH_COST: f32 = 0.05;
/// While COPC nodes stream in around an anchor, the raster is rebuilt at most
/// this often.
pub const REBUILD_INTERVAL_S: f64 = 0.5;
pub const NO_PARENT: u32 = u32::MAX;

/// One point as the raster sees it, in viewer space.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub pos: [f32; 3],
    pub intensity: u16,
    pub class: u8,
}

/// How much each channel counts. See the module note.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Weights {
    pub step: f32,
    pub height_range: f32,
    pub intensity: f32,
    pub class: f32,
    pub occupancy: f32,
}

impl Default for Weights {
    fn default() -> Self {
        // Step and intensity carry most road features. The others are there to
        // be turned up when the job is buildings, vegetation or the cloud's
        // own boundary.
        Weights { step: 1.0, height_range: 0.4, intensity: 0.8, class: 0.4, occupancy: 0.2 }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Params {
    /// Side of the square window around the anchor, in metres.
    pub window_m: f32,
    pub weights: Weights,
    /// A height step this large between neighbouring cells is a full-strength
    /// edge. A fifth of it is treated as noise and ignored.
    pub step_m: f32,
    /// A spread of Z within one cell this large is fully vertical.
    pub range_m: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { window_m: 40.0, weights: Weights::default(), step_m: 0.10, range_m: 1.0 }
    }
}

/// Cell size for a square window of side `window` holding `n` points.
///
/// Chosen from the density rather than fixed: a mobile-mapping run at a
/// thousand points per square metre and an airborne survey at twenty want
/// cells an order of magnitude apart, and a fixed size would make one of them
/// all noise and the other all blur.
pub fn cell_size(window: f32, n: usize) -> f32 {
    let finest = (window / MAX_CELLS as f32).max(MIN_CELL_M);
    let coarsest = window / MIN_CELLS;
    if n == 0 { return coarsest.max(finest); }
    let density = n as f64 / (window as f64 * window as f64);
    ((TARGET_POINTS_PER_CELL / density).sqrt() as f32).min(coarsest).max(finest)
}

/// Per-cell summaries of the points in a window.
pub struct Raster {
    /// Viewer-space lower corner of cell (0, 0).
    pub origin: [f32; 2],
    pub cell: f32,
    pub nx: usize,
    pub ny: usize,
    pub count: Vec<u32>,
    /// Lowest and highest Z in each cell; infinite when it is empty.
    pub z_min: Vec<f32>,
    pub z_max: Vec<f32>,
    /// Mean intensity.
    pub intensity: Vec<f32>,
    /// Majority class, by Boyer–Moore vote: exact whenever one class holds more
    /// than half the cell, and a plausible one otherwise — which is all a
    /// boundary channel needs, at one byte and one counter per cell.
    pub class: Vec<u8>,
}

impl Raster {
    /// Rasterise the samples in the square window of side `window` centred on
    /// `center`. Samples outside it are ignored, so a caller may over-collect.
    pub fn build(center: [f32; 2], window: f32, samples: &[Sample]) -> Raster {
        let half = window * 0.5;
        let origin = [center[0] - half, center[1] - half];
        let inside = |p: &[f32; 3]| {
            p[0] >= origin[0] && p[0] < origin[0] + window
                && p[1] >= origin[1] && p[1] < origin[1] + window
        };
        let n = samples.iter().filter(|s| inside(&s.pos)).count();
        let cell = cell_size(window, n);
        let side = ((window / cell).ceil() as usize).clamp(1, MAX_CELLS);
        let cells = side * side;
        let mut r = Raster {
            origin,
            cell,
            nx: side,
            ny: side,
            count: vec![0; cells],
            z_min: vec![f32::INFINITY; cells],
            z_max: vec![f32::NEG_INFINITY; cells],
            intensity: vec![0.0; cells],
            class: vec![0; cells],
        };
        let mut votes = vec![0u32; cells];
        for s in samples {
            let Some(c) = r.cell_of(s.pos[0], s.pos[1]) else { continue };
            r.count[c] += 1;
            r.z_min[c] = r.z_min[c].min(s.pos[2]);
            r.z_max[c] = r.z_max[c].max(s.pos[2]);
            r.intensity[c] += s.intensity as f32;
            if votes[c] == 0 {
                r.class[c] = s.class;
                votes[c] = 1;
            } else if r.class[c] == s.class {
                votes[c] += 1;
            } else {
                votes[c] -= 1;
            }
        }
        for c in 0..cells {
            if r.count[c] > 0 { r.intensity[c] /= r.count[c] as f32; }
        }
        r
    }

    pub fn cell_of(&self, x: f32, y: f32) -> Option<usize> {
        let fx = (x - self.origin[0]) / self.cell;
        let fy = (y - self.origin[1]) / self.cell;
        if !(fx >= 0.0 && fy >= 0.0) { return None; }
        let (cx, cy) = (fx as usize, fy as usize);
        if cx >= self.nx || cy >= self.ny { return None; }
        Some(cy * self.nx + cx)
    }

    pub fn center_of(&self, c: usize) -> [f32; 2] {
        [
            self.origin[0] + ((c % self.nx) as f32 + 0.5) * self.cell,
            self.origin[1] + ((c / self.nx) as f32 + 0.5) * self.cell,
        ]
    }
}

/// The edge-strength field over a raster, 0..1 per cell.
///
/// The channels are combined as a weighted sum divided by the **largest**
/// weight, not by their total, and clamped: a full-strength edge in the most
/// trusted channel is a full-strength edge, whether or not any other channel
/// agrees. Dividing by the total would cap a kerb that only shows as a step at
/// a third of the scale.
pub fn edge_field(r: &Raster, p: &Params) -> Vec<f32> {
    let (nx, ny) = (r.nx, r.ny);
    let n = nx * ny;
    let w = p.weights;
    let w_max = w.step.max(w.height_range).max(w.intensity).max(w.class).max(w.occupancy);
    if w_max <= 0.0 { return vec![0.0; n]; }

    let noise = p.step_m * 0.2;
    let mut step = vec![0f32; n];
    let mut range = vec![0f32; n];
    let mut idiff = vec![0f32; n];
    let mut classb = vec![0f32; n];
    let mut occ = vec![0f32; n];

    for cy in 0..ny {
        for cx in 0..nx {
            let c = cy * nx + cx;
            let here = r.count[c] > 0;
            let (mut neighbours, mut occ_diff, mut both, mut class_diff) = (0u32, 0u32, 0u32, 0u32);
            let (mut dz, mut di) = (0f32, 0f32);
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                if x < 0 || y < 0 || x >= nx as isize || y >= ny as isize { continue; }
                let m = y as usize * nx + x as usize;
                neighbours += 1;
                let there = r.count[m] > 0;
                if here != there { occ_diff += 1; }
                if here && there {
                    both += 1;
                    dz = dz.max((r.z_min[c] - r.z_min[m]).abs());
                    di = di.max((r.intensity[c] - r.intensity[m]).abs());
                    if r.class[c] != r.class[m] { class_diff += 1; }
                }
            }
            occ[c] = occ_diff as f32 / neighbours.max(1) as f32;
            if here {
                step[c] = ((dz - noise) / (p.step_m - noise).max(1e-6)).clamp(0.0, 1.0);
                range[c] = ((r.z_max[c] - r.z_min[c]) / p.range_m.max(1e-6)).clamp(0.0, 1.0);
                idiff[c] = di;
                if both > 0 { classb[c] = class_diff as f32 / both as f32; }
            }
        }
    }

    // Intensity has no physical scale — sensors differ by orders of magnitude,
    // and some write 8-bit values into a 16-bit field — so it is scaled by the
    // window's own strong differences rather than by a constant.
    let scale = percentile_nonzero(&idiff, 0.95);
    let mut edge: Vec<f32> = (0..n).map(|c| {
        let i = if scale > 0.0 { (idiff[c] / scale).min(1.0) } else { 0.0 };
        ((w.step * step[c] + w.height_range * range[c] + w.intensity * i
            + w.class * classb[c] + w.occupancy * occ[c]) / w_max).min(1.0)
    }).collect();
    // A light blur closes the one-cell gaps a sparse raster leaves in an edge,
    // which would otherwise send the path off the feature and back.
    blur_121(&mut edge, nx, ny);
    edge
}

/// The `q`-quantile of the non-zero values, or 0 when there are none.
fn percentile_nonzero(values: &[f32], q: f32) -> f32 {
    let mut nz: Vec<f32> = values.iter().copied().filter(|v| *v > 0.0).collect();
    if nz.is_empty() { return 0.0; }
    let k = ((nz.len() - 1) as f32 * q) as usize;
    let (_, v, _) = nz.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *v
}

/// Separable [1 2 1] / 4 blur, clamping at the border.
fn blur_121(field: &mut [f32], nx: usize, ny: usize) {
    let mut tmp = vec![0f32; field.len()];
    for y in 0..ny {
        for x in 0..nx {
            let at = |xx: usize| field[y * nx + xx];
            tmp[y * nx + x] = (at(x.saturating_sub(1)) + 2.0 * at(x) + at((x + 1).min(nx - 1))) * 0.25;
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            let at = |yy: usize| tmp[yy * nx + x];
            field[y * nx + x] = (at(y.saturating_sub(1)) + 2.0 * at(y) + at((y + 1).min(ny - 1))) * 0.25;
        }
    }
}

/// Cheapest paths from one cell to every other, as parent pointers. The costs
/// themselves are only needed while searching and are not kept.
pub struct Paths {
    pub parent: Vec<u32>,
    pub start: usize,
}

/// Dijkstra from `start` over the 8-connected grid.
///
/// Entering a cell costs its step length times `LENGTH_COST + EPS / (edge +
/// EPS)`: about one unit per cell with no edge at all, a tenth of that along a
/// full-strength edge.
///
/// Costs are fixed point, and the heap holds a cost and a cell index packed
/// into one `u64`, so ordering is a single integer compare rather than a float
/// `total_cmp` with a tie-break. The step costs are worked out once per cell
/// rather than on every relaxation. The fixed point cannot overflow: no
/// shortest path in a 512² grid costs more than the straight diagonal, about
/// 760 units, which is 50 M at this scale — well inside 32 bits.
pub fn shortest_paths(edge: &[f32], nx: usize, ny: usize, start: usize) -> Paths {
    const SCALE: f32 = 65_536.0;
    let n = nx * ny;
    let mut parent = vec![NO_PARENT; n];
    if start >= n { return Paths { parent, start }; }

    let axis: Vec<u32> = edge.iter()
        .map(|&e| ((LENGTH_COST + EDGE_EPS / (e.max(0.0) + EDGE_EPS)) * SCALE) as u32)
        .collect();
    let diagonal: Vec<u32> = axis.iter()
        .map(|&a| (a as f32 * std::f32::consts::SQRT_2) as u32)
        .collect();

    let mut dist = vec![u32::MAX; n];
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(n / 4);
    dist[start] = 0;
    heap.push(Reverse(start as u64));
    while let Some(Reverse(key)) = heap.pop() {
        let (cost, c) = ((key >> 32) as u32, (key & 0xFFFF_FFFF) as usize);
        if cost > dist[c] { continue; }
        let (cx, cy) = (c % nx, c / nx);
        let (left, right, down, up) = (cx > 0, cx + 1 < nx, cy > 0, cy + 1 < ny);
        let mut relax = |m: usize, step: u32| {
            let next = cost + step;
            if next < dist[m] {
                dist[m] = next;
                parent[m] = c as u32;
                heap.push(Reverse(((next as u64) << 32) | m as u64));
            }
        };
        if left { relax(c - 1, axis[c - 1]); }
        if right { relax(c + 1, axis[c + 1]); }
        if down {
            relax(c - nx, axis[c - nx]);
            if left { relax(c - nx - 1, diagonal[c - nx - 1]); }
            if right { relax(c - nx + 1, diagonal[c - nx + 1]); }
        }
        if up {
            relax(c + nx, axis[c + nx]);
            if left { relax(c + nx - 1, diagonal[c + nx - 1]); }
            if right { relax(c + nx + 1, diagonal[c + nx + 1]); }
        }
    }
    Paths { parent, start }
}

impl Paths {
    /// The cells from the start to `target`, both included, or `None` when the
    /// target is not in the grid or was never reached.
    pub fn trace(&self, target: usize) -> Option<Vec<usize>> {
        if target >= self.parent.len() { return None; }
        if target != self.start && self.parent[target] == NO_PARENT { return None; }
        let mut cells = vec![target];
        let mut cur = target;
        while cur != self.start {
            cur = self.parent[cur] as usize;
            cells.push(cur);
            // A parent chain can only be as long as the grid; anything longer
            // is a cycle, which Dijkstra cannot produce but a bug could.
            if cells.len() > self.parent.len() { return None; }
        }
        cells.reverse();
        Some(cells)
    }
}

/// Douglas–Peucker: the indices of the points to keep so that no dropped point
/// is further than `tol` from the simplified line. The ends are always kept.
pub fn simplify(points: &[[f32; 2]], tol: f32) -> Vec<usize> {
    let n = points.len();
    if n <= 2 { return (0..n).collect(); }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 { continue; }
        let (mut worst, mut at) = (0f32, a);
        for (i, p) in points.iter().enumerate().take(b).skip(a + 1) {
            let d = point_segment_distance(*p, points[a], points[b]);
            if d > worst { worst = d; at = i; }
        }
        if worst > tol {
            keep[at] = true;
            stack.push((a, at));
            stack.push((at, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

fn point_segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 <= 0.0 { 0.0 } else {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    ((a[0] + t * dx - p[0]).powi(2) + (a[1] + t * dy - p[1]).powi(2)).sqrt()
}

/// Where a traced vertex takes its height from **for display**. On commit every
/// traced vertex is re-snapped at full resolution through the backends; this
/// only has to put the ghost path roughly where it will land.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ZSource {
    Lowest,
    Highest,
    Fixed(f64),
    Interpolated,
}

/// The raster, field and paths grown from one anchor.
pub struct Trace {
    /// World position of the anchor.
    pub anchor: [f64; 3],
    /// Viewer origin the raster was built in.
    pub view_origin: [f64; 3],
    /// A signature of the data the window was built over: how many points the
    /// backends held around it, folded together with the height band and the
    /// visible classes. When it changes — COPC nodes arriving, a class hidden
    /// to steer the tracer — the raster is stale.
    pub data_sig: usize,
    pub params: Params,
    pub raster: Raster,
    pub edge: Vec<f32>,
    pub paths: Paths,
    pub built_at: f64,
    /// Raster, field and paths, in milliseconds — shown in the panel, because
    /// the window size is a trade the operator makes against it.
    pub build_ms: f32,
    heatmap: Option<Texture2D>,
}

fn to_view(p: [f64; 3], origin: [f64; 3]) -> [f32; 3] {
    [(p[0] - origin[0]) as f32, (p[1] - origin[1]) as f32, (p[2] - origin[2]) as f32]
}

impl Trace {
    pub fn build(
        anchor: [f64; 3],
        view_origin: [f64; 3],
        samples: &[Sample],
        data_sig: usize,
        params: Params,
        now: f64,
    ) -> Trace {
        let t0 = std::time::Instant::now();
        let a = to_view(anchor, view_origin);
        let raster = Raster::build([a[0], a[1]], params.window_m, samples);
        let mut t = Trace {
            anchor,
            view_origin,
            data_sig,
            params,
            raster,
            edge: Vec::new(),
            paths: Paths { parent: Vec::new(), start: 0 },
            built_at: now,
            build_ms: 0.0,
            heatmap: None,
        };
        t.refield(params);
        t.build_ms = t0.elapsed().as_secs_f32() * 1000.0;
        t
    }

    /// Recompute the field and paths for new weights, keeping the raster —
    /// which depends only on the window and the data.
    pub fn refield(&mut self, params: Params) {
        let t0 = std::time::Instant::now();
        self.params = params;
        self.edge = edge_field(&self.raster, &params);
        let a = to_view(self.anchor, self.view_origin);
        let start = self.raster.cell_of(a[0], a[1])
            .unwrap_or(self.raster.ny / 2 * self.raster.nx + self.raster.nx / 2);
        self.paths = shortest_paths(&self.edge, self.raster.nx, self.raster.ny, start);
        self.heatmap = None;
        self.build_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// The traced route from the anchor to `target`, simplified, in world
    /// coordinates and including both ends — or `None` when the target is
    /// outside the window.
    ///
    /// The path runs from cell centre to cell centre; its first and last points
    /// are replaced by the real anchor and target, so the route meets the
    /// vertices exactly rather than half a cell off them.
    pub fn route(&self, target: [f64; 3], z: ZSource) -> Option<Vec<[f64; 3]>> {
        let tv = to_view(target, self.view_origin);
        let goal = self.raster.cell_of(tv[0], tv[1])?;
        let cells = self.paths.trace(goal)?;
        let a = to_view(self.anchor, self.view_origin);
        let mut xy: Vec<[f32; 2]> = cells.iter().map(|&c| self.raster.center_of(c)).collect();
        let last = xy.len() - 1;
        xy[0] = [a[0], a[1]];
        xy[last] = [tv[0], tv[1]];
        if last == 0 {
            return Some(vec![self.anchor, target]);
        }

        let mut along = vec![0f32; xy.len()];
        for i in 1..xy.len() {
            along[i] = along[i - 1] + ((xy[i][0] - xy[i - 1][0]).powi(2) + (xy[i][1] - xy[i - 1][1]).powi(2)).sqrt();
        }
        let total = along[last].max(1e-6);
        let o = self.view_origin;

        Some(simplify(&xy, self.raster.cell * 0.75).into_iter().map(|i| {
            let height = if i == 0 {
                self.anchor[2]
            } else if i == last {
                target[2]
            } else {
                let c = cells[i];
                let occupied = self.raster.count[c] > 0;
                let interp = self.anchor[2] + (target[2] - self.anchor[2]) * (along[i] / total) as f64;
                match z {
                    ZSource::Fixed(h) => h,
                    ZSource::Lowest if occupied => self.raster.z_min[c] as f64 + o[2],
                    ZSource::Highest if occupied => self.raster.z_max[c] as f64 + o[2],
                    _ => interp,
                }
            };
            [xy[i][0] as f64 + o[0], xy[i][1] as f64 + o[1], height]
        }).collect())
    }

    /// Highest point in the window, in viewer space: where the heatmap is drawn
    /// so that nothing in the window hides it.
    pub fn top(&self) -> f32 {
        let top = self.raster.z_max.iter().copied().filter(|z| z.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        if top.is_finite() { top } else { to_view(self.anchor, self.view_origin)[2] }
    }

    /// The field as a translucent heatmap over the window, under the 3D camera,
    /// at viewer-space height `z`.
    ///
    /// Drawn flat above the window rather than draped: it is a tuning aid, read
    /// from above, and a quad over everything cannot be hidden by the points it
    /// is describing. Draw it after the other overlays, so its depth does not
    /// hide the ghost path beneath it.
    pub fn draw_heatmap(&mut self, z: f32) {
        let (nx, ny) = (self.raster.nx, self.raster.ny);
        let tex = self.heatmap.get_or_insert_with(|| {
            let bytes: Vec<u8> = self.edge.iter().flat_map(|&e| heat_color(e)).collect();
            let t = Texture2D::from_rgba8(nx as u16, ny as u16, &bytes);
            t.set_filter(FilterMode::Nearest);
            t
        }).clone();
        let (x0, y0) = (self.raster.origin[0], self.raster.origin[1]);
        let (x1, y1) = (x0 + nx as f32 * self.raster.cell, y0 + ny as f32 * self.raster.cell);
        let v = |x: f32, y: f32, u: f32, w: f32| Vertex::new(x, y, z, u, w, WHITE);
        draw_mesh(&Mesh {
            vertices: vec![v(x0, y0, 0.0, 0.0), v(x1, y0, 1.0, 0.0), v(x1, y1, 1.0, 1.0), v(x0, y1, 0.0, 1.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            texture: Some(tex),
        });
    }
}

/// Heatmap colour for an edge value: transparent where there is no edge,
/// through violet and orange to pale yellow at full strength, and more opaque
/// the stronger it is.
fn heat_color(e: f32) -> [u8; 4] {
    const STOPS: [(f32, [f32; 3]); 4] = [
        (0.0, [0.10, 0.05, 0.35]),
        (0.35, [0.65, 0.10, 0.55]),
        (0.7, [0.98, 0.45, 0.10]),
        (1.0, [1.0, 0.95, 0.35]),
    ];
    let e = e.clamp(0.0, 1.0);
    if e < 0.03 { return [0, 0, 0, 0]; }
    let mut rgb = STOPS[3].1;
    for w in STOPS.windows(2) {
        if e <= w[1].0 {
            let t = (e - w[0].0) / (w[1].0 - w[0].0);
            rgb = [0, 1, 2].map(|k| w[0].1[k] + t * (w[1].1[k] - w[0].1[k]));
            break;
        }
    }
    let alpha = 0.25 + 0.65 * e;
    [(rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8, (alpha * 255.0) as u8]
}

/// Edge-following settings and the trace for the current anchor.
pub struct EdgeFollow {
    pub enabled: bool,
    pub show_heatmap: bool,
    pub params: Params,
    pub trace: Option<Trace>,
}

impl Default for EdgeFollow {
    fn default() -> Self {
        // Off: a straight segment is what digitising has always done, and
        // tracing is the speculative part of the tool. The heatmap is on for
        // whoever turns tracing on, because judging the field is the first
        // thing to do with it.
        EdgeFollow { enabled: false, show_heatmap: true, params: Params::default(), trace: None }
    }
}

impl EdgeFollow {
    /// The viewer-space box the raster for `anchor` covers.
    pub fn window_box(&self, anchor: [f64; 3], view_origin: [f64; 3]) -> ([f32; 2], [f32; 2]) {
        let a = to_view(anchor, view_origin);
        let half = self.params.window_m * 0.5;
        ([a[0] - half, a[1] - half], [a[0] + half, a[1] + half])
    }

    /// Bring the trace up to date for `anchor`.
    ///
    /// `gather` collects the samples for a window box and is called only when
    /// the raster really has to be rebuilt: a new anchor, a new window size, a
    /// moved viewer origin, or — at most every [`REBUILD_INTERVAL_S`] — a change
    /// in `data_sig`, the caller's signature of what the window holds (see
    /// [`Trace::data_sig`]). That last one is how a raster built over a thinned
    /// COPC sample picks up the full-resolution nodes that stream in after the
    /// click. A weight change alone recomputes the field and paths and keeps
    /// the raster.
    pub fn update(
        &mut self,
        anchor: [f64; 3],
        view_origin: [f64; 3],
        data_sig: usize,
        now: f64,
        gather: impl FnOnce(([f32; 2], [f32; 2])) -> Vec<Sample>,
    ) {
        let rebuild = match &self.trace {
            None => true,
            Some(t) => t.anchor != anchor
                || t.view_origin != view_origin
                || t.params.window_m != self.params.window_m
                || (t.data_sig != data_sig && now - t.built_at >= REBUILD_INTERVAL_S),
        };
        if rebuild {
            let samples = gather(self.window_box(anchor, view_origin));
            self.trace = Some(Trace::build(anchor, view_origin, &samples, data_sig, self.params, now));
        } else if let Some(t) = self.trace.as_mut()
            && t.params != self.params {
            t.refield(self.params);
        }
    }

    pub fn clear(&mut self) {
        self.trace = None;
    }
}

/// The edge-following block of the digitising panel.
pub fn controls(ui: &mut egui_macroquad::egui::Ui, ef: &mut EdgeFollow) {
    use egui_macroquad::egui;
    ui.horizontal(|ui| {
        ui.checkbox(&mut ef.enabled, "Edge following (E)")
            .on_hover_text("Trace the strongest edge from the last vertex to the cursor, \
                            instead of a straight segment");
        if ef.enabled {
            ui.checkbox(&mut ef.show_heatmap, "edge heatmap (H)");
        }
    });
    if !ef.enabled { return; }

    ui.horizontal(|ui| {
        ui.label("Window:");
        ui.add(egui::DragValue::new(&mut ef.params.window_m)
            .speed(0.5).range(MIN_WINDOW_M..=MAX_WINDOW_M).max_decimals(0).suffix(" m"))
            .on_hover_text("Side of the square traced around the last vertex. The cursor has \
                            to be inside it for a trace.");
    });
    let w = &mut ef.params.weights;
    egui::Grid::new("edge_weights").num_columns(2).show(ui, |ui| {
        for (label, value, hint) in [
            ("Height step", &mut w.step, "Kerbs, breaklines, ditch edges, embankment tops"),
            ("Height spread", &mut w.height_range, "Walls, building footprints — anything vertical"),
            ("Intensity", &mut w.intensity, "Painted road markings"),
            ("Class change", &mut w.class, "Vegetation or buildings against ground"),
            ("Data edge", &mut w.occupancy, "Where points stop: the cloud boundary, roof outlines"),
        ] {
            ui.label(label).on_hover_text(hint);
            ui.add(egui::Slider::new(value, 0.0..=2.0).max_decimals(2));
            ui.end_row();
        }
        ui.label("Full step at");
        ui.add(egui::DragValue::new(&mut ef.params.step_m)
            .speed(0.005).range(0.02..=1.0).max_decimals(3).suffix(" m"));
        ui.end_row();
        ui.label("Full spread at");
        ui.add(egui::DragValue::new(&mut ef.params.range_m)
            .speed(0.02).range(0.1..=5.0).max_decimals(2).suffix(" m"));
        ui.end_row();
    });
    if let Some(t) = &ef.trace {
        ui.label(format!(
            "{}×{} cells of {:.2} m, {:.0} ms",
            t.raster.nx, t.raster.ny, t.raster.cell, t.build_ms,
        ));
    }
    ui.label("Shift-click still places a free, straight vertex.");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A regular grid of samples over `[0, w)²`, offset half a spacing so no
    /// sample sits on a cell boundary.
    fn grid(w: f32, spacing: f32, f: impl Fn(f32, f32) -> (f32, u16, u8)) -> Vec<Sample> {
        let n = (w / spacing) as usize;
        let mut out = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let (x, y) = ((i as f32 + 0.5) * spacing, (j as f32 + 0.5) * spacing);
                let (z, intensity, class) = f(x, y);
                out.push(Sample { pos: [x, y, z], intensity, class });
            }
        }
        out
    }

    #[test]
    fn the_cell_size_follows_the_density() {
        // 250 pts/m²: four points a cell wants a 0.126 m cell.
        assert!((cell_size(40.0, 400_000) - 0.1265).abs() < 1e-3);
        // Very dense: never finer than 512 cells across.
        assert!((cell_size(40.0, 50_000_000) - 40.0 / 512.0).abs() < 1e-6);
        // Very sparse: never coarser than 16 cells across.
        assert_eq!(cell_size(40.0, 100), 2.5);
        assert_eq!(cell_size(40.0, 0), 2.5);
    }

    #[test]
    fn a_minority_class_does_not_win_a_cell() {
        let s = |class| Sample { pos: [0.5, 0.5, 0.0], intensity: 0, class };
        let r = Raster::build([0.5, 0.5], 1.0, &[s(6), s(2), s(2)]);
        let c = r.cell_of(0.5, 0.5).unwrap();
        assert_eq!(r.class[c], 2);
        assert_eq!(r.count[c], 3);
    }

    #[test]
    fn a_step_in_the_ground_is_an_edge_and_flat_ground_is_not() {
        let samples = grid(20.0, 0.1, |_, y| (if y >= 10.0 { 0.15 } else { 0.0 }, 1000, 2));
        let r = Raster::build([10.0, 10.0], 20.0, &samples);
        let edge = edge_field(&r, &Params::default());
        let at = |x: f32, y: f32| edge[r.cell_of(x, y).unwrap()];
        assert!(at(10.0, 9.9) > 0.5, "at the kerb: {}", at(10.0, 9.9));
        assert!(at(10.0, 5.0) < 0.01, "on flat ground: {}", at(10.0, 5.0));
        assert!(at(10.0, 15.0) < 0.01, "on the raised side: {}", at(10.0, 15.0));
    }

    #[test]
    fn a_painted_line_is_an_edge_only_while_intensity_counts() {
        let samples = grid(20.0, 0.1, |x, _| (0.0, if (9.6..10.4).contains(&x) { 20_000 } else { 3_000 }, 2));
        let r = Raster::build([10.0, 10.0], 20.0, &samples);
        let mut p = Params::default();
        let edge = edge_field(&r, &p);
        assert!(edge[r.cell_of(9.55, 10.0).unwrap()] > 0.5);
        assert!(edge[r.cell_of(5.0, 10.0).unwrap()] < 0.01);

        p.weights.intensity = 0.0;
        let edge = edge_field(&r, &p);
        assert!(edge[r.cell_of(9.55, 10.0).unwrap()] < 0.01, "the weight is what turns it off");
    }

    #[test]
    fn the_trace_follows_a_curved_kerb_rather_than_cutting_across() {
        let kerb = |x: f32| 10.0 + 3.0 * (x / 3.0).sin();
        let samples = grid(20.0, 0.1, |x, y| (if y >= kerb(x) { 0.15 } else { 0.0 }, 1000, 2));
        let (a, b) = ([1.0, kerb(1.0) as f64, 0.0], [19.0, kerb(19.0) as f64, 0.0]);
        let t = Trace::build(a, [0.0; 3], &samples, samples.len(), Params::default(), 0.0);

        let route = t.route(b, ZSource::Interpolated).expect("the target is in the window");
        assert_eq!(route.first(), Some(&a));
        assert_eq!(route.last(), Some(&b));
        assert!(route.len() >= 4, "a curve needs more than a straight line: {route:?}");
        // Every point along the route, not just its vertices, stays on the kerb.
        for w in route.windows(2) {
            for k in 0..=20 {
                let s = k as f64 / 20.0;
                let (x, y) = (w[0][0] + s * (w[1][0] - w[0][0]), w[0][1] + s * (w[1][1] - w[0][1]));
                let off = (y - kerb(x as f32) as f64).abs();
                assert!(off < 0.5, "{off:.2} m off the kerb at x = {x:.2}");
            }
        }
    }

    #[test]
    fn outside_the_window_there_is_no_trace() {
        let samples = grid(10.0, 0.2, |_, _| (0.0, 0, 2));
        let t = Trace::build([5.0, 5.0, 0.0], [0.0; 3], &samples, 0, Params { window_m: 10.0, ..Default::default() }, 0.0);
        assert!(t.route([5.0, 12.0, 0.0], ZSource::Lowest).is_none());
        assert!(t.route([5.0, 9.0, 0.0], ZSource::Lowest).is_some());
    }

    #[test]
    fn simplifying_keeps_corners_and_drops_the_points_between() {
        let pts: Vec<[f32; 2]> = (0..=10).map(|i| [i as f32, 0.0])
            .chain((1..=10).map(|j| [10.0, j as f32]))
            .collect();
        assert_eq!(simplify(&pts, 0.1), vec![0, 10, 20]);
        // A wobble within the tolerance is dropped, one outside it is kept.
        let wobbly = [[0.0, 0.0], [1.0, 0.05], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]];
        assert_eq!(simplify(&wobbly, 0.1), vec![0, 2, 3, 4]);
    }

    #[test]
    fn the_raster_is_rebuilt_only_when_it_has_to_be() {
        let samples = grid(20.0, 0.25, |_, _| (0.0, 0, 2));
        let gathers = std::cell::Cell::new(0);
        let gather = |_| { gathers.set(gathers.get() + 1); samples.clone() };
        let mut ef = EdgeFollow::default();
        let a = [10.0, 10.0, 0.0];

        ef.update(a, [0.0; 3], 100, 0.0, gather);
        ef.update(a, [0.0; 3], 100, 0.1, gather);
        assert_eq!(gathers.get(), 1, "the same anchor reuses the raster");

        ef.params.weights.step = 0.3;
        ef.update(a, [0.0; 3], 100, 0.2, gather);
        assert_eq!(gathers.get(), 1, "a weight is a new field, not a new raster");
        assert_eq!(ef.trace.as_ref().unwrap().params.weights.step, 0.3);

        ef.update(a, [0.0; 3], 150, 0.3, gather);
        assert_eq!(gathers.get(), 1, "nodes arriving are picked up, but not every frame");
        ef.update(a, [0.0; 3], 150, 0.6, gather);
        assert_eq!(gathers.get(), 2);

        ef.update([11.0, 10.0, 0.0], [0.0; 3], 150, 0.61, gather);
        assert_eq!(gathers.get(), 3, "a new anchor always rebuilds");
    }
}
