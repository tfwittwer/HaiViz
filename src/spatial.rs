//! Flat 2D XY grid over a resident point set, built by counting sort.
//!
//! Every query in haiviz used to be a linear scan over all resident points.
//! That is fine for a click and fatal for snapping, which needs a radius query
//! on every mouse move. This is the index those queries run on.
//!
//! The layout follows `haicloudrs::voxelgrid::Voxelgrid`: a CSR bucket table
//! (`starts` + `idx`) filled by counting each point once, then placing each
//! point once. O(n) build, no allocation per bucket, no tree.
//!
//! ## Coordinate system
//!
//! The grid stores **viewer-space f32**, not world f64. That is the space
//! `Stage::positions`, `LoadedTile::positions` and `CopcLoadedNode::positions`
//! already live in, the space the camera ray is built in, and the space the
//! existing pick code works in — so nothing has to be converted on the way in
//! or on the way out. A caller that wants world coordinates adds its own
//! `viewer_origin`, which it already owns. Viewer space is cloud-local (a few
//! km at most), so f32 costs nothing here.
//!
//! ## Filtering
//!
//! Every query takes a `filter: impl Fn(u32) -> bool` over the point index.
//! The filter runs *inside* the grid walk rather than on the result, so a
//! `ZClip` or a hidden-class test prunes candidates before they can win a
//! nearest-point contest — otherwise a query would return a point the operator
//! cannot see, or return nothing at all because the one hidden point was
//! closest. Callers that want no filtering pass [`NO_FILTER`].
//!
//! The cached per-cell `z_min` / `z_max` are computed over **all** points in
//! the cell, filter or no filter. They are only ever used to *prune* — never
//! to answer — so a filter can shrink the true extent without making an answer
//! wrong.

// Phase 0 builds this index and wires it into all three backends but has no
// user-visible consumer yet: snapping (phase 4) and edge following (phase 5)
// are what call the queries. Without this the whole module reads as dead.
#![allow(dead_code)]

/// Filter that accepts every point. `|_| true`, named so call sites read.
pub const NO_FILTER: fn(u32) -> bool = |_| true;

/// Target average points per cell. Small enough that a radius query touches
/// few wasted points, large enough that the per-cell arrays do not dominate
/// memory: at 2 points/cell the fixed cost is 24 B/cell + 4 B/point ≈ 16 B per
/// point, which is the ~12 B/point the design budgeted for, plus the argmin /
/// argmax indices.
const TARGET_OCCUPANCY: f64 = 2.0;

/// Bounds on the automatically chosen cell size, in metres. The lower bound
/// stops a tiny, dense cloud from allocating millions of cells for thousands of
/// points; the upper bound stops a sparse one from degenerating into a linear
/// scan.
const MIN_CELL: f32 = 0.05;
const MAX_CELL: f32 = 5.0;

/// Hard ceiling on cells per indexed point.
///
/// `cell = sqrt(area * occupancy / n)` is self-scaling: substitute it back and
/// the cell count is `n / occupancy` whatever the footprint. Clamping breaks
/// that, and [`MAX_CELL`] breaks it in the expensive direction — a cell forced
/// *smaller* than the formula asked for means *more* cells than there are
/// points.
///
/// That is not hypothetical. A COPC root node is a thinned sample of the whole
/// file: 50 k points spread over a 30 km extent asks for a 130 m cell, gets
/// clamped to 5 m, and allocates 36 M cells — 720 MB — for 50 k points, on the
/// loader thread, while the camera is moving. This is the floor that stops it.
const MAX_CELLS_PER_POINT: f64 = 4.0;

/// Sentinel for "this cell is empty" in `arg_min` / `arg_max`.
const NO_POINT: u32 = u32::MAX;

/// A point returned by a query: its index into the source `positions` slice,
/// and its viewer-space position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub idx: u32,
    pub pos: [f32; 3],
}

/// A point returned by [`SpatialGrid::nearest_ray`], with how far along the
/// ray it sits. `t` is what lets a caller merge hits from several grids — the
/// tiles or COPC nodes a ray crosses each answer for their own points, and the
/// frontmost of those answers is the one the operator meant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub idx: u32,
    pub pos: [f32; 3],
    /// Distance along the ray, in metres.
    pub t: f32,
}

/// Counting-sort XY bucket grid over one resident point set.
///
/// One grid belongs to exactly one contiguous `positions` slice and holds
/// indices into it, so it is only valid as long as that slice keeps its order.
/// All three backends satisfy that: `Stage::positions` is immutable for the
/// life of the stage, and a tile's or a COPC node's positions are built once at
/// load and dropped wholesale at unload.
pub struct SpatialGrid {
    /// Lower corner of cell (0, 0), in viewer space.
    origin: [f32; 2],
    /// Cell size in metres, square in XY.
    cell: f32,
    nx: usize,
    ny: usize,
    /// CSR bucket offsets, length `nx * ny + 1`.
    starts: Vec<u32>,
    /// Point indices, bucketed. Length = number of indexed points.
    idx: Vec<u32>,
    /// Per-cell Z extent over *all* points in the cell, and the points
    /// achieving it. `f32::INFINITY` / `NEG_INFINITY` and [`NO_POINT`] for an
    /// empty cell. Pruning and raster input only — see the module note on
    /// filtering.
    z_min: Vec<f32>,
    z_max: Vec<f32>,
    arg_min: Vec<u32>,
    arg_max: Vec<u32>,
    /// Z extent of the whole grid, for the ray prune.
    z_lo: f32,
    z_hi: f32,
}

impl SpatialGrid {
    /// An index over nothing. Queries on it return nothing; building the real
    /// one later is a plain assignment.
    pub fn empty() -> SpatialGrid {
        SpatialGrid {
            origin: [0.0, 0.0],
            cell: 1.0,
            nx: 0,
            ny: 0,
            starts: vec![0],
            idx: Vec::new(),
            z_min: Vec::new(),
            z_max: Vec::new(),
            arg_min: Vec::new(),
            arg_max: Vec::new(),
            z_lo: 0.0,
            z_hi: 0.0,
        }
    }

    /// Build an index over `positions`, choosing the cell size from the point
    /// density so the average bucket holds [`TARGET_OCCUPANCY`] points.
    pub fn build(positions: &[[f32; 3]]) -> SpatialGrid {
        let cell = Self::auto_cell(positions);
        Self::build_with_cell(positions, cell)
    }

    /// Cell size that puts roughly [`TARGET_OCCUPANCY`] points in a bucket:
    /// `cell = sqrt(area * occupancy / n)`. Derived from the actual footprint
    /// rather than fixed, because a 40 m tile and a 3 km strip want very
    /// different cells for the same point count.
    fn auto_cell(positions: &[[f32; 3]]) -> f32 {
        if positions.is_empty() {
            return 1.0;
        }
        let (min, max) = bounds_xy(positions);
        let w = (max[0] - min[0]).max(0.0) as f64;
        let h = (max[1] - min[1]).max(0.0) as f64;
        let area = w * h;
        if area <= 0.0 {
            // A single point, or a perfectly axis-aligned line of them. Any
            // cell size gives one column of buckets; take the middle.
            return 1.0;
        }
        let n = positions.len() as f64;
        let ideal = (area * TARGET_OCCUPANCY / n).sqrt();
        // The smallest cell the budget can pay for. Applied *after* the
        // clamps, so it can override `MAX_CELL`: a sparse cloud over a huge
        // extent genuinely wants cells bigger than 5 m, and refusing to give
        // it them is what costs the memory.
        let affordable = (area / (MAX_CELLS_PER_POINT * n)).sqrt();
        ((ideal as f32).clamp(MIN_CELL, MAX_CELL)).max(affordable as f32)
    }

    /// Build with an explicit cell size. Mostly for tests and the bench; normal
    /// callers want [`SpatialGrid::build`].
    pub fn build_with_cell(positions: &[[f32; 3]], cell: f32) -> SpatialGrid {
        assert!(cell > 0.0, "cell size must be positive");
        assert!(
            positions.len() <= u32::MAX as usize,
            "SpatialGrid indexes points with u32",
        );
        if positions.is_empty() {
            return SpatialGrid::empty();
        }

        let (min, max) = bounds_xy(positions);
        let origin = [min[0], min[1]];
        // The budget again, here rather than only in `auto_cell`, because this
        // is `pub`: an explicit cell size must not be able to ask for a grid
        // that does not fit in memory.
        let mut cell = cell;
        loop {
            let w = ((max[0] - min[0]) / cell) as f64 + 1.0;
            let h = ((max[1] - min[1]) / cell) as f64 + 1.0;
            if w * h <= MAX_CELLS_PER_POINT * positions.len() as f64 { break; }
            cell *= 2.0;
        }
        // +1 so a point exactly on the upper bound lands in the last cell
        // rather than one past it.
        let nx = (((max[0] - min[0]) / cell) as usize) + 1;
        let ny = (((max[1] - min[1]) / cell) as usize) + 1;
        let n_cells = nx * ny;

        let mut starts = vec![0u32; n_cells + 1];
        let mut z_min = vec![f32::INFINITY; n_cells];
        let mut z_max = vec![f32::NEG_INFINITY; n_cells];
        let mut arg_min = vec![NO_POINT; n_cells];
        let mut arg_max = vec![NO_POINT; n_cells];

        let cell_of = |p: &[f32; 3]| -> usize {
            let cx = (((p[0] - origin[0]) / cell) as usize).min(nx - 1);
            let cy = (((p[1] - origin[1]) / cell) as usize).min(ny - 1);
            cy * nx + cx
        };

        // Pass 1: count, and accumulate the per-cell Z extent while we are
        // already touching every point.
        let mut z_lo = f32::INFINITY;
        let mut z_hi = f32::NEG_INFINITY;
        for (i, p) in positions.iter().enumerate() {
            let c = cell_of(p);
            // `starts` doubles as the counter here; the prefix sum below turns
            // it into offsets. Shifted by one so the sum can be in place.
            starts[c + 1] += 1;
            if p[2] < z_min[c] {
                z_min[c] = p[2];
                arg_min[c] = i as u32;
            }
            if p[2] > z_max[c] {
                z_max[c] = p[2];
                arg_max[c] = i as u32;
            }
            z_lo = z_lo.min(p[2]);
            z_hi = z_hi.max(p[2]);
        }

        // Prefix sum → bucket start offsets.
        for c in 0..n_cells {
            starts[c + 1] += starts[c];
        }

        // Pass 2: place. `fill` advances a cursor per cell; reusing `starts`
        // would destroy the offsets we just built.
        let mut fill = vec![0u32; n_cells];
        let mut idx = vec![0u32; positions.len()];
        for (i, p) in positions.iter().enumerate() {
            let c = cell_of(p);
            idx[(starts[c] + fill[c]) as usize] = i as u32;
            fill[c] += 1;
        }

        SpatialGrid { origin, cell, nx, ny, starts, idx, z_min, z_max, arg_min, arg_max, z_lo, z_hi }
    }

    pub fn cell_size(&self) -> f32 { self.cell }
    pub fn nx(&self) -> usize { self.nx }
    pub fn ny(&self) -> usize { self.ny }
    pub fn len(&self) -> usize { self.idx.len() }
    pub fn is_empty(&self) -> bool { self.idx.is_empty() }

    /// Viewer-space XY bounds of the indexed set, as `(min, max)`. Used to
    /// reject a whole tile or COPC node before walking its grid.
    pub fn bounds_xy(&self) -> ([f32; 2], [f32; 2]) {
        if self.nx == 0 {
            return ([0.0, 0.0], [0.0, 0.0]);
        }
        (
            self.origin,
            [
                self.origin[0] + self.nx as f32 * self.cell,
                self.origin[1] + self.ny as f32 * self.cell,
            ],
        )
    }

    /// Viewer-space Z extent of the indexed set.
    pub fn bounds_z(&self) -> (f32, f32) { (self.z_lo, self.z_hi) }

    /// True when the query circle can touch this grid at all. The cheap test a
    /// caller runs over every resident COPC node before descending into one.
    pub fn circle_overlaps(&self, x: f32, y: f32, r: f32) -> bool {
        if self.nx == 0 { return false; }
        let (min, max) = self.bounds_xy();
        let dx = (min[0] - x).max(0.0).max(x - max[0]);
        let dy = (min[1] - y).max(0.0).max(y - max[1]);
        dx * dx + dy * dy <= r * r
    }

    /// True when a viewer-space XY box can touch this grid at all — the
    /// rectangle counterpart of [`SpatialGrid::circle_overlaps`], for the
    /// edge-following raster window.
    pub fn box_overlaps(&self, min: [f32; 2], max: [f32; 2]) -> bool {
        if self.nx == 0 { return false; }
        let (lo, hi) = self.bounds_xy();
        min[0] <= hi[0] && max[0] >= lo[0] && min[1] <= hi[1] && max[1] >= lo[1]
    }

    /// The points in one cell, as indices into the source slice.
    #[inline]
    fn bucket(&self, c: usize) -> &[u32] {
        &self.idx[self.starts[c] as usize..self.starts[c + 1] as usize]
    }

    /// Clamp a viewer-space XY box to the cell index ranges it covers.
    /// Returns `None` when the box misses the grid entirely.
    fn cell_range(&self, min: [f32; 2], max: [f32; 2]) -> Option<(usize, usize, usize, usize)> {
        if self.nx == 0 { return None; }
        let fx0 = (min[0] - self.origin[0]) / self.cell;
        let fx1 = (max[0] - self.origin[0]) / self.cell;
        let fy0 = (min[1] - self.origin[1]) / self.cell;
        let fy1 = (max[1] - self.origin[1]) / self.cell;
        if fx1 < 0.0 || fy1 < 0.0 {
            return None;
        }
        if fx0 >= self.nx as f32 || fy0 >= self.ny as f32 {
            return None;
        }
        let x0 = fx0.max(0.0) as usize;
        let y0 = fy0.max(0.0) as usize;
        let x1 = (fx1.max(0.0) as usize).min(self.nx - 1);
        let y1 = (fy1.max(0.0) as usize).min(self.ny - 1);
        Some((x0, y0, x1, y1))
    }

    /// Append every point within `r` of `(x, y)` in XY — a cylinder, not a
    /// sphere, because a snap radius is a plan-view tolerance and the whole
    /// point is to reach the ground under a canopy or the kerb below the eye.
    ///
    /// `out` is appended to, not cleared, so a caller can accumulate across
    /// several tiles or COPC nodes into one buffer.
    pub fn points_in_radius(
        &self,
        positions: &[[f32; 3]],
        x: f32,
        y: f32,
        r: f32,
        filter: impl Fn(u32) -> bool,
        out: &mut Vec<u32>,
    ) {
        let Some((x0, y0, x1, y1)) = self.cell_range([x - r, y - r], [x + r, y + r]) else {
            return;
        };
        let r2 = r * r;
        for cy in y0..=y1 {
            let row = cy * self.nx;
            for cx in x0..=x1 {
                for &i in self.bucket(row + cx) {
                    let p = &positions[i as usize];
                    let dx = p[0] - x;
                    let dy = p[1] - y;
                    if dx * dx + dy * dy <= r2 && filter(i) {
                        out.push(i);
                    }
                }
            }
        }
    }

    /// The `pct`-th percentile of Z among the points within `r` of `(x, y)`,
    /// and the point that realises it.
    ///
    /// `pct` is in 0..=100. This is a *nearest-rank* percentile — the answer is
    /// always a real point, never an interpolated Z that nothing in the cloud
    /// sits at, because the whole purpose is to put a vertex on measured data.
    /// `0` and `100` therefore give back the raw minimum and maximum.
    ///
    /// Percentiles rather than raw extrema because raw extrema on LiDAR are
    /// usually noise: one stray low return puts a ditch invert 30 cm under the
    /// ground. A few hundred points come back from a 0.2 m radius, so the sort
    /// is free.
    pub fn percentile_in_radius(
        &self,
        positions: &[[f32; 3]],
        x: f32,
        y: f32,
        r: f32,
        pct: f32,
        filter: impl Fn(u32) -> bool,
    ) -> Option<Hit> {
        // One small allocation per call. A snap query runs at most once a
        // frame over a 0.2 m radius — a few hundred points — so this is far
        // below the cost of the grid walk that fills it, and threading a
        // scratch buffer through every backend wrapper is not worth it.
        let mut cands = Vec::new();
        self.points_in_radius(positions, x, y, r, filter, &mut cands);
        percentile_of(positions, &cands, pct)
    }

    /// Nearest point to a ray, within `max_dist` of it measured perpendicular.
    ///
    /// Returns the **frontmost** qualifying point — the smallest distance along
    /// the ray, not the smallest distance to it. That is what picking means
    /// from an oblique view: `max_dist` is the tolerance, the same role screen
    /// distance plays in `find_nearest_point`, and within that tolerance the
    /// operator means the surface facing them, not whatever happens to sit a
    /// millimetre closer to the ray axis on the far side of the wall.
    ///
    /// `dir` need not be normalised; `max_dist` and the returned distance are
    /// in metres regardless.
    pub fn nearest_ray(
        &self,
        positions: &[[f32; 3]],
        origin: [f32; 3],
        dir: [f32; 3],
        max_dist: f32,
        filter: impl Fn(u32) -> bool,
    ) -> Option<RayHit> {
        if self.nx == 0 { return None; }
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        // NaN as well as zero: a degenerate direction has no answer, and
        // normalising by one would poison every comparison below.
        if len <= 0.0 || len.is_nan() { return None; }
        let d = [dir[0] / len, dir[1] / len, dir[2] / len];

        let mut best: Option<RayHit> = None;
        let mut best_t = f32::INFINITY;

        // Test one point against the ray: perpendicular distance within
        // tolerance, and in front of the origin rather than behind it.
        let mut consider = |i: u32| {
            let p = &positions[i as usize];
            let vx = p[0] - origin[0];
            let vy = p[1] - origin[1];
            let vz = p[2] - origin[2];
            let t = vx * d[0] + vy * d[1] + vz * d[2];
            if t < 0.0 || t >= best_t { return; }
            let px = vx - t * d[0];
            let py = vy - t * d[1];
            let pz = vz - t * d[2];
            if px * px + py * py + pz * pz > max_dist * max_dist { return; }
            if !filter(i) { return; }
            best_t = t;
            best = Some(RayHit { idx: i, pos: *p, t });
        };

        let dxy = (d[0] * d[0] + d[1] * d[1]).sqrt();
        // A ray whose XY projection is shorter than a thousandth of a cell over
        // the full Z span of the grid is looking straight down: its footprint
        // is a disc, and the sweep below would divide by ~zero.
        let z_span = (self.z_hi - self.z_lo).abs().max(1.0);
        if dxy * z_span < self.cell * 1e-3 {
            let (x0, y0, x1, y1) = self.cell_range(
                [origin[0] - max_dist, origin[1] - max_dist],
                [origin[0] + max_dist, origin[1] + max_dist],
            )?;
            for cy in y0..=y1 {
                let row = cy * self.nx;
                for cx in x0..=x1 {
                    for &i in self.bucket(row + cx) { consider(i); }
                }
            }
            return best;
        }

        // Sweep the corridor column by column along the ray's major XY axis, so
        // every cell in it is visited exactly once — a thick-line DDA would
        // revisit the overlap between consecutive kernels, and deduping that
        // needs a scratch stamp array the query does not have.
        //
        // For each column the corridor covers, the ray's cross-axis value is
        // monotonic across the column, so its span there is bounded by the
        // values at the two column edges, widened by `max_dist`.
        let x_major = d[0].abs() >= d[1].abs();
        let (maj, min_ax) = if x_major { (0usize, 1usize) } else { (1usize, 0usize) };
        let (n_maj, n_min) = if x_major { (self.nx, self.ny) } else { (self.ny, self.nx) };

        // Clip the ray to the grid's major-axis extent, keeping only t >= 0.
        let lo = self.origin[maj] - max_dist;
        let hi = self.origin[maj] + n_maj as f32 * self.cell + max_dist;
        let (mut t0, mut t1) = (
            (lo - origin[maj]) / d[maj],
            (hi - origin[maj]) / d[maj],
        );
        if t0 > t1 { std::mem::swap(&mut t0, &mut t1); }
        t0 = t0.max(0.0);
        if t1 < t0 { return None; }
        // And to its Z extent, which is what makes an oblique ray through a
        // tall scene cheap: below the ground and above the canopy there is
        // nothing to find.
        if d[2].abs() > 1e-6 {
            let (mut tz0, mut tz1) = (
                (self.z_lo - max_dist - origin[2]) / d[2],
                (self.z_hi + max_dist - origin[2]) / d[2],
            );
            if tz0 > tz1 { std::mem::swap(&mut tz0, &mut tz1); }
            t0 = t0.max(tz0);
            t1 = t1.min(tz1);
            if t1 < t0 { return None; }
        } else if origin[2] < self.z_lo - max_dist || origin[2] > self.z_hi + max_dist {
            return None;
        }

        let a_maj = origin[maj] + t0 * d[maj];
        let b_maj = origin[maj] + t1 * d[maj];
        let (seg_lo, seg_hi) = if a_maj <= b_maj { (a_maj, b_maj) } else { (b_maj, a_maj) };

        let c_lo = (((seg_lo - max_dist - self.origin[maj]) / self.cell).max(0.0) as usize).min(n_maj - 1);
        let c_hi_f = (seg_hi + max_dist - self.origin[maj]) / self.cell;
        if c_hi_f < 0.0 { return None; }
        let c_hi = (c_hi_f as usize).min(n_maj - 1);

        // Cross-axis value of the ray at a given major-axis world coordinate.
        let cross_at = |w: f32| origin[min_ax] + (w - origin[maj]) / d[maj] * d[min_ax];

        for c in c_lo..=c_hi {
            let w0 = self.origin[maj] + c as f32 * self.cell;
            let w1 = w0 + self.cell;
            // Clamp to the segment: past its ends the ray does not exist, and
            // extrapolating `cross_at` there would widen the corridor for no
            // reason.
            let e0 = w0.clamp(seg_lo, seg_hi);
            let e1 = w1.clamp(seg_lo, seg_hi);
            let (v0, v1) = (cross_at(e0), cross_at(e1));
            let (mut lo_v, mut hi_v) = if v0 <= v1 { (v0, v1) } else { (v1, v0) };
            lo_v -= max_dist;
            hi_v += max_dist;

            let f_lo = (lo_v - self.origin[min_ax]) / self.cell;
            let f_hi = (hi_v - self.origin[min_ax]) / self.cell;
            if f_hi < 0.0 || f_lo >= n_min as f32 { continue; }
            let m_lo = (f_lo.max(0.0) as usize).min(n_min - 1);
            let m_hi = (f_hi.max(0.0) as usize).min(n_min - 1);

            for m in m_lo..=m_hi {
                let cell_idx = if x_major { m * self.nx + c } else { c * self.nx + m };
                for &i in self.bucket(cell_idx) { consider(i); }
            }
        }

        best
    }

    /// Visit every cell meeting the viewer-space XY box, handing the callback
    /// the cell's grid coordinates, its cached Z extent and its points.
    ///
    /// This is the raster feed for edge following: one pass over a window turns
    /// into count / z_min / z_max / intensity / class channels without the
    /// caller having to know the bucket layout. The Z extent handed over is the
    /// unfiltered one — a callback applying a filter recomputes from `points`.
    pub fn cells_in_box(
        &self,
        min: [f32; 2],
        max: [f32; 2],
        mut visit: impl FnMut(CellRef<'_>),
    ) {
        let Some((x0, y0, x1, y1)) = self.cell_range(min, max) else { return };
        for cy in y0..=y1 {
            let row = cy * self.nx;
            for cx in x0..=x1 {
                let c = row + cx;
                let points = self.bucket(c);
                if points.is_empty() { continue; }
                visit(CellRef {
                    cx,
                    cy,
                    center: [
                        self.origin[0] + (cx as f32 + 0.5) * self.cell,
                        self.origin[1] + (cy as f32 + 0.5) * self.cell,
                    ],
                    z_min: self.z_min[c],
                    z_max: self.z_max[c],
                    arg_min: self.arg_min[c],
                    arg_max: self.arg_max[c],
                    points,
                });
            }
        }
    }
}

/// One occupied cell, as handed to [`SpatialGrid::cells_in_box`].
pub struct CellRef<'a> {
    pub cx: usize,
    pub cy: usize,
    /// Viewer-space XY centre of the cell.
    pub center: [f32; 2],
    /// Z extent over all points in the cell — unfiltered, see the module note.
    pub z_min: f32,
    pub z_max: f32,
    /// Points achieving `z_min` / `z_max`, as indices into the source slice.
    pub arg_min: u32,
    pub arg_max: u32,
    /// Every point in the cell, as indices into the source slice.
    pub points: &'a [u32],
}

/// The point under the cursor: the frontmost hit in the **thinnest** tube
/// around the ray that holds anything. `query` is tried at `r_min`, then at
/// double that, and so on up to `r_max`.
///
/// [`SpatialGrid::nearest_ray`] on its own returns the frontmost point within
/// its whole tolerance — and wherever a surface is not square to the view, that
/// point sits at the *edge* of the tolerance rather than on the axis. The tube
/// meets a slanted façade in a long ellipse, and the ellipse's nearest end is
/// off to the side facing the camera: a snap marker a full radius left or right
/// of the cursor, flipping with the side of the building it is over. Growing
/// the tube from about a pixel keeps "the surface facing you" and bounds the
/// offset by the thinnest tube that finds a point — on the order of the point
/// spacing — while a cursor over a gap still reaches out to the full radius.
pub fn surface_hit<T>(r_min: f32, r_max: f32, mut query: impl FnMut(f32) -> Option<T>) -> Option<T> {
    let r_max = r_max.max(1e-4);
    let mut r = r_min.clamp(1e-4, r_max);
    loop {
        if let Some(hit) = query(r) { return Some(hit); }
        if r >= r_max { return None; }
        r = (r * 2.0).min(r_max);
    }
}

/// Nearest-rank `pct`-th percentile of Z over `candidates`, and the point that
/// realises it. Shared by [`SpatialGrid::percentile_in_radius`] and by callers
/// that have merged candidates from several tiles or COPC nodes into one list —
/// a percentile over a union cannot be assembled from per-grid percentiles.
pub fn percentile_of(positions: &[[f32; 3]], candidates: &[u32], pct: f32) -> Option<Hit> {
    let mut zs: Vec<(f32, u32)> = candidates
        .iter()
        .map(|&i| (positions[i as usize][2], i))
        .collect();
    let (_, idx) = percentile_by_z(&mut zs, pct)?;
    Some(Hit { idx, pos: positions[idx as usize] })
}

/// Nearest-rank `pct`-th percentile over `(z, payload)` samples, sorting them
/// in place and returning the winner.
///
/// The multi-source form: a tile manager or COPC stage gathers candidates from
/// every grid the query circle touches and asks here. A percentile over a union
/// cannot be assembled from per-grid percentiles, so the union has to be built
/// before the rank is taken — which is also why the payload is generic, since
/// each backend identifies a point differently.
pub fn percentile_by_z<T: Copy>(samples: &mut [(f32, T)], pct: f32) -> Option<(f32, T)> {
    if samples.is_empty() { return None; }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Nearest-rank: rank r = ceil(pct/100 * n), clamped to 1..=n, element r-1.
    let n = samples.len();
    let rank = ((pct.clamp(0.0, 100.0) as f64 / 100.0) * n as f64).ceil() as usize;
    Some(samples[rank.clamp(1, n) - 1])
}

fn bounds_xy(positions: &[[f32; 3]]) -> ([f32; 2], [f32; 2]) {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for p in positions {
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    (min, max)
}

#[cfg(test)]
mod surface_hit_tests {
    // Unused when the bench includes this file by path; see `tests` below.
    #[allow(unused_imports)]
    use super::*;

    fn perp(p: [f32; 3], o: [f32; 3], d: [f32; 3]) -> f32 {
        let v = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        let t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        ((v[0] - t * d[0]).powi(2) + (v[1] - t * d[1]).powi(2) + (v[2] - t * d[2]).powi(2)).sqrt()
    }

    #[test]
    fn the_snap_lands_under_the_cursor_on_a_slanted_wall() {
        // A façade on y = 10 at 2 cm spacing, looked at 14° off its surface —
        // an oblique view of a house front.
        let mut pts = Vec::new();
        for i in 0..500 {
            for k in 0..300 {
                pts.push([-5.0 + i as f32 * 0.02, 10.0, k as f32 * 0.02]);
            }
        }
        let grid = SpatialGrid::build(&pts);
        let l = (1.0f32 + 0.25 * 0.25 + 0.1 * 0.1).sqrt();
        let d = [1.0 / l, 0.25 / l, -0.1 / l];
        let aim = [0.31, 10.0, 3.05];
        let o = [aim[0] - d[0] * 50.0, aim[1] - d[1] * 50.0, aim[2] - d[2] * 50.0];

        // The whole radius at once: the frontmost point is at the tube's edge.
        // This is the bug as reported — a marker a radius to one side.
        let edge = grid.nearest_ray(&pts, o, d, 0.2, NO_FILTER).unwrap();
        assert!(perp(edge.pos, o, d) > 0.15, "expected the edge of the tube, got {}", perp(edge.pos, o, d));

        let hit = surface_hit(0.01, 0.2, |r| grid.nearest_ray(&pts, o, d, r, NO_FILTER)).unwrap();
        assert!(perp(hit.pos, o, d) < 0.045, "{} m off the line of sight", perp(hit.pos, o, d));
        assert!((hit.pos[1] - 10.0).abs() < 1e-4, "and still on the wall");
    }

    #[test]
    fn over_a_gap_the_tube_widens_to_the_radius_and_no_further() {
        let pts = vec![[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]];
        let grid = SpatialGrid::build(&pts);
        let (o, d) = ([0.1, 0.0, 10.0], [0.0, 0.0, -1.0]);

        let hit = surface_hit(0.01, 0.2, |r| grid.nearest_ray(&pts, o, d, r, NO_FILTER));
        assert_eq!(hit.map(|h| h.pos), Some([0.15, 0.0, 0.0]), "the nearer of the two, found by widening");

        let mut tried = Vec::new();
        let none = surface_hit(0.01, 0.03, |r| { tried.push(r); grid.nearest_ray(&pts, o, d, r, NO_FILTER) });
        assert!(none.is_none());
        assert_eq!(tried, vec![0.01, 0.02, 0.03], "doubles, then stops at the radius");
    }
}

#[cfg(test)]
mod tests {
    // The criterion bench includes this file by path (`benches/spatial.rs`),
    // which compiles this module without a test harness and makes rustc read
    // the glob as unused there. It is very much used by `cargo test`.
    #[allow(unused_imports)]
    use super::*;

    /// Deterministic pseudo-random cloud — no rand dependency, and a failing
    /// test reproduces exactly.
    fn synthetic(n: usize, extent: f32) -> Vec<[f32; 3]> {
        let mut s: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        (0..n)
            .map(|_| [next() * extent, next() * extent, next() * 20.0])
            .collect()
    }

    fn brute_in_radius(pts: &[[f32; 3]], x: f32, y: f32, r: f32) -> Vec<u32> {
        let mut v: Vec<u32> = Vec::new();
        for (i, p) in pts.iter().enumerate() {
            let (dx, dy) = (p[0] - x, p[1] - y);
            if dx * dx + dy * dy <= r * r { v.push(i as u32); }
        }
        v.sort();
        v
    }

    #[test]
    fn empty_grid_answers_nothing() {
        let g = SpatialGrid::build(&[]);
        assert!(g.is_empty());
        let mut out = Vec::new();
        g.points_in_radius(&[], 0.0, 0.0, 5.0, NO_FILTER, &mut out);
        assert!(out.is_empty());
        assert_eq!(g.nearest_ray(&[], [0.0; 3], [0.0, 0.0, -1.0], 1.0, NO_FILTER), None);
        assert!(!g.circle_overlaps(0.0, 0.0, 1.0));
    }

    #[test]
    fn every_point_is_bucketed_once() {
        let pts = synthetic(20_000, 60.0);
        let g = SpatialGrid::build(&pts);
        assert_eq!(g.len(), pts.len());
        let mut seen = vec![false; pts.len()];
        let (min, max) = bounds_xy(&pts);
        g.cells_in_box(min, max, |c| {
            for &i in c.points {
                assert!(!seen[i as usize], "point {} bucketed twice", i);
                seen[i as usize] = true;
            }
        });
        assert!(seen.iter().all(|&b| b), "some point was never bucketed");
    }

    #[test]
    fn radius_query_matches_brute_force() {
        let pts = synthetic(20_000, 60.0);
        let g = SpatialGrid::build(&pts);
        let mut out = Vec::new();
        for (x, y, r) in [
            (30.0, 30.0, 0.2_f32),
            (30.0, 30.0, 2.0),
            (0.0, 0.0, 1.0),      // grid corner
            (60.0, 60.0, 1.0),    // other corner
            (-50.0, 30.0, 1.0),   // wholly outside
            (30.0, 30.0, 200.0),  // encloses everything
        ] {
            out.clear();
            g.points_in_radius(&pts, x, y, r, NO_FILTER, &mut out);
            out.sort();
            assert_eq!(out, brute_in_radius(&pts, x, y, r), "radius {r} at ({x},{y})");
        }
    }

    #[test]
    fn filter_is_applied_inside_the_walk() {
        let pts = synthetic(5_000, 30.0);
        let g = SpatialGrid::build(&pts);
        let mut out = Vec::new();
        // Keep only points above 10 m — the same shape as a ZClip.
        g.points_in_radius(&pts, 15.0, 15.0, 5.0, |i| pts[i as usize][2] > 10.0, &mut out);
        assert!(!out.is_empty());
        assert!(out.iter().all(|&i| pts[i as usize][2] > 10.0));

        // A filter that hides everything must yield nothing, not a fallback.
        out.clear();
        g.points_in_radius(&pts, 15.0, 15.0, 5.0, |_| false, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn percentile_returns_a_real_point() {
        // A tidy column: z = 0, 1, .. 100 at the same XY.
        let pts: Vec<[f32; 3]> = (0..=100).map(|i| [5.0, 5.0, i as f32]).collect();
        let g = SpatialGrid::build(&pts);
        for (pct, want_z) in [(0.0_f32, 0.0_f32), (5.0, 5.0), (50.0, 50.0), (95.0, 95.0), (100.0, 100.0)] {
            let hit = g
                .percentile_in_radius(&pts, 5.0, 5.0, 1.0, pct, NO_FILTER)
                .expect("column is in range");
            assert_eq!(hit.pos[2], want_z, "p{pct}");
            // The answer must be a point that exists, not an interpolation.
            assert_eq!(pts[hit.idx as usize], hit.pos);
        }
        // Nothing in range is None, never a silent fallback to the nearest cell.
        assert!(g.percentile_in_radius(&pts, 500.0, 500.0, 1.0, 50.0, NO_FILTER).is_none());
    }

    #[test]
    fn percentile_ignores_a_single_low_outlier_at_p5() {
        // Ground at 10 m, one noise return at 0 m: the whole reason percentiles
        // exist rather than raw minima.
        let mut pts: Vec<[f32; 3]> = (0..100).map(|i| [5.0 + i as f32 * 1e-3, 5.0, 10.0]).collect();
        pts.push([5.0, 5.0, 0.0]);
        let g = SpatialGrid::build(&pts);
        let p5 = g.percentile_in_radius(&pts, 5.05, 5.0, 1.0, 5.0, NO_FILTER).unwrap();
        assert_eq!(p5.pos[2], 10.0, "p5 must not be the outlier");
        let p0 = g.percentile_in_radius(&pts, 5.05, 5.0, 1.0, 0.0, NO_FILTER).unwrap();
        assert_eq!(p0.pos[2], 0.0, "p0 gives the raw minimum back");
    }

    /// Brute-force reference for `nearest_ray`: frontmost point within
    /// `max_dist` of the ray.
    fn brute_ray(pts: &[[f32; 3]], o: [f32; 3], d: [f32; 3], max_dist: f32) -> Option<u32> {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let d = [d[0] / len, d[1] / len, d[2] / len];
        let mut best = None;
        let mut best_t = f32::INFINITY;
        for (i, p) in pts.iter().enumerate() {
            let v = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
            let t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            if t < 0.0 || t >= best_t { continue; }
            let perp = [v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2]];
            if perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2] > max_dist * max_dist { continue; }
            best_t = t;
            best = Some(i as u32);
        }
        best
    }

    #[test]
    fn ray_query_matches_brute_force() {
        let pts = synthetic(20_000, 60.0);
        let g = SpatialGrid::build(&pts);
        let cases: [([f32; 3], [f32; 3]); 11] = [
            ([30.0, 30.0, 100.0], [0.0, 0.0, -1.0]),        // straight down
            ([30.0, 30.0, 100.0], [0.0, 0.0, 1.0]),         // straight up, away
            ([-40.0, 30.0, 10.0], [1.0, 0.0, 0.0]),         // axis-aligned, x-major
            ([30.0, -40.0, 10.0], [0.0, 1.0, 0.0]),         // axis-aligned, y-major
            ([-40.0, -40.0, 40.0], [1.0, 1.0, -0.5]),       // oblique, diagonal
            ([-40.0, 80.0, 40.0], [1.0, -0.7, -0.4]),       // oblique, other way
            ([200.0, 200.0, 10.0], [1.0, 1.0, 0.0]),        // pointing away entirely
            // Near-degenerate directions: just off vertical and just off an
            // axis. These are what decide whether the disc branch or the sweep
            // runs, and the sweep divides by the major-axis component — so the
            // switchover has to be safe from both sides.
            ([30.0, 30.0, 100.0], [1.0e-4, 0.0, -1.0]),
            ([30.0, 30.0, 100.0], [0.0, -1.0e-4, -1.0]),
            ([-40.0, 30.0, 10.0], [1.0, 1.0e-5, 0.0]),
            ([-40.0, 30.0, 10.0], [1.0, -1.0e-5, -1.0e-5]),
        ];
        for (o, d) in cases {
            for max_dist in [0.2_f32, 1.0, 3.0] {
                let got = g.nearest_ray(&pts, o, d, max_dist, NO_FILTER).map(|h| h.idx);
                assert_eq!(
                    got,
                    brute_ray(&pts, o, d, max_dist),
                    "ray {o:?} → {d:?} at {max_dist} m",
                );
            }
        }
    }

    #[test]
    fn ray_query_fuzz_against_brute_force() {
        // The column sweep in `nearest_ray` is the only non-obvious geometry in
        // the module: it has to cover the corridor exactly once and miss
        // nothing at any slope. Random rays from every direction are the only
        // way to have any confidence in that.
        let pts = synthetic(8_000, 50.0);
        let g = SpatialGrid::build(&pts);
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        for case in 0..2_000 {
            // Origins both inside and well outside the cloud, and directions
            // over the whole sphere including the near-axis-aligned ones.
            let o = [
                next() * 140.0 - 45.0,
                next() * 140.0 - 45.0,
                next() * 80.0 - 30.0,
            ];
            let d = [
                next() * 2.0 - 1.0,
                next() * 2.0 - 1.0,
                next() * 2.0 - 1.0,
            ];
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 1e-6 { continue; }
            let max_dist = 0.1 + next() * 2.9;
            let got = g.nearest_ray(&pts, o, d, max_dist, NO_FILTER).map(|h| h.idx);
            assert_eq!(
                got,
                brute_ray(&pts, o, d, max_dist),
                "case {case}: ray {o:?} -> {d:?} at {max_dist} m",
            );
        }
    }

    #[test]
    fn ray_query_picks_the_frontmost_not_the_closest_to_the_axis() {
        // Two surfaces on the same ray: a near one slightly off-axis and a far
        // one dead centre. Picking must land on the near one.
        let pts = vec![
            [0.0, 0.0, 5.0],   // 5 m away, 0.1 m off the axis
            [0.0, 0.0, 20.0],  // 20 m away, dead centre
        ];
        let mut pts = pts;
        pts[0][0] = 0.1;
        let g = SpatialGrid::build_with_cell(&pts, 0.5);
        let hit = g
            .nearest_ray(&pts, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.3, NO_FILTER)
            .unwrap();
        assert_eq!(hit.idx, 0);
    }

    #[test]
    fn ray_filter_skips_hidden_points() {
        let pts = vec![[0.0, 0.0, 5.0], [0.0, 0.0, 20.0]];
        let g = SpatialGrid::build_with_cell(&pts, 0.5);
        // Hide the near point: the query must fall through to the far one
        // rather than returning the hidden one or nothing.
        let hit = g
            .nearest_ray(&pts, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.3, |i| i != 0)
            .unwrap();
        assert_eq!(hit.idx, 1);
    }

    #[test]
    fn cell_z_extent_and_args_agree() {
        let pts = synthetic(10_000, 40.0);
        let g = SpatialGrid::build(&pts);
        let (min, max) = bounds_xy(&pts);
        let mut cells = 0;
        g.cells_in_box(min, max, |c| {
            cells += 1;
            let lo = c.points.iter().map(|&i| pts[i as usize][2]).fold(f32::INFINITY, f32::min);
            let hi = c.points.iter().map(|&i| pts[i as usize][2]).fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(c.z_min, lo);
            assert_eq!(c.z_max, hi);
            assert_eq!(pts[c.arg_min as usize][2], lo);
            assert_eq!(pts[c.arg_max as usize][2], hi);
        });
        assert!(cells > 0);
    }

    #[test]
    fn cells_in_box_covers_exactly_the_box() {
        let pts = synthetic(5_000, 40.0);
        let g = SpatialGrid::build(&pts);
        // Every point returned must be within one cell of the box — cells are
        // whole, so the walk is conservative but must not be wild.
        let (bmin, bmax) = ([10.0_f32, 10.0], [20.0_f32, 20.0]);
        let slack = g.cell_size();
        let mut n = 0;
        g.cells_in_box(bmin, bmax, |c| {
            for &i in c.points {
                let p = pts[i as usize];
                assert!(p[0] >= bmin[0] - slack && p[0] <= bmax[0] + slack);
                assert!(p[1] >= bmin[1] - slack && p[1] <= bmax[1] + slack);
                n += 1;
            }
        });
        // and every point actually inside the box must have been visited.
        let inside = pts.iter().filter(|p| {
            p[0] >= bmin[0] && p[0] <= bmax[0] && p[1] >= bmin[1] && p[1] <= bmax[1]
        }).count();
        assert!(n >= inside, "{n} visited < {inside} inside");
    }

    #[test]
    fn degenerate_clouds_build() {
        // One point.
        let one = vec![[3.0, 4.0, 5.0]];
        let g = SpatialGrid::build(&one);
        let mut out = Vec::new();
        g.points_in_radius(&one, 3.0, 4.0, 0.1, NO_FILTER, &mut out);
        assert_eq!(out, vec![0]);

        // A perfectly axis-aligned line — zero area, so `auto_cell` cannot
        // divide by it.
        let line: Vec<[f32; 3]> = (0..1000).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();
        let g = SpatialGrid::build(&line);
        out.clear();
        g.points_in_radius(&line, 50.0, 0.0, 0.25, NO_FILTER, &mut out);
        assert_eq!(out.len(), brute_in_radius(&line, 50.0, 0.0, 0.25).len());

        // All points coincident.
        let same = vec![[1.0, 1.0, 1.0]; 500];
        let g = SpatialGrid::build(&same);
        out.clear();
        g.points_in_radius(&same, 1.0, 1.0, 0.01, NO_FILTER, &mut out);
        assert_eq!(out.len(), 500);
    }

    #[test]
    fn the_cell_count_stays_within_budget_however_sparse_the_cloud() {
        // The shape that used to cost 720 MB: a thin sample over a huge
        // extent, which is exactly what a COPC root node is.
        let mut pts: Vec<[f32; 3]> = Vec::new();
        let mut s: u64 = 12345;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f32 / (1u64 << 53) as f32 };
        for _ in 0..5_000 {
            pts.push([next() * 30_000.0, next() * 30_000.0, next() * 50.0]);
        }
        let g = SpatialGrid::build(&pts);
        let cells = g.nx() * g.ny();
        assert!(
            cells as f64 <= MAX_CELLS_PER_POINT * pts.len() as f64,
            "{cells} cells for {} points (cell {} m)", pts.len(), g.cell_size(),
        );
        assert!(g.cell_size() > MAX_CELL, "the budget must be allowed to override MAX_CELL");

        // And it still answers correctly at that cell size.
        let mut out = Vec::new();
        g.points_in_radius(&pts, pts[7][0], pts[7][1], 1.0, NO_FILTER, &mut out);
        assert!(out.contains(&7), "a big cell must not lose the point it holds");
        assert_eq!(out, brute_in_radius(&pts, pts[7][0], pts[7][1], 1.0));

        // An explicit cell size cannot bypass the budget either.
        let g = SpatialGrid::build_with_cell(&pts, 0.01);
        assert!((g.nx() * g.ny()) as f64 <= MAX_CELLS_PER_POINT * pts.len() as f64);
    }

    #[test]
    fn circle_overlap_rejects_distant_grids() {
        let pts = synthetic(1_000, 10.0);
        let g = SpatialGrid::build(&pts);
        assert!(g.circle_overlaps(5.0, 5.0, 0.2));
        assert!(g.circle_overlaps(-0.5, 5.0, 1.0), "just outside, within radius");
        assert!(!g.circle_overlaps(-50.0, 5.0, 1.0));
        assert!(!g.circle_overlaps(5.0, 500.0, 10.0));
    }
}
