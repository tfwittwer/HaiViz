//! Bench for the snap-query spatial index.
//!
//! Phase 0 of the vectorisation plan is the risk concentration: nothing above
//! it works without a fast radius query, and it is the only part whose
//! performance cannot be predicted by reading the code. So the numbers that
//! matter are measured here rather than argued about.
//!
//! The linear-scan baselines are deliberately kept alongside the indexed
//! queries. They are what the three `find_nearest_point` implementations do
//! today, and what a snap running on every mouse move would have cost — the
//! ratio is the whole justification for the module.
//!
//! `haiviz` is a binary crate with no library target, so the module is included
//! by path rather than imported. `spatial.rs` deliberately depends on nothing
//! else in the crate, which is what makes that work.

#[path = "../src/spatial.rs"]
mod spatial;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use spatial::{NO_FILTER, SpatialGrid};

const N: usize = 10_000_000;
/// 200 m square — about 250 pts/m² at 10 M, in the range a mobile mapping run
/// produces and dense enough that a 0.2 m snap radius holds a few hundred
/// points, which is the case the percentile sort has to survive.
const EXTENT: f32 = 200.0;
/// The default snap radius from the plan (decision 11).
const SNAP_R: f32 = 0.2;

/// Deterministic synthetic cloud: a gently rolling surface with vegetation-like
/// returns above it, so the per-cell Z extent and the ray prune have something
/// to bite on. A uniform slab would make the Z prune look better than it is.
fn synthetic(n: usize) -> Vec<[f32; 3]> {
    let mut s: u64 = 0x2545F4914F6CDD1D;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..n)
        .map(|_| {
            let x = next() * EXTENT;
            let y = next() * EXTENT;
            let ground = 3.0 * (x * 0.05).sin() + 2.0 * (y * 0.04).cos();
            // One return in five sits above the surface — canopy, façades,
            // wires. The rest is the surface itself, with a few cm of noise.
            let z = if next() < 0.2 {
                ground + next() * 20.0
            } else {
                ground + (next() - 0.5) * 0.06
            };
            [x, y, z]
        })
        .collect()
}

fn linear_nearest_ray(pts: &[[f32; 3]], o: [f32; 3], d: [f32; 3], max_dist: f32) -> Option<u32> {
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

fn linear_in_radius(pts: &[[f32; 3]], x: f32, y: f32, r: f32, out: &mut Vec<u32>) {
    let r2 = r * r;
    for (i, p) in pts.iter().enumerate() {
        let (dx, dy) = (p[0] - x, p[1] - y);
        if dx * dx + dy * dy <= r2 { out.push(i as u32); }
    }
}

fn bench_spatial(c: &mut Criterion) {
    eprintln!("building {} synthetic points…", N);
    let pts = synthetic(N);

    // --- Build ---
    // Once per stage, once per tile load, once per COPC node ingest. The COPC
    // case is the one that matters: it runs on the loader's critical path
    // while the camera is moving, so it has to stay well under a frame for a
    // node-sized set.
    let mut build = c.benchmark_group("build");
    build.sample_size(10);
    build.bench_function("10M points", |b| {
        b.iter(|| SpatialGrid::build(black_box(&pts)))
    });
    // A COPC node holds far fewer points than a whole file; this is the size
    // that has to be cheap.
    let node: Vec<[f32; 3]> = pts[..100_000].to_vec();
    build.bench_function("100k points (COPC node)", |b| {
        b.iter(|| SpatialGrid::build(black_box(&node)))
    });
    build.finish();

    let grid = SpatialGrid::build(&pts);
    eprintln!(
        "index: cell {:.3}m, {}x{} cells, {} points",
        grid.cell_size(), grid.nx(), grid.ny(), grid.len(),
    );

    let cx = EXTENT * 0.5;
    let cy = EXTENT * 0.37;

    // --- Radius queries: what runs on every mouse move while snapping ---
    let mut radius = c.benchmark_group("radius");
    for r in [SNAP_R, 0.5, 2.0] {
        radius.bench_function(format!("indexed r={r}"), |b| {
            let mut out = Vec::with_capacity(4096);
            b.iter(|| {
                out.clear();
                grid.points_in_radius(&pts, black_box(cx), black_box(cy), r, NO_FILTER, &mut out);
                out.len()
            })
        });
    }
    // A ZClip-shaped filter, to confirm that running it inside the walk does
    // not cost more than the walk it prunes.
    radius.bench_function("indexed r=0.2 filtered", |b| {
        let mut out = Vec::with_capacity(4096);
        b.iter(|| {
            out.clear();
            grid.points_in_radius(
                &pts, black_box(cx), black_box(cy), SNAP_R,
                |i| pts[i as usize][2] > 0.0,
                &mut out,
            );
            out.len()
        })
    });
    radius.sample_size(10);
    radius.bench_function("linear scan r=0.2", |b| {
        let mut out = Vec::with_capacity(4096);
        b.iter(|| {
            out.clear();
            linear_in_radius(&pts, black_box(cx), black_box(cy), SNAP_R, &mut out);
            out.len()
        })
    });
    radius.finish();

    // --- Percentile: the actual snap mode, sort included ---
    let mut pct = c.benchmark_group("percentile");
    for r in [SNAP_R, 0.5] {
        pct.bench_function(format!("p5 r={r}"), |b| {
            b.iter(|| {
                grid.percentile_in_radius(&pts, black_box(cx), black_box(cy), r, 5.0, NO_FILTER)
            })
        });
    }
    pct.finish();

    // --- Ray: the nearest-point snap, from the two camera attitudes that
    //     behave completely differently in the grid walk ---
    let top = ([cx, cy, 200.0f32], [0.0, 0.0, -1.0f32]);
    let oblique = ([cx - 120.0, cy - 120.0, 90.0f32], [1.0, 1.0, -0.7f32]);
    let mut ray = c.benchmark_group("ray");
    for (name, (o, d)) in [("top-down", top), ("oblique 40deg", oblique)] {
        ray.bench_function(format!("indexed {name}"), |b| {
            b.iter(|| grid.nearest_ray(&pts, black_box(o), black_box(d), SNAP_R, NO_FILTER))
        });
    }
    ray.sample_size(10);
    for (name, (o, d)) in [("top-down", top), ("oblique 40deg", oblique)] {
        ray.bench_function(format!("linear scan {name}"), |b| {
            b.iter(|| linear_nearest_ray(&pts, black_box(o), black_box(d), SNAP_R))
        });
    }
    ray.finish();

    // --- Raster window: one Dijkstra anchor for edge following (phase 5) ---
    // 40 m window is the plan's default. This runs once per committed vertex,
    // not per frame, so the budget is a few milliseconds rather than a few
    // hundred microseconds.
    let mut raster = c.benchmark_group("raster");
    raster.sample_size(20);
    raster.bench_function("40m window", |b| {
        b.iter(|| {
            let mut cells = 0usize;
            let mut points = 0usize;
            grid.cells_in_box(
                black_box([cx - 20.0, cy - 20.0]),
                black_box([cx + 20.0, cy + 20.0]),
                |c| { cells += 1; points += c.points.len(); },
            );
            (cells, points)
        })
    });
    raster.finish();
}

criterion_group!(benches, bench_spatial);
criterion_main!(benches);

