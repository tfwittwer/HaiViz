//! Bench for edge following.
//!
//! The plan's claim to check: a 512² raster is "a few ms per anchor, nothing
//! per frame". The per-anchor cost is the raster, the field and Dijkstra; the
//! per-frame cost is walking the parent pointers to the cursor and simplifying
//! the result.
//!
//! Included by path for the same reason as the spatial bench: `haiviz` has no
//! library target, and `edge_follow.rs` depends on nothing else in the crate.

#[path = "../src/edge_follow.rs"]
#[allow(dead_code)]
mod edge_follow;

use criterion::{Criterion, criterion_group, criterion_main};
use edge_follow::{Params, Raster, Sample, Trace, ZSource, edge_field, shortest_paths};
use std::hint::black_box;

/// A 40 m window at 1000 pts/m² — dense enough to hit the 512-cell cap — with
/// a rolling surface, a meandering kerb, painted stripes and a building
/// footprint, so every channel has something to do.
fn window(density: f32) -> Vec<Sample> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let n = (40.0 * 40.0 * density) as usize;
    (0..n).map(|_| {
        let (x, y) = (next() * 40.0, next() * 40.0);
        let kerb = 20.0 + 4.0 * (x / 6.0).sin();
        let mut z = 0.02 * (x * 0.3).sin() + if y > kerb { 0.15 } else { 0.0 };
        let mut class = 2;
        if (28.0..36.0).contains(&x) && (4.0..12.0).contains(&y) {
            z += 6.0 * next();
            class = 6;
        }
        let intensity = if ((x % 8.0) - 4.0).abs() < 0.12 { 30_000 } else { 6_000 };
        Sample { pos: [x, y, z + 0.01 * next()], intensity, class }
    }).collect()
}

fn bench(c: &mut Criterion) {
    let samples = window(1000.0);
    let mut g = c.benchmark_group("edge_follow");
    g.sample_size(10);

    g.bench_function("raster, 40 m window, 1.6 M points (512²)", |b| {
        b.iter(|| Raster::build(black_box([20.0, 20.0]), 40.0, black_box(&samples)))
    });

    let raster = Raster::build([20.0, 20.0], 40.0, &samples);
    assert_eq!(raster.nx, 512, "the bench is meant to exercise the cell cap");
    let params = Params::default();
    g.bench_function("edge field, 512²", |b| {
        b.iter(|| edge_field(black_box(&raster), black_box(&params)))
    });

    let edge = edge_field(&raster, &params);
    let start = raster.cell_of(20.0, 20.0).unwrap();
    g.bench_function("dijkstra, 512²", |b| {
        b.iter(|| shortest_paths(black_box(&edge), raster.nx, raster.ny, start))
    });

    // The window is centred on the anchor, so a target 18 m out is near its edge.
    let trace = Trace::build([20.0, 20.0, 0.0], [0.0; 3], &samples, samples.len(), params, 0.0);
    assert!(trace.route([38.0, 22.0, 0.0], ZSource::Lowest).is_some(), "the target must be inside the window");
    g.bench_function("route centre to edge (per frame)", |b| {
        b.iter(|| trace.route(black_box([38.0, 22.0, 0.0]), ZSource::Lowest))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
