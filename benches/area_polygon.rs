/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Benchmark for calculating the area of polygons.
//!
//! These benchmarks will test polygons of different sizes to compare the performance of multiple
//! implementations. Using the results, we can find thresholds for which implementation is the most
//! performant in each case.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion}; //To run the benchmark.

use apex::operations::area::{area_polygon_st, area_polygon_mt, area_polygon_gpu}; //The functions being benchmarked.
use apex::test::polygon;

fn bench_area_polygon_st(runner: &mut Criterion) {
	let mut group = runner.benchmark_group("area_polygon_st");
	for size in [10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000] {
		let mut poly = polygon::regular(size);
		group.bench_with_input(
			BenchmarkId::from_parameter(size), &size,
			|bencher, &_size| bencher.iter(
				|| area_polygon_st(&mut poly)
			)
		);
	}
}

fn bench_area_polygon_mt(runner: &mut Criterion) {
	let mut group = runner.benchmark_group("area_polygon_mt");
	for size in [10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000] {
		let mut poly = polygon::regular(size);
		group.bench_with_input(
			BenchmarkId::from_parameter(size), &size,
			|bencher, &_size| bencher.iter(
				|| area_polygon_mt(&mut poly)
			)
		);
	}
}

fn bench_area_polygon_gpu(runner: &mut Criterion) {
	let mut group = runner.benchmark_group("area_polygon_gpu");
	for size in [10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000] {
		let mut poly = polygon::regular(size);
		group.bench_with_input(
			BenchmarkId::from_parameter(size), &size,
			|bencher, &_size| bencher.iter(
				|| area_polygon_gpu(&mut poly)
			)
		);
	}
}

criterion_group!(benches, bench_area_polygon_st, bench_area_polygon_mt, bench_area_polygon_gpu);
criterion_main!(benches);