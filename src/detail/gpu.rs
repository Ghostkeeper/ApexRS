/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines resources that help to track the state of the GPU during computations.

use pollster;
use std::sync::LazyLock;
use wgpu::{Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue, RequestAdapterOptions};

pub(crate) static WGPU: LazyLock<Instance> = LazyLock::new(|| {
	Instance::new(&InstanceDescriptor::default())
});
pub(crate) static GPU: LazyLock<(Device, Queue)> = LazyLock::new(|| {
	let adapter = pollster::block_on(WGPU.request_adapter(&RequestAdapterOptions::default())).expect("Failed to find a graphics card adapter.");
	pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).expect("Failed to create GPU device.")
});