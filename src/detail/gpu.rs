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
use wgpu::{Device, DeviceDescriptor, DownlevelFlags, Instance, InstanceDescriptor, Queue, RequestAdapterOptions};


/// All the resources that belong to a particular graphics card.
///
/// This contains the objects that we need to control a graphics card: One device to store memory
/// with, upload and download data, and one queue to send instructions to.
pub struct GPUResources {
	/// WGPU's "Device" construct, which is used to interact with the GPU's memory.
	pub device: Device,

	/// WGPU's "Queue" construct, which is used to send instructions to the device.
	pub queue: Queue,
}

/// The WGPU instance is the entrypoint for all GPU operations.
pub(crate) static WGPU: LazyLock<Instance> = LazyLock::new(|| {
	Instance::new(&InstanceDescriptor::default())
});

/// The resources to access the default GPU.
pub(crate) static GPU: LazyLock<GPUResources> = LazyLock::new(|| {
	let adapter = pollster::block_on(WGPU.request_adapter(&RequestAdapterOptions::default())).expect("Failed to find a graphics card adapter.");
	let downlevel = adapter.get_downlevel_capabilities();
	if !downlevel.flags.contains(DownlevelFlags::COMPUTE_SHADERS) {
		panic!("Adapter does not support compute shaders.");
	}
	let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).expect("Failed to create GPU device.");
	GPUResources { device, queue }
});