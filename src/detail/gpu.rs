/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines resources that help to track the state of the GPU during computations.

use pollster;
use std::num::NonZeroU64; //For communicating buffer sizes to the GPU.
use std::sync::LazyLock;
use wgpu::{
	BindGroupDescriptor,
	BindGroupEntry,
	BindGroupLayoutDescriptor,
	BindGroupLayoutEntry,
	BindingType,
	Buffer,
	BufferBindingType,
	BufferDescriptor,
	BufferUsages,
	CommandEncoderDescriptor,
	ComputePassDescriptor,
	ComputePipelineDescriptor,
	Device,
	DeviceDescriptor,
	DownlevelFlags,
	Instance,
	InstanceDescriptor,
	MapMode,
	PipelineCompilationOptions,
	PipelineLayoutDescriptor,
	PollType,
	Queue,
	RequestAdapterOptions,
	ShaderModule,
	ShaderStages,
};


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

/// Execute a compute kernel on the default GPU.
///
/// This method is intended to be a general-purpose GPU calculation. As such it requires quite a lot
/// of inputs from the source and still puts a few requirements on the kernels that can be executed:
/// * The GPU kernel entrypoint must always be called "main". Other entrypoints will be ignored.
/// * We will always bind the given buffers at indexes 0, 1, 2, and so on. We can't skip an index.
/// * We only use uniform and storage buffers.
///
/// # Arguments
/// * `shader_module` - The kernel to execute on the GPU.
/// * `buffers` - the buffers to bind to the kernel. The buffers will be bound in the same order as
///   given in this list. The buffers' usages will determine how the buffer is bound to the kernel.
/// * `output_buffer` - If given, this function will return the contents of this buffer after the
///   kernel has finished executing. The output buffer, if given, must also be included in the
///   `buffers` argument.
/// * `num_threads` - How many separate threads to start. The threads will be started in work groups
///   of 256. Not all threads may be started at the same time, depending on hardware. For instance,
///   if you have 2 million items that can be processed all separately, the thread count can be 2
///   million. Each thread will have its own global invocaton ID.
///
/// # Return
/// The contentsof the `output_buffer` after executing the kernel, if given.
pub(crate) fn execute_kernel(shader_module: &ShaderModule, buffers: &[&Buffer], output_buffer: Option<&Buffer>, num_threads: u64) -> Option<Vec<u8>> {
	let mut layout_entries = vec!();
	let mut bind_group_entries = vec!();
	for (binding, buffer) in buffers.iter().enumerate() {
		//Let's get the correct bind type automatically.
		let usage = buffer.usage();
		let bind_type = if usage.contains(BufferUsages::UNIFORM) {
			BufferBindingType::Uniform { }
		} else if usage.contains(BufferUsages::STORAGE) {
			if usage.contains(BufferUsages::COPY_SRC) {
				BufferBindingType::Storage { read_only: false }
			} else {
				BufferBindingType::Storage { read_only: false }
			}
		} else {
			panic!("Unknown buffer usage type, can't assign any binding type.")
		};
		layout_entries.push(BindGroupLayoutEntry {
			binding: binding as u32,
			visibility: ShaderStages::COMPUTE,
			ty: BindingType::Buffer {
				ty: bind_type,
				min_binding_size: Some(NonZeroU64::new(buffer.size()).unwrap()),
				has_dynamic_offset: false,
			},
			count: None,
		});
		bind_group_entries.push(BindGroupEntry {
			binding: binding as u32,
			resource: buffer.as_entire_binding(),
		});
	}
	let bind_group_layout = GPU.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("Bind Group Layout"),
		entries: &layout_entries,
	});
	let bind_group = GPU.device.create_bind_group(&BindGroupDescriptor {
		label: Some("Bind Group"),
		layout: &bind_group_layout,
		entries: &bind_group_entries,
	});

	//Also communicated to the GPU is the pipeline: Compiled code telling it what to do.
	let pipeline_layout = GPU.device.create_pipeline_layout(&PipelineLayoutDescriptor {
		label: Some("Pipeline Layout"),
		bind_group_layouts: &[&bind_group_layout],
		immediate_size: 0,
	});
	let pipeline = GPU.device.create_compute_pipeline(&ComputePipelineDescriptor {
		label: Some("Pipeline"),
		layout: Some(&pipeline_layout),
		module: &shader_module,
		entry_point: Some("main"),
		compilation_options: PipelineCompilationOptions::default(),
		cache: None,
	});
	let mut encoder = GPU.device.create_command_encoder(&CommandEncoderDescriptor {
		label: Some("Encoder"),
	});
	let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
		label: Some("Compute Pass"),
		timestamp_writes: None,
	});
	compute_pass.set_pipeline(&pipeline);
	compute_pass.set_bind_group(0, &bind_group, &[]);
	compute_pass.dispatch_workgroups(((num_threads + 255) / 256) as u32, 1, 1);
	drop(compute_pass); //Now that we've dispatched the workgroups, we can drop the compute pass so that we can access the encoder again.
	if output_buffer.is_some() {
		//If we want output, then also create a readback buffer.
		let readback_buffer = GPU.device.create_buffer(&BufferDescriptor {
			label: Some("Readback buffer"),
			size: output_buffer.unwrap().size(),
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		encoder.copy_buffer_to_buffer(&output_buffer.unwrap(), 0, &readback_buffer, 0, output_buffer.unwrap().size());
		let command_buffer = encoder.finish(); //Finish the compilation.
		GPU.queue.submit([command_buffer]); //Execute the commands.
		//Read the output.
		readback_buffer.slice(..).map_async(MapMode::Read, |_| {});
		let _ = GPU.device.poll(PollType::wait_indefinitely());
		let slice: &[u8] = &readback_buffer.slice(..).get_mapped_range();
		Some(bytemuck::cast_slice(slice).to_vec())
	} else {
		let command_buffer = encoder.finish(); //Finish the compilation.
		GPU.queue.submit([command_buffer]); //Execute the commands.
		let _ = GPU.device.poll(PollType::wait_indefinitely());
		None
	}
}