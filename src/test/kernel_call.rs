/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Helper functions for calling upon kernel functions during tests.

use std::num::NonZeroU64; //For communicating buffer sizes to the GPU.
use wgpu::{
	BindGroupDescriptor,
	BindGroupEntry,
	BindGroupLayoutDescriptor,
	BindGroupLayoutEntry,
	BindingType,
	BufferBindingType,
	BufferDescriptor,
	BufferUsages,
	CommandEncoderDescriptor,
	ComputePassDescriptor,
	ComputePipelineDescriptor,
	MapMode,
	PipelineCompilationOptions,
	PipelineLayoutDescriptor,
	PollType,
	ShaderModule,
	ShaderStages,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::detail::gpu::GPU; //To perform calculations on the GPU.

pub fn kernel_call(module: &ShaderModule, main_function: &str, input: &[u8], input_binding_index: u32, output_binding_index: u32, output_size: u64) -> Vec<u8> {
	//The input buffer.
	let input_buffer = GPU.device.create_buffer_init(&BufferInitDescriptor {
		label: Some("input"),
		contents: input,
		usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
	});
	//The output buffer.
	let output_buffer = vec![0; output_size as usize];
	let output_resource = GPU.device.create_buffer_init(&BufferInitDescriptor {
		label: Some("output"),
		contents: &output_buffer,
		usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
	});
	let layout_entries: Vec<BindGroupLayoutEntry> = vec![
		//The input buffer.
		BindGroupLayoutEntry {
			binding: input_binding_index,
			visibility: ShaderStages::COMPUTE,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Uniform { },
				min_binding_size: Some(NonZeroU64::new(input.len() as u64).unwrap()),
				has_dynamic_offset: false,
			},
			count: None,
		},
		//The output buffer.
		BindGroupLayoutEntry {
			binding: output_binding_index,
			visibility: ShaderStages::COMPUTE,
			ty: BindingType::Buffer {
				ty: BufferBindingType::Storage { read_only: false },
				min_binding_size: Some(NonZeroU64::new(output_size).unwrap()),
				has_dynamic_offset: false,
			},
			count: None,
		},
	];
	let bind_group_layout = GPU.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
		label: Some("Bind Group Layout"),
		entries: layout_entries.as_array::<2>().unwrap(),
	});
	//Then bind the actual buffers according to the layout above.
	let binding_entries = vec![
		BindGroupEntry {
			binding: input_binding_index,
			resource: input_buffer.as_entire_binding(),
		},
		BindGroupEntry {
			binding: output_binding_index,
			resource: output_resource.as_entire_binding(),
		},
	];
	let bind_group = GPU.device.create_bind_group(&BindGroupDescriptor {
		label: Some("Bind Group"),
		layout: &bind_group_layout,
		entries: binding_entries.as_array::<2>().unwrap(),
	});

	let readback_resource = GPU.device.create_buffer(&BufferDescriptor {
		label: Some("Readback buffer"),
		size: output_size,
		usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
		mapped_at_creation: false,
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
		module: &module,
		entry_point: Some(main_function),
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
	compute_pass.dispatch_workgroups(1, 1, 1);
	drop(compute_pass); //Now that we've dispatched the workgroups, we can drop the compute pass so that we can access the encoder again.
	encoder.copy_buffer_to_buffer(&output_resource, 0, &readback_resource, 0, output_size);
	let command_buffer = encoder.finish(); //Finish the compilation.

	GPU.queue.submit([command_buffer]); //Execute the commands.

	//Read the output.
	readback_resource.slice(..).map_async(MapMode::Read, |_| {});
	let _ = GPU.device.poll(PollType::wait_indefinitely());
	readback_resource.slice(..).get_mapped_range().to_owned()
}