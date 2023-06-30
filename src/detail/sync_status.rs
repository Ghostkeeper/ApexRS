/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Provides an enum to denote the status of which copy of a resource is currently leading.

/// Denotes the status of synchronisation of a resource that can be either on the CPU or the GPU.
///
/// If a memory resource (like a geometric object) is created, it is first created in normal RAM,
/// accessible by the CPU. This is denoted by the `HOST` value. In order to use this resource in a
/// GPU, it must first be copied to its graphical memory. If the resource is then edited by the GPU,
/// the copy that is in the host system is outdated. This status is denoted by the `GPU` value.
/// Before the CPU can use it again, it must be copied back to the host system. If the resource was
/// synchronised but not edited yet on either the CPU or the GPU, the two copies are in sync. This
/// state is denoted by the `SYNCED` value.
#[derive(Debug, PartialEq, Eq)]
pub enum SyncStatus {
	/// The copy on the host (CPU/RAM) is leading.
	///
	/// The resource was not yet copied to the GPU, or after copying the CPU has edited it. The copy
	/// on the GPU, if any, needs to be synced from the host before it can be used.
	HOST,

	/// The copy on the GPU (VRAM) is leading.
	///
	/// The resource was copied to the GPU and has been edited there. The copy in the host needs to
	/// be synced from the GPU before it can be used.
	GPU,

	/// Both copies of the resource are up-to-date.
	///
	/// The CPU and the GPU can both perform an operation on this resource without needing to sync
	/// first.
	SYNCED
}