Apex
====
Apex is a geometric operations library aimed at heterogenous computing. It leverages different computational resources on your system (CPUs, GPUs and accelerators) to improve performance of your applications.

Status
----
This library is not in a useable state.

I am properly overengineering this. I'm giving it a 95% chance that it will never really come into a useable state. That is not the aim of this project, either. The primary goals at the moment are:
* To practice setting up projects properly, using proper data structures, design patterns, tests and documentation.
* To learn Rust.
* To have fun implementing interesting algorithms and inventing new ones.
* To specialise myself in high-performance computations, with a mix of practicality and science.

The goal of eventually maybe having a useable library is only secondary, but it serves as a motivator to keep working on the project.

Requirements
----
Apex has a requirements list specified in its Cargo file, but those are not the only requirements. There is an additional requirement outside of the Rust ecosystem: ArrayFire.

To properly compile your application with this library, the following steps need to be undertaken:
* Download the ArrayFire library binaries from their website: https://arrayfire.com/binaries/
* Run their installer to unpack the library.
* Compile using: `AF_PATH=/path/to/ArrayFire LD_LIBRARY_PATH=$AF_PATH/lib64 cargo test`.
