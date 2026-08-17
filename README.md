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

Development
----
Currently, this project is not open to online collaboration since I'm using it to practice developing in Rust and developing for the GPU. The rest of the instructions here are currently mostly for myself, but should serve as a good starting point should you want to work on Apex' code yourself.

To get started, first download and install [Cargo](https://rust-lang.org/), the Rust build system and package manager. Usually this is done by installing `rustup` and using that to install Cargo. When Cargo is installed, navigate a terminal to the directory with Rust's source code and run the following commands:

```
cargo test
```

This causes Apex' dependencies to be downloaded, Apex to be compiled and the automated tests to be ran.
