# 3D Gaussian Splatting Library

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-core)](https://crates.io/crates/wgpu-3dgs-core) [![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-core)](https://docs.rs/wgpu-3dgs-core/latest/wgpu_3dgs_core) [![License](https://img.shields.io/github/license/lioqing/wgpu-3dgs-core)](./LICENSE)

## Overview

This is the backbone of [wgpu-3dgs-viewer](https://crates.io/crates/wgpu-3dgs-viewer) and [wgpu-3dgs-editor](https://crates.io/crates/wgpu-3dgs-editor).

This library provides helper functions and abstractions for working with 3D Gaussian Splatting, including:
- Loading and parsing PLY files containing 3D Gaussian data.
- Configuring Gaussian structures for loading into GPU buffers.
- Utilities for creating and managing GPU resources related to 3D Gaussian Splatting.
- Compute pipeline abstractions for processing 3D Gaussian data on the GPU.