# Changelog

Please also check out the [`wgpu-3dgs-viewer` changelog](https://github.com/LioQing/wgpu-3dgs-viewer/blob/master/CHANGELOG.md) and [`wgpu-3dgs-editor` changelog](https://github.com/LioQing/wgpu-3dgs-editor/blob/master/CHANGELOG.md).

## [Unreleased]

### Added

- 🔢 Add native and GPU support for SH degrees 0 to 4. SPZ remains limited to degrees 0 to 3. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🔄 Add `InternalGaussians::from_gaussians` and `PlyGaussians::from_gaussians` for inferred-degree conversion, and `convert_sh_degree` for truncation or zero-extension. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 📥 Add `PlyGaussianStream::from_header` and typed streaming through `GaussiansStream::into_typed`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🛠️ Add borrowed iteration through `InternalGaussians::iter` and `iter_mut`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🔄 Add `Gaussians::into_internal`, `into_ply`, `into_spz`, and `into_spz_with_options` for source conversion. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🖥️ Add `GaussiansBuffer::try_*` methods for degree-checked runtime uploads and updates. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🔄 Add `BatchRead`, `BatchWrite`, and `BatchProgress` for bounded-step PLY and SPZ reads and writes, including `GaussiansBatchReader` and `GaussiansBatchWriter` for unified models. [#34](https://github.com/LioQing/wgpu-3dgs-core/pull/34) [#36](https://github.com/LioQing/wgpu-3dgs-core/pull/36)
- 📥 Add `GaussianStream`, `PlyGaussianStream`, and `GaussiansStream` to consume PLY Gaussians before the whole file is loaded, using `Iterator::next` or `next_batch` for caller-owned batches, SPZ does not support streaming Gaussian delivery. [#34](https://github.com/LioQing/wgpu-3dgs-core/pull/34) [#36](https://github.com/LioQing/wgpu-3dgs-core/pull/36)
- 🔢 Add support for PLY SH degrees 0 to 4, preserving the degree and coefficients in whole-model, batch, and streaming reads and writes. [#44](https://github.com/LioQing/wgpu-3dgs-core/pull/44)

### Changed

- 🐛 Fix overflow in Gaussian buffer update-range checks. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🛡️ Reject SH quantization bit counts above 8 when encoding a Gaussian to SPZ. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- 🧩 Update `GaussianPodWithSh...Cov3d...Configs` to `PackedGaussian<Sh..., Cov...>` so it is more ergonomic now. The old names remain available but are deprecated, and are planned to be removed in version 0.10. [#38](https://github.com/LioQing/wgpu-3dgs-core/pull/38)
- 🐛 Fix `SpzGaussians::from_iter` panicking on empty input for zero-point headers, and reject unsupported SPZ SH degrees in headers. [#40](https://github.com/LioQing/wgpu-3dgs-core/pull/40)

### Breaking Changes

- Change `PlyGaussians::read_gaussians` to return a complete collection, retaining the SH degree for empty files. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Rename `PlyGaussians::try_from_iter` to `from_iter(records, degree)`. `PlyGaussians::new` now returns `PlyGaussiansFromIterError` instead of `std::io::Error`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Make `Gaussian` and SH encodings generic over storage degree, defaulting to `ShDegree3`. Add `GaussianShConfig::Degree` and `GaussianPod::ShDegree`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Add item-type and degree metadata to `IterGaussian`. Unified iteration and streaming yield `AnyGaussian`, and `Gaussians::Internal` stores `InternalGaussians`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Make `Gaussian::from_ply`, `from_spz`, `to_spz`, and unified collection helpers return `Result`. Degrees must match, and `ShNone` requires degree-zero data. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Make SPZ native constructors accept `ToAnyGaussian` and return `Result` with `SpzGaussiansFromGaussiansError`. Replace native `FromIterator` implementations for PLY and SPZ with `from_gaussians`. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Make source iteration panic on unsupported PLY SH counts or mismatched SPZ header/SH degrees. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Require storage-degree flags alongside encoding flags when compiling shaders. `GaussianPod::features()` and `wesl_features()` include both. [#45](https://github.com/LioQing/wgpu-3dgs-core/pull/45)
- Update `Gaussian::color` to use `Vec4` with floating-point linear RGB and opacity instead of `U8Vec4`. [#35](https://github.com/LioQing/wgpu-3dgs-core/pull/35)
- Update `SpzGaussians::read_from` to validate the gzip trailer and rejects extra decompressed SPZ data, `SpzGaussians::write_to` to reject field lengths or variants that disagree with the header instead of writing an invalid file. [#34](https://github.com/LioQing/wgpu-3dgs-core/pull/34)
- Replace `PlyGaussianPod` with `PlyGaussian`, using `Vec3`, `Quat`, and `Vec<f32>` fields instead of fixed-size arrays, and validate more strictly when reading from PLY. The new type is no longer `Copy`, `Pod`, or `Zeroable`. [#44](https://github.com/LioQing/wgpu-3dgs-core/pull/44)
- Replace the `PlyHeader::Inria` and `PlyHeader::Custom` variants with a struct containing `header` and `sh_degree`. [#44](https://github.com/LioQing/wgpu-3dgs-core/pull/44)
- Change `PlyGaussians` from a tuple struct to named `gaussians` and `sh_degree` fields. Replace infallible construction from PLY records with `PlyGaussians::new`, `TryFrom<Vec<PlyGaussian>>`, and `PlyGaussians::try_from_iter`, rejecting unsupported or inconsistent SH counts with `PlyGaussiansFromIterError` for the latter two. Inferred empty collections default to degree 3. [#44](https://github.com/LioQing/wgpu-3dgs-core/pull/44)
- Update `PlyBatchWriter::from_iter` to require an explicit `sh_degree` argument before the iterator, and reject records whose SH coefficient counts disagree with the declared degree. [#44](https://github.com/LioQing/wgpu-3dgs-core/pull/44)

## [0.8.0](https://crates.io/crates/wgpu-3dgs-core/0.8.0) - 2026-08-23

### Changed

- ⚡ Upgrade `wgpu` to 30.0, `wesl` to 0.4, and `bytemuck` to 1.25. [#31](https://github.com/LioQing/wgpu-3dgs-core/pull/31)

## [0.7.0](https://crates.io/crates/wgpu-3dgs-core/0.7.0) - 2026-05-15

### Added

- 📦 Re-export `wesl` from the crate root. [#30](https://github.com/LioQing/wgpu-3dgs-core/pull/30)

### Changed

- ⚡ Upgrade `wgpu` to 29.0 and `glam` to 0.32. [#30](https://github.com/LioQing/wgpu-3dgs-core/pull/30)

## [0.6.0](https://crates.io/crates/wgpu-3dgs-core/0.6.0) - 2026-01-11

### Added

- 🤖 CI workflow. [#29](https://github.com/LioQing/wgpu-3dgs-core/pull/29)
- 🧵 Add `async_trait` dependency for `BufferWrapper`. [#28](https://github.com/LioQing/wgpu-3dgs-core/pull/28)

### Changed

- ⚡ Upgrade `wgpu` to 28.0, `wesl` to 0.3, and `bytemuck` to 1.24. [#26](https://github.com/LioQing/wgpu-3dgs-core/pull/26)

## [0.5.0](https://crates.io/crates/wgpu-3dgs-core/0.5.0) - 2025-12-30

### Added

- 🎉 Support for [SPZ file format](https://github.com/nianticlabs/spz/) with read/write examples. [#18](https://github.com/LioQing/wgpu-3dgs-core/pull/18)
- 📦 `Gaussians` enum type with `GaussiansSource` and `GaussiansIter` for unified Gaussian representation. [#21](https://github.com/LioQing/wgpu-3dgs-core/pull/21)
- 🔄 `ReadIterGaussians` and `WriteIterGaussians` traits for easier source format implementation. [#23](https://github.com/LioQing/wgpu-3dgs-core/pull/23)
- 🛠️ `download_single` method for `FixedSizeBufferWrapper`. [#11](https://github.com/LioQing/wgpu-3dgs-core/pull/11)
- ⚙️ Optional `workgroup_size` configuration for `ComputeBundle`. [#16](https://github.com/LioQing/wgpu-3dgs-core/pull/16)
- 🔢 `GaussianMaxStdDev` type for `GaussianTransform::max_std_dev`. [#12](https://github.com/LioQing/wgpu-3dgs-core/pull/12)

### Changed

- ⚡ Upgrade `wgpu` to 27.0 and `half` to 2.7. [#25](https://github.com/LioQing/wgpu-3dgs-core/pull/25)
- 🎯 Make `IterGaussians` require `ExactSizeIterator`. [#24](https://github.com/LioQing/wgpu-3dgs-core/pull/24)
- 🔧 Use `Vec3A` instead of `Vec3` in buffer wrappers for proper alignment. [#13](https://github.com/LioQing/wgpu-3dgs-core/pull/13)
- 📝 Refactor `DownloadableBufferWrapper` into `BufferWrapper` with function-level trait bounds. [#11](https://github.com/LioQing/wgpu-3dgs-core/pull/11)
- 📐 Simplify `GaussianShNorm8Config` to use 8-bit signed normalization. [#19](https://github.com/LioQing/wgpu-3dgs-core/pull/19)
- 🔍 Use zero-based indexing for `gaussian_unpack_sh`. [#15](https://github.com/LioQing/wgpu-3dgs-core/pull/15)
- 🎨 Replace `ReadPlyError` with `std::io::Error` for simpler error handling. [#10](https://github.com/LioQing/wgpu-3dgs-core/pull/10)

### Breaking Changes

- Rename `GaussianTransform::std_dev` → `max_std_dev` and `GaussianShDegree::degree` → `get`. [#12](https://github.com/LioQing/wgpu-3dgs-core/pull/12)
- Make `GaussianShDegree::new_unchecked` unsafe and add `Default` implementations. [#12](https://github.com/LioQing/wgpu-3dgs-core/pull/12)
- Make the WESL function `gaussian_unpack_sh` zero-based indexing. [#15](https://github.com/LioQing/wgpu-3dgs-core/pull/15)
- Major refactor: `Gaussians` now stores source format types (`Vec<Gaussian>`, `PlyGaussians`, `SpzGaussians`) instead of `Gaussian` directly, enabling lossless conversion. [#9](https://github.com/LioQing/wgpu-3dgs-core/pull/9), [#18](https://github.com/LioQing/wgpu-3dgs-core/pull/18), [#23](https://github.com/LioQing/wgpu-3dgs-core/pull/23)

## [0.4.1](https://crates.io/crates/wgpu-3dgs-core/0.4.1) - 2025-10-01

### Added

- 📑 Add example modules documentations.
- ✅ Add coverage script and reports.
- 🧪 Add tests.

### Changed

- 🐛 Fix `Gaussians::read_ply_gaussians` in specific scenario failed to read custom format.

## [0.4.0](https://crates.io/crates/wgpu-3dgs-core/0.4.0) - 2025-09-20

### Added

- 🛬 Things are moved from `wgpu-3dgs-viewer` to here.
- 🖥️ `ComputeBundle` and `ComputeBundleBuilder` for simplifying creating compute pipelines for processing.
