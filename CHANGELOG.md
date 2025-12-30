# Changelog

Please also check out the [`wgpu-3dgs-viewer` changelog](https://github.com/LioQing/wgpu-3dgs-viewer/blob/master/CHANGELOG.md) and [`wgpu-3dgs-editor` changelog](https://github.com/LioQing/wgpu-3dgs-editor/blob/master/CHANGELOG.md).

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
