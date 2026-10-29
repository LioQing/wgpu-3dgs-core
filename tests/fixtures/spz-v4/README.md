# SPZ v4 fixtures

`upstream.hex` contains five hex-encoded SPZ files, one line per SH degree 0
through 4. Each contains two antialiased Gaussians. Inputs and generation code
are in `generate.mjs`. The upstream runtime both wrote and successfully read
each fixture, independently of this repository's codec.

Generated on October 25, 2026 using the official self-contained WASM module:

- URL: https://nianticlabs.github.io/spz/spz.js
- SHA-256: `238f0a666478d64f2610246e01d2d0bab80f74e74c2a41e037be207440a06872`
- Format reference: https://github.com/nianticlabs/spz/tree/v3.0.0

To regenerate, download the module as `spz.mjs`, then run:

```sh
node tests/fixtures/spz-v4/generate.mjs /path/to/spz.mjs
```

Copy the five `SHn:` hex values into `upstream.hex`, in order. The official
hosted module may change, so compare its printed hash with the one above.

Positions are `[1, -2, 0.5]` and `[-4, 8, -0.25]`. Rotations are normalized
`[1, 2, 3, 4]` and `[4, -3, 2, -1]`, testing two omitted-component indices.
Scales are supplied in log space, alphas in logit space, and colors as SH0.
Non-DC coefficients vary across channels, points, and bands. SH0 omits the
empty SH stream. SH3 and SH4 exercise compressed ZSTD blocks, not only raw blocks.
