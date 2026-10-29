// Usage: node tests/fixtures/spz-v4/generate.mjs /path/to/upstream/spz.mjs
// Use the official self-contained WASM module from https://nianticlabs.github.io/spz/spz.js.
// This prints hex fixtures, without depending on this repository's SPZ codec.
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const modulePath = process.argv[2];
const { default: createSpzModule } = await import(pathToFileURL(modulePath));
const spz = await createSpzModule();
const hash = createHash('sha256').update(readFileSync(modulePath)).digest('hex');
console.log(`Upstream WASM module SHA-256: ${hash}`);
for (const degree of [0, 1, 2, 3, 4]) {
  const coefficients = (degree + 1) ** 2 - 1;
  const cloud = {
    numPoints: 2,
    shDegree: degree,
    antialiased: true,
    extensions: [],
    positions: new Float32Array([1, -2, 0.5, -4, 8, -0.25]),
    scales: new Float32Array([0, 0.5, -0.5, 1, -1, 0]),
    rotations: Float32Array.from([1, 2, 3, 4, 4, -3, 2, -1], value => value / Math.sqrt(30)),
    alphas: new Float32Array([0, 1]),
    colors: new Float32Array([0, 1, -1, 0.5, -0.5, 0]),
    sh: Float32Array.from({ length: 2 * coefficients * 3 }, (_, i) => (i % 17 - 8) / 16),
  };
  const bytes = spz.saveSpzToBuffer(cloud, {
    version: 4,
    from: spz.CoordinateSystem.RUB,
    sh1Bits: 5,
    shRestBits: 4,
  });
  const decoded = spz.loadSpzFromBuffer(bytes, { to: spz.CoordinateSystem.RUB });
  if (decoded.numPoints !== 2 || decoded.shDegree !== degree) {
    throw new Error('Upstream fixture round trip failed');
  }
  console.log(`SH${degree}: ${Buffer.from(bytes).toString('hex')}`);
}
