/**
 * ToRSh Node.js Binding Tests
 *
 * Plain-JavaScript Jest tests — no ts-jest required.
 *
 * These tests validate the NAPI binding layer end-to-end and require the
 * native `.node` module to be built first:
 *
 *   cd ../../../  &&  cargo build --release --features nodejs
 *   cp target/release/librstorch.dylib ../nodejs/native/torsh_native.node
 *   # (path and extension vary by platform)
 *
 * Then run:
 *   npm test
 */

'use strict';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** True if every element of `a` and `b` are within `eps` of each other. */
function allClose(a, b, eps = 1e-4) {
  if (a.length !== b.length) return false;
  return a.every((v, i) => Math.abs(v - b[i]) <= eps);
}

// ── Module loading (skip gracefully if native addon is not built) ─────────────

let Tensor;
let utils;
let nativeUnavailable = false;

try {
  const mod = require('../lib/index');
  Tensor = mod.Tensor;
  utils = mod.utils;
} catch (err) {
  nativeUnavailable = true;
}

// Jest's `test.skip` equivalent when the addon isn't present.
const maybeTest = nativeUnavailable
  ? (name, _fn) => test.skip(name, () => {})
  : (name, fn) => test(name, fn);

// ── Tensor creation ───────────────────────────────────────────────────────────

describe('Tensor.eye', () => {
  maybeTest('eye(3) produces 3×3 identity matrix', () => {
    const t = Tensor.eye(3);
    expect(t.shape()).toEqual([3, 3]);
    const expected = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    expect(allClose(t.data(), expected)).toBe(true);
  });

  maybeTest('eye(1) is [[1]]', () => {
    const t = Tensor.eye(1);
    expect(t.shape()).toEqual([1, 1]);
    expect(allClose(t.data(), [1])).toBe(true);
  });
});

describe('Tensor.linspace', () => {
  maybeTest('linspace(0, 1, 5) produces [0, 0.25, 0.5, 0.75, 1]', () => {
    const t = Tensor.linspace(0, 1, 5);
    expect(t.shape()).toEqual([5]);
    expect(allClose(t.data(), [0, 0.25, 0.5, 0.75, 1])).toBe(true);
  });

  maybeTest('linspace(0, 10, 3) produces [0, 5, 10]', () => {
    const t = Tensor.linspace(0, 10, 3);
    expect(allClose(t.data(), [0, 5, 10])).toBe(true);
  });
});

describe('Tensor.randn', () => {
  maybeTest('randn(3, 3) returns a 3×3 tensor', () => {
    const t = Tensor.randn(3, 3);
    expect(t.shape()).toEqual([3, 3]);
    expect(t.numel()).toBe(9);
  });
});

// ── Reshape ───────────────────────────────────────────────────────────────────

describe('tensor.reshape', () => {
  maybeTest('reshape([4]) changes shape to [4]', () => {
    const t = Tensor.tensor([[1, 2], [3, 4]]);
    const r = t.reshape(4);
    expect(r.shape()).toEqual([4]);
    expect(allClose(r.data(), [1, 2, 3, 4])).toBe(true);
  });

  maybeTest('reshape round-trip [4] → [2,2] preserves data', () => {
    const t = Tensor.tensor([1, 2, 3, 4]);
    const r = t.reshape(2, 2);
    expect(r.shape()).toEqual([2, 2]);
    expect(allClose(r.data(), [1, 2, 3, 4])).toBe(true);
  });
});

// ── Softmax ───────────────────────────────────────────────────────────────────

describe('tensor.softmax', () => {
  maybeTest('softmax(0) on 1-D tensor sums to 1', () => {
    const t = Tensor.tensor([1, 2, 3, 4]);
    const s = t.softmax(0);
    const sum = s.data().reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-5);
  });

  maybeTest('softmax values are all in (0, 1)', () => {
    const t = Tensor.tensor([1, 2, 3]);
    const s = t.softmax(0);
    s.data().forEach(v => {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    });
  });
});

// ── Clone / detach ────────────────────────────────────────────────────────────

describe('tensor.clone', () => {
  maybeTest('clone() is a distinct object', () => {
    const t = Tensor.tensor([10, 20, 30]);
    const c = t.clone();
    expect(c).not.toBe(t);
    expect(allClose(c.data(), t.data())).toBe(true);
  });
});

describe('tensor.detach', () => {
  maybeTest('detach() returns tensor with same data', () => {
    const t = Tensor.tensor([5, 6, 7]);
    const d = t.detach();
    expect(allClose(t.data(), d.data())).toBe(true);
  });
});

// ── Scalar operations ─────────────────────────────────────────────────────────

describe('scalar operations', () => {
  maybeTest('add(scalar) adds to every element', () => {
    const t = Tensor.tensor([1, 2, 3, 4]);
    expect(allClose(t.add(10).data(), [11, 12, 13, 14])).toBe(true);
  });

  maybeTest('mul(scalar) multiplies every element', () => {
    const t = Tensor.tensor([1, 2, 3, 4]);
    expect(allClose(t.mul(2).data(), [2, 4, 6, 8])).toBe(true);
  });

  maybeTest('div(scalar) divides every element', () => {
    const t = Tensor.tensor([2, 4, 6, 8]);
    expect(allClose(t.div(2).data(), [1, 2, 3, 4])).toBe(true);
  });

  maybeTest('sub(scalar) subtracts from every element', () => {
    const t = Tensor.tensor([5, 6, 7, 8]);
    expect(allClose(t.sub(3).data(), [2, 3, 4, 5])).toBe(true);
  });
});

// ── Tensor–tensor subtraction ─────────────────────────────────────────────────

describe('tensor subtraction', () => {
  maybeTest('sub(tensor) subtracts element-wise', () => {
    const a = Tensor.tensor([5, 6, 7, 8]);
    const b = Tensor.tensor([1, 2, 3, 4]);
    expect(allClose(a.sub(b).data(), [4, 4, 4, 4])).toBe(true);
  });

  maybeTest('sub(tensor) with 2D tensors', () => {
    const a = Tensor.tensor([[3, 4], [5, 6]]);
    const b = Tensor.tensor([[1, 1], [2, 2]]);
    expect(allClose(a.sub(b).data(), [2, 3, 3, 4])).toBe(true);
  });
});

// ── Activations ───────────────────────────────────────────────────────────────

describe('activations', () => {
  maybeTest('sigmoid([0]) ≈ 0.5', () => {
    const t = Tensor.tensor([0]);
    const s = t.sigmoid();
    expect(Math.abs(s.data()[0] - 0.5)).toBeLessThan(1e-5);
  });

  maybeTest('tanh([0]) ≈ 0', () => {
    const t = Tensor.tensor([0]);
    const r = t.tanh();
    expect(Math.abs(r.data()[0])).toBeLessThan(1e-5);
  });
});

// ── Reductions ────────────────────────────────────────────────────────────────

describe('reductions', () => {
  maybeTest('sum() of [1,2,3,4] is 10', () => {
    const t = Tensor.tensor([1, 2, 3, 4]);
    expect(Math.abs(t.sum().data()[0] - 10)).toBeLessThan(1e-4);
  });

  maybeTest('mean() of [2,4,6,8] is 5', () => {
    const t = Tensor.tensor([2, 4, 6, 8]);
    expect(Math.abs(t.mean().data()[0] - 5)).toBeLessThan(1e-4);
  });
});

// ── Transpose ─────────────────────────────────────────────────────────────────

describe('transpose', () => {
  maybeTest('transpose of 2×3 is 3×2', () => {
    const t = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    const tr = t.transpose();
    expect(tr.shape()).toEqual([3, 2]);
  });
});

// ── Utilities ─────────────────────────────────────────────────────────────────

describe('utils', () => {
  maybeTest('manualSeed() does not throw', () => {
    expect(() => utils.manualSeed(42)).not.toThrow();
  });

  maybeTest('cudaAvailable() returns a boolean', () => {
    expect(typeof utils.cudaAvailable()).toBe('boolean');
  });

  maybeTest('cudaDeviceCount() returns a non-negative integer', () => {
    const c = utils.cudaDeviceCount();
    expect(Number.isInteger(c)).toBe(true);
    expect(c).toBeGreaterThanOrEqual(0);
  });

  maybeTest('saveTensor / loadTensor round-trip', () => {
    const os = require('os');
    const path = require('path');
    const fs = require('fs');
    const tmp = path.join(os.tmpdir(), `torsh_test_${Date.now()}.bin`);
    try {
      const t = Tensor.tensor([3.14, 2.72, 1.41]);
      utils.saveTensor(t, tmp);
      const loaded = utils.loadTensor(tmp);
      expect(loaded.shape()).toEqual([3]);
      expect(allClose(loaded.data(), t.data())).toBe(true);
    } finally {
      if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
    }
  });
});
