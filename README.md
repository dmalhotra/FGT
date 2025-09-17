# Fast Gaussian Transform (FGT)

A high-performance implementation of the **tensor-product Fast Gaussian Transform (FGT)** in 1D, 2D, and 3D using a **sum of exponentials approximation** with SIMD vectorization.

---

## Requirements
- Compiler: g++ ≥ 9, clang++ ≥ 10, or icpc (GCC ≥ 9 compatible)
- Standard: C++17
- Dependencies: [SCTL](https://github.com/dmalhotra/SCTL)

---

## Build
- Release: `make`
- Clean: `make clean`

---

## Run Tests
```bash
make && ./bin/test
