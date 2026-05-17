# Modal GPU comparison

This note records same-benchmark comparisons on Modal between a Tesla T4 and an
NVIDIA B200 for the CUDA/Kokkos benchmark harness.

## Environment

- ExaMPM branch: `seb/kokkos5-cuda-tests`
- Kokkos: 5.1.1
- Cabana: `master` at build time
- T4 image: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- T4 Kokkos architecture: `Kokkos_ARCH_TURING75`
- B200 image: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- B200 Kokkos architecture: `Kokkos_ARCH_BLACKWELL100`
- ExaMPM CUDA memory space on Modal: `Kokkos::SharedSpace`

The `SharedSpace` choice is intentional for this Modal image stack because the
OpenMPI package available through apt is not CUDA-aware.

## Size Sweep Commands

```sh
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA dense 1250000 200
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA dense 5000000 100
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA dense 20000000 50
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA particle_init 16 20 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA particle_init 32 10 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA particle_init 64 5 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA step 16 20 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA step 32 10 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA step 64 5 2
```

The dense sizes use a 4x ratio. The particle benchmarks use `cells_per_dim` of
16, 32, and 64, which gives an 8x particle-count ratio between adjacent sizes
with `ppc=2`.

## Size Sweep Results

Modal runs: T4 `ap-8VyKCbLKsLLXAJh0GTWnUz`, B200 `ap-vOb0Yb2a2P3Uyj69tds7n8`.

| Benchmark | Label | Size | Iterations | T4 seconds | T4 rate/s | B200 seconds | B200 rate/s | B200/T4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | small | 1250000 | 200 | 0.16038113 | 1.5587869e+09 | 0.003492352 | 7.1584995e+10 | 45.92x |
| `dense` | medium | 5000000 | 100 | 0.25294193 | 1.9767383e+09 | 0.004200182 | 1.1904246e+11 | 60.22x |
| `dense` | large | 20000000 | 50 | 0.31472324 | 3.1773948e+09 | 0.006838276 | 1.4623569e+11 | 46.02x |
| `particle_init` | small | 16 | 20 | 0.11474516 | 1.4847511e+07 | 0.043028397 | 3.9594317e+07 | 2.67x |
| `particle_init` | medium | 32 | 10 | 0.18189622 | 2.4133322e+07 | 0.059161894 | 7.4199112e+07 | 3.07x |
| `particle_init` | large | 64 | 5 | 0.49185130 | 2.7894609e+07 | 0.14810396 | 9.2637632e+07 | 3.32x |
| `step` | small | 16 | 20 | 0.042306979 | 4.0269479e+07 | 0.004805454 | 3.5453050e+08 | 8.80x |
| `step` | medium | 32 | 10 | 0.15199925 | 2.8880143e+07 | 0.006481061 | 6.7732120e+08 | 23.45x |
| `step` | large | 64 | 5 | 0.34015667 | 4.0334356e+07 | 0.018676302 | 7.3462080e+08 | 18.21x |

## Interpretation

The `dense` benchmark scales the most, so the CUDA backend is active and the
B200 is clearly executing the hot device loop much faster than the T4. The
stabilized sweep shows roughly 46x to 60x higher dense-loop throughput on B200.

The full `step` benchmark scales less than the dense loop but still gains about
9x to 23x. That is the most relevant signal for ExaMPM because it includes the
particle-grid path, scatter/gather style work, Cabana data movement, and the
communication/migration calls used by the time integrator.

The smaller `particle_init` speedup, about 2.7x to 3.3x, suggests this path is
not purely limited by raw GPU arithmetic. For follow-up optimization work,
prioritize profiling the real `step` path before spending time on standalone
dense math.

## Follow-up checks

- Run the same benchmark with larger grids/particle counts to reduce fixed
  overhead in `particle_init` and `step`.
- Compare `Kokkos::CudaSpace` versus `Kokkos::SharedSpace` on a CUDA-aware MPI
  stack. Modal's apt OpenMPI path currently needs `SharedSpace`, but a cluster
  deployment may not.
- Add kernel-level profiling around particle-to-grid and grid-to-particle
  phases to split scatter, interpolation, boundary, and migration costs.
