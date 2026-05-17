# Modal GPU comparison

This note records a same-benchmark comparison on Modal between a Tesla T4 and
an NVIDIA B200 for the CUDA/Kokkos benchmark harness.

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

## Commands

```sh
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA dense 5000000 20
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA particle_init 24 5 2
/opt/ExaMPM/install/bin/ExaMPM_Benchmarks CUDA step 16 5 2
```

## Results

| Benchmark | Size | Iterations | T4 seconds | T4 rate/s | B200 seconds | B200 rate/s | B200/T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 5000000 | 20 | 0.078194539 | 1.2788617e+09 | 0.002499364 | 4.0010179e+10 | 31.29x |
| `particle_init` | 24 | 5 | 0.098540486 | 1.0959962e+07 | 0.024847694 | 4.3464798e+07 | 3.97x |
| `step` | 16 | 5 | 0.019330127 | 2.2033999e+07 | 0.001873356 | 2.2735668e+08 | 10.32x |

## Interpretation

The `dense` benchmark scales the most, so the CUDA backend is active and the
B200 is clearly executing the hot device loop much faster than the T4.

The full `step` benchmark scales less than the dense loop but still gains about
10x. That is the most relevant signal for ExaMPM because it includes the
particle-grid path, scatter/gather style work, Cabana data movement, and the
communication/migration calls used by the time integrator.

The smaller `particle_init` speedup suggests this path is not purely limited by
raw GPU arithmetic. For follow-up optimization work, prioritize profiling the
real `step` path before spending time on standalone dense math.

## Follow-up checks

- Run the same benchmark with larger grids/particle counts to reduce fixed
  overhead in `particle_init` and `step`.
- Compare `Kokkos::CudaSpace` versus `Kokkos::SharedSpace` on a CUDA-aware MPI
  stack. Modal's apt OpenMPI path currently needs `SharedSpace`, but a cluster
  deployment may not.
- Add kernel-level profiling around particle-to-grid and grid-to-particle
  phases to split scatter, interpolation, boundary, and migration costs.
