# ExaMPM Architecture and Blackwell Notes

## Current Architecture

ExaMPM is a small, mostly header-only MPM library built on Cabana, Kokkos, and
MPI. The compiled library currently contains only the explicit `Mesh`
instantiations in `src/ExaMPM_Mesh.cpp`; most numerical behavior is expressed
as templated kernels in headers.

The main layers are:

- `Mesh<MemorySpace>` wraps a uniform Cabana grid, domain padding for
  non-periodic boundaries, halo width, and domain node bounds.
- `ProblemManager<MemorySpace>` owns particle AoSoA data, grid arrays,
  Cabana halo objects, typed field accessors, scatter/gather, and particle
  migration.
- `ParticleInit` initializes particles over owned cells and compacts rejected
  candidates.
- `DenseLinearAlgebra` provides inline 3x3 determinant, inverse, transpose,
  matrix-vector, and matrix-matrix operations used inside GPU kernels.
- `VelocityInterpolation` implements APIC P2G and G2P interpolation helpers.
- `TimeIntegrator` orchestrates `p2g`, `fieldSolve`, `g2p`, position
  correction, halo exchange, and density marking.
- `Solver` is the application-facing backend factory and solve loop for
  Serial, OpenMP, CUDA, and HIP.

For Kokkos 5 + Cabana master, the CUDA path now uses `Kokkos::Cuda` execution
with `Kokkos::SharedSpace` memory. This preserves GPU execution while allowing
the non-CUDA-aware OpenMPI package used in the Modal image to access halo and
particle buffers safely.

## Test Base

CTest is enabled at the top level. Tests are split into:

- Unit tests in `tests/unit_tests.cpp`, currently covering
  `DenseLinearAlgebra` and `BoundaryCondition` on the selected Kokkos backend.
- Example smoke tests in `examples/CMakeLists.txt`, currently covering
  `DamBreak` and `FreeFall`.
- Optional benchmark smoke coverage when `EXAMPM_ENABLE_BENCHMARKS=ON`.

Useful configure flags:

```bash
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DEXAMPM_TEST_BACKEND=CUDA \
  -DEXAMPM_ENABLE_BENCHMARKS=ON
ctest --test-dir build -L CUDA --output-on-failure
```

## Benchmark Base

`benchmarks/ExaMPM_Benchmarks` is dependency-light and prints CSV:

```text
benchmark,backend,size,iterations,seconds,rate_per_second
```

Current benchmark modes:

- `dense`: large parallel loop over inline 3x3 dense linear algebra.
- `particle_init`: constructs deterministic `ProblemManager` instances and
  measures particle initialization throughput.
- `step`: measures direct `TimeIntegrator::step` plus particle communication,
  avoiding `Solver::solve` output overhead.

Example:

```bash
ExaMPM_Benchmarks CUDA dense 1000000 10
ExaMPM_Benchmarks CUDA particle_init 32 5 2
ExaMPM_Benchmarks CUDA step 16 10 2
```

## Blackwell Notes

Blackwell should not require algorithmic changes for this code path if Kokkos
and Cabana are built with a CUDA toolkit and Kokkos architecture flag that know
the target GPU.

Recommended build direction:

- For B100/B200, use CUDA Toolkit 12.8 or newer and a native Blackwell target
  such as `sm_100` / `Kokkos_ARCH_BLACKWELL100` when available.
- For B300-class devices, use the matching Kokkos architecture flag such as
  `Kokkos_ARCH_BLACKWELL103` when available in the Kokkos release.
- Include PTX for forward compatibility when distributing binaries across
  unknown Blackwell variants.
- Keep `CUDA_FORCE_PTX_JIT=1` as a compatibility check for prebuilt binaries.

Performance priorities for Blackwell:

- The ExaMPM hot path is likely memory and scatter/halo dominated, not dense
  floating-point dominated.
- Compare `dense`, `particle_init`, and `step`; if `dense` scales but `step`
  does not, the bottleneck is likely Cabana scatter/gather, atomics, shared
  memory behavior, or MPI/halo traffic.
- Re-test whether `SharedSpace` is still the right CUDA memory choice on a
  CUDA-aware MPI stack. On a CUDA-aware MPI, `CudaSpace` may be worth
  re-benchmarking.
- For multi-GPU Blackwell systems, add 2-rank and multi-rank benchmark labels
  to separate compute throughput from halo and migration costs.

References:

- NVIDIA Blackwell Tuning Guide:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- NVIDIA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- Kokkos configuration guide:
  https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html
