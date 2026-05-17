#include <ExaMPM_BoundaryConditions.hpp>
#include <ExaMPM_DenseLinearAlgebra.hpp>
#include <ExaMPM_Mesh.hpp>
#include <ExaMPM_ProblemManager.hpp>
#include <ExaMPM_TimeIntegrator.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace
{
template <class ExecutionSpace>
struct BenchmarkMemorySpace
{
    using type = typename ExecutionSpace::memory_space;
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct BenchmarkMemorySpace<Kokkos::Cuda>
{
    using type = Kokkos::SharedSpace;
};
#endif

struct FullCellParticleInit
{
    double volume;
    double mass;

    FullCellParticleInit( const double cell_size, const int ppc,
                          const double density )
        : volume( cell_size * cell_size * cell_size / ( ppc * ppc * ppc ) )
        , mass( volume * density )
    {
    }

    template <class ParticleType>
    KOKKOS_INLINE_FUNCTION bool operator()( const double x[3],
                                            ParticleType& particle ) const
    {
        for ( int d0 = 0; d0 < 3; ++d0 )
            for ( int d1 = 0; d1 < 3; ++d1 )
                Cabana::get<0>( particle, d0, d1 ) = 0.0;

        Cabana::get<1>( particle, 0 ) = 0.1 * x[0];
        Cabana::get<1>( particle, 1 ) = 0.1 * x[1];
        Cabana::get<1>( particle, 2 ) = -0.1 * x[2];

        for ( int d = 0; d < 3; ++d )
            Cabana::get<2>( particle, d ) = x[d];

        Cabana::get<3>( particle ) = mass;
        Cabana::get<4>( particle ) = volume;
        Cabana::get<5>( particle ) = 1.0;
        return true;
    }
};

void printResult( const std::string& benchmark, const std::string& backend,
                  const int size, const int iterations, const double seconds,
                  const double rate )
{
    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( 0 == rank )
    {
        std::cout << std::setprecision( 8 ) << "benchmark,backend,size,"
                  << "iterations,seconds,rate_per_second\n"
                  << benchmark << "," << backend << "," << size << ","
                  << iterations << "," << seconds << "," << rate << "\n";
    }
}

template <class ExecutionSpace>
int benchmarkDense( const std::string& backend, const int size,
                    const int iterations )
{
    using memory_space = typename ExecutionSpace::memory_space;

    Kokkos::View<double*, memory_space> result( "dense_result", size );
    double checksum = 0.0;

    ExecutionSpace().fence();
    Kokkos::Timer timer;
    for ( int iter = 0; iter < iterations; ++iter )
    {
        Kokkos::parallel_for(
            "benchmark_dense_linear_algebra",
            Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0, size ),
            KOKKOS_LAMBDA( const int i ) {
                const double scale = 1.0 + 1.0e-6 * i;
                const double a[3][3] = {
                    { 2.0 * scale, 0.0, 1.0 },
                    { 1.0, 1.0 * scale, 0.0 },
                    { 0.0, 3.0, 1.0 * scale }
                };
                const double x[3] = { 1.0, 2.0, -1.0 };
                double inv_a[3][3];
                double y[3];
                const double det_a =
                    ExaMPM::DenseLinearAlgebra::determinant( a );
                ExaMPM::DenseLinearAlgebra::inverse( a, det_a, inv_a );
                ExaMPM::DenseLinearAlgebra::matVecMultiply( inv_a, x, y );
                result( i ) = y[0] + y[1] + y[2];
            } );
    }
    ExecutionSpace().fence();
    const double seconds = timer.seconds();

    Kokkos::parallel_reduce(
        "benchmark_dense_checksum",
        Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0, size ),
        KOKKOS_LAMBDA( const int i, double& local_sum ) {
            local_sum += result( i );
        },
        checksum );
    ExecutionSpace().fence();

    printResult( "dense", backend, size, iterations, seconds,
                 static_cast<double>( size ) * iterations / seconds );
    if ( 0.0 == checksum )
        return 1;
    return 0;
}

template <class ExecutionSpace>
auto createProblemManager( const int cells_per_dim, const int ppc )
{
    using memory_space = typename BenchmarkMemorySpace<ExecutionSpace>::type;
    using mesh_type = ExaMPM::Mesh<memory_space>;
    using problem_manager_type = ExaMPM::ProblemManager<memory_space>;

    const double cell_size = 1.0 / cells_per_dim;
    const Kokkos::Array<double, 6> global_box = { 0.0, 0.0, 0.0,
                                                  1.0, 1.0, 1.0 };
    const std::array<int, 3> global_num_cell = { cells_per_dim, cells_per_dim,
                                                 cells_per_dim };
    const std::array<bool, 3> periodic = { false, false, false };

    int comm_size = 1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    const std::array<int, 3> ranks_per_dim = { 1, comm_size, 1 };
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

    auto mesh = std::make_shared<mesh_type>( global_box, global_num_cell,
                                             periodic, partitioner, 0, 3,
                                             MPI_COMM_WORLD );

    return std::make_shared<problem_manager_type>(
        ExecutionSpace(), mesh,
        FullCellParticleInit( cell_size, ppc, 1000.0 ), ppc, 1.0e5, 1000.0,
        7.0, 100.0 );
}

template <class ExecutionSpace>
int benchmarkParticleInit( const std::string& backend, const int cells_per_dim,
                           const int iterations, const int ppc )
{
    ExecutionSpace().fence();
    Kokkos::Timer timer;
    std::size_t particles = 0;
    for ( int iter = 0; iter < iterations; ++iter )
    {
        auto pm = createProblemManager<ExecutionSpace>( cells_per_dim, ppc );
        particles += pm->numParticle();
    }
    ExecutionSpace().fence();

    const double seconds = timer.seconds();
    printResult( "particle_init", backend, cells_per_dim, iterations, seconds,
                 static_cast<double>( particles ) / seconds );
    return particles ? 0 : 1;
}

template <class ExecutionSpace>
int benchmarkStep( const std::string& backend, const int cells_per_dim,
                   const int iterations, const int ppc )
{
    auto pm = createProblemManager<ExecutionSpace>( cells_per_dim, ppc );

    ExaMPM::BoundaryCondition bc;
    for ( int d = 0; d < 6; ++d )
        bc.boundary[d] = ExaMPM::BoundaryType::FREE_SLIP;
    bc.min = pm->mesh()->minDomainGlobalNodeIndex();
    bc.max = pm->mesh()->maxDomainGlobalNodeIndex();

    ExecutionSpace().fence();
    Kokkos::Timer timer;
    for ( int iter = 0; iter < iterations; ++iter )
    {
        ExaMPM::TimeIntegrator::step( ExecutionSpace(), *pm, 1.0e-4, 9.81,
                                      bc );
        pm->communicateParticles( 3 );
    }
    ExecutionSpace().fence();

    const double seconds = timer.seconds();
    printResult( "step", backend, cells_per_dim, iterations, seconds,
                 static_cast<double>( pm->numParticle() ) * iterations /
                     seconds );
    return pm->numParticle() ? 0 : 1;
}

template <class ExecutionSpace>
int runBenchmark( const std::string& backend, const std::string& benchmark,
                  const int size, const int iterations, const int ppc )
{
    if ( "dense" == benchmark )
        return benchmarkDense<ExecutionSpace>( backend, size, iterations );
    if ( "particle_init" == benchmark )
        return benchmarkParticleInit<ExecutionSpace>( backend, size, iterations,
                                                      ppc );
    if ( "step" == benchmark )
        return benchmarkStep<ExecutionSpace>( backend, size, iterations, ppc );

    std::cerr << "Unknown benchmark: " << benchmark << "\n";
    return 1;
}

int dispatchBenchmark( const std::string& backend,
                       const std::string& benchmark, const int size,
                       const int iterations, const int ppc )
{
    if ( "serial" == backend || "Serial" == backend || "SERIAL" == backend )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return runBenchmark<Kokkos::Serial>( backend, benchmark, size,
                                             iterations, ppc );
#else
        std::cerr << "Serial backend not enabled\n";
        return 1;
#endif
    }
    if ( "openmp" == backend || "OpenMP" == backend || "OPENMP" == backend )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return runBenchmark<Kokkos::OpenMP>( backend, benchmark, size,
                                             iterations, ppc );
#else
        std::cerr << "OpenMP backend not enabled\n";
        return 1;
#endif
    }
    if ( "cuda" == backend || "Cuda" == backend || "CUDA" == backend )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return runBenchmark<Kokkos::Cuda>( backend, benchmark, size, iterations,
                                           ppc );
#else
        std::cerr << "CUDA backend not enabled\n";
        return 1;
#endif
    }
    if ( "hip" == backend || "Hip" == backend || "HIP" == backend )
    {
#ifdef KOKKOS_ENABLE_HIP
        return runBenchmark<Kokkos::Experimental::HIP>(
            backend, benchmark, size, iterations, ppc );
#else
        std::cerr << "HIP backend not enabled\n";
        return 1;
#endif
    }

    std::cerr << "Unknown backend: " << backend << "\n";
    return 1;
}
} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
        if ( argc < 3 )
        {
            std::cerr << "Usage: ExaMPM_Benchmarks backend benchmark "
                         "[size] [iterations] [ppc]\n"
                      << "  benchmark: dense, particle_init, step\n";
            Kokkos::finalize();
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        const std::string backend = argv[1];
        const std::string benchmark = argv[2];
        const int size = ( argc > 3 ) ? std::atoi( argv[3] ) : 1000000;
        const int iterations = ( argc > 4 ) ? std::atoi( argv[4] ) : 10;
        const int ppc = ( argc > 5 ) ? std::atoi( argv[5] ) : 2;

        const int result =
            dispatchBenchmark( backend, benchmark, size, iterations, ppc );

        Kokkos::finalize();
        MPI_Finalize();
        return result ? EXIT_FAILURE : EXIT_SUCCESS;
    }
}
