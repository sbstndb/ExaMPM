#include <ExaMPM_BoundaryConditions.hpp>
#include <ExaMPM_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
template <class ExecutionSpace>
int testDenseLinearAlgebra()
{
    int failures = 0;
    Kokkos::parallel_reduce(
        "test_dense_linear_algebra",
        Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0, 1 ),
        KOKKOS_LAMBDA( const int, int& local_failures ) {
            const double a[3][3] = {
                { 2.0, 0.0, 1.0 },
                { 1.0, 1.0, 0.0 },
                { 0.0, 3.0, 1.0 }
            };
            const double x[3] = { 1.0, 2.0, -1.0 };
            const double identity[3][3] = {
                { 1.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0 },
                { 0.0, 0.0, 1.0 }
            };

            const double tolerance = 1.0e-12;
            double y[3];
            double inv_a[3][3];
            double product[3][3];
            double transpose_a[3][3];

            const double det_a = ExaMPM::DenseLinearAlgebra::determinant( a );
            if ( Kokkos::abs( det_a - 5.0 ) > tolerance )
                ++local_failures;

            ExaMPM::DenseLinearAlgebra::matVecMultiply( a, x, y );
            if ( Kokkos::abs( y[0] - 1.0 ) > tolerance ||
                 Kokkos::abs( y[1] - 3.0 ) > tolerance ||
                 Kokkos::abs( y[2] - 5.0 ) > tolerance )
                ++local_failures;

            ExaMPM::DenseLinearAlgebra::inverse( a, det_a, inv_a );
            ExaMPM::DenseLinearAlgebra::matMatMultiply( a, inv_a, product );
            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    if ( Kokkos::abs( product[i][j] - identity[i][j] ) >
                         tolerance )
                        ++local_failures;

            ExaMPM::DenseLinearAlgebra::transpose( a, transpose_a );
            if ( Kokkos::abs( transpose_a[0][2] - a[2][0] ) > tolerance ||
                 Kokkos::abs( transpose_a[2][1] - a[1][2] ) > tolerance )
                ++local_failures;
        },
        failures );

    ExecutionSpace().fence();
    return failures;
}

template <class ExecutionSpace>
int testBoundaryConditions()
{
    ExaMPM::BoundaryCondition bc;
    bc.boundary[0] = ExaMPM::BoundaryType::FREE_SLIP;
    bc.boundary[1] = ExaMPM::BoundaryType::NONE;
    bc.boundary[2] = ExaMPM::BoundaryType::NONE;
    bc.boundary[3] = ExaMPM::BoundaryType::NONE;
    bc.boundary[4] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[5] = ExaMPM::BoundaryType::FREE_SLIP;
    bc.min = { 0, 0, 0 };
    bc.max = { 10, 20, 30 };

    int failures = 0;
    Kokkos::parallel_reduce(
        "test_boundary_conditions",
        Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0, 4 ),
        KOKKOS_LAMBDA( const int case_id, int& local_failures ) {
            const double tolerance = 1.0e-12;
            double ux = 1.0;
            double uy = 2.0;
            double uz = 3.0;

            if ( 0 == case_id )
            {
                bc( 0, 5, 5, ux, uy, uz );
                if ( Kokkos::abs( ux ) > tolerance ||
                     Kokkos::abs( uy - 2.0 ) > tolerance ||
                     Kokkos::abs( uz - 3.0 ) > tolerance )
                    ++local_failures;
            }
            else if ( 1 == case_id )
            {
                bc( 5, 20, 5, ux, uy, uz );
                if ( Kokkos::abs( ux ) > tolerance ||
                     Kokkos::abs( uy ) > tolerance ||
                     Kokkos::abs( uz ) > tolerance )
                    ++local_failures;
            }
            else if ( 2 == case_id )
            {
                bc( 5, 5, 30, ux, uy, uz );
                if ( Kokkos::abs( ux - 1.0 ) > tolerance ||
                     Kokkos::abs( uy - 2.0 ) > tolerance ||
                     Kokkos::abs( uz ) > tolerance )
                    ++local_failures;
            }
            else
            {
                bc( 5, 5, 5, ux, uy, uz );
                if ( Kokkos::abs( ux - 1.0 ) > tolerance ||
                     Kokkos::abs( uy - 2.0 ) > tolerance ||
                     Kokkos::abs( uz - 3.0 ) > tolerance )
                    ++local_failures;
            }
        },
        failures );

    ExecutionSpace().fence();
    return failures;
}

template <class ExecutionSpace>
int runTests()
{
    int failures = 0;
    failures += testDenseLinearAlgebra<ExecutionSpace>();
    failures += testBoundaryConditions<ExecutionSpace>();
    return failures;
}

int dispatchTests( const std::string& backend )
{
    if ( "serial" == backend || "Serial" == backend || "SERIAL" == backend )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return runTests<Kokkos::Serial>();
#else
        std::cerr << "Serial backend not enabled\n";
        return 1;
#endif
    }
    if ( "openmp" == backend || "OpenMP" == backend || "OPENMP" == backend )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return runTests<Kokkos::OpenMP>();
#else
        std::cerr << "OpenMP backend not enabled\n";
        return 1;
#endif
    }
    if ( "cuda" == backend || "Cuda" == backend || "CUDA" == backend )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return runTests<Kokkos::Cuda>();
#else
        std::cerr << "CUDA backend not enabled\n";
        return 1;
#endif
    }
    if ( "hip" == backend || "Hip" == backend || "HIP" == backend )
    {
#ifdef KOKKOS_ENABLE_HIP
        return runTests<Kokkos::Experimental::HIP>();
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
    Kokkos::initialize( argc, argv );
    {
        std::string backend = ( argc > 1 ) ? argv[1] : "Serial";
        int failures = dispatchTests( backend );
        if ( failures )
            std::cerr << failures << " unit test checks failed\n";
        else
            std::cout << "All unit test checks passed on " << backend << "\n";

        Kokkos::finalize();
        return failures ? EXIT_FAILURE : EXIT_SUCCESS;
    }
}
