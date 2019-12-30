/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/*
 * This sample compares performance between serial matrix multiplication and
 * OpenACC.
 */

/*
 * nvcc -ccbin pgc++ -O2 -Xcompiler "-V19.4 -Bstatic_pgi -acc -ta=tesla=nordc"
 * openacc.cpp -o openacc -run
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Verify input matrices
void verify( int const &n, float const *C_ref, float const *C_test ) {

    // Check result against reference
    float error_norm = 0.0f;
    float ref_norm   = 0.0f;
    float diff       = 0.0f;

    for ( int i = 0; i < n * n; i++ ) {
        diff = C_test[i] - C_ref[i];
        error_norm += diff * diff;
        ref_norm += C_test[i] * C_test[i];
    } // i

    error_norm = static_cast<float>( std::sqrt( static_cast<double>( error_norm ) ) );
    ref_norm   = static_cast<float>( std::sqrt( static_cast<double>( ref_norm ) ) );

    if ( std::fabs( ref_norm ) < 1e-7 )
        std::printf( "Reference norm is 0.\t" );

    if ( error_norm / ref_norm < 1e-5f )
        std::printf( "Test passed.\n" );
    else
        std::printf( "Test failed.\n" );
} // verify

void normalC( int const &  n,
              float const &alpha,
              float const *A,
              float const *B,
              float const &beta,
              float *      C,
              int const &  loops ) {

    auto start = std::chrono::high_resolution_clock::now( );

    for ( int l = 0; l < loops; l++ ) {
        for ( int i = 0; i < n; ++i ) {
            for ( int j = 0; j < n; ++j ) {
                float prod = 0.0f;
                for ( int k = 0; k < n; ++k ) {
                    prod += A[k * n + i] * B[j * n + k];
                } // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            } // j
        }     // i
    }         // loops

    auto                                      stop       = std::chrono::high_resolution_clock::now( );
    std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
    std::printf( "%0.2f ms\n", elapsed_ms.count( ) / loops );
} // normalC

void openACC( int const    n,
              float const  alpha,
              float const *A,
              float const *B,
              float const  beta,
              float *      C,
              int const &  loops ) {

    for ( int l = 0; l < loops; l++ ) {

#pragma acc kernels copyin( A [0:( n * n )], B [0:( n * n )] ) copyout( C [0:( n * n )] )
#pragma acc loop independent
        for ( int i = 0; i < n; ++i ) {
#pragma acc loop independent
            for ( int j = 0; j < n; ++j ) {
                float prod = 0.0f;
#pragma acc loop independent reduction( + : prod )
                for ( int k = 0; k < n; ++k ) {
                    prod += A[k * n + i] * B[j * n + k];
                } // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            } // j
        }     // i
    }         // loops
} // openACC

int main( int argc, char **argv ) {

    int n = 1024;
    if ( argc > 1 )
        n = std::atoi( argv[1] );
    printf( "Running with N = %d\n\n", n );

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Declare host variables
    float *h_A     = new float[sizeof( float ) * n * n];
    float *h_B     = new float[sizeof( float ) * n * n];
    float *h_C     = new float[sizeof( float ) * n * n];
    float *h_C_acc = new float[sizeof( float ) * n * n];

    // Initialize values
    for ( int i = 0; i < n * n; i++ ) {
        h_A[i] = 2.0f;
        h_B[i] = 1.0f;
    } // i

    // Benchmark normal C matrix multiplication
    printf( "Running Normal C: " );
    normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

    // Benchmark and verify OpenACC matrix multiplication
    printf( "Running OpenACC: " );
    openACC( n, alpha, h_A, h_B, beta, h_C_acc, 5 );
    verify( n, h_C, h_C_acc );

    // Memory clean up
    delete[]( h_A );
    delete[]( h_B );
    delete[]( h_C );
    delete[]( h_C_acc );
} // main
