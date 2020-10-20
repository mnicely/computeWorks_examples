/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample compares performance between serial matrix multiplication and
 * BLAS.
 */

#include <chrono>
#include <cmath>  // std::sqrt, std::fabs
#include <cstdio>
#include <cstdlib>  // std::atoi

#include "blas.h"
#include "cublas.h"
#include "cuda.cuh"
#include "openacc.h"
#include "openmp.h"
#include "timer.h"

void verify( int const &n, float const *C_ref, float const *C_test ) {

    // Check result against reference
    float error_norm = 0.0f;
    float ref_norm   = 0.0f;
    float diff       = 0.0f;

    for ( int i = 0; i < n * n; i++ ) {
        diff = C_test[i] - C_ref[i];
        error_norm += diff * diff;
        ref_norm += C_test[i] * C_test[i];
    }  // i

    error_norm = static_cast<float>( std::sqrt( static_cast<double>( error_norm ) ) );
    ref_norm   = static_cast<float>( std::sqrt( static_cast<double>( ref_norm ) ) );

    if ( std::fabs( ref_norm ) < 1e-7 )
        std::printf( "Reference norm is 0.\t" );

    if ( error_norm / ref_norm < 1e-5f )
        std::printf( "--> Test passed.\n" );
    else
        std::printf( "--> Test failed.\n" );
}  // verify

void serial( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

    // Timer
    Timer timer {};

    timer.startCPUTimer( );

    for ( int l = 0; l < loops; l++ ) {
        for ( int i = 0; i < n; ++i ) {
            for ( int j = 0; j < n; ++j ) {
                float prod = 0.0f;
                for ( int k = 0; k < n; ++k ) {
                    prod += A[k * n + i] * B[j * n + k];
                }  // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }  // j
        }      // i
    }          // loops

    timer.stopAndPrintCPU( loops );
    std::printf( "\n" );

}  // serial

int main( int argc, char **argv ) {

    int n = 512;
    if ( argc > 1 )
        n = std::atoi( argv[1] );
    printf( "Running with N = %d\n\n", n );

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Declare host variables
    float *h_A        = new float[sizeof( float ) * n * n];
    float *h_B        = new float[sizeof( float ) * n * n];
    float *h_C        = new float[sizeof( float ) * n * n];
    float *h_C_mp     = new float[sizeof( float ) * n * n];
    float *h_C_blas   = new float[sizeof( float ) * n * n];
    float *h_C_acc    = new float[sizeof( float ) * n * n];
    float *h_C_cublas = new float[sizeof( float ) * n * n];
    float *h_C_cuda   = new float[sizeof( float ) * n * n];

    // Initialize values
    for ( int i = 0; i < n * n; i++ ) {
        h_A[i] = 2.0f;
        h_B[i] = 1.0f;
    }  // i

    const int loops = 4;

    // Benchmark normal C matrix multiplication
    std::printf( "Running Serial:\t" );
    serial( n, alpha, h_A, h_B, beta, h_C, loops );

    // Benchmark and verify OpenMP matrix multiplication
    std::printf( "Running OpenMP:\t" );
    openMP( n, alpha, h_A, h_B, beta, h_C_mp, loops );
    verify( n, h_C, h_C_mp );

    // Benchmark and verify BLAS matrix multiplication
    std::printf( "Running BLAS:\t" );
    blas( n, alpha, h_A, h_B, beta, h_C_blas, loops );
    verify( n, h_C, h_C_blas );

    // Benchmark and verify CUBLAS matrix multiplication
    std::printf( "Running CUBLAS:\t" );
    cublas( n, alpha, h_A, h_B, beta, h_C_cublas, loops );
    verify( n, h_C, h_C_cublas );

    // Benchmark and verify CUDA matrix multiplication
    std::printf( "Running CUDA:\t" );
    cuda( n, alpha, h_A, h_B, beta, h_C_cuda, loops );
    verify( n, h_C, h_C_cuda );

    // Benchmark and verify OpenACC matrix multiplication
    std::printf( "Running OpenACC:\t" );
    openACC( n, alpha, h_A, h_B, beta, h_C_acc, loops );
    verify( n, h_C, h_C_acc );

    // Memory clean up
    delete[]( h_A );
    delete[]( h_B );
    delete[]( h_C );
    delete[]( h_C_mp );
    delete[]( h_C_acc );
    delete[]( h_C_cublas );
    delete[]( h_C_cuda );
}  // main
