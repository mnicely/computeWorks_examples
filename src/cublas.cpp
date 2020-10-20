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
 * the cuBLAS API.
 */

/*
 * nvcc -O2 -lcublas cublas.cpp -o cublas -run
 */

#include <stdexcept>
#include <string>

#include <cublas_v2.h>

#include "cublas.h"
#include "timer.h"

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

void cublas( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

    // Timer
    Timer timer {};

    // Declare device pointers and cublas handle
    float *        d_A, *d_B, *d_C;
    cublasHandle_t handle;
    CUDA_RT_CALL( cublasCreate( &handle ) );

    // Allocate memory on device
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_A, sizeof( float ) * n * n ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_B, sizeof( float ) * n * n ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_C, sizeof( float ) * n * n ) );

    // Copy host memory to device
    CUDA_RT_CALL( cudaMemcpy( d_A, A, sizeof( float ) * n * n, cudaMemcpyHostToDevice ) );
    CUDA_RT_CALL( cudaMemcpy( d_B, B, sizeof( float ) * n * n, cudaMemcpyHostToDevice ) );

    timer.startGPUTimer( );

    for ( int l = 0; l < loops; l++ )
        CUDA_RT_CALL( cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    timer.stopAndPrintGPU( loops );

    // Copy results from device to host
    CUDA_RT_CALL( cudaMemcpy( C, d_C, sizeof( float ) * n * n, cudaMemcpyDeviceToHost ) );

    CUDA_RT_CALL( cudaFree( d_A ) );
    CUDA_RT_CALL( cudaFree( d_B ) );
    CUDA_RT_CALL( cudaFree( d_C ) );
    CUDA_RT_CALL( cublasDestroy( handle ) );

}  // cublas
