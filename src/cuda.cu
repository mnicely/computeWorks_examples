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
 * an optimized CUDA kernel.
 */

/*
 * nvcc -O2 cuda.cu -o cuda -run
 */

#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <helper_cuda.h>

#include "cuda.cuh"
#include "timer.h"

// Thread block size
#define BLOCK_SIZE 16

// Matrix multiplication kernel called by MatMul()
__global__ void
MatMulKernel( int const n, float const *__restrict__ A, float const *__restrict__ B, float *__restrict__ C ) {

    auto const block = cooperative_groups::this_thread_block( );

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread computes one element of Csub by accumulating results into Cvalue
    float Cvalue = 0.0f;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float *Csub = &C[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for ( int m = 0; m < ( n / BLOCK_SIZE ); ++m ) {

        // Get sub-matrix Asub of A
        float const *Asub = &A[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * m];

        // Get sub-matrix Bsub of B
        float const *Bsub = &B[n * BLOCK_SIZE * m + BLOCK_SIZE * blockCol];

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row * n + col];
        Bs[row][col] = Bsub[row * n + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        block.sync( );

        // Multiply Asub and Bsub together
        for ( int e = 0; e < BLOCK_SIZE; ++e )
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        block.sync( );
    } // m

    // Write Csub to device memory each thread writes one element
    Csub[row * n + col] = Cvalue;
} // MatMulKernel

void cuda( int const &  n,
           float const &alpha,
           float const *A,
           float const *B,
           float const &beta,
           float *      C,
           int const &  loops ) {

    // Timer
    Timer timer {};

    // Declare device result pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on device
    checkCudaErrors( cudaMalloc( ( void ** )&d_A, sizeof( float ) * n * n ) );
    checkCudaErrors( cudaMalloc( ( void ** )&d_B, sizeof( float ) * n * n ) );
    checkCudaErrors( cudaMalloc( ( void ** )&d_C, sizeof( float ) * n * n ) );

    // Copy host memory to device
    checkCudaErrors( cudaMemcpy( d_A, A, sizeof( float ) * n * n, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_B, B, sizeof( float ) * n * n, cudaMemcpyHostToDevice ) );

    // setup the dimensions
    dim3 blocksPerGrid( ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE, ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE );
    dim3 threadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );

    timer.startGPUTimer( );

    for ( int l = 0; l < loops; l++ )
        MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>( n, d_A, d_B, d_C );

    checkCudaErrors( cudaDeviceSynchronize( ) );

    timer.stopAndPrintGPU( loops );

    // Copy results from device to host
    checkCudaErrors( cudaMemcpy( C, d_C, sizeof( float ) * n * n, cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaFree( d_A ) );
    checkCudaErrors( cudaFree( d_B ) );
    checkCudaErrors( cudaFree( d_C ) );

} // cuda
