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
 * OpenMP. The number of OpenMP threads can by setting the environment variable
 * OMP_NUM_THREADS, using export OMP_NUM_THREADS=X, where X is the number of
 * CPU threads you wish to use. The environment variable can be ignored by
 * using the runtime command omp_set_num_threads ( X ), where X is the number
 * of CPU threads you wish to use. You must include omp.h in order to use
 * runtime commands.
 */

/*
 * nvcc -ccbin pgc++ -O2 -Xcompiler "-V19.4 -mp" openmp.cpp -o openmp -run
 */

#include <omp.h>

#include "openmp.h"
#include "timer.h"

void openMP( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

    // Timer
    Timer timer {};

    // Request number of threads at runtime
    omp_set_num_threads( omp_get_max_threads( ) );

    timer.startCPUTimer( );

    for ( int l = 0; l < loops; l++ ) {

        int i, j, k;

        // Create parallel region and worksharing
#pragma omp parallel for shared( A, B, C, n ) private( i, j, k ) schedule( static )
        for ( i = 0; i < n; ++i ) {
            for ( j = 0; j < n; ++j ) {
                float prod = 0.0f;
                for ( k = 0; k < n; ++k ) {
                    prod += A[k * n + i] * B[j * n + k];
                }  // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }  // j
        }      // i
    }          // loops

    timer.stopAndPrintCPU( loops );

}  // openMP
