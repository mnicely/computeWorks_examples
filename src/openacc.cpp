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
 * OpenACC.
 *
 * Timer.h is not needed. Profiling is accomplised by compiling ta=tesla:time
 */

/*
 * nvcc -ccbin pgc++ -O2 -Xcompiler "-V19.4 -Bstatic_pgi -acc -ta=tesla=nordc"
 * openacc.cpp -o openacc -run
 */

#include "openacc.h"

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
                }  // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }  // j
        }      // i
    }          // loops
}  // openACC
