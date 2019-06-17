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
 * BLAS.
 */

/*
 * nvcc -ccbin pgc++ -O2 -Xlinker "-lblas" -Xcompiler "V19.4"
 * blas.cpp -o blas -run
 */

#include <cblas.h>
#include "helper.h"

void normalC(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	auto start = getTimeCPU();

	for ( int l = 0; l < loops; l++ ) {
		for ( int i = 0; i < n; ++i ) {
			for ( int j = 0; j < n; ++j ) {
				float prod = 0.0f;
				for ( int k = 0; k < n; ++k ) {
					prod += A[k * n + i] * B[j * n + k];
				} // k
				C[j * n + i] = alpha * prod + beta * C[j * n + i];
			} // j
		} // i
	} // loops

	auto end = getTimeCPU();

	printCPUTime( start, end, loops );
} // normalC

void blas(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	openblas_set_num_threads( 6 );

	auto start = getTimeCPU();

	for ( int l = 0; l < loops; l++ )
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, beta, C, n );
	auto end = getTimeCPU();

	printCPUTime( start, end, loops );
} // blas

int main( int argc, char** argv ) {

	int n = 1024;
	if ( argc > 1)
		n = std::atoi( argv[1] );
	printf( "Running with N = %d\n\n", n );

	float alpha = 1.0f;
	float beta = 0.0f;

	// Declare host variables
	float *h_A = new float[sizeof(float) * n * n];
	float *h_B = new float[sizeof(float) * n * n];
	float *h_C = new float[sizeof(float) * n * n];
	float *h_C_blas = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	} // i

	// Benchmark normal C matrix multiplication
	printf( "Running Normal C: " );
	normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

	// Benchmark and verify BLAS matrix multiplication
	printf( "Running BLAS: " );
	blas( n, alpha, h_A, h_B, beta, h_C_blas, 5 );
	verify( n, h_C, h_C_blas );

	// Memory clean up
	delete[] ( h_A );
	delete[] ( h_B );
	delete[] ( h_C );
	delete[] ( h_C_blas );
} // main
