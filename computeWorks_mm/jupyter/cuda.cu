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
 * a naive CUDA kernel.
 */

/*
 * nvcc -O2 cuda.cu -o cuda -run
 */

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

__global__ void cudaKernel(
		int const n,
		float const * __restrict__ A,
		float const * __restrict__ B,
		float * __restrict__ C ) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if ( row < n && col < n ) {
		// each thread computes one element of the block sub-matrix
		for ( int i = 0; i < n; i++ ) {
			tmpSum += A[row * n + i] * B[i * n + col];
		} // i
		C[row * n + col] = tmpSum;
	} // row & col
} // cudaKernel

void cuda(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	// Declare device result pointers
	float *d_A, *d_B, *d_C;

	// Allocate memory on device
	cudaMalloc( (void **) &d_A, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_B, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_C, sizeof(float) * n * n );

	// Copy host memory to device
	cudaMemcpy( d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, B, sizeof(float) * n * n, cudaMemcpyHostToDevice );

	// setup the dimensions
	int threads = 16;
	dim3 blocksPerGrid( ( n + threads - 1 ) / threads, ( n + threads - 1 ) / threads );
	dim3 threadsPerBlock( threads, threads );

	auto startEvent = startGPUTimer();
	for ( int l = 0; l < loops; l++ )
		cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_A, d_B, d_C);
	auto stopEvent = stopGPUTimer();

	cudaDeviceSynchronize();

	// Copy results from device to host
	cudaMemcpy( C, d_C, sizeof(float) * n * n, cudaMemcpyDeviceToHost );

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	printGPUTime( startEvent, stopEvent, loops );
} // cuda

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
	float *h_C_cuda = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	} // i

	// Benchmark normal C matrix multiplication
	printf( "Running Normal C: " );
	normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

	// Benchmark and verify CUDA matrix multiplication
	printf( "Running CUDA: " );
	cuda( n, alpha, h_A, h_B, beta, h_C_cuda, 5 );
	verify( n, h_C, h_C_cuda );

	// Memory clean up
	delete[] ( h_A );
	delete[] ( h_B );
	delete[] ( h_C );
	delete[] ( h_C_cuda );
} // main
