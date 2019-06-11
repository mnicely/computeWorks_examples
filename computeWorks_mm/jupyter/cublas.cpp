#include <cublas_v2.h>	// cuBLAS
#include "helper.h"

void normalC(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	auto start = startTimer();

	for ( int l = 0; l < loops; l++ ) {
		for ( int i = 0; i < n; ++i ) {
			for ( int j = 0; j < n; ++j ) {

				float prod = 0.0f;

				for ( int k = 0; k < n; ++k ) {
					prod += A[k * n + i] * B[j * n + k];
				}

				C[j * n + i] = alpha * prod + beta * C[j * n + i];
			} // j
		} // i
	} // loops

	auto end = stopTimer();

	printTime( start, end, loops );
}

void cublas(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	// Declare device pointers and cublas handle
	float *d_A, *d_B, *d_C;
	cublasHandle_t handle;
	cublasCreate( &handle );

	// Allocate memory on device
	cudaMalloc( (void **) &d_A, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_B, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_C, sizeof(float) * n * n );

	// Copy host memory to device
	cudaMemcpy( d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, B, sizeof(float) * n * n, cudaMemcpyHostToDevice );

	auto startEvent = startGPUTimer();
	for ( int l = 0; l < loops; l++ )
		cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n );
	auto stopEvent = stopGPUTimer();

	cudaDeviceSynchronize();

	// Copy results from device to host
	cudaMemcpy( C, d_C, sizeof(float) * n * n, cudaMemcpyDeviceToHost );

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );
	cublasDestroy( handle );

	printGPUTime( startEvent, stopEvent, loops );
}

int main( int argc, char** argv ) {

	int n;
	if ( argc < 2 ) {
		n = 1024;
		printf( "No input given.\n" );
		printf( "Running with N = %d\n\n", n );
	} else {
		n = std::atoi( argv[1] );
		printf( "Running with N = %d\n\n", n );
	}

	float alpha = 1.0f;
	float beta = 0.0f;

	// Declare host variables
	float *h_A = new float[sizeof(float) * n * n];
	float *h_B = new float[sizeof(float) * n * n];
	float *h_C = new float[sizeof(float) * n * n];
	float *h_C_cublas = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	}

	// Benchmark normal C matrix multiplication
	printf( "Running Normal C: " );
	normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

	// Benchmark and verify CUBLAS matrix multiplication
	printf( "Running CUBLAS: " );
	cublas( n, alpha, h_A, h_B, beta, h_C_cublas, 500 );
	verify( n, h_C, h_C_cublas );

	// Memory clean up
	delete[] ( h_A );
	delete[] ( h_B );
	delete[] ( h_C );
	delete[] ( h_C_cublas );
}
