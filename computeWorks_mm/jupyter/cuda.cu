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
		}
		C[row * n + col] = tmpSum;
	}
}

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
	float *h_C_cuda = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	}

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
}
