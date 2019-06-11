#include <chrono>		// std::chrono
#include <cmath> 		// std::sqrt, std::fabs
#include <cstdio>		// std::printf
#include <cstdlib>		// std::atoi
#include <stdexcept>	// std::runtime_error
#include <omp.h>		// OpenMP - PGI
#include <cblas.h>		// OpenBLAS - PGI
#include <cublas_v2.h>	// cuBLAS

auto startTimer( ) {
	return ( std::chrono::high_resolution_clock::now() );
}

auto stopTimer( ) {
	return ( std::chrono::high_resolution_clock::now() );
}

auto startGPUTimer( ) {
	cudaEvent_t startEvent = nullptr;
	cudaEventCreate( &startEvent, cudaEventBlockingSync );
	cudaEventRecord( startEvent );
	return ( startEvent );
}

auto stopGPUTimer( ) {
	cudaEvent_t stopEvent = nullptr;
	cudaEventCreate( &stopEvent, cudaEventBlockingSync );
	cudaEventRecord( stopEvent );
	cudaEventSynchronize( stopEvent );
	return ( stopEvent );
}

template<typename T>
void printTime( T start, T stop, int const & loops ) {
	std::chrono::duration<double, std::milli> elapsed_ms;
	elapsed_ms = stop - start;
	std::printf( "%0.2f ms\n", elapsed_ms.count() / loops );
}

template<typename T>
void printGPUTime( T startEvent, T stopEvent, int const & loops ) {
	float elapsed_ms;
	cudaEventElapsedTime( &elapsed_ms, startEvent, stopEvent );
	std::printf( "%0.2f ms\n", elapsed_ms / loops );
}

void print( int const & n, float const * C ) {
	for ( int i = 0; i < n; i++ ) {
		for ( int j = 0; j < n; j++ ) {
			printf( "%0.0f ", C[i * n + j] );
		}
		printf( "\n" );
	}
	printf( "\n" );
}

void verify( int const & n, float const * C_ref, float const * C_test ) {

	// Check result against reference
	float error_norm = 0.0f;
	float ref_norm = 0.0f;
	float diff = 0.0f;

	for ( int i = 0; i < n * n; i++ ) {
		diff = C_test[i] - C_ref[i];
		error_norm += diff * diff;
		ref_norm += C_test[i] * C_test[i];
	}

	error_norm = static_cast<float>( std::sqrt( static_cast<double>( error_norm ) ) );
	ref_norm = static_cast<float>( std::sqrt( static_cast<double>( ref_norm ) ) );

	if ( std::fabs( ref_norm ) < 1e-7 )
		std::printf( "Reference norm is 0.\t" );

	if ( error_norm / ref_norm < 1e-5f )
		std::printf( "Test passed.\n" );
	else
		std::printf( "Test failed.\n" );
}

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

void openMP(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	// Request number of threads at runtime
	omp_set_num_threads( 6 );

	auto start = startTimer();

	for ( int l = 0; l < loops; l++ ) {

		// Create parallel region and worksharing
#pragma omp parallel for shared(A, B, C, n) schedule(static)
		for ( int i = 0; i < n; ++i ) {
			for ( int j = 0; j < n; ++j ) {
				float prod = 0.0f;
				for ( int k = 0; k < n; ++k ) {
					prod += A[k * n + i] * B[j * n + k];
				}

				C[j * n + i] = alpha * prod + beta * C[j * n + i];
			}
		}
	}

	auto end = stopTimer();

	printTime( start, end, loops );
}

void blas(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C,
		int const & loops ) {

	openblas_set_num_threads( 6 );

	auto start = startTimer();

	for ( int l = 0; l < loops; l++ )
		cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, beta, C, n );
	auto end = stopTimer();

	printTime( start, end, loops );
}

void openACC(
		int const n,
		float const alpha,
		float const * A,
		float const * B,
		float const beta,
		float * C,
		int const & loops ) {

	for ( int l = 0; l < loops; l++ ) {

#pragma acc kernels copyin(A[0:(n*n)], B[0:(n*n)]) copyout(C[0:(n*n)])
#pragma acc loop independent
		for ( int i = 0; i < n; ++i ) {
#pragma acc loop independent
			for ( int j = 0; j < n; ++j ) {
				float prod = 0.0f;
#pragma acc loop independent reduction(+:prod)
				for ( int k = 0; k < n; ++k ) {
					prod += A[k * n + i] * B[j * n + k];
				}

				C[j * n + i] = alpha * prod + beta * C[j * n + i];
			}
		}
	}
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
	float *h_C_mp = new float[sizeof(float) * n * n];
	float *h_C_blas = new float[sizeof(float) * n * n];
	float *h_C_acc = new float[sizeof(float) * n * n];
	float *h_C_cublas = new float[sizeof(float) * n * n];
	float *h_C_cuda = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	}

	// Benchmark normal C matrix multiplication
	printf( "Running Normal C: " );
	normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

	// Benchmark and verify OpenMP matrix multiplication
	printf( "Running OpenMP: " );
	openMP( n, alpha, h_A, h_B, beta, h_C_mp, 5 );
	verify( n, h_C, h_C_mp );

	// Benchmark and verify BLAS matrix multiplication
	printf( "Running BLAS: " );
	blas( n, alpha, h_A, h_B, beta, h_C_blas, 500 );
	verify( n, h_C, h_C_blas );

	// Benchmark and verify CUBLAS matrix multiplication
	printf( "Running CUBLAS: " );
	cublas( n, alpha, h_A, h_B, beta, h_C_cublas, 500 );
	verify( n, h_C, h_C_cublas );

	// Benchmark and verify CUDA matrix multiplication
	printf( "Running CUDA: " );
	cuda( n, alpha, h_A, h_B, beta, h_C_cuda, 500 );
	verify( n, h_C, h_C_cuda );

	// Benchmark and verify OpenACC matrix multiplication
	printf( "Running OpenACC: " );
	openACC( n, alpha, h_A, h_B, beta, h_C_acc, 500 );
	verify( n, h_C, h_C_acc );

	// Memory clean up
	delete[] ( h_A );
	delete[] ( h_B );
	delete[] ( h_C );
	delete[] ( h_C_mp );
	delete[] ( h_C_acc );
	delete[] ( h_C_cublas );
	delete[] ( h_C_cuda );
}
