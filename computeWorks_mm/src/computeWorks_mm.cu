#include <chrono>		// std::chrono
#include <cmath> 		// std::sqrt, std::fabs
#include <cstdio>		// std::printf
#include <cstdlib>		// std::atoi, std::free
#include <functional>	// std::function
#include <stdexcept>	// std::runtime_error
#include <omp.h>		// OpenMP
#include <cublas_v2.h>	// cuBLAS

auto constexpr tPB = 16;

auto benchmark = [](auto fcnPtr, int const & loops) {

	// Warmup run
		fcnPtr();

		// Host timing variables
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		std::chrono::duration<double, std::milli> elapsed_ms;

		start = std::chrono::high_resolution_clock::now();

		for ( int l = 0; l < loops; l++ ) {
			fcnPtr();
		}

		end = std::chrono::high_resolution_clock::now();
		elapsed_ms = end - start;
		std::printf( "%0.2f ms\n", elapsed_ms.count() / loops );

	};

auto benchmarkGPU = [](auto fcnPtr, int const & loops) {

	// Warmup run
		fcnPtr();

		// Create timers
		cudaEvent_t startEvent = nullptr, stopEvent = nullptr;

		cudaEventCreate( &startEvent, cudaEventBlockingSync );
		cudaEventCreate( &stopEvent, cudaEventBlockingSync );

		cudaEventRecord( startEvent );

		for ( int l = 0; l < loops; l++ ) {
			fcnPtr();
		}

		cudaEventRecord( stopEvent );
		cudaEventSynchronize( stopEvent );
		float elapsed_ms;
		cudaEventElapsedTime( &elapsed_ms, startEvent, stopEvent );

		std::printf( "%0.2f ms\n", elapsed_ms / loops );

	};

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

	error_norm = static_cast<float>( std::sqrt(
			static_cast<double>( error_norm ) ) );
	ref_norm =
			static_cast<float>( std::sqrt( static_cast<double>( ref_norm ) ) );

	if ( std::fabs( ref_norm ) < 1e-7 )
		throw std::runtime_error( "!!!! reference norm is 0\n" );

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
		float * C ) {

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

void openMP(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C ) {

	// Request number of threads at runtime
	omp_set_num_threads( 4 );

	// Create parallel region and worksharing
#pragma omp parallel for shared(A, B, C, n)
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

void openACC(
		int const n,
		float const alpha,
		float const * A,
		float const * B,
		float const beta,
		float * C ) {


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

void cublas(
		int const & n,
		float const & alpha,
		float const * A,
		float const * B,
		float const & beta,
		float * C ) {

	// Declare device result pointers
	float *d_A, *d_B, *d_C;

	// Allocate memory on device
	cudaMalloc( (void **) &d_A, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_B, sizeof(float) * n * n );
	cudaMalloc( (void **) &d_C, sizeof(float) * n * n );

	// Copy host memory to device
	cudaMemcpy( d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, B, sizeof(float) * n * n, cudaMemcpyHostToDevice );

	cublasHandle_t handle;
	cublasCreate( &handle );

	cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B,
			n, &beta, d_C, n );

	cudaDeviceSynchronize();

	// Copy results from device to host
	cudaMemcpy( C, d_C, sizeof(float) * n * n, cudaMemcpyDeviceToHost );

	cublasDestroy( handle );
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
		float * C ) {

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
	dim3 blocksPerGrid( ( n + tPB - 1 ) / tPB, ( n + tPB - 1 ) / tPB );
	dim3 threadsPerBlock( tPB, tPB );

	cudaKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_A, d_B, d_C);
	cudaDeviceSynchronize();

	// Copy results from device to host
	cudaMemcpy( C, d_C, sizeof(float) * n * n, cudaMemcpyDeviceToHost );
}

int main( int argc, char** argv ) {

	int n;
	if ( argc < 2 ) {
		n = 1024;
		printf("No input given.\n");
		printf("Running with N = %d\n\n", n);
	} else {
		n = std::atoi( argv[1] );
		printf("Running with N = %d\n\n", n);
	}

	float alpha = 1.0f;
	float beta = 0.0f;

	// Declare host variables
	float *h_A = new float[sizeof(float) * n * n];
	float *h_B = new float[sizeof(float) * n * n];
	float *h_C = new float[sizeof(float) * n * n];
	float *h_C_mp = new float[sizeof(float) * n * n];
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
	benchmark( [&] {normalC( n, alpha, h_A, h_B, beta, h_C );}, 2 );

	// Benchmark and verify OpenMP matrix multiplication
	printf( "Running OpenMP: " );
	benchmark( [&] {openMP( n, alpha, h_A, h_B, beta, h_C_mp );}, 10 );
	verify( n, h_C, h_C_mp );

	// Benchmark and verify OpenACC matrix multiplication
	printf( "Running OpenACC: " );
	benchmark( [&] {openACC( n, alpha, h_A, h_B, beta, h_C_acc );}, 10 );
	verify( n, h_C, h_C_acc );

	// Benchmark and verify CUBLAS matrix multiplication
	printf( "Running CUBLAS: " );
	benchmarkGPU( [&] {cublas( n, alpha, h_A, h_B, beta, h_C_cublas );}, 10 );
	verify( n, h_C, h_C_cublas );

	// Benchmark and verify CUDA matrix multiplication
	printf( "Running CUDA: " );
	benchmarkGPU( [&] {cuda( n, alpha, h_A, h_B, beta, h_C_cuda );}, 10 );
	verify( n, h_C, h_C_cuda );

	// Memory clean up
	delete ( h_A );
	delete ( h_B );
	delete ( h_C );
	delete ( h_C_mp );
	delete ( h_C_acc );
	delete ( h_C_cublas );
	delete ( h_C_cuda );
}
