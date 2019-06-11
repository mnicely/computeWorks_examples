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
	float *h_C_acc = new float[sizeof(float) * n * n];

	// Initialize values
	for ( int i = 0; i < n * n; i++ ) {
		h_A[i] = 2.0f;
		h_B[i] = 1.0f;
	}

	// Benchmark normal C matrix multiplication
	printf( "Running Normal C: " );
	normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

	// Benchmark and verify OpenACC matrix multiplication
	printf( "Running OpenACC: " );
	openACC( n, alpha, h_A, h_B, beta, h_C_acc, 5 );
	verify( n, h_C, h_C_acc );

	// Memory clean up
	delete[] ( h_A );
	delete[] ( h_B );
	delete[] ( h_C );
	delete[] ( h_C_acc );
}
