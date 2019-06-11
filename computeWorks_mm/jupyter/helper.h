#pragma once

#include <chrono>		// std::chrono
#include <cmath> 					// std::sqrt, std::fabs
#include <cstdio>					// std::printf
#include <cstdlib>				// std::atoi
#include <stdexcept>			// std::runtime_error
#include <cuda_runtime.h>	// CUDA runtime

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
