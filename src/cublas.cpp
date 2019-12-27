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
 * the cuBLAS API.
 */

/*
 * nvcc -O2 -lcublas cublas.cpp -o cublas -run
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

// Verify input matrices
void verify( int const &n, float const *C_ref, float const *C_test ) {

    // Check result against reference
    float error_norm = 0.0f;
    float ref_norm   = 0.0f;
    float diff       = 0.0f;

    for ( int i = 0; i < n * n; i++ ) {
        diff = C_test[i] - C_ref[i];
        error_norm += diff * diff;
        ref_norm += C_test[i] * C_test[i];
    } // i

    error_norm = static_cast<float>( std::sqrt( static_cast<double>( error_norm ) ) );
    ref_norm   = static_cast<float>( std::sqrt( static_cast<double>( ref_norm ) ) );

    if ( std::fabs( ref_norm ) < 1e-7 )
        std::printf( "Reference norm is 0.\t" );

    if ( error_norm / ref_norm < 1e-5f )
        std::printf( "Test passed.\n" );
    else
        std::printf( "Test failed.\n" );
} // verify

void normalC( int const &  n,
              float const &alpha,
              float const *A,
              float const *B,
              float const &beta,
              float *      C,
              int const &  loops ) {

    auto start = std::chrono::high_resolution_clock::now( );

    for ( int l = 0; l < loops; l++ ) {
        for ( int i = 0; i < n; ++i ) {
            for ( int j = 0; j < n; ++j ) {
                float prod = 0.0f;
                for ( int k = 0; k < n; ++k ) {
                    prod += A[k * n + i] * B[j * n + k];
                } // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            } // j
        }     // i
    }         // loops

    auto                                      stop       = std::chrono::high_resolution_clock::now( );
    std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
    std::printf( "%0.2f ms\n", elapsed_ms.count( ) / loops );
} // normalC

void cublas( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

    // Declare timer variables
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent  = nullptr;
    cudaEventCreate( &startEvent, cudaEventBlockingSync );
    cudaEventCreate( &stopEvent, cudaEventBlockingSync );

    // Declare device pointers and cublas handle
    float *        d_A, *d_B, *d_C;
    cublasHandle_t handle;
    cublasCreate( &handle );

    // Allocate memory on device
    cudaMalloc( ( void ** )&d_A, sizeof( float ) * n * n );
    cudaMalloc( ( void ** )&d_B, sizeof( float ) * n * n );
    cudaMalloc( ( void ** )&d_C, sizeof( float ) * n * n );

    // Copy host memory to device
    cudaMemcpy( d_A, A, sizeof( float ) * n * n, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, B, sizeof( float ) * n * n, cudaMemcpyHostToDevice );

    cudaEventRecord( startEvent );

    for ( int l = 0; l < loops; l++ )
        cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n );

    cudaDeviceSynchronize( );
    cudaEventSynchronize( stopEvent );

    // Copy results from device to host
    cudaMemcpy( C, d_C, sizeof( float ) * n * n, cudaMemcpyDeviceToHost );

    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_C );
    cublasDestroy( handle );

    float elapsed_ms;
    cudaEventElapsedTime( &elapsed_ms, startEvent, stopEvent );
    std::printf( "%0.2f ms\n", elapsed_ms / loops );
} // cublas

int main( int argc, char **argv ) {

    int n = 1024;
    if ( argc > 1 )
        n = std::atoi( argv[1] );
    printf( "Running with N = %d\n\n", n );

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Declare host variables
    float *h_A        = new float[sizeof( float ) * n * n];
    float *h_B        = new float[sizeof( float ) * n * n];
    float *h_C        = new float[sizeof( float ) * n * n];
    float *h_C_cublas = new float[sizeof( float ) * n * n];

    // Initialize values
    for ( int i = 0; i < n * n; i++ ) {
        h_A[i] = 2.0f;
        h_B[i] = 1.0f;
    } // i

    // Benchmark normal C matrix multiplication
    printf( "Running Normal C: " );
    normalC( n, alpha, h_A, h_B, beta, h_C, 2 );

    // Benchmark and verify CUBLAS matrix multiplication
    printf( "Running CUBLAS: " );
    cublas( n, alpha, h_A, h_B, beta, h_C_cublas, 500 );
    verify( n, h_C, h_C_cublas );

    // Memory clean up
    delete[]( h_A );
    delete[]( h_B );
    delete[]( h_C );
    delete[]( h_C_cublas );
} // main
