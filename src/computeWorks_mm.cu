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
 * This sample compares performance between serial matrix multiplication the
 * following parallel alternatives
 *
 * 1. OpenMP
 * 2. BLAS
 * 3. cuBLAS
 * 4. CUDA
 * 5. OpenACC
 */

#include <cblas.h>              // OpenBLAS - PGI
#include <chrono>               // std::chrono
#include <cmath>                // std::sqrt, std::fabs
#include <cooperative_groups.h> // Cooperative Groups
#include <cstdio>               // std::printf
#include <cstdlib>              // std::atoi
#include <cublas_v2.h>          // cuBLAS
#include <omp.h>                // OpenMP - PGI
#include <stdexcept>            // std::runtime_error

// Thread block size
#define BLOCK_SIZE 32

auto getTimeCPU( ) {
    return ( std::chrono::high_resolution_clock::now( ) );
} // getTimeCPU

auto startGPUTimer( ) {
    cudaEvent_t startEvent = nullptr;
    cudaEventCreate( &startEvent, cudaEventBlockingSync );
    cudaEventRecord( startEvent );
    return ( startEvent );
} // startGPUTimer

auto stopGPUTimer( ) {
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate( &stopEvent, cudaEventBlockingSync );
    cudaEventRecord( stopEvent );
    cudaEventSynchronize( stopEvent );
    return ( stopEvent );
} // stopGPUTimer

template<typename T>
void printCPUTime( T start, T stop, int const &loops ) {
    std::chrono::duration<double, std::milli> elapsed_ms;
    elapsed_ms = stop - start;
    std::printf( "%0.2f ms\n", elapsed_ms.count( ) / loops );
} // printCPUTime

template<typename T>
void printGPUTime( T startEvent, T stopEvent, int const &loops ) {
    float elapsed_ms;
    cudaEventElapsedTime( &elapsed_ms, startEvent, stopEvent );
    std::printf( "%0.2f ms\n", elapsed_ms / loops );
} // printGPUTime

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

    auto start = getTimeCPU( );

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

    auto end = getTimeCPU( );

    printCPUTime( start, end, loops );
} // normalC

void openMP( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

    // Request number of threads at runtime
    omp_set_num_threads( omp_get_max_threads( ) );

    auto start = getTimeCPU( );

    for ( int l = 0; l < loops; l++ ) {

        // Create parallel region and worksharing
#pragma omp parallel for shared( A, B, C, n ) schedule( static )
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

    auto end = getTimeCPU( );

    printCPUTime( start, end, loops );
} // openMP

void blas( int const &  n,
           float const &alpha,
           float const *A,
           float const *B,
           float const &beta,
           float *      C,
           int const &  loops ) {

    openblas_set_num_threads( 6 );

    auto start = getTimeCPU( );

    for ( int l = 0; l < loops; l++ )
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, beta, C, n );
    auto end = getTimeCPU( );

    printCPUTime( start, end, loops );
} // blas

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
                } // k
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            } // j
        }     // i
    }         // loops
} // openACC

void cublas( int const &  n,
             float const &alpha,
             float const *A,
             float const *B,
             float const &beta,
             float *      C,
             int const &  loops ) {

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

    auto startEvent = startGPUTimer( );
    for ( int l = 0; l < loops; l++ )
        cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n );
    auto stopEvent = stopGPUTimer( );

    cudaDeviceSynchronize( );

    // Copy results from device to host
    cudaMemcpy( C, d_C, sizeof( float ) * n * n, cudaMemcpyDeviceToHost );

    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_C );
    cublasDestroy( handle );

    printGPUTime( startEvent, stopEvent, loops );
} // cublas

__global__ void
MatMulKernel( int const n, float const *__restrict__ A, float const *__restrict__ B, float *__restrict__ C ) {

    auto const block = cooperative_groups::this_thread_block( );

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread computes one element of Csub by accumulating results into Cvalue
    float Cvalue = 0.0f;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float *Csub = &C[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for ( int m = 0; m < ( n / BLOCK_SIZE ); ++m ) {

        // Get sub-matrix Asub of A
        float const *Asub = &A[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * m];

        // Get sub-matrix Bsub of B
        float const *Bsub = &B[n * BLOCK_SIZE * m + BLOCK_SIZE * blockCol];

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row * n + col];
        Bs[row][col] = Bsub[row * n + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        block.sync( );

        // Multiply Asub and Bsub together
        for ( int e = 0; e < BLOCK_SIZE; ++e )
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        block.sync( );
    } // m

    // Write Csub to device memory each thread writes one element
    Csub[row * n + col] = Cvalue;
} // MatMulKernel

void cuda( int const &  n,
           float const &alpha,
           float const *A,
           float const *B,
           float const &beta,
           float *      C,
           int const &  loops ) {

    // Declare device result pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc( ( void ** )&d_A, sizeof( float ) * n * n );
    cudaMalloc( ( void ** )&d_B, sizeof( float ) * n * n );
    cudaMalloc( ( void ** )&d_C, sizeof( float ) * n * n );

    // Copy host memory to device
    cudaMemcpy( d_A, A, sizeof( float ) * n * n, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, B, sizeof( float ) * n * n, cudaMemcpyHostToDevice );

    // setup the dimensions
    dim3 blocksPerGrid( ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE, ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE );
    dim3 threadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );

    auto startEvent = startGPUTimer( );
    for ( int l = 0; l < loops; l++ )
        MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>( n, d_A, d_B, d_C );
    auto stopEvent = stopGPUTimer( );

    cudaDeviceSynchronize( );

    // Copy results from device to host
    cudaMemcpy( C, d_C, sizeof( float ) * n * n, cudaMemcpyDeviceToHost );

    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_C );

    printGPUTime( startEvent, stopEvent, loops );
} // cuda

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
    float *h_C_mp     = new float[sizeof( float ) * n * n];
    float *h_C_blas   = new float[sizeof( float ) * n * n];
    float *h_C_acc    = new float[sizeof( float ) * n * n];
    float *h_C_cublas = new float[sizeof( float ) * n * n];
    float *h_C_cuda   = new float[sizeof( float ) * n * n];

    // Initialize values
    for ( int i = 0; i < n * n; i++ ) {
        h_A[i] = 2.0f;
        h_B[i] = 1.0f;
    } // i

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
    delete[]( h_A );
    delete[]( h_B );
    delete[]( h_C );
    delete[]( h_C_mp );
    delete[]( h_C_acc );
    delete[]( h_C_cublas );
    delete[]( h_C_cuda );
} // main