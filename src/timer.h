/*
 * timer.h
 *
 *  Created on: Jul 20, 2019
 *      Author: mnicely
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

class Timer {

  public:
    // CPU Timer
    void startCPUTimer( ) {
        start = std::chrono::high_resolution_clock::now( );
    }  // startCPUTimer

    void stopCPUTimer( ) {
        stop = std::chrono::high_resolution_clock::now( );
    }  // stopCPUTimer

    void printCPUTime( ) {
        elapsed_cpu_ms = stop - start;
        std::printf( "%0.2f ms ", elapsed_cpu_ms.count( ) );
    }  // printCPUTime

    void printCPUTime( int const &loops ) {
        elapsed_cpu_ms = stop - start;
        std::printf( "%0.2f ms ", elapsed_cpu_ms.count( ) / loops );
    }  // printCPUTime

    void stopAndPrintCPU( ) {
        stopCPUTimer( );
        printCPUTime( );
    }

    void stopAndPrintCPU( int const &loops ) {
        stopCPUTimer( );
        printCPUTime( loops );
    }

    // GPU Timer
    void startGPUTimer( ) {
        cudaEventCreate( &startEvent, cudaEventBlockingSync );
        cudaEventRecord( startEvent );
    }  // startGPUTimer

    void stopGPUTimer( ) {
        cudaEventCreate( &stopEvent, cudaEventBlockingSync );
        cudaEventRecord( stopEvent );
        cudaEventSynchronize( stopEvent );
    }  // stopGPUTimer

    void printGPUTime( ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms ", elapsed_gpu_ms );
    }  // printGPUTime

    void printGPUTime( int const &loops ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms ", elapsed_gpu_ms / loops );
    }  // printGPUTime

    void stopAndPrintGPU( ) {
        stopGPUTimer( );
        printGPUTime( );
    }

    void stopAndPrintGPU( int const &loops ) {
        stopGPUTimer( );
        printGPUTime( loops );
    }

  private:
    std::chrono::high_resolution_clock::time_point start {};
    std::chrono::high_resolution_clock::time_point stop {};
    std::chrono::duration<double, std::milli>      elapsed_cpu_ms {};

    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};
};

#endif /* TIMER_H_ */
