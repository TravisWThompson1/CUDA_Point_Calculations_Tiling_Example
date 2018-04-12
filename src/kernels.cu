// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include "../include/kernels.cuh"

#include <stdio.h>
#include <iostream>
#include <cmath>


/**
 * Point to point interaction calculation model. For this example, the interaction is simply adding the two terms.
 * However, it could be a more complicated interaction such as a force or potential.
 * @param point1 Single value of interacting object 1.
 * @param point2 Single value of interacting object 2.
 * @return Returns their calculated interaction.
 */
__device__ float interaction_calculation(float point1, float point2){
    return point1 + point2;
}



/**
 * Device kernel tiling function used to calculate interactions between all points in p (p[i] and p[j], where i!=j).
 * @param p Array of points p[i] to calculation interactions between (p[i] and p[j], where i!=j).
 * @param interactions Matrix of resulting interaction terms between p[i] and p[j], where i!=j.
 * @param NUM_OF_POINTS Number of points in array p.
 */
template <unsigned int BLOCKSIZE>
__global__ void tiling_Kernel(float *p, float *interactions, int NUM_OF_POINTS){

    /////////////////////////// PARAMETERS //////////////////////////////

    // Define block parameters for ease of use.
    unsigned int MATRIX_SIZE = NUM_OF_POINTS * NUM_OF_POINTS;
    unsigned int index_ij;

    /////////////////////////// THREAD ID ///////////////////////////////

    // Calculate the initial thread index in x direction as i in p[i].
    unsigned int i = BLOCKSIZE * blockIdx.x + threadIdx.x;
    // Calculate the initial thread index in y direction as j for the starting value of j in p[j].
    unsigned int j = BLOCKSIZE * blockIdx.y;

    ////////////////////// MEMORY ACCESS SETUP //////////////////////////

    // Initialize shared memory with size of number of threads per block.
    extern __shared__ float points[];

    // Initialize variables for interacting points.
    float point1, point2;

    // Check for overreach in x direction.
    if ( i < NUM_OF_POINTS ) {

        // Load this thread's point (p[i]) from global memory to local variable.
        point1 = p[i];

        // Each thread loads a secondary point from global memory to shared memory.
        points[threadIdx.x] =  p[j + threadIdx.x];

        // Sync after memory load.
        __syncthreads();

    /////////////////// POINT-TO-POINT CALCULATIONS /////////////////////

        #pragma unroll
        // Calculate point1 and point2 interactions.
        for(int iter = 0; iter < BLOCKSIZE; iter++){

            // Determine proper linear index of interactions[i][j].
            index_ij = (j + iter) * NUM_OF_POINTS + i;

            // Load point2 from shared memory.
            point2 = points[iter];

            // Check for out of bounds indexing.
            if ( index_ij < MATRIX_SIZE ) {

                // No same index calculations
                if (i != j) {

                    // Calculate interaction.
                    interactions[index_ij] = interaction_calculation(point1, point2);

                }
            }
        }
    }
}



/**
 * Wrapper function to call CUDA function tiling_Kernel(). Allocates memory on device and transfers array of points to
 * device. Calls kernel function and transfers calculated interactions back to host memory.
 * @param points Array of points used in point-to-point interaction calculations.
 * @param interactions Martix of point[i] and point[j] calculated interactions.
 * @param NUM_OF_POINTS Number of points in array points.
 * @param BLOCKSIZE Blocksize to be used in CUDA kernel.
 */
void tiling_calculation(float *points, float *interactions, int NUM_OF_POINTS, int BLOCKSIZE) {

    // Initialize device pointers
    float *d_points, *d_interactions;

    // Set up CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device.
    cudaMalloc((void **) &d_points, NUM_OF_POINTS * sizeof(float));
    cudaMalloc((void **) &d_interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float));

    // Transfer variables from cpu to gpu.
    cudaMemcpy(d_points, points, NUM_OF_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interactions, interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float), cudaMemcpyHostToDevice);

    // Determine blocksize and gridsize.
    dim3 numThreads(BLOCKSIZE, 1, 1);
    dim3 numBlocks(ceil(NUM_OF_POINTS / (float) numThreads.x), ceil(NUM_OF_POINTS / (float) numThreads.x));


    // Call CUDA timers.
    cudaEventRecord(start);

    // Call CUDA kernel.
    switch( BLOCKSIZE ) {
        case 1:
            tiling_Kernel <1> <<< numBlocks, numThreads, 1*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 2:
            tiling_Kernel <2> <<< numBlocks, numThreads, 2*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 4:
            tiling_Kernel <4> <<< numBlocks, numThreads, 4*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 8:
            tiling_Kernel <8> <<< numBlocks, numThreads, 8*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 16:
            tiling_Kernel <16> <<< numBlocks, numThreads, 16*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 32:
            tiling_Kernel <32> <<< numBlocks, numThreads, 32*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 64:
            tiling_Kernel <64> <<< numBlocks, numThreads, 64*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 128:
            tiling_Kernel <128> <<< numBlocks, numThreads, 128*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 256:
            tiling_Kernel <256> <<< numBlocks, numThreads, 256*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 512:
            tiling_Kernel <512> <<< numBlocks, numThreads, 512*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
        case 1024:
            tiling_Kernel <1024> <<< numBlocks, numThreads, 1024*sizeof(float) >>> (d_points, d_interactions, NUM_OF_POINTS);
            break;
    }

    // Stop CUDA timer.
    cudaEventRecord(stop);

    // End CUDA timers.
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Effective bandwidth calculation. ((2 reads + #BLOCKSIZE writes) * 4 bytes) / 10^9 / time(seconds)
    float effectiveBandwidth = (( 2 + BLOCKSIZE ) * 4 ) / (milliseconds / (float) 1000) / (float) pow(10,9);
    // Effective GLOPS calculation. ((# FLOPS in one thread) * (# of threads total) / 10^9 / time(seconds)
    float effectiveGFLOPS = (( 5 + BLOCKSIZE * 4 ) * (BLOCKSIZE * numBlocks.x * numBlocks.y)) / (milliseconds / (float) 1000) / (float) pow(10,9);

    // Print out results.
    std::cout << "BLOCKSIZE = " << BLOCKSIZE << "   \tTime [ms] = " << milliseconds << "\tEff. Bandwidth = " << effectiveBandwidth <<"\tEff. GFLOPS = " << effectiveGFLOPS << std::endl;

    // Transfer variables from gpu to cpu.
    cudaMemcpy(interactions, d_interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float), cudaMemcpyDeviceToHost);

}












