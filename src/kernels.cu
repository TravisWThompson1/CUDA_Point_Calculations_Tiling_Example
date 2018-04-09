// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include "../include/kernels.cuh"

#include <stdio.h>


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
__global__ void d_tiling_calculation(float *p, float *interactions, int NUM_OF_POINTS){

    /////////////////////////// PARAMETERS //////////////////////////////

    // Define blockwidth parameters for ease of use.
    int BLOCKWIDTH = blockDim.x;
    int BLOCKS_PER_ROW = ceilf(NUM_OF_POINTS / (float) BLOCKWIDTH);
    unsigned int INTERACTION_SIZE = NUM_OF_POINTS * NUM_OF_POINTS;
    unsigned int interaction_ij;

    /////////////////////////// THREAD ID ///////////////////////////////

    // Initialize blockId and threadId.
    uint2 threadId, blockId;

    // Calculate blockId in x and y.
    blockId.x = (blockIdx.y * gridDim.x + blockIdx.x) % BLOCKS_PER_ROW;
    blockId.y = (blockIdx.y * gridDim.x + blockIdx.x) / (float) BLOCKS_PER_ROW;

    // Calculate initial threadId in x and y.
    threadId.x = BLOCKWIDTH * blockId.x + threadIdx.x;
    threadId.y = BLOCKWIDTH * blockId.y;

    ////////////////////// MEMORY ACCESS SETUP //////////////////////////

    // Initialize shared memory with size of number of threads per block.
    extern __shared__ float points[];

    // Initialize variables for interacting points.
    float point1, point2;

    // Check for overreach in x direction.
    if ( threadId.x < NUM_OF_POINTS ) {

        // Load this thread's point from global memory to local variable.
        point1 = p[threadId.x];

        // Load secondary point.
        point2 = p[threadId.y + threadIdx.x];

        // Load secondary points from global memory to shared memory.
        points[threadIdx.x] = point2;

        // Sync after memory load.
        __syncthreads();

    /////////////////// POINT-TO-POINT CALCULATIONS /////////////////////

        // Calculate point1 and point2 interactions.
        for(int i = 0; i < BLOCKWIDTH; i++){

            // Determine proper linear index of interactions[i][j].
            interaction_ij = blockId.y * BLOCKWIDTH * NUM_OF_POINTS + i * NUM_OF_POINTS + threadId.x;

            // Load point2 from shared memory.
            point2 = points[i];

            // Check for out of bounds indexing.
            if ( interaction_ij < (NUM_OF_POINTS * NUM_OF_POINTS) ) {

                // No same index calculations
                //if (threadId.x != blockId.y * BLOCKWIDTH + i) {

                // Calculate interaction.
                interactions[blockId.y * BLOCKWIDTH * NUM_OF_POINTS + i * NUM_OF_POINTS +
                             threadId.x] = interaction_calculation(point1, point2);

                //}
            }
        }
    }
}




void tiling_calculation(float *p, float *interactions, int NUM_OF_POINTS) {

    // Initialize device pointers
    float *d_p, *d_interactions;

    // Allocate memory on device.
    cudaMalloc((void **) &d_p, NUM_OF_POINTS * sizeof(float));
    cudaMalloc((void **) &d_interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float));

    // Transfer variables from cpu to gpu.
    cudaMemcpy(d_p, p, NUM_OF_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interactions, interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float), cudaMemcpyHostToDevice);

    // Determine blocksize and gridsize.
    dim3 numThreads(BLOCKSIZE, 1, 1);
    dim3 numBlocks(ceil(NUM_OF_POINTS / (float) numThreads.x), ceil(NUM_OF_POINTS / (float) numThreads.x));

    // Call tiling routine.
    d_tiling_calculation<<< numBlocks, numThreads, BLOCKSIZE*sizeof(float) >>>(d_p, d_interactions, NUM_OF_POINTS);

    // Transfer variables from gpu to cpu.
    cudaMemcpy(interactions, d_interactions, NUM_OF_POINTS * NUM_OF_POINTS * sizeof(float), cudaMemcpyDeviceToHost);

}












