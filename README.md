# N-Body Interactions

## Overview
N-body interacting systems are commonly used in describing the interactions between N particles via the gravtiational and/or electromagnetic potentials. In these systems, each particle is influenced by every other particle in the system; usually by an inverse law <a href="https://www.codecogs.com/eqnedit.php?latex=r^{-a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;r^{-a}" title="&space;r^{-a}" /></a>, where a=1,2,...N depending on the value observable being calculated (potential, force, etc.) and <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is a constant. All of the calculations are dependent on the radial distance between two particles <a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}" title="p_{i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=p_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}" title="p_{j}" /></a>, where their distance is <a href="https://www.codecogs.com/eqnedit.php?latex=r_{ij}&space;=&space;|r_{j}&space;-&space;r_{j}|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{ij}&space;=&space;|r_{j}&space;-&space;r_{j}|" title="r_{ij} = |r_{j} - r_{j}|" /></a>.

## CUDA Tiling Method

One way to efficiently calculate the interaction between each point is to build a a matrix of interactions <a href="https://www.codecogs.com/eqnedit.php?latex=V_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V_{ij}" title="V_{ij}" /></a> and calculate each interaction individually. The so called tiliing method commonly used in CUDA is one of the most efficient ways to populate the interactions matrix. Here is a small example of using tiling to transpose a matrix: https://www.youtube.com/watch?v=pP-1nJEp4Qc

We will use this method to have a block of threads read in data for points i and j in a coalesced manner to be saved to the efficient shared memory. From here, intereactions can be calculated between points i and j and outputted efficiently to the resultant matrix. 

## CUDA Kernel Code Breakdown

### Declaration
```
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
    unsigned int interaction_ij;
```

First, we initialize the function with the ```__global__``` keyword, allowing this function to be called on the GPU from the CPU. Next, it is helpful to intialize a few useful variables that are esentially parameters of the function, such as the ```BLOCKWIDTH``` and how many blocks will be along one dimension ```BLOCKS_PER_ROW```. These variables are not necessary, but included for convenience. 




### Thread ID and Block ID
```
    /////////////////////////// THREAD ID ///////////////////////////////

    // Initialize blockId and threadId.
    uint2 threadId, blockId;

    // Calculate blockId in x and y.
    blockId.x = (blockIdx.y * gridDim.x + blockIdx.x) % BLOCKS_PER_ROW;
    blockId.y = (blockIdx.y * gridDim.x + blockIdx.x) / (float) BLOCKS_PER_ROW;

    // Calculate initial threadId in x and y.
    threadId.x = BLOCKWIDTH * blockId.x + threadIdx.x;
    threadId.y = BLOCKWIDTH * blockId.y;
```

Next, the first step that is found in every CUDA kernel is determining the thread ID. It is useful to calculate values such as the thread number and block number in the x,y, and z directions, if more than one dimension is used. For this case, we use two dimensions so we calculated both the x and y values.



### Coalesced Memory Accesses

```
   ////////////////////// MEMORY ACCESS SETUP //////////////////////////

    // Initialize shared memory with size of number of threads per block.
    extern __shared__ float points[];

    // Initialize variables for interacting points.
    float point1, point2;

    // Check for overreach in x direction.
    if ( threadId.x < NUM_OF_POINTS ) {

        // Load this thread's point from global memory to local variable.
        point1 = p[threadId.x];

        // Load secondary points from global memory to shared memory.
        points[threadIdx.x] =  p[threadId.y + threadIdx.x];

        // Sync after memory load.
        __syncthreads();
```

Now, we discuss the coalesced memory accesses that is so critical in tiling calculations. The variable (```points[]```) is declared as a shared memory variable that is accessable to all threads in a block. The size of ```points[]``` is declared externally as the number of threads in a block in the CUDA kernel call. 

Before reading the point data, we ensure that all of our threads are within the bounds of ```p[]``` with a simple ```if``` conditional statement. The threads that pass this condition are then allowed to read points i and j. Each thread will read its <a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}" title="p_{i}" /></a> from global memory to ```point1```. Next, each thread will read one of the <a href="https://www.codecogs.com/eqnedit.php?latex=p_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}" title="p_{j}" /></a> that will be calculated in an interaction with the other points in the tile. Each thread reads a <a href="https://www.codecogs.com/eqnedit.php?latex=p_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}" title="p_{j}" /></a> and writes it into shared memory so that each thread can access it later at a faster speed than a regular global memory read. The memory accesses resemble the following schematic if ```blocksize=4```, ```i=0:3```, and ```j=8:11```:

```
p[N] = |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |  11 |  12 | ... |  N-1  |
          ^     ^     ^     ^                             ^     ^     ^     ^
          |     |     |     |                             |     |     |     |                           
          t0    t1    t2    t3                            t0    t1    t2    t3                   
                 point1                                   |     |     |     |                     
                                                          |     |     |     |                  
                                                          v     v     v     v
                                                          __shared__ points[4] 
```
```
_____________________________________________________________________________________
       |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |  11 | ...
|  0  | 
|  1  |
|  2  |
|  3  |
|  4  |   t0    t1    t2    t3
|  5  |   |     |     |     |  
|  6  |   v     v     v     v 
|  7  |   
|  8  |  {  }  {  }  {  }  {  }    <-----|
|  9  |  {  }  {  }  {  }  {  }    <-----|------------ __shared__ points[4] 
|  10 |  {  }  {  }  {  }  {  }    <-----|
|  11 |  {  }  {  }  {  }  {  }    <-----|
|  12 |
|  13 |
| ... |
| N-1 |
_____________________________________________________________________________________
```

















## CUDA Kernel Code

```
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

        // Load secondary points from global memory to shared memory.
        points[threadIdx.x] =  p[threadId.y + threadIdx.x];

        // Sync after memory load.
        __syncthreads();

    /////////////////// POINT-TO-POINT CALCULATIONS /////////////////////

        #pragma unroll
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
```






