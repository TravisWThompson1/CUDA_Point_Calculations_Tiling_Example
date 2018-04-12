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
 * Device kernel tiling function used to calculate interactions between all points in p (p[i] and
 * p[j], where i!=j).
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

```

First, we initialize the function with the ```__global__``` keyword, allowing this function to be called on the GPU from the CPU. The function takes in a template parameter that is the blocksize (number of threads per block), which will be used may be used by the compiler in further optimizations. Having the blocksize as a known constant value can be used to improve efficiency in later steps that are not included here. Next, it is helpful to intialize a few useful variables such as the total matrix size so that we only compute it once, rather than many times in a for loop. 




### Thread ID and Block ID
```
    /////////////////////////// THREAD ID ///////////////////////////////

    // Calculate the initial thread index in x direction as i in p[i].
    unsigned int i = BLOCKSIZE * blockIdx.x + threadIdx.x;
    // Calculate the initial thread index in y direction as j for the starting value of j in p[j].
    unsigned int j = BLOCKSIZE * blockIdx.y;
```

Next, the first step that is found in every CUDA kernel is determining the thread ID. It is useful to calculate values such as the thread number and block number in the x,y, and z directions, if more than one dimension is used. For this case, we use two dimensions so we calculated both the x and y values of the thread ID's, which are in fact our indices ```i``` and ```j``` that were mentioned earlier.



### Coalesced Memory Accesses

```
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


### Point-to-Point Interaction

```
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
```

Lastly, each thread calculates a few point-to-point interactions, specifically, the number of calculations is the number of threads in a block. In each iteration, the new output <a href="https://www.codecogs.com/eqnedit.php?latex=V_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V_{ij}" title="V_{ij}" /></a> index ```index_ij``` is calculated (these indices can become a little complicated). Next, the appropiate ```point2``` is loaded from the shared array ```points```. Some point-to-point calculations forbid a self-interaction (where ```i==j```). This specification is included but can be commented out in the code above allowing self-interactions, but this is typically unusual. Finally, the interaction is calculated between ```point1``` and ```point2``` in the function ```interaction_calculation()``` that is not shown here. This is the function that is particular to the interaction between point i and j. For the example in this code, we use a simple interaction as a proof of concept, where ```interaction_calculation()``` simply returns the sum of ```point1``` and ```point2```. 

A schematic of the calculations performed by one block looks like the following:
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



### Launching the CUDA Kernel

```
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
```

Finally, launching the CUDA kernel is shown in the code above, where a switch statement is used to call the ```tiling_Kernel()``` function with a specific blocksize. Launching the kernel in a switch statement like this is not necessary. However, in using a switch statement in this way, the template argument (```BLOCKSIZE```) is a constant. Therefore the kernel can be further optimized for certain blocksize inputs, such as if the blocksize is greater than or less than the actual size of the working data. This optimization is not added into the code here, but it is a good exercise to include this regardless (although it does increase compile time).


### What BLOCKSIZE Should Be Used?

This is always a question that should be asked for every CUDA kernel. For this code, a test case was written to deteremine the fastest blocksize. The result turned out that a blocksize of ```128``` yielded the fastest results with the highest GFLOPS. Feel free to compile and run the test yourself.



## CUDA Kernel Code in kernels.cu

```
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
```






