#include <iostream>
#include "include/kernels.cuh"



int main(){

    // Initialize tile size and number of tiles.
    int NUM_OF_POINTS = 40;

    // Initialize array of points p_i
    float p[NUM_OF_POINTS];

    // Initialize matrix r_ij (of size NUM_OF_POINTS^2) that represents the interaction calculated between r_i and r_j.
    float r[NUM_OF_POINTS * NUM_OF_POINTS];

    // Initialize each point with a value.
    for (int i = 0; i < NUM_OF_POINTS; i++) {
        p[i] = float(i);
        for (int j = 0; j < NUM_OF_POINTS; j++) {
            r[i * NUM_OF_POINTS + j] = 0.0;
        }
    }


    // Run tiling calculations on GPU.
    tiling_calculation(p, r, NUM_OF_POINTS);

    // Print output.
    for (int i = 0; i < NUM_OF_POINTS; i++){
        for (int j = 0; j < NUM_OF_POINTS; j++){
            std::cout << r[i * NUM_OF_POINTS + j] << " ";
        }
        std::cout << "\n";
    }








    return 0;
}






























