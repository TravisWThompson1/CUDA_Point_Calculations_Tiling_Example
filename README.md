# N-Body Interactions

## Overview
N-body interacting systems are commonly used in describing the interactions between N particles via the gravtiational and/or electromagnetic potentials. In these systems, each particle is influenced by every other particle in the system; usually by an inverse law <a href="https://www.codecogs.com/eqnedit.php?latex=r^{-a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;r^{-a}" title="&space;r^{-a}" /></a>, where a=1,2,...N depending on the value observable being calculated (potential, force, etc.) and <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is a constant. All of the calculations are dependent on the radial distance between two particles <a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{i}" title="p_{i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}" title="p_{j}" /></a>, where their distance is <a href="https://www.codecogs.com/eqnedit.php?latex=r_{ij}&space;=&space;|r_{j}&space;-&space;r_{j}|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{ij}&space;=&space;|r_{j}&space;-&space;r_{j}|" title="r_{ij} = |r_{j} - r_{j}|" /></a>.

## CUDA Tiling Method

One way to efficiently calculate the interaction between each point is to build a a matrix of interactions <a href="https://www.codecogs.com/eqnedit.php?latex=V_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V_{ij}" title="V_{ij}" /></a> and calculate each interaction individually. The so called tiliing method commonly used in CUDA is one of the most efficient ways to populate the interactions matrix. Here is a small example of using tiling to transpose a matrix: https://www.youtube.com/watch?v=pP-1nJEp4Qc

We will use this method to have a block of threads read in data for points i and j in a coalesced manner to be saved to the efficient shared memory. From here, intereactions can be calculated between points i and j and outputted efficiently to the resultant matrix. 

## CUDA Kernel Breakdown













