// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

#define SIZE 1024 
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE) 
#define TILE_SIZE 16
#define HALO 2 

// Sets up the heatmap
void Ped::Model::setupHeatmapCuda()
{
    // We only need to allocate blurred_heatmap for host as only host will use that to show on screen
    // GPU takes care of rest, scaled and regular heatmap.
    int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
    blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
    for (int i = 0; i < SCALED_SIZE; i++) {
        blurred_heatmap[i] = bhm + SCALED_SIZE*i;
    }
	// Cuda version setup.
	cudaMalloc(&d_heatmap, SIZE * SIZE * sizeof(int));
	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));

	cudaMalloc(&d_agentX, agents.size() * sizeof(int));
	cudaMalloc(&d_agentY, agents.size() * sizeof(int));

	agentX_h = new int[agents.size()];
	agentY_h = new int[agents.size()];
}

__global__ void fadeHeatMap(int* heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * SIZE + x;


    if (x < SIZE && y < SIZE)
        heatmap[idx] = (int)roundf(heatmap[idx] * 0.80f);

}

__global__ void incrementHeatMap(int* heatmap, int* agentX, int* agentY, int numAgents) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numAgents) {
		return;
	}

	int x = agentX[i];
	int y = agentY[i];

	if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
		atomicAdd(&heatmap[y * SIZE + x], 40);
	}
}

__global__ void scaleHeatMap(int *heatmap, int *scaledHeatMap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Check bounds against the 5120x5120 grid
    if (x < SCALED_SIZE && y < SCALED_SIZE) {
        // Map the thread's coordinate back to the 1024x1024 original grid
        int origX = x / CELLSIZE;
        int origY = y / CELLSIZE;
        
        // Read the value from the original 1024x1024 heatmap
        int val = heatmap[origY * SIZE + origX];
        
        // Cap the heat value at 255 for correct ARGB color rendering
        val = val > 255 ? 255 : val;
        
        // Write exactly ONCE per thread (Fully coalesced memory write!)
        scaledHeatMap[y * SCALED_SIZE + x] = val;
    } 
}

// STEP 4: BLUR KERNEL
__global__ void blurHeatMap(const int *scaled, int *blurred) {
    __shared__ int tile[TILE_SIZE + 2 * HALO][TILE_SIZE + 2 * HALO];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x = blockIdx.x * TILE_SIZE + tx - HALO;
    int y = blockIdx.y * TILE_SIZE + ty - HALO;

    int loadX = max(0, min(x, SCALED_SIZE - 1));
    int loadY = max(0, min(y, SCALED_SIZE - 1));
    tile[ty][tx] = scaled[loadY * SCALED_SIZE + loadX];
    
    __syncthreads(); 

    if (tx >= HALO && tx < TILE_SIZE + HALO && ty >= HALO && ty < TILE_SIZE + HALO) {
        
        
        if (x >= 2 && x < SCALED_SIZE - 2 && y >= 2 && y < SCALED_SIZE - 2) {
            
            const int w[5][5] = {
                { 1, 4, 7, 4, 1 },
                { 4, 16, 26, 16, 4 },
                { 7, 26, 41, 26, 7 },
                { 4, 16, 26, 16, 4 },
                { 1, 4, 7, 4, 1 }
            };
            
            int sum = 0;
            // loop indices (-2 to 2)
            for (int k = -2; k < 3; k++) {
                for (int l = -2; l < 3; l++) {
                    sum += w[2 + k][2 + l] * tile[ty + k][tx + l];
                }
            }
            
            //division and bitshift
            int value = sum / 273; 
            blurred[y * SCALED_SIZE + x] = 0x00FF0000 | value << 24;
        }
    }
}

bool eventsCreated = false;
cudaEvent_t startFade, stopFade, startInc, stopInc, startScale, stopScale, startBlur, stopBlur;
cudaEvent_t startTotal, stopTotal;

void Ped::Model::updateHeatmapCuda() {
    // 1. CHRONO: True Total Time 
    auto tickStart = std::chrono::high_resolution_clock::now();

    int numAgents = agents.size();

    // Create separate events for each step
    if (!eventsCreated) {
        cudaEventCreate(&startFade); cudaEventCreate(&stopFade);
        cudaEventCreate(&startInc); cudaEventCreate(&stopInc);
        cudaEventCreate(&startScale); cudaEventCreate(&stopScale);
        cudaEventCreate(&startBlur); cudaEventCreate(&stopBlur);
        cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal);
        eventsCreated = true;
    }    
    // Prepare data for GPU
    for(int i = 0; i < numAgents; i++) {
        agentX_h[i] = agents[i]->getDesiredX();
        agentY_h[i] = agents[i]->getDesiredY();
    }

    cudaEventRecord(startTotal);

    cudaMemcpyAsync(d_agentX, agentX_h, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_agentY, agentY_h, numAgents * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 grid(SIZE/16, SIZE/16);
    dim3 blurGrid(SCALED_SIZE/16, SCALED_SIZE/16);

	cudaEventRecord(startFade);
    fadeHeatMap<<<grid, threads>>>(d_heatmap);
	cudaEventRecord(stopFade);

	cudaEventRecord(startInc);
    incrementHeatMap<<<(numAgents+127)/128, 128>>>(d_heatmap, d_agentX, d_agentY, numAgents);
	cudaEventRecord(stopInc);
	
	cudaEventRecord(startScale);
	scaleHeatMap<<<grid, threads>>>(d_heatmap, d_scaled_heatmap);
	cudaEventRecord(stopScale);
    
	cudaEventRecord(startBlur);
    blurHeatMap<<<blurGrid, threads>>>(d_scaled_heatmap, d_blurred_heatmap);
	cudaEventRecord(stopBlur);

    // 3. CHRONO: Measure pure CPU time
    auto cpuStart = std::chrono::high_resolution_clock::now();
     for (int i = 0; i < numAgents; i++) {
        agents[i]-> computeNextDesiredPosition();
		move(agents[i]);
    }
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpuStop - cpuStart;

    // This cudaMemcpy is implicitly synchronous, so it will wait for the GPU to finish
    cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stopTotal);

    cudaEventSynchronize(stopTotal);
    
    // Calculate Final Tick Time
    auto tickStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTickTime = tickStop - tickStart;

    // Calculate all times
    float gpuTotalTime, msFade, msInc, msScale, msBlur;
    cudaEventElapsedTime(&gpuTotalTime, startTotal, stopTotal);
    cudaEventElapsedTime(&msFade, startFade, stopFade);
    cudaEventElapsedTime(&msInc, startInc, stopInc);
    cudaEventElapsedTime(&msScale, startScale, stopScale);
    cudaEventElapsedTime(&msBlur, startBlur, stopBlur);

    // Shows GPU and CPU run concurrently, heterogenous computation!
    /*
    static int frame = 0;
    if (frame++ % 50 == 0) {
        printf("GPU: %f ms | CPU: %f ms | Total Tick: %f ms\n", 
               gpuTotalTime, cpuTime.count(), totalTickTime.count());
    }
    */
    
    // Print raw CSV data for EVERY frame
    // Format: Frame, Fade, Inc, Scale, Blur, GPUTotal, CPUTotal, TickTotal
    static int frame = 0;
    printf("%d,%f,%f,%f,%f,%f,%f,%f\n", 
           frame++, msFade, msInc, msScale, msBlur, gpuTotalTime, cpuTime.count(), totalTickTime.count());
    
}
