#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cstdio>
#include <chrono>
// 1. EXACT CONSTANTS MATCHING ped_model.h
#define SIZE 1024 
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE) 
#define TILE_SIZE 16
#define HALO 2 

// CUDA timing
static cudaEvent_t startCreate, stopCreate, startScale, stopScale, startBlur, stopBlur;
static cudaEvent_t startTotal, stopTotal;
static bool eventsCreated = false;
static int frameCount = 0;

// STEP 1: FADE KERNEL
__global__ void fadeKernel(int *heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < SIZE && y < SIZE) {
        int idx = y * SIZE + x;
        // Use roundf() to match CPU exactly, use f as in float 32 bit, as GPU is heavily optimized
        // for 32 bit operations 
        heatmap[idx] = (int)roundf(heatmap[idx] * 0.80f);
    }
}

// STEP 2: ADD HEAT KERNEL
__global__ void addHeatKernel(int *heatmap, const int *agentX, const int *agentY, int numAgents) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents) {
        int x = agentX[i];
        int y = agentY[i];
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            atomicAdd(&heatmap[y * SIZE + x], 40);
        }
    }
}

// STEP 3: CAP AND SCALE KERNEL
// Instead of 1 thread per scaled pixel, we use 1 thread per ORIGINAL cell.
// It caps the value, saves it, and writes to a 5x5 block.
/*
__global__ void scaleKernel(int *heatmap, int *scaled) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < SIZE && y < SIZE) {
        int idx = y * SIZE + x;
        int val = heatmap[idx];
        
        // Cap at 255 and save back to the original array 
        if (val > 255) {
            val = 255;
            heatmap[idx] = val;
        }
        
        // Scale: Write the value to a 5x5 area in the scaled array
        int startX = x * CELLSIZE;
        int startY = y * CELLSIZE;
        for (int cy = 0; cy < CELLSIZE; cy++) {
            for (int cx = 0; cx < CELLSIZE; cx++) {
                scaled[(startY + cy) * SCALED_SIZE + (startX + cx)] = val;
            }
        }
    }
}
*/
// STEP 3: CAP AND SCALE KERNEL (CORRECTED)
// 1 thread per SCALED pixel (5120x5120 grid)
__global__ void scaleKernel(const int *heatmap, int *scaled) {
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
        scaled[y * SCALED_SIZE + x] = val;
    }
}

// STEP 4: BLUR KERNEL
__global__ void blurKernel(const int *scaled, int *blurred) {
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

// Global device pointers hidden safely inside the CUDA file
static int* d_heatmap = nullptr;
static int* d_scaled = nullptr;
static int* d_blurred = nullptr;
static int* d_agentX = nullptr;
static int* d_agentY = nullptr;
static int current_num_agents = 0;

// Wrapper 1: Start the GPU work (Asynchronous)
extern "C" void startHeatmapCUDA(int* h_agentX, int* h_agentY, int numAgents) {
    // 1. Allocate GPU memory ONLY ONCE
    if (d_heatmap == nullptr) {
        cudaMalloc(&d_heatmap, SIZE * SIZE * sizeof(int));
        cudaMemset(d_heatmap, 0, SIZE * SIZE * sizeof(int));
        cudaMalloc(&d_scaled, SCALED_SIZE * SCALED_SIZE * sizeof(int));
        cudaMalloc(&d_blurred, SCALED_SIZE * SCALED_SIZE * sizeof(int));

        // Initialize Stopwatches on the first run
        if (!eventsCreated) {
            cudaEventCreate(&startCreate); cudaEventCreate(&stopCreate);
            cudaEventCreate(&startScale); cudaEventCreate(&stopScale);
            cudaEventCreate(&startBlur); cudaEventCreate(&stopBlur);
            cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal);
            eventsCreated = true;
        }
    }

    cudaEventRecord(startTotal);
    // Allocate agent coordinate arrays (resize if agent count grows)
    if (numAgents > current_num_agents) {
        if (d_agentX) cudaFree(d_agentX);
        if (d_agentY) cudaFree(d_agentY);
        cudaMalloc(&d_agentX, numAgents * sizeof(int));
        cudaMalloc(&d_agentY, numAgents * sizeof(int));
        current_num_agents = numAgents;
    }

    // 2. Copy data Host -> Device
    if (numAgents > 0) {
    cudaMemcpy(d_agentX, h_agentX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentY, h_agentY, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    }
    // 3. Launch Kernels

    // 1. Grid for Fade and Scale (1024 / 16 = 64 blocks)
    int gridX = (int)ceil((float)SIZE / 16.0f);
    int gridY = (int)ceil((float)SIZE / 16.0f);
    // Amount of blocks for this grid, 64 blocks.
    dim3 grid(gridX, gridY);
    // Threads per block, so 16x16 threads per block, 256 threads.
    dim3 block(16, 16);

    int threadsPerBlock = 256;
    int blocksCreate = 0;
    if (numAgents > 0) {
        blocksCreate = (int)ceil((float)numAgents / (float)threadsPerBlock);
    }

    // 3. Grid for Scale/Blur (5120 / 16 = 320 blocks)
    int blurGridX = (int)ceil((float)SCALED_SIZE / (float)TILE_SIZE);
    int blurGridY = (int)ceil((float)SCALED_SIZE / (float)TILE_SIZE);
    dim3 gridBlur(blurGridX, blurGridY);
    // 20x20, hardcoded best.
    dim3 blockBlur(TILE_SIZE + 2 * HALO, TILE_SIZE + 2 * HALO);
    
    // Step 1 timing fading + add heat
    cudaEventRecord(startCreate);
    fadeKernel<<<grid, block>>>(d_heatmap);
    addHeatKernel<<<blocksCreate, threadsPerBlock>>>(d_heatmap, d_agentX, d_agentY, numAgents);
    cudaEventRecord(stopCreate);

    // Step 2 timing scale kernel
    cudaEventRecord(startScale);
    scaleKernel<<<gridBlur, block>>>(d_heatmap, d_scaled);
    cudaEventRecord(stopScale);

    // Step 3 timing blurkernel
    cudaEventRecord(startBlur);
    blurKernel<<<gridBlur, blockBlur>>>(d_scaled, d_blurred);
    cudaEventRecord(stopBlur);
}

// Wrapper 2: Wait for GPU and fetch results
extern "C" void finishHeatmapCUDA(int* h_blurred) {
    cudaDeviceSynchronize();
    cudaMemcpy(h_blurred, d_blurred, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    // Record end of all GPU operations
    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);

    // Calculate elapsed time
    float msCreate, msScale, msBlur, msTotal;
    cudaEventElapsedTime(&msCreate, startCreate, stopCreate);
    cudaEventElapsedTime(&msScale, startScale, stopScale);
    cudaEventElapsedTime(&msBlur, startBlur, stopBlur);
    cudaEventElapsedTime(&msTotal, startTotal, stopTotal);

    // Print the results once every 50 frames to avoid spamming the terminal
    if (frameCount++ % 50 == 0) {
        printf("GPU Times [Frame %d] - Create: %.3f ms | Scale: %.3f ms | Blur: %.3f ms\n", 
               frameCount, msCreate, msScale, msBlur);
        printf("GPU Time: %.3f ms\n", msTotal);

    }
    
}