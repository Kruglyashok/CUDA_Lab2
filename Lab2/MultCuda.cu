#include "MultCuda.cuh"


void gpuMatrMult(float* Ad, float* Bd, float* Cd, int rowsA, int colsA, int colsB, float *A, float *B, float *Ch)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(rowsA / dimBlock.y,colsB / dimBlock.x);
	printf("x: %d  y: %d \n", dimGrid.x, dimGrid.y);
	printf("bx: %d  by: %d\n", dimBlock.x, dimBlock.y);
	cudaMalloc((void**)&Ad, sizeof(float)*rowsA*colsA);
	cudaMalloc((void**)&Bd, sizeof(float)*colsA*colsB);
	cudaMemcpy(Ad, A, sizeof(float)*rowsA*colsA, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, sizeof(float)*colsA*colsB, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Cd, sizeof(float)*rowsA*colsB);
	gpuMatrMultD<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, rowsA, colsA, colsB);
	cudaDeviceSynchronize();
	cudaMemcpy(Ch, Cd, rowsA*colsB * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(Cd); cudaFree(Ad); cudaFree(Bd);
	printf(cudaGetErrorString(cudaGetLastError()));
	printf("\n");
	cudaDeviceSynchronize();
}

__global__ void gpuMatrMultD(float* Ad, float* Bd, float* Cd, int rowsA, int colsA, int colsB)
{
	//printf("start pguMatrMult\n");
	
	int bIndx = blockIdx.x;
	int bIndy = blockIdx.y;
	int tIndx = threadIdx.x;
	int tIndy = threadIdx.y;
	Cd[(blockDim.x*bIndx + tIndx)*colsB + blockDim.y*bIndy + tIndy] = 0;
	for (int k = 0; k < colsA; ++k) {
		Cd[(blockDim.x*bIndx+tIndx)*colsB + blockDim.y*bIndy + tIndy] += Ad[(blockDim.x*bIndx + tIndx)*colsA + k] * Bd[k*colsB + blockDim.y*bIndy + tIndy];
	}
	//printf("i am %d  %d  %d  %d\n", bIndx, bIndy, tIndx, tIndy);
}
void Mul(const float* A, const float* B, int hA, int wA, int wB, float* C) {
	int size;
	float* Ad; size = hA * wA * sizeof(float); cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd; size = wA * wB * sizeof(float); cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);
	// Compute the execution configuration assuming the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);
	// Launch the device computation
	Muld<<<dimGrid, dimBlock>>> (Ad, Bd, wA, wB, Cd);
	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
}
__global__ void Muld(float* A, float* B, int wA, int wB, float* C) {
		int bx = blockIdx.x; // Block index
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int aBegin = wA * BLOCK_SIZE * by; // Index of the first sub-matrix of A processed by the block
		int aEnd = aBegin + wA - 1; // Index of the last sub-matrix of A processed by the block
		int aStep = BLOCK_SIZE; // Step size used to iterate through the sub-matrices of A
		int bBegin = BLOCK_SIZE * bx; // Index of the first sub-matrix of B processed by the block
		int bStep = BLOCK_SIZE * wB; // Step size used to iterate through the sub-matrices of B
		float Csub = 0; // The element of the block sub-matrix that is computed by the thread
		for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
			// Shared memory for the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
			// Shared memory for the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
			As[ty][tx] = A[a + wA * ty + tx]; // Load the matrices from global memory to shared memory;
			Bs[ty][tx] = B[b + wB * ty + tx]; // each thread loads one element of each matrix
			__syncthreads(); // Synchronize to make sure the matrices are loaded
			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
			for (int k = 0; k < BLOCK_SIZE; ++k)
				Csub += As[ty][k] * Bs[k][tx];
			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}
		// Write the block sub-matrix to global memory;
		// each thread writes one element
		int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + wB * ty + tx] = Csub;
}

