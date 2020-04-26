//=========================================================================================
//							DETAILS ABOUT THE SUBMISSION
//=========================================================================================
//
// Name: Obaid Ur-Rahmaan
// ID: 1807611
//
// Goals Achieved:
//	1. Block scan
//	2. Full scan for large vectors
//
// My time, in milliseconds, to execute the different scans on a vector of 10,000,000 entries:
//	1. Block scan without BCAO: 4.08781mSecs
//	2. Full scan without BCAO: 4.72147mSecs
//
// (Lab Machine)
// CPU model: Intel® Core™ i7-8700 CPU @ 3.20GHz × 12
// GPU model: GeForce GTX 2060
//
//=========================================================================================
//=========================================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>

#define NUM_ELEMENTS 10000000

// Block size of 128 showed the best performance
#define BLOCK_SIZE 128

// A helper macro to simplify handling of CUDA errors
#define CUDA_ERROR(err, msg) { \
		if (err != cudaSuccess) {\
			printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
			exit( EXIT_FAILURE );\
		}\
}

// Function to compare the results with the host version
__host__
static void compare_results(const int *vector1, const int *vector2, int numElements) {
	for (int i = 0; i < numElements; i++) {
		if (vector1[i] != vector2[i]) {
			printf("%d ----------- %d\n", vector1[i], vector2[i]);
			fprintf(stderr, "Verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}

// Sequential version to compare results with Parallel version later
__host__
void sequential_scan(int *g_idata, int *g_odata, int n) {

	g_odata[0] = 0;
	for (int i = 1; i < n; i++) {
		g_odata[i] = g_odata[i - 1] + g_idata[i - 1];
	}
}

__global__
void block_add(int *block, int len_block, int *SUM) {

	// Value of element to be added to the main vector
	int s = SUM[blockIdx.x];

	// get address of the vector to update
	int addr = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

	__syncthreads();
	// update two elements in the vector
	if (addr < len_block)
		block[addr] += s;
	if (addr + blockDim.x < len_block)
		block[addr + blockDim.x] += s;
}

__device__
void transfer_memory(int blockId, int thid, int n, int* temp, int* g_idata, int* g_odata, bool global_to_shared) {

	if (global_to_shared) {
		if (blockId + (thid * 2) < n)
			temp[thid * 2] = g_idata[blockId + (thid * 2)];

		if (blockId + (thid * 2) + 1 < n)
			temp[(thid * 2) + 1] = g_idata[blockId + (thid * 2) + 1];
	} else {
		if (blockId + (thid * 2) < n)
			g_odata[blockId + (thid * 2)] = temp[thid * 2];
		if (blockId + (thid * 2) + 1 < n)
			g_odata[blockId + (thid * 2) + 1] = temp[(thid * 2) + 1];
	}

}

__global__
void block_scan(int *g_idata, int *g_odata, int n, int *SUM, int add_last) {

	// shared memory initialisation
	__shared__ int temp[BLOCK_SIZE * 2];

	int thid = threadIdx.x;
	int blockId = blockDim.x * blockIdx.x * 2;
	int offset = 1;
	int last = 0;

	// load the elements from global memory into the shared memory
	transfer_memory(blockId, thid, n, temp, g_idata, g_odata, true);

	// save the last element for later to improve the performance
	if (add_last && thid == BLOCK_SIZE - 1)
		last = temp[(thid * 2) + 1];

	// build sum in place up the tree (reduction phase)
	for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (thid == 0)
		temp[(BLOCK_SIZE * 2) - 1] = 0;

	// traverse down tree & build scan (distribution phase)
	for (int d = 1; d < (BLOCK_SIZE * 2); d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	// extract the sum (merged to improve the performance) [Only applicable for full scan]
	if (add_last && thid == BLOCK_SIZE - 1)
		SUM[blockIdx.x] = temp[(thid * 2) + 1] + last;

	// load the shared memory back into the global memory
	transfer_memory(blockId, thid, n, temp, g_idata, g_odata, false);

}

__host__
void full_scan(int *h_IN, int *h_OUT, int len) {

	size_t level_1_memory = (1 + ((len - 1) / (BLOCK_SIZE * 2))) * sizeof(int);
	size_t level_2_memory = (BLOCK_SIZE * 2) * sizeof(int);

	// error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	// allocate memory for all the possible vectors needed for the execution
	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, len * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, len * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1 = NULL;
	err = cudaMalloc((void**) &d_SUM_1, level_1_memory);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_1_Scanned, level_1_memory);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2 = NULL;
	err = cudaMalloc((void**) &d_SUM_2, level_2_memory);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_2_Scanned, level_2_memory);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	// copy the memory from the host to the device
	err = cudaMemcpy(d_IN, h_IN, len * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	// size of the grid for each level
	int level_1_blocks_per_grid = 1 + ((len - 1) / (BLOCK_SIZE * 2));
	int level_2_blocks_per_grid = 1 + ceil(level_1_blocks_per_grid / (BLOCK_SIZE * 2));
	int level_3_blocks_per_grid = 1 + ceil(level_2_blocks_per_grid / (BLOCK_SIZE * 2));

	if (level_2_blocks_per_grid == 1) {
		printf("HIIIIIIIII\n");
	}

	// Only 1 scan is required to scan the whole array
	if (level_1_blocks_per_grid == 1) {

		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernel
		block_scan<<<level_1_blocks_per_grid, BLOCK_SIZE>>>(d_IN, d_OUT, len, NULL, 0);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// 2 scans are required to scan the whole array
	else if (level_2_blocks_per_grid == 1) {

		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernels
		block_scan<<<level_1_blocks_per_grid, BLOCK_SIZE>>>(d_IN, d_OUT, len, d_SUM_1, 1);
		block_scan<<<level_2_blocks_per_grid, BLOCK_SIZE>>>(d_SUM_1, d_SUM_1_Scanned, level_1_blocks_per_grid, NULL, 0);
		block_add<<<level_1_blocks_per_grid, BLOCK_SIZE>>>(d_OUT, len, d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// 3 scans are required to scan the whole array
	else if (level_3_blocks_per_grid == 1) {

		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernels
		block_scan<<<level_1_blocks_per_grid, BLOCK_SIZE>>>(d_IN, d_OUT, len, d_SUM_1, 1);
		block_scan<<<level_2_blocks_per_grid, BLOCK_SIZE>>>(d_SUM_1, d_SUM_1_Scanned, level_1_blocks_per_grid, d_SUM_2, 1);
		block_scan<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned, level_2_blocks_per_grid, NULL, 0);
		block_add<<<level_2_blocks_per_grid, BLOCK_SIZE>>>(d_SUM_1_Scanned, level_1_blocks_per_grid, d_SUM_2_Scanned);
		block_add<<<level_1_blocks_per_grid, BLOCK_SIZE>>>(d_OUT, len, d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// Print error message is array is too large
	else {
		fprintf(stderr, "The array size=%d is too large to be scanned.\n", len);
		goto cleanup;
	}

	// check whether the run was successful
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	// get the duration it took for the kernels to execute
	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	// print the time elapsed
	printf("Full block scan of %d elements took %.5fmSecs\n", len, d_msecs);

	// copy the result from the device back to the host
	err = cudaMemcpy(h_OUT, d_OUT, len * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

	// -----------------------------------------------------------
	// 							CLEANUP
	// -----------------------------------------------------------

	cleanup:
	// Free device global memory
	CUDA_ERROR(cudaFree(d_IN), "Failed to free device vector IN");
	CUDA_ERROR(cudaFree(d_OUT), "Failed to free device vector OUT");
	CUDA_ERROR(cudaFree(d_SUM_1), "Failed to free device vector SUM_1");
	CUDA_ERROR(cudaFree(d_SUM_1_Scanned), "Failed to free device vector SUM_1_Scanned");
	CUDA_ERROR(cudaFree(d_SUM_2), "Failed to free device vector SUM_2");
	CUDA_ERROR(cudaFree(d_SUM_2_Scanned), "Failed to free device vector SUM_2_Scanned");

	// Clean up the Device timer event objects
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
}

/**
 * Host main routine
 */
int main(void) {

	// allocate memory on the host for the arrays
	int *h_IN = (int *) malloc(NUM_ELEMENTS * sizeof(int));
	int *h_OUT = (int *) malloc(NUM_ELEMENTS * sizeof(int));
	int *h_OUT_CUDA = (int *) malloc(NUM_ELEMENTS * sizeof(int));

	// verify the host allocations
	if (h_IN == NULL || h_OUT == NULL || h_OUT_CUDA == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// -----------------------------------------------------------
	// 					SEQUENTIAL SCAN
	// -----------------------------------------------------------

	// create host stop-watch times
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// initialise the host input
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		h_IN[i] = rand() % 10;
	}

	// Time Sequential scan
	sdkStartTimer(&timer);
	sequential_scan(h_IN, h_OUT, NUM_ELEMENTS);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Sequential scan on host of %d elements took %.5fmSecs\n", NUM_ELEMENTS, h_msecs);

	// -----------------------------------------------------------
	// 					BLOCK SCAN
	// -----------------------------------------------------------

	// error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	// allocate memory for all the possible vectors needed for the execution
	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, NUM_ELEMENTS * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, NUM_ELEMENTS * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	// copy the memory from the host to the device
	err = cudaMemcpy(d_IN, h_IN, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	// size of the grid for each level
	int blocksPerGridLevel1 = 1 + ((NUM_ELEMENTS - 1) / (BLOCK_SIZE * 2));

	// record the start time
	cudaEventRecord(d_start, 0);

	// execute the kernel
	block_scan<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, NUM_ELEMENTS, NULL, 0);

	// record the stop time
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);
	cudaDeviceSynchronize();

	// check whether the run was successful
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	// get the duration it took to execute kernel
	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	// print the time elapsed
	printf("Block scan of %d elements took %.5fmSecs\n", NUM_ELEMENTS, d_msecs);

	// Free device global memory
	CUDA_ERROR(cudaFree(d_IN), "Failed to free device vector IN");
	CUDA_ERROR(cudaFree(d_OUT), "Failed to free device vector OUT");

	// Clean up the Device timer event objects
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	// -----------------------------------------------------------
	// 				  FULL SCAN
	// -----------------------------------------------------------

	full_scan(h_IN, h_OUT_CUDA, NUM_ELEMENTS);
	compare_results(h_OUT, h_OUT_CUDA, NUM_ELEMENTS);

	// Free host memory
	free(h_IN);
	free(h_OUT);
	free(h_OUT_CUDA);

	// Clean up the Host timer
	sdkDeleteTimer(&timer);

	return 0;
}



