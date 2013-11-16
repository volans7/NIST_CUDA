#include <stdio.h>
#include <math.h>
#include <string.h>
#include "include\externs.h"
#include "include\cephes.h"

#include <cuda_runtime.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> // ��� threadIdx
#include <device_functions.h> // ��� __syncthreads()

#include <ctime>
#include <time.h>
#include <Windows.h>

#pragma comment(lib, "cudart") // ������������ ����������� ��� CUDA runtime API (���������������)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                    B L O C K  F R E Q U E N C Y  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define BLOCK_SIZE 10 // BLOCK_SIZE = M. ��� ����������� ��������� ����� ����� � �����. 128, 256, 512, 1024
#define NUM_BLOCKS 10 // NUM_BLOCKS = n / M
/*
__global__ void sumOnesInBlock (int nn, int MM, int * inData, int * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;

	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	if (i + blockDim.x < nn) // �������� �����, �.�. �� ��� ������� ������������������ ������ 1024
		data[tid] = inData[i] + inData[i + blockDim.x];
	else
		data[tid] = inData[i];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 32; s = s / 2 )
	{
		if (tid < s)
			if (i + s < nn) // �������� �����, �.�. �� ��� ������� ������������������ ������ 1024. ����� �� ������������� ������
				data[tid] += data [tid +s];
		__syncthreads();
	}
	
	if (tid < 32) // �.�. � � ����� 32 ����, �� ������������� �� ���������
	{
		data[tid] += data[tid + 32];
		data[tid] += data[tid + 16];
		data[tid] += data[tid + 8];
		data[tid] += data[tid + 4];
		data[tid] += data[tid + 2];
		data[tid] += data[tid + 1];
	}
	if (tid == 0) // ��������� ����� ��������� �����
		outData [blockIdx.x] = data [0];
}
*/
__global__ void sumOnesInBlock (int nn, int MM, int * inData, int * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();
	/*
	for (int s = blockDim.x / 2; s > 0; s = s / 2 )
	{
		if (tid < s)
			if (i + s < nn) // �������� �����, �.�. �� ��� ������� ������������������ ������ 1024. ����� �� ������������� ������
				data[tid] += data [tid +s];
		__syncthreads();
	}
	*/
	//////////////////////////////////////////////////////////////������� ��� � ������ 1
	for (int s = 0; s < blockDim.x; s++)
	{
		if (tid % 2 == 1)
			if (i + 1 < nn) // �������� �����, �.�. �� ��� ������� ������������������ ������ 1024. ����� �� ������������� ������
				data[tid] += data [tid + 1];
		__syncthreads();
	}
	if (tid == 0) // ��������� ����� ��������� �����
		outData [blockIdx.x] = data [0];
}

__global__ void sumBlocks (int nn, int MM, int * inData, int * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ int data[NUM_BLOCKS];
	int tid = threadIdx.x;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s = s / 2 )
	{
		if (tid < s)
			if (i + s < nn) // �������� �����, �.�. �� ��� ������� ������������������ ������ 1024. ����� �� ������������� ������
				data[tid] += data [tid +s];
		__syncthreads();
	}
	
	if (tid == 0) // ��������� ����� ��������� �����
		outData [blockIdx.x] = data [0];
}

int BF (int * data, int n, int M)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // ���������� ����� � �����
	//int NumBloks = n / BLOCK_SIZE; // ����������� ���������� ������ ��� �������� n � ������� ����� � ����������� � ������� �������
	int NumBloks = NUM_BLOCKS;

	double	p_value, sum, pi, v, chi_squared;
	
	//�������� �������
	size_t free = 0;
	size_t total = 0;

	// ������� ������ GPU ��� ������
	cudaSetDevice(0);

	// �������� ������ �� CPU
	int * inD = new int [n];
	int * outD = new int [n];

	// �������� ������ �� GPU
	int * inDev = NULL;
	int * outDev = NULL;

	// �������� ������ �� GPU
	cudaMalloc ((void**)&inDev, numBytes);
	cudaMalloc ((void**)&outDev, numBytes);

	// �������� ������ �� GPU
	cudaMemcpyAsync (inDev, data, numBytes, cudaMemcpyHostToDevice);

	// ������ ����
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumBloks);

	// �������� �����
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	// ������� ������� ������ � ��������� ���������� ����
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	// ����������� ������� strat � �������� �����
	cudaEventRecord (start, 0);

	// ������ ����
	sumOnesInBlock<<<bloks, threads>>> (n, M, inDev, outDev); 
	//sumBlocks<<<bloks, threads>>> (n, M, inDev, outDev); 

	// ����������� ������� stop � ������� �����
	cudaEventRecord (stop, 0);
	// ���������� ��������� ��������� ���������� ����, ��������� ����������� ������������� �� ������� stop
	cudaEventSynchronize (stop);
	// ����������� ����� ����� ��������� start � stop
	cudaEventElapsedTime ( &gpuTime, start, stop);
	printf("time spent executing by the GPU: %.9f millseconds\n", gpuTime);
	// ���������� ��������� �������
	cudaEventDestroy (start);
	cudaEventDestroy (stop);

	// �������� ��������� ������� � CPU
	cudaMemcpy (outD, outDev, numBytes, cudaMemcpyDeviceToHost);

	// ����������� ������
	cudaFree (inDev);
	cudaFree (outDev);

	sum = 0;
	//if (NewNumBloks > BLOCK_SIZE) // ���� ������ �����, �� ��������� ��������� ������������ �� GPU
	//	sum = reduce(outD, NewNumBloks);
	//else // ���� ����, �� ������������ ��������� �� CPU
	//{			
		for (int i = 0; i < NumBloks; i++)
		{
			printf("%d) %d \n", i, outD[i]);
			pi = (double) outD[i]/(double)M;
			v = pi - 0.5;
			sum += v*v;
		}
		delete [] outD;
	//}
		
	// ���������� � ������
	cudaMemGetInfo (&free, &total);
	printf("���������� ��������� ������: %lld, ����� %lld\n", free, total); 

	cudaFree(outDev);
	//cudaFree(inDev2);
	cudaMemGetInfo (&free, &total);
	printf("���������� ��������� ������: %lld, ����� %lld\n", free, total); 

	return sum;
}

void
BlockFrequency(int M, int n)
{
	int		i, j, N, blockSum;
	double	p_value, sum, pi, v, chi_squared;
	
	N = n / M; // ��������� �� ���������������������
	sum = 0.0;
	
	// �������� ������ �� CPU
	int * eps = new int [n];

	// ������� ������� 
	printf("Data: \n");
	for (int i = 0; i < n; i++)
	{
		eps[i] = epsilon[i]; 
		printf("%d) %d\n", i, eps[i]);
	}

	sum = BF (eps, n, M);
	/*
	for (i = 0; i < N; i++) 
	{
		blockSum = 0; // ��������� ���� ������ � ������ ���������������������
		for (j = 0; j < M; j++)
		{
			blockSum += epsilon[j+i*M];
			printf("%d.%d) blockSum = %d ", i, j, blockSum);
		}
		pi = (double)blockSum/(double)M;
		v = pi - 0.5;
		sum += v*v;
	}
	*/
	chi_squared = 4.0 * M * sum;
	p_value = cephes_igamc(N/2.0, chi_squared/2.0);

	//������� ���������
	printf("\n Result sum = %lf, NumBloks = %d, chi_squared=%lf, P-value=%lf\n", sum, N, chi_squared, p_value);

	/*
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t\tBLOCK FREQUENCY TEST\n");
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t---------------------------------------------\n");
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\tCOMPUTATIONAL INFORMATION:\n");
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t---------------------------------------------\n");
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t(a) Chi^2           = %f\n", chi_squared);
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t(b) # of substrings = %d\n", N);
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t(c) block length    = %d\n", M);
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t(d) Note: %d bits were discarded.\n", n % M);
	fprintf(stats[TEST_BLOCK_FREQUENCY], "\t\t---------------------------------------------\n");

	fprintf(stats[TEST_BLOCK_FREQUENCY], "%s\t\tp_value = %f\n\n", p_value < ALPHA ? "FAILURE" : "SUCCESS", p_value); fflush(stats[TEST_BLOCK_FREQUENCY]);
	fprintf(results[TEST_BLOCK_FREQUENCY], "%f\n", p_value); fflush(results[TEST_BLOCK_FREQUENCY]);	*/
	
	printf("Press key...\n");
	getchar();
}
