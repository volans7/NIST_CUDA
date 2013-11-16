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
                          F R E Q U E N C Y  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define BLOCK_SIZE 1024 // ��� ����������� ��������� ����� ����� � �����. 128, 256, 512, 1024

__global__ void reduce (int nn, int * inData, int * outData)
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

int reduce (int * data, int n)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // ���������� ����� � �����
	int NumBloks = ceil((float) n / NumThreads); // ����������� ���������� ������ ��� �������� n � ������� ����� � ����������� � ������� �������
	
	int sum = 0;
	
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
	int NewNumBloks = ceil((float) NumBloks / 2 );  // ����������� ���������� ������ ��� �������� n � ������� ����� � ����������� � ������� �������
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NewNumBloks);

	// �������� �����
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	// ������� ������� ������ � ��������� ���������� ����
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	// ����������� ������� strat � �������� �����
	cudaEventRecord (start, 0);

	// ������ ����
	reduce<<<bloks, threads>>> (n, inDev, outDev); // ������� �����

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
	if (NewNumBloks > BLOCK_SIZE) // ���� ������ �����, �� ��������� ��������� ������������ �� GPU
		sum = reduce(outD, NewNumBloks);
	else // ���� ����, �� ������������ ��������� �� CPU
	{			
		for (int i = 0; i < NewNumBloks; i++)
			sum += outD[i];
		delete [] outD;
	}
		
	// ���������� � ������
	cudaMemGetInfo (&free, &total);
	printf("���������� ��������� ������: %lld, ����� %lld\n", free, total); 

	cudaFree(outDev);
	//cudaFree(inDev2);
	cudaMemGetInfo (&free, &total);
	printf("���������� ��������� ������: %lld, ����� %lld\n", free, total); 

	return sum;
}

__global__ void epsilonToX (int nn, int * inData, int * outData)
{
	int tid = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	outData[i] = 2 * inData[i] - 1;
}

void
Frequency(int n)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // ���������� ����� � �����
	int NumBloks = ceil((float) n / NumThreads); // ����������� ���������� ������ ��� �������� n � ������� ����� � ����������� � ������� �������

	int sum;
	double	f, s_obs, p_value, sqrt2 = 1.41421356237309504880;
	
	// ������� ������ GPU ��� ������
	cudaSetDevice(0);

	// �������� ������ �� CPU
	int * a = new int [n];
	int * aa = new int [n];
	int * d = new int [n];
	int * dd = new int [n];

	// ������� ������� 
	printf("Data: \n");
	for (int i = 0; i < n; i++)
	{
		a[i] = epsilon[i]; 
		aa[i] = epsilon[i];
	}
	
	// �������� ������ �� GPU
	int * aDev = NULL;
	int * dDev = NULL;

	// �������� ������ �� GPU
	cudaMalloc ((void**)&aDev, numBytes);
	cudaMalloc ((void**)&dDev, numBytes);
	
	// �������� ������ �� GPU
	cudaMemcpyAsync (aDev, a, numBytes, cudaMemcpyHostToDevice);

	// ������ ����
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumBloks);
	
	// ������ ����
	epsilonToX<<<bloks, threads>>> (n, aDev, dDev); // ������� x

	// ����������� ������
	cudaFree (aDev);
	cudaMemcpy (d, dDev, numBytes, cudaMemcpyDeviceToHost);
	// ����������� ������
	cudaFree (dDev);

	// ��������� ������
	int startGPU = GetTickCount();      
	sum = reduce (d, n);	
	//������� ����� ���������� ������� �� CPU (� ������������)
    printf ("GPU compute time: %i\n", GetTickCount() - startGPU);

	// ��������� s(obs)
	s_obs = fabs(1.0*sum)/sqrt(1.0*n); // �������� �� 1.0, ����� ���������� �������, ����� ������� ������������ (��� ������������)
	f = s_obs/sqrt2;
	p_value = cephes_erfc(1.0*f);

	// ����������� ������
	cudaFree (aDev);
	cudaFree (dDev);

	//������� ���������
	printf("\n Result sumA = %d, NumBloks = %d, s(obs)=%lf, P-value=%lf\n", sum, NumBloks, s_obs, p_value);

	// ��������� ������
	int startCPU = GetTickCount();      
	// �� �� ���������� �� CPU
	int summ = 0;
	for (int i = 0; i < n; i++)
	{
		aa[i] = 2 * aa[i] - 1;
		summ += aa[i];
	}
	//������� ����� ���������� ������� �� CPU (� ������������)
    printf ("CPU compute time: %i\n", GetTickCount() - startCPU);

	printf("Result sumAA = %d\n", summ);
	
	delete [] a;
	delete [] d;

	printf("Press key...\n");
	getchar();
	
}

