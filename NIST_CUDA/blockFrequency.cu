#include <stdio.h>
#include <math.h>
#include <string.h>
#include "include\externs.h"
#include "include\cephes.h"

#include <cuda_runtime.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> // для threadIdx
#include <device_functions.h> // для __syncthreads()

#include <ctime>
#include <time.h>
#include <Windows.h>

#pragma comment(lib, "cudart") // динамическая бибилиотека для CUDA runtime API (высокоуровневый)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                    B L O C K  F R E Q U E N C Y  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define BLOCK_SIZE 10 // BLOCK_SIZE = M. это максимально возможное число нитей в блоке. 128, 256, 512, 1024
#define NUM_BLOCKS 10 // NUM_BLOCKS = n / M
/*
__global__ void sumOnesInBlock (int nn, int MM, int * inData, int * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;

	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	if (i + blockDim.x < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024
		data[tid] = inData[i] + inData[i + blockDim.x];
	else
		data[tid] = inData[i];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 32; s = s / 2 )
	{
		if (tid < s)
			if (i + s < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	
	if (tid < 32) // т.к. в в варпе 32 нити, то синхронизация не требуется
	{
		data[tid] += data[tid + 32];
		data[tid] += data[tid + 16];
		data[tid] += data[tid + 8];
		data[tid] += data[tid + 4];
		data[tid] += data[tid + 2];
		data[tid] += data[tid + 1];
	}
	if (tid == 0) // сохранить сумму элементов блока
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
			if (i + s < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	*/
	//////////////////////////////////////////////////////////////сделать как в редьюс 1
	for (int s = 0; s < blockDim.x; s++)
	{
		if (tid % 2 == 1)
			if (i + 1 < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid + 1];
		__syncthreads();
	}
	if (tid == 0) // сохранить сумму элементов блока
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
			if (i + s < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	
	if (tid == 0) // сохранить сумму элементов блока
		outData [blockIdx.x] = data [0];
}

int BF (int * data, int n, int M)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	//int NumBloks = n / BLOCK_SIZE; // высчитываем количество блоков при заданном n и размере блока с округлением в большую сторону
	int NumBloks = NUM_BLOCKS;

	double	p_value, sum, pi, v, chi_squared;
	
	//счетчики времени
	size_t free = 0;
	size_t total = 0;

	// Выбрать первый GPU для работы
	cudaSetDevice(0);

	// Выделить память на CPU
	int * inD = new int [n];
	int * outD = new int [n];

	// выделяем память на GPU
	int * inDev = NULL;
	int * outDev = NULL;

	// выделяем память на GPU
	cudaMalloc ((void**)&inDev, numBytes);
	cudaMalloc ((void**)&outDev, numBytes);

	// копируем данные на GPU
	cudaMemcpyAsync (inDev, data, numBytes, cudaMemcpyHostToDevice);

	// запуск ядра
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumBloks);

	// Засекаем время
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	// создаем события начала и окончания выполнения ядра
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	// Привязываем событие strat к текущему месту
	cudaEventRecord (start, 0);

	// запуск ядра
	sumOnesInBlock<<<bloks, threads>>> (n, M, inDev, outDev); 
	//sumBlocks<<<bloks, threads>>> (n, M, inDev, outDev); 

	// привязываем событие stop к данному месту
	cudaEventRecord (stop, 0);
	// Дожидаемся реального окончания выполнения ядра, используя возможность синхронизации по событию stop
	cudaEventSynchronize (stop);
	// Запрашиваем время между событиями start и stop
	cudaEventElapsedTime ( &gpuTime, start, stop);
	printf("time spent executing by the GPU: %.9f millseconds\n", gpuTime);
	// Уничтожаем созданные события
	cudaEventDestroy (start);
	cudaEventDestroy (stop);

	// копируем результат обратно в CPU
	cudaMemcpy (outD, outDev, numBytes, cudaMemcpyDeviceToHost);

	// освобождаем память
	cudaFree (inDev);
	cudaFree (outDev);

	sum = 0;
	//if (NewNumBloks > BLOCK_SIZE) // если блоков много, то запускаем повторное суммирование на GPU
	//	sum = reduce(outD, NewNumBloks);
	//else // если мало, то подсчитываем результат на CPU
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
		
	// Информация о памяти
	cudaMemGetInfo (&free, &total);
	printf("Количество свободной памяти: %lld, всего %lld\n", free, total); 

	cudaFree(outDev);
	//cudaFree(inDev2);
	cudaMemGetInfo (&free, &total);
	printf("Количество свободной памяти: %lld, всего %lld\n", free, total); 

	return sum;
}

void
BlockFrequency(int M, int n)
{
	int		i, j, N, blockSum;
	double	p_value, sum, pi, v, chi_squared;
	
	N = n / M; // разбиваем на подпоследовательности
	sum = 0.0;
	
	// Выделить память на CPU
	int * eps = new int [n];

	// Создаем массивы 
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
		blockSum = 0; // определим долю единиц в каждой подпоследовательности
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

	//Выводим результат
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
