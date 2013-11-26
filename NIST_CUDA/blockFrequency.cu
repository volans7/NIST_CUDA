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
#define M_SIZE 10
#define NUM_BLOCKS 10

// Считаем суммы для каждого блока
__global__ void sumOnesInAllBlocks (int nn, int * inData, int * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ int data[BLOCK_SIZE];

	int tid = threadIdx.x;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();
	for (int s = 1; s < blockDim.x; s = s * 2)
	{
		if (tid % (2*s) == 0) // проверить, участвует ли нить в данном шаге
			//if (i + s < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	if (tid == 0) // сохранить сумму элементов блока
		outData [blockIdx.x] = data [0];
}

// Считаем результирующие суммы для подпоследовательностей
__global__ void sumOnesInBlocks (int nn, int * inData, int * outData)
{
	__shared__ int data[NUM_BLOCKS];

	int tid = threadIdx.x;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();
	for (int s = 1; s < blockDim.x; s = s * 2)
	{
		if (tid % (2*s) == 0) // проверить, участвует ли нить в данном шаге
			//if (i + s < M * NUM_BLOCKS) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	if (tid == 0) // сохранить сумму элементов блока
		outData [blockIdx.x] = data [0];
}

__global__ void piBlocks (int nn, int MM, int * inData, double * outData)
{
	double tmp;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nn)
	{
		tmp = ((double)inData[i]/(double)MM) - 0.5;
		outData[i] = tmp * tmp;
	}
}

__global__ void sumBlocks (int nn, int MM, double * inData, double * outData)
{
	//__shared__ short int data[BLOCK_SIZE];
	__shared__ double data[BLOCK_SIZE];
	int tid = threadIdx.x;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = inData[i];
	__syncthreads();
	for (int s = 1; s < blockDim.x; s = s * 2)
	{
		if (tid % (2*s) == 0) // проверить, участвует ли нить в данном шаге
			//if (i + s < nn) // проверка нужна, т.к. не все входные последовательности кратны 1024. чтобы не суммировалось лишнее
				data[tid] += data [tid +s];
		__syncthreads();
	}
	if (tid == 0) // сохранить сумму элементов блока
		outData [blockIdx.x] = data [0];
}

double BF (int * data, int n, int M)
{
	//int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	int N = n / M; // оставшиеся элементы отбрасываются. N = 60000000 / 600064 = 99. 
	int NumBloks = ceil((float) M / BLOCK_SIZE); // для одной подпоследовательности. NumBloks = 600064 / 1024 = 586
	int NumAllBloks = NumBloks * N; // для всей последовательности. NumAllBloks = 586 * 99 = 58014

	double	p_value, sum, pi, v, chi_squared;
	
	//счетчики времени
	size_t free = 0;
	size_t total = 0;

	printf("\n Начальные данные: ---------------------------\n");
	printf("Длина последовательности: %d\n", n);
	printf("Длина подпоследовательности: %d = %d\n", M, M_SIZE);
	printf("Количество подпоследовательностей: %d\n", N);
	printf("Количество нитей в блоке: %d\n", NumThreads);
	printf("Количество блоков в подпоследовательности: %d = %d\n", NumBloks, NUM_BLOCKS);
	printf("Всего блоков: %d\n", NumAllBloks);


	// Выбрать первый GPU для работы
	cudaSetDevice(0);

	// Выделить память на CPU
	int * inD = new int [n];
	int * outD = new int [NumAllBloks];
	int * outD2 = new int [N];
	double * outD3 = new double [N];

	// выделяем память на GPU
	int * inDev = NULL;
	int * outDev = NULL;
	int * outDev2 = NULL;
	double * outDev3 = NULL;

	// выделяем память на GPU
	cudaMalloc ((void**)&inDev, n * sizeof (int));
	cudaMalloc ((void**)&outDev, NumAllBloks * sizeof (int));
	cudaMalloc ((void**)&outDev2, N * sizeof(int));
	cudaMalloc ((void**)&outDev3, N * sizeof(double));

	// копируем данные на GPU
	cudaMemcpyAsync (inDev, data, n * sizeof (int), cudaMemcpyHostToDevice);

	// параметры для запуска ядра
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumAllBloks);
	// параметры для запуска ядра
	dim3 threads2 = dim3(NumBloks);
	dim3 bloks2 = dim3(N);
	// параметры для запуска ядра
	dim3 threads3 = dim3(N);
	dim3 bloks3 = dim3(1);

	// Засекаем время
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	// создаем события начала и окончания выполнения ядра
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	// Привязываем событие strat к текущему месту
	cudaEventRecord (start, 0);

	// запуск ядра
	sumOnesInAllBlocks<<<bloks, threads>>> (n, inDev, outDev); 

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
	
	sum = 0;
	
	// нельзя рядом с предыдущим ядром, т.к. это ядро начинает выполняться раньше окончанияя того
	sumOnesInBlocks<<<bloks2, threads2>>> (n, outDev, outDev2); 
	
	// копируем результат обратно в CPU
	cudaMemcpy (outD2, outDev2, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		printf("%d) %d \n", i, outD2[i]);

	piBlocks<<<bloks2, threads2>>> (n, M, outDev2, outDev3); 
	// копируем результат обратно в CPU
	cudaMemcpy (outD3, outDev3, N * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		printf("%d) %lf\n", i, outD3[i]);
		
	sumBlocks<<<bloks3, threads3>>> (M, M, outDev3, outDev3); 
	// копируем результат обратно в CPU
	cudaMemcpy (outD3, outDev3, N * sizeof(double), cudaMemcpyDeviceToHost);

	sum = outD3[0];
	printf("Sum = %lf\n", sum);

	delete [] outD;
	delete [] outD2;
	delete [] inD;

	// освобождаем память
	cudaFree (inDev);
	cudaFree (outDev);
	cudaFree (outDev2);
		
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
		//printf("%d) %d\n", i, eps[i]);
	}

	sum = BF (eps, n, M);
	chi_squared = 4.0 * M * sum;
	p_value = cephes_igamc(N/2.0, chi_squared/2.0);

	//Выводим результат
	printf("\n Result sum = %lf, NumAllBloks = %d, chi_squared=%lf, P-value=%lf\n", sum, N, chi_squared, p_value);

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
