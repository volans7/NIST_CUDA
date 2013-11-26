#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "include\externs.h"
#include "include\cephes.h"
#include <iostream>

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
                              R U N S  T E S T 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define BLOCK_SIZE 1024 // это максимально возможное число нитей в блоке. 128, 256, 512, 1024

// Считаем суммы для каждого блока
__global__ void funcR (int nn, int * inData, int * outData)
{	
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
		
	if (i > 0)
	{
		if (inData[i] != inData [i - 1])
			outData[i] = 1;
		else
			outData[i] = 0;
	}
	else
		outData[i] = 0;
}

__global__ void reduce1 (int nn, int * inData, int * outData)
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

int reduce1 (int * data, int n)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	int NumBloks = ceil((float) n / NumThreads); // высчитываем количество блоков при заданном n и размере блока с округлением в большую сторону
	
	int sum = 0;
	
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
	int NewNumBloks = ceil((float) NumBloks / 2 );  // высчитываем количество блоков при заданном n и размере блока с округлением в большую сторону
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NewNumBloks);

	// Засекаем время
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	// создаем события начала и окончания выполнения ядра
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	// Привязываем событие strat к текущему месту
	cudaEventRecord (start, 0);

	// запуск ядра
	reduce1<<<bloks, threads>>> (n, inDev, outDev); // находим сумму

	// привязываем событие stop к данному месту
	cudaEventRecord (stop, 0);
	// Дожидаемся реального окончания выполнения ядра, используя возможность синхронизации по событию stop
	cudaEventSynchronize (stop);
	// Запрашиваем время между событиями start и stop
	cudaEventElapsedTime ( &gpuTime, start, stop);
	printf("GPU compute time (находим сумму): %.9f millseconds\n", gpuTime);
	// Уничтожаем созданные события
	cudaEventDestroy (start);
	cudaEventDestroy (stop);

	// копируем результат обратно в CPU
	cudaMemcpy (outD, outDev, numBytes, cudaMemcpyDeviceToHost);

	// освобождаем память
	cudaFree (inDev);
	cudaFree (outDev);

	printf("Sums of ones:\n");
	for (int i = 0; i < NewNumBloks; i++)
		printf("%d) %d\n", i, outD[i]);

	sum = 0;
	if (NewNumBloks > BLOCK_SIZE) // если блоков много, то запускаем повторное суммирование на GPU
		sum = reduce1(outD, NewNumBloks);
	else // если мало, то подсчитываем результат на CPU
	{			
		for (int i = 0; i < NewNumBloks; i++)
			sum += outD[i];
		delete [] outD;
	}
		
	return sum;
}

int funcR (int * data, int n)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	int NumBloks = ceil((float) n / NumThreads); // высчитываем количество блоков при заданном n и размере блока с округлением в большую сторону
	
	int sum = 0;
	
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
	funcR<<<bloks, threads>>> (n, inDev, outDev); // находим сумму

	// привязываем событие stop к данному месту
	cudaEventRecord (stop, 0);
	// Дожидаемся реального окончания выполнения ядра, используя возможность синхронизации по событию stop
	cudaEventSynchronize (stop);
	// Запрашиваем время между событиями start и stop
	cudaEventElapsedTime ( &gpuTime, start, stop);
	printf("GPU compute time (находим число переходов): %.9f millseconds\n", gpuTime);
	// Уничтожаем созданные события
	cudaEventDestroy (start);
	cudaEventDestroy (stop);

	// копируем результат обратно в CPU
	cudaMemcpy (outD, outDev, numBytes, cudaMemcpyDeviceToHost);

	// освобождаем память
	cudaFree (inDev);
	cudaFree (outDev);
	/*
	printf("Массив поиска переходов из 1 в 0 и из 0 в 1:\n");
	for (int i = 0; i < n; i++)
	{
		if (i % BLOCK_SIZE == 0)		
			printf("\n");
		printf("%d) %d ", i, outD[i]);
	}
	printf("\n");
	*/
	sum = reduce1(outD, n);

	printf("sum = %d\n", sum);
	
	delete [] outD;
			
	return sum;
}


void
Runs(int n)
{
	int		S, S2, k;
	double	pi, pi2, V, V2, erfc_arg, p_value;

	int * a = new int [n];

	for (int i = 0; i < n; i++)
		a[i] = epsilon[i]; 

	S = reduce1 (a, n);
	pi = (double)S / (double)n;

	// Запускаем таймер
	LARGE_INTEGER frequency0;        // ticks per second
	LARGE_INTEGER t10, t20;           // ticks
	double elapsedTime0;	
	//// get ticks per second
	QueryPerformanceFrequency(&frequency0);
	// start timer
	QueryPerformanceCounter(&t10);

	S2 = 0;
	for (k = 0; k < n; k++)
		if (epsilon[k])
			S2++;

	// stop timer
	QueryPerformanceCounter(&t20);
	// compute and print the elapsed time in millisec
	elapsedTime0 = (t20.QuadPart - t10.QuadPart) * 1000.0 / frequency0.QuadPart;
	//Выводим время выполнения функции на CPU 
	printf ("CPU compute time (находим число единиц): %f\n", elapsedTime0);

	pi2 = (double)S2 / (double)n;
	
	printf ("GPU: S = %d, pi = %lf\n", S, pi);
	printf ("CPU: S = %d, pi2 = %lf\n", S2, pi2);
	
	if ( fabs(pi - 0.5) > (2.0 / sqrt((double)n)) ) { 
		
		fprintf(stats[TEST_RUNS], "\t\t\t\tRUNS TEST\n");
		fprintf(stats[TEST_RUNS], "\t\t------------------------------------------\n");
		fprintf(stats[TEST_RUNS], "\t\tPI ESTIMATOR CRITERIA NOT MET! PI = %f\n", pi);
		
		p_value = 0.0;
	}
	else 
	{
		V = 1;
		V2 = 1;

		// Запускаем таймер
		LARGE_INTEGER frequency1;        // ticks per second
		LARGE_INTEGER t11, t21;           // ticks
		double elapsedTime1;	
		//// get ticks per second
		QueryPerformanceFrequency(&frequency1);
		// start timer
		QueryPerformanceCounter(&t11);

		for (k = 1; k < n; k++)
			if (epsilon[k] != epsilon[k-1])
				V2++;

		// stop timer
		QueryPerformanceCounter(&t21);
		// compute and print the elapsed time in millisec
		elapsedTime1 = (t21.QuadPart - t11.QuadPart) * 1000.0 / frequency1.QuadPart;
		//Выводим время выполнения функции на CPU 
		printf ("CPU compute time (находим число переходов): %f\n", elapsedTime1);

		V += funcR(a, n);
		printf ("GPU: V = %lf\n", V);
		printf ("CPU: V2 = %lf\n", V2);

		erfc_arg = fabs(V2 - 2.0 * n * pi2 * (1-pi2)) / (2.0 * pi2 * (1-pi2) * sqrt(2.0*n)); 
		p_value = cephes_erfc(erfc_arg);
		
		fprintf(stats[TEST_RUNS], "\t\t\t\tRUNS TEST\n");
		fprintf(stats[TEST_RUNS], "\t\t------------------------------------------\n");
		fprintf(stats[TEST_RUNS], "\t\tCOMPUTATIONAL INFORMATION:\n");
		fprintf(stats[TEST_RUNS], "\t\t------------------------------------------\n");
		fprintf(stats[TEST_RUNS], "\t\t(a) Pi                        = %f\n", pi);
		fprintf(stats[TEST_RUNS], "\t\t(b) V_n_obs (Total # of runs) = %d\n", (int)V);
		fprintf(stats[TEST_RUNS], "\t\t(c) V_n_obs - 2 n pi (1-pi)\n");
		fprintf(stats[TEST_RUNS], "\t\t    -----------------------   = %f\n", erfc_arg);
		fprintf(stats[TEST_RUNS], "\t\t      2 sqrt(2n) pi (1-pi)\n");
		fprintf(stats[TEST_RUNS], "\t\t------------------------------------------\n");
		if ( isNegative(p_value) || isGreaterThanOne(p_value) )
			fprintf(stats[TEST_RUNS], "WARNING:  P_VALUE IS OUT OF RANGE.\n");

		fprintf(stats[TEST_RUNS], "%s\t\tp_value = %f\n\n", p_value < ALPHA ? "FAILURE" : "SUCCESS", p_value); fflush(stats[TEST_RUNS]);
		
	}

	fprintf(results[TEST_RUNS], "%f\n", p_value); fflush(results[TEST_RUNS]);
}
