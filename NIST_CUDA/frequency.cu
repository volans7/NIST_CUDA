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
                          F R E Q U E N C Y  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define BLOCK_SIZE 1024 // это максимально возможное число нитей в блоке. 128, 256, 512, 1024

__global__ void reduce (int nn, int * inData, int * outData)
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

int reduce (int * data, int n)
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
	reduce<<<bloks, threads>>> (n, inDev, outDev); // находим сумму

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
	if (NewNumBloks > BLOCK_SIZE) // если блоков много, то запускаем повторное суммирование на GPU
		sum = reduce(outD, NewNumBloks);
	else // если мало, то подсчитываем результат на CPU
	{			
		for (int i = 0; i < NewNumBloks; i++)
			sum += outD[i];
		delete [] outD;
	}
		
	// Информация о памяти
	cudaMemGetInfo (&free, &total);
	printf("Количество свободной памяти: %lld, всего %lld\n", free, total); 

	cudaFree(outDev);
	//cudaFree(inDev2);
	cudaMemGetInfo (&free, &total);
	printf("Количество свободной памяти: %lld, всего %lld\n", free, total); 

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
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	int NumBloks = ceil((float) n / NumThreads); // высчитываем количество блоков при заданном n и размере блока с округлением в большую сторону

	int sum;
	double	f, s_obs, p_value, sqrt2 = 1.41421356237309504880;
	
	// Выбрать первый GPU для работы
	cudaSetDevice(0);

	// Выделить память на CPU
	int * a = new int [n];
	int * aa = new int [n];
	int * d = new int [n];
	int * dd = new int [n];

	// Создаем массивы 
	printf("Data: \n");
	for (int i = 0; i < n; i++)
	{
		a[i] = epsilon[i]; 
		aa[i] = epsilon[i];
	}
	
	// выделяем память на GPU
	int * aDev = NULL;
	int * dDev = NULL;

	// выделяем память на GPU
	cudaMalloc ((void**)&aDev, numBytes);
	cudaMalloc ((void**)&dDev, numBytes);
	
	// копируем данные на GPU
	cudaMemcpyAsync (aDev, a, numBytes, cudaMemcpyHostToDevice);

	// запуск ядра
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumBloks);
	
	// запуск ядра
	epsilonToX<<<bloks, threads>>> (n, aDev, dDev); // находим x

	// освобождаем память
	cudaFree (aDev);
	cudaMemcpy (d, dDev, numBytes, cudaMemcpyDeviceToHost);
	// освобождаем память
	cudaFree (dDev);

	// Запускаем таймер
	int startGPU = GetTickCount();      
	sum = reduce (d, n);	
	//Выводим время выполнения функции на CPU (в миллиекундах)
    printf ("GPU compute time: %i\n", GetTickCount() - startGPU);

	// вычисляем s(obs)
	s_obs = fabs(1.0*sum)/sqrt(1.0*n); // умножаем на 1.0, чтобы компилятор понимал, какую функцию использовать (тип вещественный)
	f = s_obs/sqrt2;
	p_value = cephes_erfc(1.0*f);

	// освобождаем память
	cudaFree (aDev);
	cudaFree (dDev);

	//Выводим результат
	printf("\n Result sumA = %d, NumBloks = %d, s(obs)=%lf, P-value=%lf\n", sum, NumBloks, s_obs, p_value);

	// Запускаем таймер
	int startCPU = GetTickCount();      
	// Те же вычисления на CPU
	int summ = 0;
	for (int i = 0; i < n; i++)
	{
		aa[i] = 2 * aa[i] - 1;
		summ += aa[i];
	}
	//Выводим время выполнения функции на CPU (в миллиекундах)
    printf ("CPU compute time: %i\n", GetTickCount() - startCPU);

	printf("Result sumAA = %d\n", summ);
	
	delete [] a;
	delete [] d;

	printf("Press key...\n");
	getchar();
	
}

