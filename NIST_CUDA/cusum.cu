#include <stdio.h>
#include <stdlib.h>
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
		    C U M U L A T I V E  S U M S  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define BLOCK_SIZE 8 // это максимально возможное число нитей в блоке. 128, 256, 512, 1024. 4
//#define N (256*256) // размер массива
#define N (16) // размер массива. N = n
#define LOG_NUM_BANKS 4 // логарифм числа банков 16 по основанию 2
#define CONFLICT_FREE_OFFS(i) ((i) >> LOG_NUM_BANKS) // макрос. сдвиг вправо на LOG_NUM_BANKS разрядов. 16 = 10000 -> 1

__global__ void scan (int * inData, int * outData, int * sums, int n)
{
	__shared__ int temp [2*BLOCK_SIZE + CONFLICT_FREE_OFFS(2*BLOCK_SIZE)];

	int tid = threadIdx.x;
	int offset = 1;
	int ai = tid;
	int bi = tid + (n / 2);
	int offsA = CONFLICT_FREE_OFFS(ai);
	int offsB = CONFLICT_FREE_OFFS(bi);

	temp[ai + offsA] = inData[ai + 2*BLOCK_SIZE*blockIdx.x];
	temp[bi + offsB] = inData[bi + 2*BLOCK_SIZE*blockIdx.x];

	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if(tid < d)
		{
			//подсчитываем индексы для сложения элементов (как в reduce)
			int ai = offset * (2 * tid + 1) - 1; 
			int bi = offset * (2 * tid + 2) - 1;

			ai += CONFLICT_FREE_OFFS(ai);
			bi += CONFLICT_FREE_OFFS(bi);
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	// сохраняем результат в массив сумм
	if (tid == 0)
	{
		int i = n - 1 + CONFLICT_FREE_OFFS(n-1);
		sums[blockIdx.x] = temp[i]; // сохраняем результат из крайнего правого элемента массива
		temp[i] = 0; // зануляем крайний правый элемент 
	}
	
	for (int d = 1; d < n; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			int t;
			ai += CONFLICT_FREE_OFFS(ai);
			bi += CONFLICT_FREE_OFFS(bi);
			t = temp[ai]; // сохраняем элемент 
			temp[ai] = temp[bi]; // на его место записываем другой
			temp[bi] += t; // на место другого записываем сумму другого с элементом из хранилища
		}
	}
	
	__syncthreads();

	outData[ai + 2*BLOCK_SIZE*blockIdx.x] = temp [ai + offsA]; 
	outData[bi + 2*BLOCK_SIZE*blockIdx.x] = temp [bi + offsB];	
}

// Ядро, осуществляющее коррекцию массива
__global__ void scanDistribute(int * data, int * sums)
{
	data[threadIdx.x + blockIdx.x*2*BLOCK_SIZE] += sums [blockIdx.x];
}

// Reduction (min/max), только для blockDim.x = степени 2:
__global__ void funcMin (int nn, int * inData, int * outData)
{
	int  thread2;
	double temp;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ double min[BLOCK_SIZE];
	int nTotalThreads = blockDim.x;	// число нитей в блоке
	int halfPoint = (nTotalThreads >> 1);

	// Получаем входные данные
	min[threadIdx.x] = inData[i];
	__syncthreads();

	while(nTotalThreads > 1)
	{		
		halfPoint = (nTotalThreads >> 1);	// делим на 2	

		if (threadIdx.x < halfPoint) // только первая половина нитей будет активна
		{
			thread2 = threadIdx.x + halfPoint;
 
			// получаем значение из разделяемой памяти для другой нити
			temp = min[thread2];
			if (temp < min[threadIdx.x]) 
				min[threadIdx.x] = temp; 		 
		}
		__syncthreads();
 
		// уменьшаем размер бинарного дерева на 2:
		nTotalThreads = halfPoint;
	}

	if (threadIdx.x == 0) // сохранить результат
		outData [blockIdx.x] = min[threadIdx.x];
}

__global__ void funcMax (int nn, int * inData, int * outData)
{
	int  thread2;
	double temp;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ double max[BLOCK_SIZE];
	int nTotalThreads = blockDim.x;	// число нитей в блоке
	int halfPoint = (nTotalThreads >> 1);

	// Получаем входные данные
	max[threadIdx.x] = inData[i];
	__syncthreads();

	while(nTotalThreads > 1)
	{		
		halfPoint = (nTotalThreads >> 1);	// делим на 2	

		if (threadIdx.x < halfPoint) // только первая половина нитей будет активна
		{
			thread2 = threadIdx.x + halfPoint;
 
			// получаем значение из разделяемой памяти для другой нити
			temp = max[thread2];
			if (temp > max[threadIdx.x]) 
				max[threadIdx.x] = temp; 		 
		}
		__syncthreads();
 
		// уменьшаем размер бинарного дерева на 2:
		nTotalThreads = halfPoint;
	}

	if (threadIdx.x == 0) // сохранить результат
		outData [blockIdx.x] = max[threadIdx.x];
}

//Осуществить scan для заданного массива
void scan (int * inData, int * outData, int n)
{
	int numBlocks = n / (2*BLOCK_SIZE);
	int * sums = NULL; // суммы элементов для каждого блока
	int * sums2 = NULL; // результат scan'а этих сумм

	if (numBlocks < 1)
		numBlocks = 1;

	cudaMalloc ((void**)&sums, numBlocks * sizeof (int));
	cudaMalloc ((void**)&sums2, numBlocks * sizeof (int));

	// осуществляем поблочный scan. одна нить на два элемента
	dim3 threads (BLOCK_SIZE, 1, 1);
	dim3 blocks (numBlocks, 1, 1);

	scan<<<blocks, threads>>>(inData, outData, sums, 2*BLOCK_SIZE);
	
	// выполняем scan для сумм
	if (n >= 2*BLOCK_SIZE)
		scan(sums, sums2, numBlocks);
	else 
		cudaMemcpy(sums2, sums, numBlocks*sizeof(int), cudaMemcpyDeviceToDevice);
	
	// Корректируем результаты
	threads = dim3 (2*BLOCK_SIZE, 1, 1);
	blocks = dim3 (numBlocks - 1, 1, 1);

	scanDistribute<<<blocks, threads>>>(outData + 2*BLOCK_SIZE, sums2 + 1);

	cudaFree(sums);
	cudaFree(sums2);
}

int funcMin(int n, int * b)
{
	int i;
	// Выделить память на CPU
	int numBlocks = n / BLOCK_SIZE;
	int * minD = new int [numBlocks];

	// выделяем память на GPU
	int * minDev = NULL;
	int * inDev = NULL;
	cudaMalloc ((void**)&minDev, numBlocks*sizeof(int));
	cudaMalloc ((void**)&inDev, n*sizeof(int));
	
	cudaMemcpy(inDev, b, n*sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads (BLOCK_SIZE, 1, 1);
	dim3 blocks (numBlocks, 1, 1);
	funcMin<<<blocks, threads>>>(n, inDev, minDev);

	// копируем результат обратно в CPU
	cudaMemcpy (minD, minDev, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
	/*
	printf("Минимумы и максимумы в блоках:\n");
	for (i = 0; i < numBlocks; i++)
		printf("%d) min=%d ", i, minD[i]);
	*/

	int tempMin = minD[0];
	if (numBlocks > BLOCK_SIZE)
	{
		cudaFree(minDev);
		cudaFree(inDev);
		funcMin(numBlocks, minD);	
	}
	else
	{
		for (i = 0; i < numBlocks; i++)
			if (minD[i] < tempMin)
				tempMin = minD[i];
		printf("Result min = %d\n", tempMin);
		delete [] minD;
		cudaFree(minDev);
		cudaFree(inDev);
		return tempMin;
	}
}

int funcMax(int n, int * b)
{
	int i;
	// Выделить память на CPU
	int numBlocks = n / BLOCK_SIZE;
	int * maxD = new int [numBlocks];

	// выделяем память на GPU
	int * inDev = NULL;
	int * maxDev = NULL;
	cudaMalloc ((void**)&inDev, n*sizeof(int));
	cudaMalloc ((void**)&maxDev, numBlocks*sizeof(int));
	
	cudaMemcpy(inDev, b, n*sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads (BLOCK_SIZE, 1, 1);
	dim3 blocks (numBlocks, 1, 1);
	funcMax<<<blocks, threads>>>(n, inDev, maxDev);

	// копируем результат обратно в CPU
	cudaMemcpy (maxD, maxDev, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
	/*
	printf("Минимумы и максимумы в блоках:\n");
	for (i = 0; i < numBlocks; i++)
		printf("%d) max=%d ", i, maxD[i]);
	*/

	int tempMax = maxD[0];
	if (numBlocks > BLOCK_SIZE)
	{
		cudaFree(maxDev);
		cudaFree(inDev);
		funcMax(numBlocks, maxD);		
	}
	else
	{
		for (i = 0; i < numBlocks; i++)
			if (maxD[i] > tempMax)
				tempMax = maxD[i];
		printf("Result max = %d\n", tempMax);
		delete [] maxD;
		cudaFree(maxDev);
		cudaFree(inDev);
		return tempMax;
	}
}

int
Cusum(int n)
{
	int numBytes = n * sizeof (int);
	int i = 0;
	int sup, inf, zrev, z;

	int * a = new int [n];
	int * b = new int [n+1];

	// Создаем массивы 
	//printf("Data: \n");
	int tempSum = 0;
	for (int ii = 0; ii < n; ii++)
	{
		a[ii] = 2*epsilon[ii]-1; 
		tempSum +=a[ii];
		//printf("a[%d] = %d\n", ii, a[ii]);
		//fprintf(results[TEST_CUSUM], "%d) %d\n", ii, tempSum);
	}
	printf("Настоящая кумулятивная сумма: %d\n", tempSum);

	int * adev[2] = {NULL, NULL};

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaMalloc((void**)&adev[0], numBytes);
	cudaMalloc((void**)&adev[1], numBytes);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	cudaMemcpy(adev[0], a, numBytes, cudaMemcpyHostToDevice);

	scan(adev[0], adev[1], n);

	cudaMemcpy (b, adev[1], numBytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	/*
	for (i = 0; i < n; i++)
		if (fabs (b[i] - 1) > 0,0001)
			printf("item at %d diff %f -> (%f %d)\n", i, b[i] - i, b[i], i);
*/
	for (i = 1; i < n; i++)
		b[i-1] = b[i];
	b[n-1] = b[n-2] + a[n-1];
	//for (i = 0; i < n; i++)
	//	printf("%d) %d\n", i, b[i]);
		//fprintf(stats[TEST_CUSUM], "%d) %d\n", i, b[i]);

	printf("Кумулятивная сумма: %d\n", b[n-1]);
	printf("\n");

	printf("time spent executing by the GPU: %.2f milliseconds\n", gpuTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Вычисляем z
	inf = funcMin(n, b);
	sup = funcMax(n, b);

	z = (sup > -inf) ? sup : -inf;		
	zrev = (sup - b[n-1] > b[n-1] - inf) ? sup - b[n-1] : b[n-1] - inf;

	int tmpMin = b[0];
	int tmpMax = b[0];
	for (i = 0; i < n; i++)
	{
		if (b[i] < tmpMin)
			tmpMin = b[i];
		if (b[i] > tmpMax)
			tmpMax = b[i];
	}
	printf("Настоящий минумум = %d \n", tmpMin);
	printf("Настоящий максимум = %d \n", tmpMax);

	cudaFree(adev[0]);
	cudaFree(adev[1]);

	delete a; 
	delete b;
	
	//for(int i = 0; i < n; i++)
	//	printf("%d ", d[i]);
	
	printf("Press key...\n");
	getchar();
	
	return 0;
}

__global__ void epsilonToX2 (int nn, int * inData, int * outData)
{
	int tid = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	outData[i] = 2 * inData[i] - 1;
}

void
CumulativeSums(int n)
{
	int		S, sup, inf, z, zrev, k;
	double	sum1, sum2, p_value;

	// CUDA -------------------------------------------------------------------------------------------
	Cusum(n);
	// END CUDA ---------------------------------------------------------------------------------------
	S = 0;
	sup = 0;
	inf = 0;
	printf("\nCPU result:\n");
	for (k = 0; k < n; k++) 
	{
		epsilon[k] ? S++ : S--;
		//printf("eps=%d S=%d ", epsilon[k], S);
		if (S > sup)
			sup++;
		if (S < inf)
			inf--;
		z = (sup > -inf) ? sup : -inf;		
		zrev = (sup-S > S-inf) ? sup-S : S-inf;
		//printf("%d) S=%d z=%d sup=%d int=%d zrev=%d ", k, S, z, sup, inf, zrev);
	}
	printf("S=%d z=%d sup=%d int=%d zrev=%d\n", S, z, sup, inf, zrev);
	// Проход вперед
	sum1 = 0.0;
	int tempk1 = (-n / z + 1) / 4, tempk2 = (n / z - 1) / 4;
	printf("Границы результирующего цикла: [ %d, %d ]\n", tempk1, tempk2);

	for (k = (-n / z + 1) / 4; k <= (n / z - 1) / 4; k++) 
	{		
		sum1 += cephes_normal(((4 * k + 1) * z) / sqrt((double)n)); 
		//printf("k=%d sum1=%lf ", k, sum1);
		sum1 -= cephes_normal(((4 * k - 1) * z) / sqrt((double)n)); 
		//printf("sum1=%lf\n",sum1);
	}
	sum2 = 0.0;
	tempk1 = (-n / z - 3) / 4; tempk2 =(n / z - 1) / 4;
	printf("Границы результирующего цикла: [ %d, %d ]\n", tempk1, tempk2);
	for (k = (-n / z - 3) / 4; k <=(n / z - 1) / 4; k++ ) {
		sum2 += cephes_normal(((4 * k + 3) * z) / sqrt((double)n)); 
		//printf("k=%d sum2=%lf ", k, sum2);
		sum2 -= cephes_normal(((4 * k + 1) * z) / sqrt((double)n)); 
		//printf("sum2=%lf\n",sum2);
	}

	p_value = 1.0 - sum1 + sum2;
	
	fprintf(stats[TEST_CUSUM], "\t\t      CUMULATIVE SUMS (FORWARD) TEST\n");
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");
	fprintf(stats[TEST_CUSUM], "\t\tCOMPUTATIONAL INFORMATION:\n");
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");
	fprintf(stats[TEST_CUSUM], "\t\t(a) The maximum partial sum = %d\n", z);
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");

	if ( isNegative(p_value) || isGreaterThanOne(p_value) )
		fprintf(stats[TEST_CUSUM], "\t\tWARNING:  P_VALUE IS OUT OF RANGE\n");

	fprintf(stats[TEST_CUSUM], "%s\t\tp_value = %f\n\n", p_value < ALPHA ? "FAILURE" : "SUCCESS", p_value);
	fprintf(results[TEST_CUSUM], "%f\n", p_value);
		
	// Проход назад
	sum1 = 0.0;
	for ( k=(-n/zrev+1)/4; k<=(n/zrev-1)/4; k++ ) {
		sum1 += cephes_normal(((4*k+1)*zrev)/sqrt((double)n)); 
		sum1 -= cephes_normal(((4*k-1)*zrev)/sqrt((double)n)); 
	}
	sum2 = 0.0;
	for ( k=(-n/zrev-3)/4; k<=(n/zrev-1)/4; k++ ) {
		sum2 += cephes_normal(((4*k+3)*zrev)/sqrt((double)n)); 
		sum2 -= cephes_normal(((4*k+1)*zrev)/sqrt((double)n)); 
	}
	p_value = 1.0 - sum1 + sum2;

	fprintf(stats[TEST_CUSUM], "\t\t      CUMULATIVE SUMS (REVERSE) TEST\n");
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");
	fprintf(stats[TEST_CUSUM], "\t\tCOMPUTATIONAL INFORMATION:\n");
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");
	fprintf(stats[TEST_CUSUM], "\t\t(a) The maximum partial sum = %d\n", zrev);
	fprintf(stats[TEST_CUSUM], "\t\t-------------------------------------------\n");

	if ( isNegative(p_value) || isGreaterThanOne(p_value) )
		fprintf(stats[TEST_CUSUM], "\t\tWARNING:  P_VALUE IS OUT OF RANGE\n");

	fprintf(stats[TEST_CUSUM], "%s\t\tp_value = %f\n\n", p_value < ALPHA ? "FAILURE" : "SUCCESS", p_value); fflush(stats[TEST_CUSUM]);
	fprintf(results[TEST_CUSUM], "%f\n", p_value); fflush(results[TEST_CUSUM]);
}