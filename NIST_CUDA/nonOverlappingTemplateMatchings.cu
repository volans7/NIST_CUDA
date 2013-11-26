#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "include\externs.h"
#include "include\utilities.h"
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
          N O N O V E R L A P P I N G  T E M P L A T E  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define BLOCK_SIZE 10 // = M это максимально возможное число нитей в блоке. 128, 256, 512, 1024

__global__ void TemplateMatchings (int nn, int m, int * inData, int * pattern, int * W)
{
	__shared__ int data[BLOCK_SIZE];
	int tid = threadIdx.x;
	int M = BLOCK_SIZE;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int match = 1;
	int k;

	data[tid] = inData[i];
	__syncthreads();

	for (k = 0; k < m; k++)
	{
		if (i + k < nn)
			if((int)data[tid + k] != (int)pattern[k])
			{
				match = 0;
				break;
			}
	}
	if ( match == 1 && i + k < nn)
		W[i] = 1;
	else 
		W[i] = 0;
}

int TemplateMatchings (int n, int m, int N, int M, int * data, int * pattern)
{
	int numBytes = n * sizeof (int);
	int NumThreads = BLOCK_SIZE; // количество нитей в блоке
	int NumBloks = n / NumThreads; 
	
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
	int * patDev = NULL;
	int * outDev = NULL;

	// выделяем память на GPU
	cudaMalloc ((void**)&inDev, n * sizeof (int));
	cudaMalloc ((void**)&patDev, m * sizeof (int));
	cudaMalloc ((void**)&outDev, n * sizeof (int));

	// копируем данные на GPU
	cudaMemcpyAsync (inDev, data, n * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync (patDev, pattern, m * sizeof (int), cudaMemcpyHostToDevice);

	// запуск ядра
	dim3 threads = dim3(NumThreads);
	dim3 bloks = dim3(NumBloks);
	
	// запуск ядра 
	TemplateMatchings<<<bloks, threads>>> (n, m, inDev, patDev, outDev); // находим сумму

	// копируем результат обратно в CPU
	cudaMemcpy (outD, outDev, numBytes, cudaMemcpyDeviceToHost);

	// освобождаем память
	cudaFree (inDev);
	cudaFree (outDev);

	for (int i = 0; i < n; i++)
	{
	//	printf("%d) %d ", i, outD[i]);
		sum += outD[i];
	}
	delete [] outD;
	
	return sum;
}


void
NonOverlappingTemplateMatchings(int m, int n)
{
	int		numOfTemplates[100] = {0, 0, 2, 4, 6, 12, 20, 40, 74, 148, 284, 568, 1116,
						2232, 4424, 8848, 17622, 35244, 70340, 140680, 281076, 562152};
	/*----------------------------------------------------------------------------
	NOTE:  Should additional templates lengths beyond 21 be desired, they must 
	first be constructed, saved into files and then the corresponding 
	number of nonperiodic templates for that file be stored in the m-th 
	position in the numOfTemplates variable.
	----------------------------------------------------------------------------*/
	unsigned int	bit, W_obs, nu[6], *Wj = NULL; 
	FILE			*fp;
	double			sum, chi2, p_value, lambda, pi[6], varWj;
	int				i, j, jj, k, match, SKIP, M, N, K = 5;
	char			directory[100];
	BitSequence		*sequence = NULL;

	int summ; // my tmp
	int * a = new int [n]; 
	int * pat = new int [n];

	N = 8; // частей последовательности
	M = n / N; // длина подпоследовательности

	// копируем эпсилон
	for(int z = 0; z < n; z++)
		a[z] = epsilon[z]; 
	

	if ( (Wj = (unsigned int*)calloc(N, sizeof(unsigned int))) == NULL ) {
		fprintf(stats[TEST_NONPERIODIC], "\tNONOVERLAPPING TEMPLATES TESTS ABORTED DUE TO ONE OF THE FOLLOWING : \n");
		fprintf(stats[TEST_NONPERIODIC], "\tInsufficient memory for required work space.\n");
		return;
	}
	lambda = (M - m + 1) / pow(2.0, m); // lambda = мю. m - размер шаблона в битах
	varWj = M * (1.0 / pow(2.0, m) - (2.0 * m - 1.0) / pow(2.0, 2.0 * m)); // сигма
	sprintf(directory, "templates/template%d", m); // обращаемся к шаблону по его длине
	// если подпоследовательность меньше, чем  шаблон
	if ( ((isNegative(lambda)) || (isZero(lambda))) ||
		 ((fp = fopen(directory, "r")) == NULL) ||
		 ((sequence = (BitSequence *) calloc(m, sizeof(BitSequence))) == NULL) ) 
	{
		fprintf(stats[TEST_NONPERIODIC], "\tNONOVERLAPPING TEMPLATES TESTS ABORTED DUE TO ONE OF THE FOLLOWING : \n");
		fprintf(stats[TEST_NONPERIODIC], "\tLambda (%f) not being positive!\n", lambda);
		fprintf(stats[TEST_NONPERIODIC], "\tTemplate file <%s> not existing\n", directory);
		fprintf(stats[TEST_NONPERIODIC], "\tInsufficient memory for required work space.\n");
		if ( sequence != NULL )
			free(sequence);
	}
	else 
	{
		fprintf(stats[TEST_NONPERIODIC], "\t\t  NONPERIODIC TEMPLATES TEST\n");
		fprintf(stats[TEST_NONPERIODIC], "-------------------------------------------------------------------------------------\n");
		fprintf(stats[TEST_NONPERIODIC], "\t\t  COMPUTATIONAL INFORMATION\n");
		fprintf(stats[TEST_NONPERIODIC], "-------------------------------------------------------------------------------------\n");
		fprintf(stats[TEST_NONPERIODIC], "\tLAMBDA = %f\tM = %d\tN = %d\tm = %d\tn = %d\n", lambda, M, N, m, n);
		fprintf(stats[TEST_NONPERIODIC], "-------------------------------------------------------------------------------------\n");
		fprintf(stats[TEST_NONPERIODIC], "\t\tF R E Q U E N C Y\n");
		fprintf(stats[TEST_NONPERIODIC], "Template   W_1  W_2  W_3  W_4  W_5  W_6  W_7  W_8    Chi^2   P_value Assignment Index\n");
		fprintf(stats[TEST_NONPERIODIC], "-------------------------------------------------------------------------------------\n");

		if ( numOfTemplates[m] < MAXNUMOFTEMPLATES )
			SKIP = 1;
		else
			SKIP = (int)(numOfTemplates[m]/MAXNUMOFTEMPLATES);
		numOfTemplates[m] = (int)numOfTemplates[m]/SKIP;
		sum = 0.0;
		for (i = 0; i < 2; i++) 
		{                      // Compute Probabilities 
			pi[i] = exp(-lambda+i*log(lambda)-cephes_lgam(i+1));
			sum += pi[i];
		}
		pi[0] = sum;
		for (i = 2; i <= K; i++) 
		{                      // Compute Probabilities 
			pi[i-1] = exp(-lambda+i*log(lambda)-cephes_lgam(i+1));
			sum += pi[i-1];
		}
		pi[K] = 1 - sum;

		printf("Количество шаблонов: %d Размер шаблона: %d", MIN(MAXNUMOFTEMPLATES, numOfTemplates[m]), m);
		for(jj = 0; jj < MIN(MAXNUMOFTEMPLATES, numOfTemplates[m]); jj++) // проходим по всем шаблонам размера m		
		{
			sum = 0;
			printf("\nШаблон: ");
			for (k = 0; k < m; k++) // считываем шаблон
			{
				fscanf(fp, "%d", &bit);
				sequence[k] = bit;

				// мой код -----------------------------------------------------------------------------------------------------------------------
				pat[k] = sequence[k];
				printf("%d", pat[k]);
				//--------------------------------------------------------------------------------------------------------------------------------
				fprintf(stats[TEST_NONPERIODIC], "%d", sequence[k]);
			}
			//printf("\nОбработка...");
			summ = TemplateMatchings (n, m, N, M, a, pat);
			printf(" Результат %d) %d", jj, summ);
			//------------------------------------------------------------------------------------------------------------------------------------
			fprintf(stats[TEST_NONPERIODIC], " ");
			//for (k = 0; k <= K; k++)
			//	nu[k] = 0;
			for (i = 0; i < N; i++) // для каждой подпоследовательности 
			{
				W_obs = 0;
				for (j = 0; j < M - m + 1; j++) // проходим всю подпоследовательность
				{
					match = 1;
					for (k = 0; k < m; k++) // побитово сравниваем 
					{
						if ((int)sequence[k] != (int)epsilon[i*M+j+k]) 
						{
							match = 0;
							//printf("%d != %d ", sequence[k], epsilon[i*M+j+k]);
								break;
						}
					}
					if ( match == 1 )
						W_obs++;
				}
				Wj[i] = W_obs;
			}
			sum = 0;
			chi2 = 0.0;                                   /* Compute Chi Square */
			for (i = 0; i < N; i++) 
			{
				if ( m == 10 )
					fprintf(stats[TEST_NONPERIODIC], "%3d  ", Wj[i]);
				else
					fprintf(stats[TEST_NONPERIODIC], "%4d ", Wj[i]);
				chi2 += pow(((double)Wj[i] - lambda)/pow(varWj, 0.5), 2);
			}
			p_value = cephes_igamc(N/2.0, chi2/2.0);
			if ( isNegative(p_value) || isGreaterThanOne(p_value) )
				fprintf(stats[TEST_NONPERIODIC], "\t\tWARNING:  P_VALUE IS OUT OF RANGE.\n");

			fprintf(stats[TEST_NONPERIODIC], "%9.6f %f %s %3d\n", chi2, p_value, p_value < ALPHA ? "FAILURE" : "SUCCESS", jj);
			if ( SKIP > 1 )
				fseek(fp, (long)(SKIP-1)*2*m, SEEK_CUR);
			fprintf(results[TEST_NONPERIODIC], "%f\n", p_value); fflush(results[TEST_NONPERIODIC]);
		}
	}
	
	fprintf(stats[TEST_NONPERIODIC], "\n"); fflush(stats[TEST_NONPERIODIC]);
	if ( sequence != NULL )
		free(sequence);

	free(Wj);

	fclose(fp);
}
