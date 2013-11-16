#include <iostream> 
#include <stdio.h> 
#include <cuda_runtime.h> 

#include <math.h>
#include <string.h>
#include "include\decls.h"
#include "include\cephes.h"  
#include "include\utilities.h"

#pragma comment(lib, "cudart") 

void	partitionResultFile(int numOfFiles, int numOfSequences, int option, int testNameID);
void	postProcessResults(int option);
int		cmp(const double *a, const double *b);
int		computeMetrics(char *s, int test);
 
using namespace std; 

#define N 10


__global__ void add2( int *a, int *b, int *c)
{
	int tid = blockIdx.x; // это CPU 0
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

__global__ void add(int a, int b, int *c) // ключевое слово, означающее, что функция должа испольняться на GPU
{ 
    *c=a + b;
} 

int DeviceInfo(void)
{
	int count;
	cudaDeviceProp prop;

	//Информация об устройстве
	cudaGetDeviceCount( &count );
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties (&prop, i);
		printf("--- Общая информация об устройстве %d ---\n", i);
		printf("Имя: %s\n", prop.name);
		printf("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
		printf("Тактовая частота: %d\n", prop.clockRate);
		printf("Перекрытие копирования: ");
		if (prop.deviceOverlap)
			printf("Разрешено\n");
		else
			printf("Запрещено\n");
		printf("Тайм-аут выполнения ядра: ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Включен\n");
		else
			printf("Выключен\n");

		printf("--- Информация о памяти для устройства %d ---\n", i);
		printf("Всего глобальной памяти: %lld\n", prop.totalGlobalMem);
		printf("Всего константной памяти: %ld\n", prop.totalConstMem);
		printf("Максимальный шаг: %ld\n", prop.memPitch);
		printf("Выравнивание текстур: %ld\n", prop.textureAlignment);
		
		printf("--- Информация о мультипроцессорах для устройства %d ---\n", i);
		printf("Количество мультипроцессоров: %d\n", prop.multiProcessorCount);
		printf("Разделяемая память на один МП: %ld\n", prop.sharedMemPerBlock);
		printf("Регистров на один МП: %d\n", prop.regsPerBlock);
		printf("Нитей в варпе: %d\n", prop.warpSize);
		printf("Макс. количество нитей в блоке: %d\n", prop.maxThreadsPerBlock);
		printf("Макс. количество нитей по измерениям: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Максимальные размеры сетки: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		size_t free = 0;
		size_t total = 0;
		cudaMemGetInfo (&free, &total);
		printf("Количество свободной памяти: %lld, всего %lld\n", free, total); 
		printf("\n");	
	}
	return 0;
}


// Main entry into the program 
int main(void) 
{ 
	setlocale (LC_ALL, ".1251"); // для вывода;

	int		i;
	int		option;			/* TEMPLATE LENGTH/STREAM LENGTH/GENERATOR*/
	char	*streamFile;	/* STREAM FILENAME     */

	DeviceInfo();

	// Определяем параметры теста (структура определена в defs.h)
	tp.n = 100; //60000000 int shared memory max. 
	tp.blockFrequencyBlockLength = 128;
	tp.nonOverlappingTemplateBlockLength = 9;
	tp.overlappingTemplateBlockLength = 9;
	tp.approximateEntropyBlockLength = 10;
	tp.serialBlockLength = 16;
	tp.linearComplexitySequenceLength = 500;
	tp.numOfBitStreams = 1;

	option = generatorOptions(&streamFile); // показываем генераторы
	chooseTests(); // выбираем тесты - все или по одному (?)
	fixParameters(); // пользователь может изменить дефолтные параметры
	openOutputStreams(option);
	invokeTestSuite(option, streamFile);

	printf("Press any key...\n");
	getchar();
	
/*	
	

	
	
	fclose(freqfp);
	for( i=1; i<=NUMOFTESTS; i++ ) {
		if ( stats[i] != NULL )
			fclose(stats[i]);
		if ( results[i] != NULL )
			fclose(results[i]);
	}
	if ( (testVector[0] == 1) || (testVector[TEST_CUSUM] == 1) ) 
		partitionResultFile(2, tp.numOfBitStreams, option, TEST_CUSUM);
	if ( (testVector[0] == 1) || (testVector[TEST_NONPERIODIC] == 1) ) 
		partitionResultFile(MAXNUMOFTEMPLATES, tp.numOfBitStreams, option, TEST_NONPERIODIC);
	if ( (testVector[0] == 1) || (testVector[TEST_RND_EXCURSION] == 1) )
		partitionResultFile(8, tp.numOfBitStreams, option, TEST_RND_EXCURSION);
	if ( (testVector[0] == 1) || (testVector[TEST_RND_EXCURSION_VAR] == 1) )
		partitionResultFile(18, tp.numOfBitStreams, option, TEST_RND_EXCURSION_VAR);
	if ( (testVector[0] == 1) || (testVector[TEST_SERIAL] == 1) )
		partitionResultFile(2, tp.numOfBitStreams, option, TEST_SERIAL);
	fprintf(summary, "------------------------------------------------------------------------------\n");
	fprintf(summary, "RESULTS FOR THE UNIFORMITY OF P-VALUES AND THE PROPORTION OF PASSING SEQUENCES\n");
	fprintf(summary, "------------------------------------------------------------------------------\n");
	fprintf(summary, "   generator is <%s>\n", streamFile);
	fprintf(summary, "------------------------------------------------------------------------------\n");
	fprintf(summary, " C1  C2  C3  C4  C5  C6  C7  C8  C9 C10  P-VALUE  PROPORTION  STATISTICAL TEST\n");
	fprintf(summary, "------------------------------------------------------------------------------\n");
	postProcessResults(option);
	fclose(summary);

	*/

	printf("Press any key...\n");
	getchar();
	return 0; 
} 
/*
void
partitionResultFile(int numOfFiles, int numOfSequences, int option, int testNameID)
{ 
	int		i, k, m, j, start, end, num, numread;
	float	c;
	FILE	**fp = (FILE **)calloc(numOfFiles+1, sizeof(FILE *));
	int		*results = (int *)calloc(numOfFiles, sizeof(int *));
	char	*s[MAXFILESPERMITTEDFORPARTITION];
	char	resultsDir[200];
	
	for ( i=0; i<MAXFILESPERMITTEDFORPARTITION; i++ )
		s[i] = (char*)calloc(200, sizeof(char));
	
	sprintf(resultsDir, "experiments/%s/%s/results.txt", generatorDir[option], testNames[testNameID]);
	
	if ( (fp[numOfFiles] = fopen(resultsDir, "r")) == NULL ) {
		printf("%s", resultsDir);
		printf(" -- file not found. Exiting program.\n");
		exit(-1);
	}
	
	for ( i=0; i<numOfFiles; i++ ) {
		if ( i < 10 )
			sprintf(s[i], "experiments/%s/%s/data%1d.txt", generatorDir[option], testNames[testNameID], i+1);
		else if (i < 100)
			sprintf(s[i], "experiments/%s/%s/data%2d.txt", generatorDir[option], testNames[testNameID], i+1);
		else
			sprintf(s[i], "experiments/%s/%s/data%3d.txt", generatorDir[option], testNames[testNameID], i+1);
	}
	numread = 0;
	m = numOfFiles/20;
	if ( (numOfFiles%20) != 0 )
		m++;
	for ( i=0; i<numOfFiles; i++ ) {
		if ( (fp[i] = fopen(s[i], "w")) == NULL ) {
			printf("%s", s[i]);
			printf(" -- file not found. Exiting program.\n");
			exit(-1);
		}
		fclose(fp[i]);
	}
	for ( num=0; num<numOfSequences; num++ ) {
		for ( k=0; k<m; k++ ) { 			// FOR EACH BATCH 
			
			start = k*20;		// BOUNDARY SEGMENTS
			end   = k*20+19;
			if ( k == (m-1) )
				end = numOfFiles-1;
			
			for ( i=start; i<=end; i++ ) {		// OPEN FILE 
				if ( (fp[i] = fopen(s[i], "a")) == NULL ) {
					printf("%s", s[i]);
					printf(" -- file not found. Exiting program.\n");
					exit(-1);
				}
			}
			
			for ( j=start; j<=end; j++ ) {		// POPULATE FILE 
				fscanf(fp[numOfFiles], "%f", &c);
				fprintf(fp[j], "%f\n", c);
				numread++;
			}

			for ( i=start; i<=end; i++ )		// CLOSE FILE
				fclose(fp[i]);
		}
	}
	fclose(fp[numOfFiles]);
	for ( i=0; i<MAXFILESPERMITTEDFORPARTITION; i++ )
		free(s[i]);

	return;
}

int
cmp(const double *a, const double *b)
{
	if ( *a < *b )
		return -1;
	if ( *a == *b )
		return 0;
	return 1;
}


void
postProcessResults(int option)
{
	int		i, k, randomExcursionSampleSize, generalSampleSize;
	int		passRate, case1, case2, numOfFiles = 2;
	double	p_hat;
	char	s[200];
	
	for ( i=1; i<=NUMOFTESTS; i++ ) {		// FOR EACH TEST
		if ( testVector[i] ) {
			// SPECIAL CASES -- HANDLING MULTIPLE FILES FOR A SINGLE TEST
			if ( ((i == TEST_CUSUM) && testVector[TEST_CUSUM] ) ||
				 ((i == TEST_NONPERIODIC) && testVector[TEST_NONPERIODIC] ) ||
				 ((i == TEST_RND_EXCURSION) && testVector[TEST_RND_EXCURSION]) ||
				 ((i == TEST_RND_EXCURSION_VAR) && testVector[TEST_RND_EXCURSION_VAR]) || 
				 ((i == TEST_SERIAL) && testVector[TEST_SERIAL]) ) {
				
				if ( (i == TEST_NONPERIODIC) && testVector[TEST_NONPERIODIC] )  
					numOfFiles = MAXNUMOFTEMPLATES;
				else if ( (i == TEST_RND_EXCURSION) && testVector[TEST_RND_EXCURSION] ) 
					numOfFiles = 8;
				else if ( (i == TEST_RND_EXCURSION_VAR) && testVector[TEST_RND_EXCURSION_VAR] ) 
					numOfFiles = 18;
				else
					numOfFiles = 2;
				for ( k=0; k<numOfFiles; k++ ) {
					if ( k < 10 )
						sprintf(s, "experiments/%s/%s/data%1d.txt", generatorDir[option], testNames[i], k+1);
					else if ( k < 100 )
						sprintf(s, "experiments/%s/%s/data%2d.txt", generatorDir[option], testNames[i], k+1);
					else
						sprintf(s, "experiments/%s/%s/data%3d.txt", generatorDir[option], testNames[i], k+1);
					if ( (i == TEST_RND_EXCURSION) || (i == TEST_RND_EXCURSION_VAR) ) 
						randomExcursionSampleSize = computeMetrics(s,i);
					else
						generalSampleSize = computeMetrics(s,i);
				}
			}
			else {
				sprintf(s, "experiments/%s/%s/results.txt", generatorDir[option], testNames[i]);
				generalSampleSize = computeMetrics(s,i);
			}
		}
	}

	fprintf(summary, "\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	case1 = 0;
	case2 = 0;
	if ( testVector[TEST_RND_EXCURSION] || testVector[TEST_RND_EXCURSION_VAR] ) 
		case2 = 1;
	for ( i=1; i<=NUMOFTESTS; i++ ) {
		if ( testVector[i] && (i != TEST_RND_EXCURSION) && (i != TEST_RND_EXCURSION_VAR) ) {
			case1 = 1;
			break;
		}
	}
	p_hat = 1.0 - ALPHA;
	if ( case1 ) {
		if ( generalSampleSize == 0 ) {
			fprintf(summary, "The minimum pass rate for each statistical test with the exception of the\n");
			fprintf(summary, "random excursion (variant) test is undefined.\n\n");
		}
		else {
			passRate = (p_hat - 3.0 * sqrt((p_hat*ALPHA)/generalSampleSize)) * generalSampleSize;
			fprintf(summary, "The minimum pass rate for each statistical test with the exception of the\n");
			fprintf(summary, "random excursion (variant) test is approximately = %d for a\n", generalSampleSize ? passRate : 0);
			fprintf(summary, "sample size = %d binary sequences.\n\n", generalSampleSize);
		}
	}
	if ( case2 ) {
		if ( randomExcursionSampleSize == 0 )
			fprintf(summary, "The minimum pass rate for the random excursion (variant) test is undefined.\n\n");
		else {
			passRate = (p_hat - 3.0 * sqrt((p_hat*ALPHA)/randomExcursionSampleSize)) * randomExcursionSampleSize;
			fprintf(summary, "The minimum pass rate for the random excursion (variant) test\n");
			fprintf(summary, "is approximately = %d for a sample size = %d binary sequences.\n\n", passRate, randomExcursionSampleSize);
		}
	}
	fprintf(summary, "For further guidelines construct a probability table using the MAPLE program\n");
	fprintf(summary, "provided in the addendum section of the documentation.\n");
	fprintf(summary, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

int
computeMetrics(char *s, int test)
{
	int		j, pos, count, passCount, sampleSize, expCount, proportion_threshold_min, proportion_threshold_max;
	int		freqPerBin[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	double	*A, *T, chi2, proportion, uniformity, p_hat, tmp;
	float	c;
	FILE	*fp;
	
	if ( (fp = fopen(s, "r")) == NULL ) {
		printf("%s",s);
		printf(" -- file not found. Exiting program.\n");
		exit(-1);
	}
	
	if ( (A = (double *)calloc(tp.numOfBitStreams, sizeof(double))) == NULL ) {
		printf("Final Analysis Report aborted due to insufficient workspace\n");
		return 0;
	}
	
	// Compute Metric 1: Proportion of Passing Sequences 
	
	count = 0; 		
	sampleSize = tp.numOfBitStreams;
	
	if ( (test == TEST_RND_EXCURSION) || (test == TEST_RND_EXCURSION_VAR) ) { // Special Case: Random Excursion Tests 
		if ( (T = (double *)calloc(tp.numOfBitStreams, sizeof(double))) == NULL ) {
			printf("Final Analysis Report aborted due to insufficient workspace\n");
			return 0;
		}
		for ( j=0; j<sampleSize; j++ ) {
			fscanf(fp, "%f", &c);
			if ( c > 0.000000 )
				T[count++] = c;
		}
		
		if ( (A = (double *)calloc(count, sizeof(double))) == NULL ) {
			printf("Final Analysis Report aborted due to insufficient workspace\n");
			return 0;
		}
		
		for ( j=0; j<count; j++ )
			A[j] = T[j];
		
		sampleSize = count;
		count = 0;
		for ( j=0; j<sampleSize; j++ )
			if ( A[j] < ALPHA )
				count++;
		free(T);
	}
	else {
		if ( (A = (double *)calloc(sampleSize, sizeof(double))) == NULL ) {
			printf("Final Analysis Report aborted due to insufficient workspace\n");
			return 0;
		}
		for ( j=0; j<sampleSize; j++ ) {
			fscanf(fp, "%f", &c);
			if ( c < ALPHA )
				count++;
			A[j] = c;
		}
	}
	if ( sampleSize == 0 )
		passCount = 0;
	else
		passCount = sampleSize - count;
	
	p_hat = 1.0 - ALPHA;
	proportion_threshold_max = (p_hat + 3.0 * sqrt((p_hat*ALPHA)/sampleSize)) * sampleSize;
	proportion_threshold_min = (p_hat - 3.0 * sqrt((p_hat*ALPHA)/sampleSize)) * sampleSize;
	
	// Compute Metric 2: Histogram 
	
	//qsort((void *)A, sampleSize, sizeof(double), (void *)cmp); --------------------------------------------------------------------------------- ПОЧИНИТЬ!

	for ( j=0; j<sampleSize; j++ ) {
		pos = (int)floor(A[j]*10);
		if ( pos == 10 )
			pos--;
		freqPerBin[pos]++;
	}
	chi2 = 0.0;
	expCount = sampleSize/10;
	if ( expCount == 0 )
		uniformity = 0.0;
	else {
		for ( j=0; j<10; j++ )
			chi2 += pow(freqPerBin[j]-expCount, 2)/expCount;
		uniformity = cephes_igamc(9.0/2.0, chi2/2.0);
	}
	
	for ( j=0; j<10; j++ )			// DISPLAY RESULTS 
		fprintf(summary, "%3d ", freqPerBin[j]);
	
	if ( expCount == 0 )
		fprintf(summary, "    ----    ");
	else if ( uniformity < 0.0001 )
		fprintf(summary, " %8.6f * ", uniformity);
	else
		fprintf(summary, " %8.6f   ", uniformity);
	
	if ( sampleSize == 0 )
		fprintf(summary, " ------     %s\n", testNames[test]);
	//	else if ( proportion < 0.96 )
	else if ( (passCount < proportion_threshold_min) || (passCount > proportion_threshold_max))
		fprintf(summary, "%4d/%-4d *  %s\n", passCount, sampleSize, testNames[test]);
	else
		fprintf(summary, "%4d/%-4d    %s\n", passCount, sampleSize, testNames[test]);
	
	fclose(fp);
	free(A);
	
	return sampleSize;
}
*/