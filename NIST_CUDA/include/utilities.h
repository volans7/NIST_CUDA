//typedef unsigned char  BYTE; // В C++ нет типа BYTE. Принято объявлять как строку. так не работает. в заголовке ниже есть какое-то объявление. найти
#include "config.h" //  Чтобы видел BYTE. В C++ нет типа BYTE. Принято объявлять как строку.
#include <stdio.h> // добавила, чтобы FILE определился

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
              U T I L I T Y  F U N C T I O N  P R O T O T Y P E S 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int		displayGeneratorOptions();
int		generatorOptions(char** streamFile);
void	chooseTests();
void	fixParameters();
void	fileBasedBitStreams(char *streamFile);
void	readBinaryDigitsInASCIIFormat(FILE *fp, char *streamFile);
void	readHexDigitsInBinaryFormat(FILE *fp);
int		convertToBits(BYTE *x, int xBitLength, int bitsNeeded, int *num_0s, int *num_1s, int *bitsRead);
void	openOutputStreams(int option);
void	invokeTestSuite(int option, char *streamFile);
void	nist_test_suite();
