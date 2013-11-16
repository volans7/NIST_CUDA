// Обычно с помощью такого приема из стандартного заголовочного файла для С получают такой файл для С++. 
#if defined(__cplusplus) // if defined - Позволяет исключить двойные включения описания классов
extern "C" { // в extern "C" упоминание С относится к порядку связывания
/* extern "C" - Служит указанием компилятору С++ создавать код строго 'С'. 
То есть не будут добавляться к именам функций дополнительные символы,
что делает компилятор C++.
Это нужно для использования в С-коде.
Ядро Unix и многие системные утилиты разработаны на языке 'С'
Поэтому нужен код С а не С++.
*/
#endif

#ifndef _CONFIG_H_
#define	_CONFIG_H_

//#define	WINDOWS32
//#define	PROTOTYPES
//#define	LITTLE_ENDIAN
//#define	LOWHI

/*
 * AUTO DEFINES (DON'T TOUCH!)
 */

#ifndef	CSTRTD // не понятно что, попробовать без них
typedef char *CSTRTD;
#endif
#ifndef	BSTRTD // не понятно что, попробовать без них
typedef unsigned char *BSTRTD;
#endif

#ifndef	BYTE // тип байт
typedef unsigned char BYTE;
#endif
#ifndef	UINT // не понятно что, попробовать без них
typedef unsigned int UINT;
#endif
#ifndef	USHORT // не понятно что, попробовать без них
typedef unsigned short USHORT;
#endif
#ifndef	ULONG // не понятно что, попробовать без них
typedef unsigned long ULONG;
#endif
#ifndef	DIGIT // не понятно что, попробовать без них
typedef USHORT DIGIT;	/* 16-bit word */
#endif
#ifndef	DBLWORD // не понятно что, попробовать без них
typedef ULONG DBLWORD;  /* 32-bit word */
#endif

#ifndef	WORD64 // не понятно что, попробовать без них
typedef ULONG WORD64[2];  /* 64-bit word */
#endif

#endif /* _CONFIG_H_ */

#if defined(__cplusplus)
}
#endif