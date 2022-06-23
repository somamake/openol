#ifndef PARA_H
#endif
#include "openol.h"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "/Users/somafujimori/codes/holo"
#endif
typedef float PREC_T;

// #define RECT_TEST

#ifndef RECT_TEST 
const int WIDTH = 1920 * 2;
const int HEIGHT = 1080 * 2;
PREC_T p = 3.74e-6 ;
PREC_T lambda = 532e-9;
PREC_T d=-0.1 + 0;

// const size_t WIDTH = 256 * 4 * 8 * 4;
// const size_t HEIGHT = 256 * 4 * 8 * 4;
// PREC_T p = 1.0e-6;
// PREC_T lambda = 520e-9;
// PREC_T d=0.05+ 0;

// const size_t WIDTH = 256 * 4 * 2;
// const size_t HEIGHT = 256 * 4 * 2;
// PREC_T p = 4.0e-6;
// PREC_T lambda = 633e-9;
// PREC_T d=0.05+ 0;

// const int WIDTH = 256 * 4;
// const int HEIGHT = 256 * 4;
// PREC_T p = 8.0e-6;
// PREC_T lambda = 633e-9;
// PREC_T d=0.10+ 0;
char path[] = PROJECT_ROOT "/out/olObject.bmp";
#endif

#ifdef RECT_TEST
const int WIDTH = 1920;
const int HEIGHT = 1080;
const PREC_T p=3.74e-6;	//サンプリング間隔
const PREC_T lambda=633.0e-9;	//赤色レーザ波長
const PREC_T d=0.01;	//開口面と観察面間の距離

#endif
const PREC_T s = 1.0;

#define PARA_H