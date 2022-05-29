#include <stdio.h>
#include <iostream>
#include "openol.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <chrono>
#ifndef PROJECT_ROOT
#define PROJECT_ROOT
#endif

///////
#define WIDTH 1920
#define HEIGHT 1080

typedef double PREC_T;

int main(){
    // clock_t start, end;
    char path3d[] = PROJECT_ROOT "/data/cube284.3d";
    std::unique_ptr<PREC_T[]> x,y,z;
    int N;
    std::unique_ptr<PREC_T[]> phase = std::make_unique<PREC_T[]>(WIDTH * HEIGHT);
    std::chrono::system_clock::time_point  start, end;

    // for 1920 x 1080
    PREC_T p = 8.0e-6;
    PREC_T lambda = 520e-9;
    // PREC_T offset[] = {960,540,1/p};
    PREC_T offset[] = {960 * p,540 * p,1};
    PREC_T scale = 30 * p;

    // for 4k
    // double p = 3.74e-6;
    // double lambda = 532e-9;
    // double offset[] = {WIDTH * 2/4 ,HEIGHT * 2/4,0.3f/p};
    // double scale = 60;

    
    auto img = std::make_unique<uint8_t[]>(WIDTH * HEIGHT);
    ol::objread(path3d,x,y,z,N,offset,scale);
    printf("objread done\n");
    ol::objinfo(x,y,z,N);
    start = std::chrono::system_clock::now(); // 計測開始時間
    ol::Cgh(x,y,z,N,phase,HEIGHT,WIDTH,p,lambda,false);
    end = std::chrono::system_clock::now();  // 計測終了時間
 
    // 処理に要した時間
    auto time = end - start;
 
    // 処理に要した時間をミリ秒に変換
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout << msec << " msec" << std::endl;
    
    ol::phase8bit(phase,img,HEIGHT,WIDTH);
    ol::bmpwrite(PROJECT_ROOT "/out/olcgh284.bmp",img,HEIGHT,WIDTH);
}