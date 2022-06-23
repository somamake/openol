#include "para.h"
#include "openol.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <chrono>
#include <algorithm>


int main(){
    char path3d[] = PROJECT_ROOT "/3ddata/horn02-f000.3d";
    // char path3d[] = PROJECT_ROOT "/3ddata/hayabusa_7877/3d/hayabusa_7877_021.3d";
    // char path3d[] = PROJECT_ROOT "/3ddata/tyrano_6215/3DF/10fps/tyrannoLow_010.3df";
    // char path3d[] = PROJECT_ROOT "/data/cube284.3d";
    ol::Vect3<PREC_T> offset((float) (WIDTH / 2 * p),(float) (HEIGHT / 2 * p),d);
    
    std::chrono::system_clock::time_point  start, end;
    auto devData = cuda::make_unique<thrust::complex<PREC_T>[]>(WIDTH * HEIGHT);
    std::unique_ptr<std::complex<PREC_T>[]> u =std::make_unique<std::complex<PREC_T>[]>(WIDTH * HEIGHT);
    // cudaHostRegister(u.get(),WIDTH * HEIGHT * sizeof(std::complex<PREC_T>),cudaHostRegisterDefault);

    auto obj = ol::objread<PREC_T>(path3d);
    ol::objset(obj,WIDTH * p * 0.8,HEIGHT * p * 0.8,offset,ol::ZMIDDLE);
    std::cout << obj.size << std::endl;
    ol::objinfo(obj);
    ol::objsort(obj);
    std::cout << obj[0].z << std::endl;

    
    ol::AsmTransferFCheck(HEIGHT,WIDTH,p,p,lambda,d);
    start = std::chrono::system_clock::now(); // 計測開始時間
    auto gobj = ol::Object2cuda<PREC_T>(obj);
    // ol::gCgh(gobj,devData,HEIGHT,WIDTH,p,lambda);
    ol::WRPMethod_D(gobj,devData,HEIGHT,WIDTH,p,lambda,d,-1.0f,ol::PROPMODE::AUTO);
    // ol::WRPStep1_D(gobj,devData,HEIGHT,WIDTH,p,lambda,0.0f);
    // ol::gCghReduction(gobj,devData,HEIGHT,WIDTH,p,lambda);
    // ol::gMedianFilter(devData,devData,HEIGHT,WIDTH,8,8);
    ol::cuda2cpu(devData,u,HEIGHT * WIDTH);
    end = std::chrono::system_clock::now();  // 計測終了時間
    
    auto time = end - start;
 
    // // 処理に要した時間をミリ秒に変換
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout << msec << " msec" << std::endl;
    
    
    ol::Save(PROJECT_ROOT "/out/olObject.bmp",u,HEIGHT,WIDTH,ol::PHASE);
    return 0;
}