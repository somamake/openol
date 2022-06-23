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
    char path3d[] = PROJECT_ROOT "/data/horn02-f000.3d";
    // char path3d[] = PROJECT_ROOT "/data/cube284.3d";
    ol::Vect3<PREC_T> offset((float) (WIDTH / 2 * p),(float) (HEIGHT / 2 * p),1.0f);
    
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

    

    start = std::chrono::system_clock::now(); // 計測開始時間
    auto gobj = ol::Object2cuda<PREC_T>(obj);
    ol::gCgh(gobj,devData,HEIGHT,WIDTH,p,lambda);
    ol::gFresnelProp(devData,HEIGHT,WIDTH,p,p,lambda,d);
    ol::cuda2cpu(devData,u,HEIGHT * WIDTH);
    end = std::chrono::system_clock::now();  // 計測終了時間
    
    std::cout << u[HEIGHT * WIDTH] << std::endl;
    auto time = end - start;
 
    // // 処理に要した時間をミリ秒に変換
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout << msec << " msec" << std::endl;
    
    
    ol::Save(PROJECT_ROOT "/out/olCghProp.bmp",u,HEIGHT,WIDTH,ol::AMP);
    return 0;
}