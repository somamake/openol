// #include "olclass.h"
// #include "olcgh.h"
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
    ol::Vect3<PREC_T> offset((float) (WIDTH / 2 * p),(float) (HEIGHT / 2 * p),1.0f);
    char path3d[] = PROJECT_ROOT "/data/horn02-f000.3d";
    std::chrono::system_clock::time_point  start, end;
    std::unique_ptr<PREC_T[]> phase = std::make_unique<PREC_T[]>(WIDTH * HEIGHT);
    auto img = std::make_unique<uint8_t[]>(WIDTH * HEIGHT);

    auto obj = ol::objread<PREC_T>(path3d);
    ol::objset(obj,WIDTH * p * 0.8,HEIGHT * p * 0.8,offset,ol::ZMIDDLE);
    std::cout << obj.size << std::endl;
    ol::objinfo(obj);
    ol::objsort(obj);
    std::cout << obj[0].z << std::endl;

    start = std::chrono::system_clock::now(); // 計測開始時間
    ol::Cgh(obj,phase,HEIGHT,WIDTH,p,p,lambda);
    end = std::chrono::system_clock::now();  // 計測終了時間

    auto time = end - start;
 
    // 処理に要した時間をミリ秒に変換
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    std::cout << msec << " msec" << std::endl;
    
    ol::phase8bit(phase,img,HEIGHT,WIDTH);
    ol::bmpwrite(PROJECT_ROOT "/out/olObject.bmp",img,HEIGHT,WIDTH);
    return 0;
}