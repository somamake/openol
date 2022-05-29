#include <fftw3.h>
#include <complex>
#include <stdio.h>
#include <memory>
#include <chrono>
#include "openol.h"
#include "para.h"
#include "molprop.h"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT
#endif
typedef std::complex<PREC_T> Complex;


int main(){
    std::unique_ptr<std::complex<PREC_T>[]> u;
    u = std::make_unique<std::complex<PREC_T>[]>(WIDTH * HEIGHT);

    auto img = std::make_unique<uint8_t[]>(HEIGHT * WIDTH);
    std::chrono::system_clock::time_point  start, end;
    #ifndef RECT_TEST
    ol::bmpread(path,img.get(),HEIGHT,WIDTH);
    ol::img2complex(img,u,HEIGHT,WIDTH,ol::PHASE);
    #else
    int rectwidth = 40;
    int rectheight = 20;

    for(int i = 0;i < HEIGHT;i++){
		for(int j = 0;j < WIDTH;j++){
			int adr = j + i * WIDTH;
			if(j > (WIDTH/2-rectwidth/2) && j < (WIDTH/2+rectwidth/2) && 
                i > (HEIGHT/2-rectheight/2) && i < (HEIGHT/2+rectheight/2) ){
				u[adr]=1.0;
			}
			else{
				u[adr]=0.0;
			}
		}
	}
    #endif

    // ol::FresnelResponseCheck(HEIGHT,WIDTH,p,p,lambda,-d);

    start = std::chrono::system_clock::now();
    // fftwf_init_threads();
    // ol::FresnelProp(u,HEIGHT,WIDTH,p,p,lambda,-d);
    // ol::AsmProp(u,HEIGHT,WIDTH,p,p,lambda,-d);
    ol::mAsmProp(u,HEIGHT,WIDTH,p,p,lambda,-d);
    // ol::shiftedFresnelProp<PREC_T>(u,HEIGHT,WIDTH,p/s,p/s,lambda,d,s,0.012,0.012);


    end = std::chrono::system_clock::now();  // 計測終了時間
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << msec << " msec" << std::endl;
    // ol::MedianFilter(u,u,HEIGHT,WIDTH,16,16);
    ol::Save(PROJECT_ROOT "/out/olProp.bmp",u,HEIGHT,WIDTH,ol::AMP);
}