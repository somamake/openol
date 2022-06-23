#include <fftw3.h>
#include <complex>
#include <stdio.h>
#include <memory>
#include <chrono>
#include "openol.h"
#include "para.h"

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
    ol::img2complex(img,u,HEIGHT,WIDTH,true);
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
    
    std::cout << "start" << std::endl;
    start = std::chrono::system_clock::now();
    ol::TiltedReferenceProp<PREC_T>(u,HEIGHT,WIDTH,p,p,lambda,-d - 0.00f,M_PI / 180 * (4.85 * 2),ol::BICUBIC,ol::PROPMODE::ASM);
    // ol::TiltedSourceProp<PREC_T>(u,HEIGHT,WIDTH,p,p,lambda,-d - 0.005f,M_PI / 180 * (-5),ol::BICUBIC,ol::PROPMODE::ASM);

    // レンズ
    // ol::AsmProp(u,HEIGHT,WIDTH,p,p,lambda,-0.01f);
    // auto u_lens = std::make_unique<std::complex<PREC_T>[]>(HEIGHT*WIDTH);
    // ol::FresnelResponse(u_lens,HEIGHT,WIDTH,p,p,lambda,-0.005f);
    // ol::mul_complex(u,u_lens,u,HEIGHT,WIDTH);
    // ol::AsmProp(u,HEIGHT,WIDTH,p,p,lambda,-0.01f);

    end = std::chrono::system_clock::now();  // 計測終了時間
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << msec << " msec" << std::endl;
    ol::Save(PROJECT_ROOT "/out/oltiltedProp.bmp",u,HEIGHT,WIDTH,ol::AMP);
}