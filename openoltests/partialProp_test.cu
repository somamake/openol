#include <fftw3.h>
#include <complex>
#include <stdio.h>
#include <memory>
#include "openol.h"
#include "para.h"
#include <chrono>

#ifndef PROJECT_ROOT
#define PROJECT_ROOT
#endif

// #define WIDTH 512
// #define HEIGHT 512
// #define RECT_TEST


// typedef float PREC_T;

typedef std::complex<PREC_T> Complex;

int main(){
    

    auto img = std::make_unique<uint8_t[]>(HEIGHT * WIDTH);
    
    
    // std::complex<PREC_T> *u1;
    // cudaMallocHost(&u1,WIDTH * HEIGHT * sizeof(std::complex<PREC_T>),cudaHostRegisterDefault);
    // std::unique_ptr<std::complex<PREC_T>[]> u(u1);
    auto u = std::make_unique<std::complex<PREC_T>[]>(WIDTH * HEIGHT);
    // auto u = std::make_unique<std::complex<PREC_T>[]>(WIDTH * HEIGHT);
    // cudaHostRegister(u.get(),WIDTH * HEIGHT * sizeof(std::complex<PREC_T>),cudaHostRegisterDefault);
    

    #ifndef RECT_TEST
    ol::bmpread(path,img.get(),HEIGHT,WIDTH);
    ol::img2complex(img,u,HEIGHT,WIDTH,true);
    int clipwidth = WIDTH * 5 / 100;
    for (int n = 0;n < HEIGHT;n++){
        for (int m = 0;m < WIDTH;m++){
            if (m >= clipwidth){
                u[n * WIDTH + m] = 0;
            }
        }
    }

    // shift 用
    // int shiftwidth = WIDTH / 2 - clipwidth / 2;
    // for (int n = 0;n < HEIGHT;n++){
    //     for (int m = clipwidth - 1;m >= 0;m--){
    //         u[n * WIDTH + m + shiftwidth] = u[n * WIDTH + m];
    //         u[n * WIDTH + m] = 0;
    //     }
    // }
    #endif

    #ifdef RECT_TEST
    int rectwidth = 40;
    int rectheight = 20;
    for(int i = 0;i < HEIGHT;i++){
		for(int j = 0;j < WIDTH;j++){
			int adr = j + i * WIDTH;
			if(j > (WIDTH - rectwidth)/2 && j < (WIDTH + rectwidth)/2 && i > (HEIGHT-rectheight)/2 && i < (HEIGHT+rectheight)/2 ){
				u[adr]=1.0;
			}
			else{
				u[adr]=0.0;
			}
		}
	}
    #endif

    auto d_u = cuda::make_unique<thrust::complex<PREC_T>[]>(HEIGHT * WIDTH);
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();
    ol::cpu2cuda(u,d_u,HEIGHT * WIDTH);
    
    // ol::gFresnelProp(d_u,HEIGHT,WIDTH,p,p,lambda,-d-0.005f);
    ol::gAsmProp(d_u,HEIGHT,WIDTH,p,p,lambda,-d-0.005f);
    // ol::gshiftedFresnelProp(d_u,HEIGHT,WIDTH,(PREC_T)p/s,(PREC_T)p/s,(PREC_T)lambda,(PREC_T)-d,(PREC_T)s,(PREC_T)0.1f,(PREC_T)0.1f);
    ol::cuda2cpu(d_u,u,HEIGHT*WIDTH);
    cudaHostUnregister(u.get());

    end = std::chrono::system_clock::now();  // 計測終了時間
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << msec << " msec" << std::endl;
    ol::complex2img(u,img,HEIGHT,WIDTH);
    ol::bmpwrite(PROJECT_ROOT "/out/olProp.bmp",img.get(),HEIGHT,WIDTH);
}