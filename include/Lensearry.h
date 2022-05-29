#ifndef LENSEARRY_H
#define LENSEARRY_H
#ifdef __NVCC__
#include "thrust/complex.h"
#include "golclass.h"
#include "curand.h"
#include "golprop.h"
#endif
#include "memory"
#include <random>

namespace ol{


template<typename _Tp=float>
class Lensearry{
    public:
        float pitch;
        int width;
        int height;
        float f;
        float lambda;
        float blockwidth;
        int blockpx;
        std::unique_ptr<std::complex<_Tp>[]> u;
        ol::HOLOGRAMSTEP step = ol::HOLOGRAMSTEP::PLAY;
        Lensearry(int height, int width, float pitch,float blockwidth,float f,float lambda){
            this->pitch = pitch;
            this->height = height;
            this->width = width;
            this->f = f;
            this->lambda = lambda;
            this->blockwidth = blockwidth;
            this->blockpx = round(blockwidth / pitch);
        }
        ~Lensearry(){
        }

        template<typename PREC_T>
        std::unique_ptr<std::complex<PREC_T>[]> duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on = 0,int om = 0){
            auto u_lens = std::make_unique<std::complex<PREC_T>[]>(this->height * this->width);
            for (int block_n = 0;block_n < this->height;block_n += this->blockpx){
                for (int block_m = 0;block_m < this->width;block_m += this->blockpx){
                    for (int local_n = 0;local_n < this->blockpx;local_n++){
                        for (int local_m = 0;local_m < this->blockpx;local_m++){
                            int m = (block_m + local_m);
                            int n = (block_n + local_n);
                            if (m >= this->width || n >= this->height){
                                break;
                            }
                            m = (m + om) % width;
                            n = (n + on) % height;
                            m = (m >= 0) ? m : m+width;
                            n = (n >= 0) ? n : n+height;
                            u_lens[m + n * this->width] = ublock[local_m + local_n * this->blockpx];
                        }
                    }
                    
                }
            }
            return u_lens;
        }
        template<typename PREC_T=float>
        void gen_complex(HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            this->step = step;
            auto ublock = std::make_unique<std::complex<PREC_T>[]>(this->blockpx * this->blockpx);
            float tmpf = f;
            if (step == ol::HOLOGRAMSTEP::RECORD){
                tmpf = -f;
            }
            ol::FresnelResponse(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf);
            auto u_lens = duplicate(ublock,oy,ox);
            this->u = std::move(u_lens);
        }
        template<typename PREC_T=float>
        void mult_complex(std::unique_ptr<std::complex<PREC_T>[]>& u, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            if (((void*)this->u.get() == NULL) || (this->step != step)){
                gen_complex(step,oy,ox);
            }
            ol::mul_complex(u,this->u,u,this->height,this->width);
        }
        template<typename PREC_T=float>
        void shift(int oy = 0,int ox = 0){
            // auto linebuf = std::make_unique<std::complex<PREC_T>[]>(height);
            if (ox != 0){
                for (int n = 0;n < this->height / 2;n++){
                    for (int m = 0;m < this->width / 2;m++){
                        std::complex<PREC_T> tmp;
                        int m1 = (m + ox) % this->width; 
                        tmp = this->u[n * width + m];
                        this->u[n * width + m] = this->u[n * width + m1];
                        this->u[n * width + m1] = tmp;
                    }
                }
            }
            if (oy != 0){
                for (int n = 0;n < this->height / 2;n++){
                    int n1 = (n + oy) % this->height; 
                    for (int m = 0;m < this->width / 2;m++){
                        std::complex<PREC_T> tmp;
                        tmp = this->u[n * width + m];
                        this->u[n * width + m] = this->u[n1 * width + m];
                        this->u[n1 * width + m] = tmp;
                    }
                }
            }
            
        }
        #ifdef __NVCC__
        template<typename PREC_T=float>
        cuda::unique_ptr<thrust::complex<PREC_T>[]> ggen_complex(HOLOGRAMSTEP step=PLAY){
            auto u_lens = cuda::make_unique<thrust::complex<PREC_T>[]>(this->height * this->width);
            ol::gFresnelResponse(u_lens,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-f);
            return u_lens;
        }
        #endif
};
}

#endif
