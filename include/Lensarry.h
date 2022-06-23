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
class Lensarry{
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
        Lensarry(int height, int width, float pitch,float blockwidth,float f,float lambda){
            this->pitch = pitch;
            this->height = height;
            this->width = width;
            this->f = f;
            this->lambda = lambda;
            this->blockwidth = blockwidth;
            this->blockpx = round(blockwidth / pitch);
        }
        ~Lensarry(){
        }
        void info(){
            printf("pitch = %lf\n",pitch);
            printf("blockpx = %d\n",blockpx);

        }

        template<typename PREC_T>
        std::unique_ptr<std::complex<PREC_T>[]> duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on = 0,int om = 0){
            auto u_lens = std::make_unique<std::complex<PREC_T>[]>(this->height * this->width);
            
            
            for (int block_n = 0;block_n < this->height;block_n += this->blockpx){
                for (int block_m = 0;block_m < this->width;block_m += this->blockpx){
                    for (int local_n = 0;local_n < this->blockpx;local_n++){
                        for (int local_m = 0;local_m < this->blockpx;local_m++){
                            int local_mshift = (local_m - om) % blockpx;
                            int local_nshift = (local_n - on) % blockpx;
                            local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+blockpx;
                            local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+blockpx;
                            int m = (block_m + local_m);
                            int n = (block_n + local_n);
                            if (m >= this->width || n >= this->height){
                                break;
                            }
                            u_lens[m + n * this->width] = ublock[local_mshift + local_nshift * this->blockpx];
                        }
                    }
                    
                }
            }
            return u_lens;
        }
        template<typename PREC_T>
        std::unique_ptr<std::complex<PREC_T>[]> duplicate_random(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on = 0,int om = 0){
            auto u_lens = std::make_unique<std::complex<PREC_T>[]>(this->height * this->width);
            PREC_T xlim;
            int wlim;
            xlim = abs(lambda * f * 0.5 / p_limit);
            wlim = floor (xlim / pitch);
            
            std::mt19937 e2(0);
            for (int block_n = 0;block_n < this->height;block_n += this->blockpx){
                for (int block_m = 0;block_m < this->width;block_m += this->blockpx){
                    for (int local_n = 0;local_n < this->blockpx;local_n++){
                        for (int local_m = 0;local_m < this->blockpx;local_m++){
                            int local_mshift = (local_m - om) % blockpx;
                            int local_nshift = (local_n - on) % blockpx;
                            local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+blockpx;
                            local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+blockpx;
                            int m = (block_m + local_m);
                            int n = (block_n + local_n);
                            if (m >= this->width || n >= this->height){
                                break;
                            }
                            if (abs(local_mshift - this->blockpx / 2) >= wlim ||  abs(local_nshift - this->blockpx / 2) >= wlim){
                                
                                int a = e2() >> 5;
                                int b = e2() >> 6;
                                double value = (a * 67108864.0 + b) / 9007199254740992.0;
                                float phase = 2 * F32_PI * (value -0.5f);
                                u_lens[m + n * this->width] = std::complex(cos(phase),sin(phase));
                                // printf("a");
                            }
                            else{
                                u_lens[m + n * this->width] = ublock[local_mshift + local_nshift * this->blockpx];
                            }
                            
                        }
                    }
                    
                }
            }
            // printf("\n");
            return u_lens;
        }

        template<typename PREC_T>
        std::unique_ptr<std::complex<PREC_T>[]> duplicate_multrandom(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on = 0,int om = 0){
            auto u_lens = std::make_unique<std::complex<PREC_T>[]>(this->height * this->width);
            PREC_T xlim;
            int wlim;
            xlim = abs(lambda * f * 0.5 / p_limit);
            wlim = floor (xlim / pitch);
            
            std::mt19937 e2(0);
            for (int block_n = 0;block_n < this->height;block_n += this->blockpx){
                for (int block_m = 0;block_m < this->width;block_m += this->blockpx){
                    for (int local_n = 0;local_n < this->blockpx;local_n++){
                        for (int local_m = 0;local_m < this->blockpx;local_m++){
                            int local_mshift = (local_m - om) % blockpx;
                            int local_nshift = (local_n - on) % blockpx;
                            local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+blockpx;
                            local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+blockpx;
                            int m = (block_m + local_m);
                            int n = (block_n + local_n);
                            if (m >= this->width || n >= this->height){
                                break;
                            }
                            if (abs(local_mshift - this->blockpx / 2) >= wlim ||  abs(local_nshift - this->blockpx / 2) >= wlim){
                                
                                int a = e2() >> 5;
                                int b = e2() >> 6;
                                double value = (a * 67108864.0 + b) / 9007199254740992.0;
                                float phase = 0.5 * F32_PI * (value -0.5f);
                                u_lens[m + n * this->width] = ublock[local_mshift + local_nshift * this->blockpx] * std::complex(cos(phase),sin(phase));
                                // printf("a");
                            }
                            else{
                                u_lens[m + n * this->width] = ublock[local_mshift + local_nshift * this->blockpx];
                            }
                            
                        }
                    }
                    
                }
            }
            // printf("\n");
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
        void gen_complex_random(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            this->step = step;
            auto ublock = std::make_unique<std::complex<PREC_T>[]>(this->blockpx * this->blockpx);
            float tmpf = f;
            if (step == ol::HOLOGRAMSTEP::RECORD){
                tmpf = -f;
            }
            ol::FresnelResponse(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf);
            auto u_lens = duplicate_multrandom(ublock,p_limit,oy,ox);
            this->u = std::move(u_lens);
        }
        template<typename PREC_T=float>
        void mult_complex_random(std::unique_ptr<std::complex<PREC_T>[]>& u, PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            if (((void*)this->u.get() == NULL) || (this->step != step)){
                gen_complex_random(p_limit,step,oy,ox);
            }
            ol::mul_complex(u,this->u,u,this->height,this->width);
        }

        template<typename PREC_T=float>
        void gen_complex_limit(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            this->step = step;
            auto ublock = std::make_unique<std::complex<PREC_T>[]>(this->blockpx * this->blockpx);
            float tmpf = f;
            if (step == ol::HOLOGRAMSTEP::RECORD){
                tmpf = -f;
            }
            ol::FresnelResponseBandLimit(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf,p_limit);
            
            auto u_lens = duplicate(ublock,oy,ox);
            this->u = std::move(u_lens);
        }
        template<typename PREC_T=float>
        void mult_complex_limit(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            if (((void*)this->u.get() == NULL) || (this->step != step)){
                gen_complex_limit(p_limit,step,oy,ox);
            }
            ol::mul_complex(u,this->u,u,this->height,this->width);
        }

        template<typename PREC_T=float>
        void gen_complex_limit_random(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            this->step = step;
            auto ublock = std::make_unique<std::complex<PREC_T>[]>(this->blockpx * this->blockpx);
            float tmpf = f;
            if (step == ol::HOLOGRAMSTEP::RECORD){
                tmpf = -f;
            }
            ol::FresnelResponseBandLimit(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf,p_limit);
            auto u_lens = duplicate_random(ublock,p_limit,oy,ox);
            this->u = std::move(u_lens);
        }
        template<typename PREC_T=float>
        void mult_complex_limit_random(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0){
            if (((void*)this->u.get() == NULL) || (this->step != step)){
                gen_complex_limit_random(p_limit,step,oy,ox);
            }
            ol::mul_complex(u,this->u,u,this->height,this->width);
        }

        // 使ってない
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
