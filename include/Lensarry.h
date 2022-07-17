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

#ifdef __NVCC__
template<typename PREC_T=float>
__global__ void gduplicateKernel(thrust::complex<PREC_T>* ublock,thrust::complex<PREC_T>* ulens,int blockpx,int height,int width,int on,int om){
    int local_m = blockIdx.x * blockDim.x + threadIdx.x;
    int local_n = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_m >= blockpx || local_n >= blockpx){
        return;
    }
    int local_mshift = (local_m - om) % blockpx;
    int local_nshift = (local_n - on) % blockpx;
    local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+blockpx;
    local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+blockpx;
    for (int block_n = 0;block_n < height;block_n += blockpx){
        for (int block_m = 0;block_m < width;block_m += blockpx){
            
            int m = (block_m + local_m);
            int n = (block_n + local_n);
            if (m >= width || n >= height){
                break;
            }
            ulens[m + n * width] = ublock[local_mshift + local_nshift * blockpx];
            
        }
    }
}

template<typename PREC_T=float>
__global__ void gduplicateCircleKernel(thrust::complex<PREC_T>* ublock,thrust::complex<PREC_T>* ulens,int blockpx,int height,int width,float pitch,float r,int stepn,int stepm,int on,int om){
    int local_m = blockIdx.x * blockDim.x + threadIdx.x;
    int local_n = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_m >= blockpx || local_n >= blockpx){
        return;
    }
    
    int local_nshift = (local_n - on) % blockpx;
    
    local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+blockpx;
    for (int block_n = 0;block_n < height;block_n += stepn){
        int local_mshift = (local_m - om + (block_n / stepn % 2) * blockpx / 2 ) % blockpx;
        local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+blockpx;
        for (int block_m = 0;block_m < width;block_m += stepm){
            int m = (block_m + local_m);
            int n = (block_n + local_n);
            if (m >= width || n >= height){
                break;
            }
            if (pow(local_mshift * pitch - r,2) + pow(local_nshift * pitch - r,2) <= pow(r,2.0f)){
                ulens[m + n * width] = ublock[local_mshift + local_nshift * blockpx];
            }
            
            
        }
    }
}
#endif

namespace ol{

template<typename PREC_T=float>
class Lensarry{
    public:
        float pitch;
        int width;
        int height;
        float f;
        float lambda;
        float blockwidth;
        int blockpx;
        std::unique_ptr<std::complex<PREC_T>[]> u;
        #ifdef __NVCC__
        cuda::unique_ptr<thrust::complex<PREC_T>[]> du;
        #endif
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
            printf("pitch[um] = %lf\n",pitch*1e6);
            printf("blockpx = %d\n",blockpx);

        }
        virtual std::unique_ptr<std::complex<PREC_T>[]> duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on = 0,int om = 0);
        std::unique_ptr<std::complex<PREC_T>[]> duplicate_random(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on = 0,int om = 0);
        std::unique_ptr<std::complex<PREC_T>[]> duplicate_multrandom(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on = 0,int om = 0);
        virtual void gen_complex(HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        virtual void mult_complex(std::unique_ptr<std::complex<PREC_T>[]>& u, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void gen_complex_random(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void mult_complex_random(std::unique_ptr<std::complex<PREC_T>[]>& u, PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void gen_complex_limit(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void mult_complex_limit(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void gen_complex_limit_random(PREC_T p_limit,HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void mult_complex_limit_random(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void shift(int oy = 0,int ox = 0);


        
        #ifdef __NVCC__
        virtual cuda::unique_ptr<thrust::complex<PREC_T>[]> gduplicate(cuda::unique_ptr<thrust::complex<PREC_T>[]>& ublock,int on = 0,int om = 0);
        virtual void ggen_complex(HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        virtual void mult_complex(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void ggen_complex_limit(PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        void mult_complex_limit(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0);
        
        #endif
};

template<typename PREC_T>
std::unique_ptr<std::complex<PREC_T>[]> Lensarry<PREC_T>::duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on,int om){
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
std::unique_ptr<std::complex<PREC_T>[]> Lensarry<PREC_T>::duplicate_random(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on,int om){
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
std::unique_ptr<std::complex<PREC_T>[]> Lensarry<PREC_T>::duplicate_multrandom(std::unique_ptr<std::complex<PREC_T>[]>& ublock,PREC_T p_limit,int on,int om){
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

template<typename PREC_T>
void Lensarry<PREC_T>::gen_complex(HOLOGRAMSTEP step,int oy,int ox){
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
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex(std::unique_ptr<std::complex<PREC_T>[]>& u, HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->u.get() == NULL) || (this->step != step)){
        gen_complex(step,oy,ox);
    }
    ol::mul_complex(u,this->u,u,this->height,this->width);
}
template<typename PREC_T>
void Lensarry<PREC_T>::gen_complex_random(PREC_T p_limit,HOLOGRAMSTEP step,int oy,int ox){
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
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex_random(std::unique_ptr<std::complex<PREC_T>[]>& u, PREC_T p_limit,HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->u.get() == NULL) || (this->step != step)){
        gen_complex_random(p_limit,step,oy,ox);
    }
    ol::mul_complex(u,this->u,u,this->height,this->width);
}
template<typename PREC_T>
void Lensarry<PREC_T>::gen_complex_limit(PREC_T p_limit,HOLOGRAMSTEP step,int oy,int ox){
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
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex_limit(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->u.get() == NULL) || (this->step != step)){
        gen_complex_limit(p_limit,step,oy,ox);
    }
    ol::mul_complex(u,this->u,u,this->height,this->width);
}

template<typename PREC_T>
void Lensarry<PREC_T>::gen_complex_limit_random(PREC_T p_limit,HOLOGRAMSTEP step,int oy,int ox){
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
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex_limit_random(std::unique_ptr<std::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->u.get() == NULL) || (this->step != step)){
        gen_complex_limit_random(p_limit,step,oy,ox);
    }
    ol::mul_complex(u,this->u,u,this->height,this->width);
}

// 使ってない
template<typename PREC_T>
void Lensarry<PREC_T>::shift(int oy,int ox){
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
template<typename PREC_T>
cuda::unique_ptr<thrust::complex<PREC_T>[]> Lensarry<PREC_T>::gduplicate(cuda::unique_ptr<thrust::complex<PREC_T>[]>& ublock,int on,int om){
    auto u_lens = cuda::make_unique<thrust::complex<PREC_T>[]>(this->height * this->width);
    
    
    dim3 block(16, 16, 1);
    dim3 grid(ceil((float)this->blockpx / block.x), ceil((float)this->blockpx / block.y), 1);
    gduplicateKernel<<<grid,block>>>(ublock.get(),u_lens.get(),this->blockpx,height,width,on,om);
    cudaDeviceSynchronize();
    return u_lens;
}
template<typename PREC_T>
void Lensarry<PREC_T>::ggen_complex(HOLOGRAMSTEP step,int oy,int ox){
    this->step = step;
    auto ublock = cuda::make_unique<thrust::complex<PREC_T>[]>(this->blockpx * this->blockpx);
    float tmpf = f;
    if (step == ol::HOLOGRAMSTEP::RECORD){
        tmpf = -f;
    }
    ol::gFresnelResponse(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf);
    auto u_lens = gduplicate(ublock,oy,ox);
    this->du = std::move(u_lens);
}
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u, HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->du.get() == NULL) || (this->step != step)){
        ggen_complex(step,oy,ox);
    }
    ol::mult(u,this->du,u,this->height,this->width);
}

template<typename PREC_T>
void Lensarry<PREC_T>::ggen_complex_limit(PREC_T p_limit, HOLOGRAMSTEP step,int oy,int ox){
    this->step = step;
    auto ublock = cuda::make_unique<thrust::complex<PREC_T>[]>(this->blockpx * this->blockpx);
    float tmpf = f;
    if (step == ol::HOLOGRAMSTEP::RECORD){
        tmpf = -f;
    }
    ol::gFresnelResponseBandLimit(ublock,this->blockpx,this->blockpx,this->pitch,this->pitch,lambda,-tmpf,p_limit);
    auto u_lens = gduplicate(ublock,oy,ox);
    this->du = std::move(u_lens);
}
template<typename PREC_T>
void Lensarry<PREC_T>::mult_complex_limit(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,PREC_T p_limit, HOLOGRAMSTEP step,int oy,int ox){
    if (((void*)this->du.get() == NULL) || (this->step != step)){
        ggen_complex_limit(p_limit,step,oy,ox);
    }
    ol::mult(u,this->du,u,this->height,this->width);
}
#endif

template<typename PREC_T>
class LensarryCircle : public Lensarry<PREC_T>{
    public:
        using Lensarry<PREC_T>::Lensarry;
        std::unique_ptr<std::complex<PREC_T>[]> duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on = 0,int om = 0) override;
        // void gen_complex(HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0) override;
        // void mult_complex(std::unique_ptr<std::complex<PREC_T>[]>& u, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0) override;

        #ifdef __NVCC__
        cuda::unique_ptr<thrust::complex<PREC_T>[]> gduplicate(cuda::unique_ptr<thrust::complex<PREC_T>[]>& ublock,int on = 0,int om = 0) override;
        // void ggen_complex(HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0) override;
        // void mult_complex(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u, HOLOGRAMSTEP step=PLAY,int oy = 0,int ox = 0) override;
        #endif
};

template<typename PREC_T>
std::unique_ptr<std::complex<PREC_T>[]> LensarryCircle<PREC_T>::duplicate(std::unique_ptr<std::complex<PREC_T>[]>& ublock,int on,int om){
    auto u_lens = std::make_unique<std::complex<PREC_T>[]>(this->height * this->width);
    
    float stepy = this->blockwidth * sqrt(3.0f) * 0.5f;
    float stepx = this->blockwidth;
    float r = this->blockwidth * 0.5f;
    int stepn = round(stepy / this->pitch);
    int stepm = round(stepx / this->pitch); 
    for (int64_t n = 0;n < this->height;n++){
        for (int64_t m = 0;m < this->width;m++){
            u_lens[n * this->width + m] = 1.0f;
        }
    }
    for (int block_n = 0;block_n < this->height;block_n+=stepn){
        for (int block_m = 0;block_m < this->width;block_m+=stepm){
            for (int local_n = 0;local_n < this->blockpx;local_n++){
                for (int local_m = 0;local_m < this->blockpx;local_m++){
                    int local_mshift = (local_m - om + (block_n / stepn % 2) * this->blockpx / 2 ) % this->blockpx ;
                    // int local_mshift = (local_m - om) % this->blockpx;
                    int local_nshift = (local_n - on) % this->blockpx;
                    local_mshift = (local_mshift >= 0) ? local_mshift : local_mshift+this->blockpx;
                    local_nshift = (local_nshift >= 0) ? local_nshift : local_nshift+this->blockpx;
                    int m = (block_m + local_m);
                    int n = (block_n + local_n);
                    if (m >= this->width || n >= this->height){
                        break;
                    }

                    if (powf32(local_mshift * this->pitch - this->blockwidth * 0.5f,2) + powf32(local_nshift * this->pitch - this->blockwidth * 0.5f,2) <= powf32(r,2.0f)){
                        u_lens[m + n * this->width] = ublock[local_mshift + local_nshift * this->blockpx];
                    }
                }
            }
            
        }
    }
    return u_lens;
}

#ifdef __NVCC__
template<typename PREC_T>
cuda::unique_ptr<thrust::complex<PREC_T>[]> LensarryCircle<PREC_T>::gduplicate(cuda::unique_ptr<thrust::complex<PREC_T>[]>& ublock,int on,int om){
    auto u_lens = cuda::make_unique<thrust::complex<PREC_T>[]>(this->height * this->width);
    
    float stepy = this->blockwidth * sqrt(3.0f) * 0.5f;
    float stepx = this->blockwidth;
    float r = this->blockwidth * 0.5f;
    int stepn = round(stepy / this->pitch);
    int stepm = round(stepx / this->pitch); 
    dim3 block(16, 16, 1);
    dim3 grid(ceil((float)this->blockpx / block.x), ceil((float)this->blockpx / block.y), 1);
    gduplicateCircleKernel<<<grid,block>>>(ublock.get(),u_lens.get(),this->blockpx,this->height,this->width,this->pitch,r,stepn,stepm,on,om);
    cudaDeviceSynchronize();
    return u_lens;
}

#endif

}

#endif
