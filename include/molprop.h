
#ifndef MOLPROP_H
#define MOLPROP_H
#include <stdio.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cufft.h>
#include "olutils.h"
#include "olimg.h"

namespace ol
{

template<typename PREC_T>
void fft2d_hyblid_base(std::unique_ptr<std::complex<PREC_T>[]>& src,std::unique_ptr<std::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx,int direction,int blocknum_y = 2,int blocknum_x = 2)
{
    int64_t blocknx = nx / blocknum_x;
    int64_t blockny = ny / blocknum_y;
    cufftHandle fftplan;
    int64_t size;
    if constexpr(std::is_same_v<PREC_T,double>){
        
    }
    else if constexpr(std::is_same_v<PREC_T,float>){
        // axis = x に沿ってfft1d
        size = blockny * nx;
        cufftPlan1d(&fftplan, nx, CUFFT_C2C,blockny);
        // cufftPlanMany(&fftplan,)
        auto dbuf = cuda::make_unique<thrust::complex<PREC_T>[]>(size);
        for (int n = 0;n < blocknum_y;n++){
            cudaMemcpy(dbuf.get(),src.get() + size * n,size * sizeof(std::complex<PREC_T>),cudaMemcpyHostToDevice);
            cufftExecC2C(fftplan, (cufftComplex*)dbuf.get(), (cufftComplex*)dbuf.get(), direction);
            cudaMemcpy(dst.get() + size * n,dbuf.get(),size * sizeof(std::complex<PREC_T>),cudaMemcpyDeviceToHost);
        }
        cufftDestroy(fftplan);
        dbuf.reset();
        // axis = y に沿ってfft1d
        size = ny * blocknx;
        cufftPlan1d(&fftplan, ny, CUFFT_C2C,blocknx);
        dbuf = cuda::make_unique<thrust::complex<PREC_T>[]>(size);
        auto hbuf = std::make_unique<std::complex<PREC_T>[]>(ny);
        for (int m = 0;m < blocknum_x;m++){
            // メモリコピー host to device
            for (int local_m = 0;local_m < blocknx;local_m++){
                for (int n = 0;n < ny;n++){
                    hbuf[n] = dst[n * nx + (local_m + blocknx * m)];
                }
                cudaMemcpy(dbuf.get() + local_m * ny,hbuf.get(),ny * sizeof(std::complex<PREC_T>),cudaMemcpyHostToDevice);
            }
            cufftExecC2C(fftplan, (cufftComplex*)dbuf.get(), (cufftComplex*)dbuf.get(), direction);
            // メモリコピー device to host
            for (int local_m = 0;local_m < blocknx;local_m++){
                cudaMemcpy(hbuf.get(),dbuf.get() + local_m * ny,ny * sizeof(std::complex<PREC_T>),cudaMemcpyDeviceToHost);
                for (int n = 0;n < ny;n++){
                    dst[n * nx + (local_m + blocknx * m)] = hbuf[n];
                }
            }
        }
        cufftDestroy(fftplan);
        dbuf.reset();
        
    }
}

template<typename PREC_T>
void fft2d_hyblid(std::unique_ptr<std::complex<PREC_T>[]>& src,std::unique_ptr<std::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx,int blocknum_y = 2,int blocknum_x = 2)
{
    fft2d_hyblid_base(src,dst,ny,nx,CUFFT_FORWARD,blocknum_y,blocknum_x);
}

/*IFFT計算関数*/
template<typename PREC_T>
void ifft2d_hyblid(std::unique_ptr<std::complex<PREC_T>[]>& src,std::unique_ptr<std::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx,int blocknum_y = 2,int blocknum_x = 2)
{
    fft2d_hyblid_base(src,dst,ny,nx,CUFFT_INVERSE,blocknum_y,blocknum_x);
}

template<typename PREC_T = double, typename COMPLEX_T>
void mAsmProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int64_t height, int64_t width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    // auto img = std::make_unique<uint8_t[]>(height * width * 4);
    int64_t height2 = height * 2;
    int64_t width2 = width * 2;
    PREC_T du = 1 / ( dx * width2);
    PREC_T dv = 1 / (dy * height2);
    auto H = std::make_unique<std::complex<COMPLEX_T>[]>(height2 * width2);
    zeropadding(u,u,height,width,height2,width2);
    AsmTransferF(H,height * 2, width * 2,dv,du,lambda, z);

    fftshift(H,height2,width2);
    fft2d_hyblid(u,u,height2,width2);
    
    mul_complex(u,H,u,height2,width2);
    mul_scalar(u,1.0/(height2*width2),height2,width2);

    ifft2d_hyblid(u,u,height2,width2);
    del_zero(u,u,height2,width2);
    mul_scalar(u,1.0/(height2*width2),height,width); 
}

} // namespace ol

#endif