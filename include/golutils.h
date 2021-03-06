#ifndef GUTILS_H
#define GUTILS_H
#include <stdio.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cufft.h>
#include "olutils.h"
#include "olimg.h"

namespace ol{

class gFFT{
	public:
	cufftHandle fftplan;
	int64_t ny,nx;
	bool flag = false;
	~gFFT(){
		cufftDestroy(fftplan);
	}
	void set(int64_t ny, int64_t nx){
		this->ny = ny;
		this->nx = nx;
		cufftPlan2d(&fftplan, ny, nx, CUFFT_C2C);
		flag = true;
	}
	gFFT(int64_t ny, int64_t nx){
		set(ny,nx);
	}
	void fft(cuda::unique_ptr<thrust::complex<float>[]>& src,cuda::unique_ptr<thrust::complex<float>[]>& dst){
		if (flag == true){
			cufftComplex* _src = reinterpret_cast<cufftComplex*>(src.get());
			cufftComplex* _dst = reinterpret_cast<cufftComplex*>(dst.get());
			cufftExecC2C(fftplan, _src, _dst, CUFFT_FORWARD);
		}
		else{
			printf("plan not set\n");
		}
		
	}
	void ifft(cuda::unique_ptr<thrust::complex<float>[]>& src,cuda::unique_ptr<thrust::complex<float>[]>& dst){
		if (flag == true){
			cufftComplex* _src = reinterpret_cast<cufftComplex*>(src.get());
			cufftComplex* _dst = reinterpret_cast<cufftComplex*>(dst.get());
			cufftExecC2C(fftplan, _src, _dst, CUFFT_INVERSE);
		}
		else{
			printf("plan not set\n");
		}
		
	}
};

template<typename PREC_T>
void fft(cuda::unique_ptr<thrust::complex<PREC_T>[]>& src,cuda::unique_ptr<thrust::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx)
{
    cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny, nx, CUFFT_C2C);
	cufftComplex* _src = reinterpret_cast<cufftComplex*>(src.get());
	cufftComplex* _dst = reinterpret_cast<cufftComplex*>(dst.get());
	cufftExecC2C(fftplan, _src, _dst, CUFFT_FORWARD);
	cufftDestroy(fftplan);
}

template<typename PREC_T>
void ifft(cuda::unique_ptr<thrust::complex<PREC_T>[]>& src,cuda::unique_ptr<thrust::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx)
{
    cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny, nx, CUFFT_C2C);
	cufftComplex* _src = reinterpret_cast<cufftComplex*>(src.get());
	cufftComplex* _dst = reinterpret_cast<cufftComplex*>(dst.get());
	cufftExecC2C(fftplan, _src, _dst, CUFFT_INVERSE);
	cufftDestroy(fftplan);
}




template<typename PREC_T>
__global__ void gfftshift(thrust::complex<PREC_T>* u, int ny, int nx){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	int hnx = nx / 2; int hny = ny / 2;
	if (w >= hnx || h >= hny){
		return;
	}
	int idx1, idx2;
	thrust::complex<PREC_T> tmp;
	idx1 = h * nx + w;
	idx2 = (h + hny) * nx + (w + hnx);
	tmp = u[idx1];
	u[idx1] = u[idx2];
	u[idx2] = tmp;

	idx1 = h * nx + (w + hnx);
	idx2 = (h + hny) * nx + w;
	tmp = u[idx1];
	u[idx1] = u[idx2];
	u[idx2] = tmp;
}
template<typename PREC_T>
void gfftshift(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u, int ny, int nx){
	dim3 block(16, 16, 1);
	int hny = ny / 2;
	int hnx = nx / 2;
	dim3 grid(ceil((float) hnx / block.x), ceil((float)hny / block.y), 1);
	gfftshift<<<grid,block>>>(u.get(), ny, nx);
	cudaDeviceSynchronize();
}

__global__ void KernelMult( cufftComplex* a, cufftComplex *b, cufftComplex *c, int height, int width)
{
	//????????????????????????????????????????????????????????????
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		c[adr]=cuCmulf(a[adr],b[adr]);
	}	
}

template<typename PREC_T>
__global__ void KernelMult( thrust::complex<PREC_T>* a, thrust::complex<PREC_T>*b, thrust::complex<PREC_T>*c, int height, int width)
{
	//????????????????????????????????????????????????????????????
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		c[adr] = a[adr] * b[adr];
	}	
}

template<typename PREC_T>
void mult(cuda::unique_ptr<thrust::complex<PREC_T>[]>& a, cuda::unique_ptr<thrust::complex<PREC_T>[]>& b,cuda::unique_ptr<thrust::complex<PREC_T>[]>& c,
	int height, int width)
{
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y), 1);
	KernelMult<<<grid,block>>>(a.get(),b.get(),c.get(),height,width);
	cudaDeviceSynchronize();
}

template<typename PREC_T>
__global__ void KernelAdd( thrust::complex<PREC_T>* a, thrust::complex<PREC_T>*b, thrust::complex<PREC_T>*c, int height, int width)
{
	//????????????????????????????????????????????????????????????
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		c[adr] = a[adr] + b[adr];
	}	
}

template<typename PREC_T=float>
void add( cuda::unique_ptr<thrust::complex<PREC_T>[]>& u1,cuda::unique_ptr<thrust::complex<PREC_T>[]>& u2,cuda::unique_ptr<thrust::complex<PREC_T>[]>& udst, int height, int width)
{
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y), 1);
	KernelAdd<<<grid,block>>>(u1.get(),u2.get(),udst.get(),height,width);
	cudaDeviceSynchronize();
}

template<typename COMPLEX_T =float, typename PREC_T=float>
__global__ void mul_scalar( thrust::complex<COMPLEX_T>* u,PREC_T scalar, int height, int width)
{
	//????????????????????????????????????????????????????????????
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		u[adr] *= scalar;
	}	
}

template<typename COMPLEX_T =float,typename PREC_T=float>
void mul_scalar( cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,PREC_T scalar, int height, int width)
{
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y), 1);
	mul_scalar<<<grid,block>>>(u.get(),scalar,height,width);
	cudaDeviceSynchronize();
}

// out must be different from in
// grid and block must be caliculated by out_height and out_width
template<typename PREC_T>
__global__ void gzeropadding(thrust::complex<PREC_T>* in,thrust::complex<PREC_T>* out,
                int in_height,int in_width)
{
	//????????????????????????????????????????????????????????????
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if (w < 2 * in_width && h < 2 * in_height){
		if(in_height/2 <= h && h < in_height*3/2 && in_width/2 <= w && w < in_width*3/2)
		{
			out[w + h * 2 * in_width] = in[w - in_width/2+(h - in_height/2) * in_width];
		}
		else{
			out[w + h * 2 * in_width] = 0;
		}
	}
}

template<typename _Tp>
__global__ void gzeropadding(_Tp* src, _Tp* dst,
                int in_height, int in_width,int out_height,int out_width){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if(w < out_width && h < out_height){
		int64_t ypadsize_l = (out_height - in_height) / 2;
		int64_t xpadsize_l = (out_width - in_width) / 2;
		int64_t ypadsize_r = (out_height - in_height + 1) / 2;
		int64_t xpadsize_r = (out_width - in_width + 1) / 2;
		if ( ypadsize_l <= h && h < (out_height - ypadsize_r) && xpadsize_l <= w && w < (out_width - xpadsize_r) ){
                dst[h * out_width + w] = src[(h - ypadsize_l) * in_width + w - xpadsize_l];
		}
		else{
			dst[h * out_width + w] = 0;
		}
	}
}

template<typename _Tp>
void gzeropadding(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,
                int in_height, int in_width,int out_height,int out_width){
    cuda::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_height * out_width);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_width / block.x), ceil((float)out_height / block.y), 1);
    gzeropadding<<<grid,block>>>(src.get(),tmp.get(),in_height,in_width,out_height,out_width);
	if (src == dst){
		src.reset();
	}
	dst = std::move(tmp);
	cudaDeviceSynchronize();
}

// out must be different from in
template<typename PREC_T>
__global__ void gdel_zero(thrust::complex<PREC_T>* in,thrust::complex<PREC_T>* out,
                int in_height,int in_width)
{
	//????????????????????????????????????????????????????????????
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if ( w < in_width && h < in_height){
		if(in_height/4 <= h && h < in_height*3/4 && in_width/4 <= w && w < in_width*3/4)
		{
			out[w - in_width/4 + (h - in_height/4) * in_width/2] = in[w + h * in_width];
		}
	}
}
template<typename _Tp>
void gdel_zero(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int ny, int nx){
	cuda::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>((ny /2) * (nx / 2));
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
    gdel_zero<<<grid,block>>>(src.get(),tmp.get(),ny,nx);
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
	cudaDeviceSynchronize();
}


template<typename _Tp>
__global__ void gcut(_Tp* src, _Tp* dst,int ny, int nx, int y, int x,int ysize,int xsize){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if (w < xsize && h < ysize){
		int idxdst = h * xsize + w;
        int idxsrc = (h + y)* nx + w + x;
        dst[idxdst] = src[idxsrc];
	}
}
template<typename _Tp>
void gcut(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int ny, int nx, int y, int x,int ysize,int xsize){
	cuda::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(ysize * xsize);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
    gcut<<<grid,block>>>(src.get(),tmp.get(),ny,nx,y,x,ysize,xsize);
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
	cudaDeviceSynchronize();
}

template<typename _Tp>
__global__ void gshrink_int(_Tp* src, _Tp* dst,int in_ny, int in_nx,int out_ny,int out_nx){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if ( w >= out_nx || h >= out_ny){
		return;
	}
	int s = (in_ny / out_ny) < (in_nx / out_nx) ? (in_ny / out_ny) : (in_nx / out_nx);
	dst[w + h * out_nx] = src[w * s + h * s * in_nx];
}

template<typename _Tp>
void gshrink(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int in_ny, int in_nx,int out_ny,int out_nx){
	cuda::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_nx / block.x), ceil((float)out_ny / block.y), 1);
    gshrink_int<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx);
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
	cudaDeviceSynchronize();
}

template<typename _Tp>
__global__ void gMedianFilter(_Tp* src, _Tp* dst,int in_ny, int in_nx,int filter_ny,int filter_nx,int out_ny,int out_nx){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if ( w >= out_nx || h >= out_ny){
		return;
	}
	_Tp sum = 0;
	for (int local_h = 0;local_h < filter_ny;local_h++){
		for (int local_w = 0;local_w < filter_nx;local_w++){
			sum += src[(w * filter_nx + local_w)+ (h * filter_ny + local_h)* in_nx];
		}
	}
	dst[w + h * out_nx] = sum / (filter_ny * filter_nx);
}

template<typename _Tp>
void gMedianFilter(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int in_ny, int in_nx,int filter_ny,int filter_nx){
	cuda::unique_ptr<_Tp[]> tmp;
	int out_ny = in_ny / filter_ny; int out_nx = in_nx / filter_nx;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_nx / block.x), ceil((float)out_ny / block.y), 1);
    gMedianFilter<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,filter_ny,filter_nx,out_ny,out_nx);
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
	cudaDeviceSynchronize();
}

template<typename _Tp>
__global__ void gexpand_int(_Tp* src, _Tp* dst,int in_ny, int in_nx,int out_ny,int out_nx){
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if (w >= in_nx || h >= in_ny){
		return;
	}
	int s = (out_ny / in_ny) < (out_nx / in_nx) ? (out_ny / in_ny) : (out_nx / in_nx);
	for(int local_h = 0;local_h < s;local_h++){
		for(int local_w = 0;local_w < s;local_w++){
			dst[w * s + local_w + (h * s + local_h) * out_nx] = src[w + h * in_nx];
		}
	}
}

template<typename _Tp>
void gexpand(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int in_ny, int in_nx,int out_ny,int out_nx){
	cuda::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)in_nx / block.x), ceil((float)in_ny / block.y), 1);
    gexpand_int<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx);
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
	cudaDeviceSynchronize();
}


template<typename _Tp,typename PREC_T>
__device__ void bilinear(_Tp* src,_Tp& dst, PREC_T dsty,PREC_T dstx,int64_t ny, int64_t nx){
    // ???????????????
    _Tp U1;
    _Tp U2;
	

    int64_t ms = floor(dstx); int64_t ns = floor(dsty);
    if ( 0 <= ms && ms < (nx - 1) && 0 <= ns && ns < (ny - 1)){
        float s = dstx- ms; float t = dsty - ns;
        U1 = (1 - s) * src[ms + ns * nx] + s * src[ (ms + 1) + ns * nx];
        U2 = (1 - s) * src[ms + (ns + 1) * nx] + s * src[ (ms + 1) + (ns + 1) * nx];

        dst = (1 - t) * U1 + t * U2;
    }
    else if(0 <= ms && ms < nx && 0 <= ns && ns < ny){
        dst =  src[ms + ns * nx];
    }
    else{
        dst = 0;
    }
}


// src???dst??????????????????????????????????????????
template<typename _Tp>
__global__ void interpolatelinear(_Tp* src, _Tp* dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){

	int m = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;
	if (m >= out_nx || n >= out_ny){
		return;
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 

	float dstx = m / xscale; 
	float dsty = n / yscale;
	bilinear(src,dst[n * out_nx + m],dsty,dstx,in_ny,in_nx);
}

template<typename _Tp>
void interpolatelinear(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    cuda::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_nx / block.x), ceil((float)out_ny / block.y), 1);
	interpolatelinear<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx);
    
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

// src???dst??????????????????????????????????????????
template<typename _Tp>
__global__ void NearestNeighborInterpolation(_Tp* src, _Tp* dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){

	int m = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;
	if (m >= out_nx || n >= out_ny){
		return;
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 

	float dstx = m / xscale; 
	float dsty = n / yscale;
	int64_t ms = round(dstx); int64_t ns = round(dsty);
	// int64_t ms = (dstx + 0.0001f); int64_t ns = (dsty + 0.0001f);
	// int64_t ms = (dstx); int64_t ns = (dsty);
    if(0 <= ms && ms < in_nx && 0 <= ns && ns < in_ny){
        dst[m + n * out_nx] =  src[ms + ns * in_nx];
		// dst = 0;
    }
    else{
        dst = 0;
    }
}

template<typename _Tp>
void NearestNeighborInterpolation(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    cuda::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_nx / block.x), ceil((float)out_ny / block.y), 1);
	NearestNeighborInterpolation<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx);
    
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}
template<typename _Tp>
__global__ void AreaAverageKernel(_Tp* src, _Tp* dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx,float yscale,float xscale,float S){
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y*blockDim.y + threadIdx.y;
    if (m >= out_nx || n >= out_ny){
        return;
    }
    float src_x = m * xscale; 
    float src_y = n * yscale;
    int src_m = floor(src_x);
    int src_n = floor(src_y);
    _Tp sum = 0;
    int window_m = (int)(src_x + xscale + 0.9999f) - src_m;
    int window_n = (int)(src_y + yscale + 0.9999f) - src_n;
    float lyl = (src_n + 1) - src_y;
    float lyr = (src_y + yscale) - (src_n + window_n - 1);
    float lxl = (src_m + 1) - src_x;
    float lxr = (src_x + xscale) -(src_m + window_m - 1);
    for (int local_n = 0; local_n < window_n;local_n++){
        float ly = 1.f;
        if (local_n == 0){
            ly = lyl;
        }
        else if (local_n == window_n - 1){
            ly = lyr;
        }
        
        for (int local_m = 0; local_m < window_m;local_m++){
            float lx = 1;
            if (local_m == 0){
                lx = lxl;
            }
            else if (local_m == window_m - 1){
                lx = lxr;
            }
            int in_n = local_n + src_n;
            int in_m = local_m + src_m;
            if (in_n >= in_ny || in_n < 0 || in_m >= in_nx || in_m < 0){
                break;
            }
            int64_t in_idx = in_n * in_nx + in_m;
            sum += src[in_idx] * lx * ly;
        }
    }
    dst[n * out_nx + m] = sum / S;
}

template<typename _Tp>
__global__ void AreaAverageKernel(_Tp* src, _Tp* dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    int64_t m = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t n = blockIdx.y*blockDim.y + threadIdx.y;
    if (m >= out_nx || n >= out_ny){
        return;
    }
    float xscale = (float)in_nx / out_nx;
    float yscale = (float)in_ny / out_ny; 
    float S = xscale * yscale;
    float src_x = m * xscale; 
    float src_y = n * yscale;
    int64_t src_m = floor(src_x);
    int64_t src_n = floor(src_y);
    _Tp sum = 0;
    int64_t window_m = (int)(src_x + xscale + 0.9999f) - src_m;
    int64_t window_n = (int)(src_y + yscale + 0.9999f) - src_n;
    float lyl = (src_n + 1) - src_y;
    float lyr = (src_y + yscale) - (src_n + window_n - 1);
    float lxl = (src_m + 1) - src_x;
    float lxr = (src_x + xscale) -(src_m + window_m - 1);
    for (int64_t local_n = 0; local_n < window_n;local_n++){
        float ly = 1.f;
        if (local_n == 0){
            ly = lyl;
        }
        else if (local_n == window_n - 1){
            ly = lyr;
        }
        
        for (int64_t local_m = 0; local_m < window_m;local_m++){
            float lx = 1;
            if (local_m == 0){
                lx = lxl;
            }
            else if (local_m == window_m - 1){
                lx = lxr;
            }
            int64_t in_n = local_n + src_n;
            int64_t in_m = local_m + src_m;
            if (in_n >= in_ny || in_n < 0 || in_m >= in_nx || in_m < 0){
                break;
            }
            int64_t in_idx = in_n * in_nx + in_m;
            sum += src[in_idx] * lx * ly;
        }
    }
    dst[n * out_nx + m] = sum / S;
}

// ???????????????
template<typename _Tp>
void AreaAverage(cuda::unique_ptr<_Tp[]>& src, cuda::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    cuda::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = cuda::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    float xscale = (float)in_nx / out_nx;
    float yscale = (float)in_ny / out_ny; 
    float S = xscale * yscale;
    dim3 block(16, 16, 1);
	dim3 grid(ceil((float)out_nx / block.x), ceil((float)out_ny / block.y), 1);
    AreaAverageKernel<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx,yscale,xscale,S);
    // AreaAverageKernel<<<grid,block>>>(src.get(),tmp.get(),in_ny,in_nx,out_ny,out_nx);
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}


}

#endif