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
	//スレッド・ブロック番号を元にアドレス計算
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
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		c[adr] = a[adr] * b[adr];
	}	
}

template<typename PREC_T>
__global__ void KernelAdd( thrust::complex<PREC_T>* a, thrust::complex<PREC_T>*b, thrust::complex<PREC_T>*c, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
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

template<typename PREC_T=float>
__global__ void mul_scalar( thrust::complex<PREC_T>* u,PREC_T scalar, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
		u[adr] *= scalar;
	}	
}

template<typename PREC_T=float>
void mul_scalar( cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,PREC_T scalar, int height, int width)
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
	//スレッド・ブロック番号を元にアドレス計算
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
		int ypadsize = (out_height - in_height) / 2;
    	int xpadsize = (out_width - in_width) / 2;
		if ( ypadsize <= h && h < (out_height - ypadsize) && xpadsize <= w && w < (out_width - xpadsize) ){
                dst[h * out_width + w] = src[(h - ypadsize) * in_width + w - xpadsize];
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
	//スレッド・ブロック番号を元にアドレス計算
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
    // 双一次補間
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


// srcとdstのサイズは整数倍でなくてよい
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

// srcとdstのサイズは整数倍でなくてよい
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

}

#endif