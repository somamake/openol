#ifndef OLUTILS_H
#define OLUTILS_H

#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>
#include <omp.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include "olimg.h"



namespace ol{

template<typename PREC_T>
void fft(std::unique_ptr<std::complex<PREC_T>[]>& src,std::unique_ptr<std::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx)
{
    if constexpr(std::is_same_v<PREC_T,double>){
        fftw_complex* _usrc = reinterpret_cast<fftw_complex*>(src.get());
        fftw_complex* _udst = reinterpret_cast<fftw_complex*>(dst.get());
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
        fftw_plan p = fftw_plan_dft_2d( ny, nx, _usrc, _udst, FFTW_FORWARD, FFTW_ESTIMATE );
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
    else if constexpr(std::is_same_v<PREC_T,float>){
        fftwf_complex* _usrc = reinterpret_cast<fftwf_complex*>(src.get());
        fftwf_complex* _udst = reinterpret_cast<fftwf_complex*>(dst.get());
        fftwf_init_threads();
        fftwf_plan_with_nthreads(omp_get_max_threads());
        // fftwf_plan_with_nthreads(1);
        fftwf_plan p = fftwf_plan_dft_2d( ny, nx, _usrc, _udst, FFTW_FORWARD, FFTW_ESTIMATE );
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }
}

/*IFFT計算関数*/
template<typename PREC_T>
void ifft(std::unique_ptr<std::complex<PREC_T>[]>& src,std::unique_ptr<std::complex<PREC_T>[]>& dst, int64_t ny, int64_t nx)
{
    if constexpr(std::is_same_v<PREC_T,double>){
        fftw_complex* _usrc = reinterpret_cast<fftw_complex*>(src.get());
        fftw_complex* _udst = reinterpret_cast<fftw_complex*>(dst.get());
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
        fftw_plan p = fftw_plan_dft_2d( ny, nx, _usrc, _udst, FFTW_BACKWARD, FFTW_ESTIMATE );
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
    else if constexpr(std::is_same_v<PREC_T,float>){
        fftwf_complex* _usrc = reinterpret_cast<fftwf_complex*>(src.get());
        fftwf_complex* _udst = reinterpret_cast<fftwf_complex*>(dst.get());
        fftwf_init_threads();
        fftwf_plan_with_nthreads(omp_get_max_threads());
        fftwf_plan p = fftwf_plan_dft_2d( ny, nx, _usrc, _udst, FFTW_BACKWARD, FFTW_ESTIMATE );
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }
    
}
/*IFFT計算関数終了*/

/*規格化関数*/
template<typename PREC_T>
void standardization(std::unique_ptr<std::complex<PREC_T>[]>& u,int64_t ny,int64_t nx)
{
    float c = 1.0 / (ny * nx);
	for(int64_t n = 0;n < ny;n++){
		for(int64_t m = 0;m < nx;m++){
			u[m + n * nx] *= c;
		}
	}
}

template<typename COMPLEX_T, typename PREC_T>
void mul_scalar(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,PREC_T c, int64_t ny,int64_t nx)
{
	for(int64_t n = 0;n < ny;n++){
		for(int64_t m = 0;m < nx;m++){
			u[m + n * nx] *= c;
		}
	}
}



/*fftshift関数*/

template<typename PREC_T>
void fftshift(std::unique_ptr<std::complex<PREC_T>[]>& u,int64_t height,int64_t width)
{
    int64_t halfheight = height / 2;
    int64_t halfwidth = width / 2;
    for(int64_t h = 0;h < halfheight;h++){
        for(int64_t w = 0;w < halfwidth;w++){	
            size_t idx1, idx2;
            std::complex<PREC_T> tmp;
            idx1 = h * width + w;
            idx2 = (h + halfheight) * width + (w + halfwidth);
            tmp = u[idx1];
            u[idx1] = u[idx2];
            u[idx2] = tmp;

            idx1 = h * width + (w + halfwidth);
            idx2 = (h + halfheight) * width + w;
            tmp = u[idx1];
            u[idx1] = u[idx2];
            u[idx2] = tmp;
        }
    }
}
/*fftshift関数終了*/

/*複素積関数*/
template<typename PREC_T>
void mul_complex(std::unique_ptr<std::complex<PREC_T>[]>& in1, std::unique_ptr<std::complex<PREC_T>[]>& in2, 
                    std::unique_ptr<std::complex<PREC_T>[]>& out,size_t height, size_t width)
{ 
    for(size_t n = 0; n < height;n++){
		for(size_t m = 0;m < width;m++){
			out[m + n * width] = in1[m + n * width] * in2[m + n * width];
        }
    }
}

/*ゼロパディング関数*/
template<typename PREC_T>
void zeropadding(std::unique_ptr<std::complex<PREC_T>[]>& in,std::unique_ptr<std::complex<PREC_T>[]>& out,
                size_t in_height,size_t in_width)
{
    auto tmp = std::make_unique<std::complex<PREC_T>[]>(in_height * in_width * 4);
    for(size_t n=0;n<2*in_height;n++){
        for(size_t m=0;m<2*in_width;m++){
            if(in_height/2 <= n && n < in_height*3/2 && in_width/2 <= m && m < in_width*3/2)
            {
                tmp[m + n * 2 * in_width] = in[m - in_width/2+(n - in_height/2) * in_width];
            }
            else{
                tmp[m + n * 2 * in_width] = 0;
            }
        }
    }
    if (out == in){
        in.reset();
    }
    out = std::move(tmp);
}

template<typename _Tp>
void zeropadding(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,
                int64_t in_height, int64_t in_width,int64_t out_height,int64_t out_width){
    int64_t ypadsize = (out_height - in_height) / 2;
    int64_t xpadsize = (out_width - in_width) / 2;
    auto tmp = std::make_unique<_Tp[]>(out_height * out_width);
    for (int64_t h = 0;h < out_height;h++){
        for (int64_t w = 0;w < out_width;w++){
            if ( ypadsize <= h && h < (out_height - ypadsize) && xpadsize <= w && w < (out_width - xpadsize) ){
                tmp[h * out_width + w] = src[(h - ypadsize) * in_width + w - xpadsize];
            }
            else{
                tmp[h * out_width + w] = 0;
            }
            
        }
    }
    if (dst == src){
        dst.reset();
    }
    dst = std::move(tmp);
}
/*ゼロパディング関数終了*/

template<typename PREC_T>
void del_zero(std::unique_ptr<std::complex<PREC_T>[]>& in,std::unique_ptr<std::complex<PREC_T>[]>& out,
                int64_t in_height,int64_t in_width)
{
    auto tmp = std::make_unique<std::complex<PREC_T>[]>(in_height * in_width / 4);
	for(int64_t n=0;n<in_height;n++){
		for(int64_t m=0;m<in_width;m++){
            if(in_height/4 <= n && n < in_height*3/4 && in_width/4 <= m && m < in_width*3/4)
            {
                tmp[m-in_width/4+(n-in_height/4)*in_width/2] = in[m+n*in_width];
            }
		}
	}
    if (out == in){
        in.reset();
    }
    out = std::move(tmp);
}

template<typename _Tp>
void cut(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t height, int64_t width, int64_t y, int64_t x,int64_t ysize,int64_t xsize){
    auto tmp = std::make_unique<_Tp[]>(ysize * xsize);
    for (int64_t h = 0; h < ysize;h++){
        for (int64_t w = 0;w < xsize;w++){
            int64_t idxdst = h * xsize + w;
            int64_t idxsrc = (h + y)* width + w + x;
            tmp[idxdst] = src[idxsrc];
        }
    }
    dst = std::move(tmp);
}

template<typename _Tp,typename PREC_T=float>
_Tp bilinear(std::unique_ptr<_Tp[]>& src, PREC_T dsty,PREC_T dstx,int64_t ny, int64_t nx){
    // 双一次補間
    _Tp U1;
    _Tp U2;

    int64_t ms = floor(dstx); int64_t ns = floor(dsty);
    if ( 0 <= ms && ms < (nx - 1) && 0 <= ns && ns < (ny - 1)){
        PREC_T s = dstx - ms; PREC_T t = dsty - ns;
        U1 = (1 - s) * src[ms + ns * nx] + s * src[ (ms + 1) + ns * nx];
        U2 = (1 - s) * src[ms + (ns + 1) * nx] + s * src[ (ms + 1) + (ns + 1) * nx];

        return (1 - t) * U1 + t * U2;
    }
    else if(0 <= ms && ms < nx && 0 <= ns && ns < ny){
        return src[ms + ns * nx];
    }
    else{
        return 0;
    }
}

template<typename _Tp,typename PREC_T>
void bilinear(std::unique_ptr<_Tp[]>& src,_Tp& dst, PREC_T dsty,PREC_T dstx,int64_t ny, int64_t nx){
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


// inline Circulation

template<typename _Tp,typename PREC_T>
_Tp bilinearCirculation(std::unique_ptr<_Tp[]>& src, PREC_T dsty,PREC_T dstx,int64_t ny, int64_t nx,float range){
    // 双一次補間
    _Tp U1;
    _Tp U2;
    _Tp dst;
    float half = range / 2;
    int64_t ms = floor(dstx); int64_t ns = floor(dsty);
    if ( 0 <= ms && ms < (nx - 1) && 0 <= ns && ns < (ny - 1)){
        float s = dstx- ms; float t = dsty - ns;
        float addnum = 0;

        addnum = (abs(src[ms + ns * nx] - src[ (ms + 1) + ns * nx]) <  half) ? 0 : range;
        U1 = (1 - s) * (src[ms + ns * nx] + addnum) + s * src[ (ms + 1) + ns * nx];
        
        addnum = (abs(src[ms + (ns + 1) * nx] - src[ (ms + 1) + (ns + 1)* nx]) <  half) ? 0 : range;
        U2 = (1 - s) * (src[ms + (ns + 1) * nx] + addnum) + s * src[ (ms + 1) + (ns + 1) * nx];

        addnum = (abs(U1 - U2) <  half) ? 0 : range;
        dst = (1 - t) * (U1 + addnum) + t * U2;
    }
    else if(0 <= ms && ms < nx && 0 <= ns && ns < ny){
        dst =  src[ms + ns * nx];
    }
    else{
        dst = 0;
    }
    return dst;
}

template<typename _Tp,typename PREC_T>
_Tp bicubic(std::unique_ptr<_Tp[]>& src, PREC_T dsty,PREC_T dstx,int64_t ny, int64_t nx,PREC_T a = -1){
    const int windowsize = 4;

    int64_t ms = floor(dstx); int64_t ns = floor(dsty);
    if (0 <= ms && ms < nx && 0 <= ns && ns < ny){
        _Tp avg = 0;
        for (int64_t j = 0;j < windowsize; j++){
            for (int64_t i = 0;i < windowsize;i++){
                int64_t m = (ms - 1 + i); int64_t n = (ns - 1 + j);
                if ( 0 <= m && m < nx && 0 <= n && n < ny){
                    PREC_T d = sqrt( (m - dstx)*(m - dstx) + (n - dsty)*(n - dsty));
                    PREC_T k;
                    if (0 <= d && d < 1){
                        k = 1 - ( a + 3)*d*d + (a + 2)*d*d*d;
                    }
                    else if ( 1 <= d && d < 2){
                        k = -4*a + 8*a*d - 5*a*d*d + a*d*d*d; 
                    }
                    else{
                        k = 0;
                    }
                    avg += k * src[m + n * nx];
                }
            }
        }
        return avg;
    }
    else{
        return 0;
    }   
}

template<typename _Tp>
void interpolatelinear(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    std::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 
    for(int n = 0;n < out_ny;n++){
        for (int m = 0;m < out_nx;m++){
            float dstx = m / xscale; 
            float dsty = n / yscale;
            tmp[n * out_nx + m] = bilinear(src,dsty,dstx,in_ny,in_nx);
            // dst[n * out_nx + m] = 1;
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}


// 最近傍補間
template<typename _Tp>
void NearestNeighborInterpolation(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    std::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 
    for(int n = 0;n < out_ny;n++){
        for (int m = 0;m < out_nx;m++){
            float dstx = m / xscale; 
            float dsty = n / yscale;
            int64_t ms = round(dstx); int64_t ns = round(dsty);
            if(0 <= ms && ms < in_nx && 0 <= ns && ns < in_ny){
                tmp[m + n * out_nx] =  src[ms + ns * in_nx];
            }
            else{
                tmp = 0;
            }
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

// 角度のような周期的な値の補間
template<typename _Tp>
void interpolatelinearCirculationInt(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx,float range){
    std::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 
    for(int n = 0;n < out_ny;n++){
        for (int m = 0;m < out_nx;m++){
            float dstx = m / xscale; 
            float dsty = n / yscale;
            tmp[n * out_nx + m] = round( bilinearCirculation(src,dsty,dstx,in_ny,in_nx,range) );
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
} 

template<typename _Tp>
void MedianFilter(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t filter_ny,int64_t filter_nx){
	std::unique_ptr<_Tp[]> tmp;
	int64_t out_ny = in_ny / filter_ny; int64_t out_nx = in_nx / filter_nx;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    for (int64_t h = 0;h < out_ny;h++){
        for (int64_t w = 0;w < out_nx;w++){
            _Tp sum = 0;
            for (int64_t local_h = 0;local_h < filter_ny;local_h++){
                for (int64_t local_w = 0;local_w < filter_nx;local_w++){
                    sum += src[(w * filter_nx + local_w)+ (h * filter_ny + local_h)* in_nx];
                }
            }
            tmp[w + h * out_nx] = sum / (float)(filter_ny * filter_nx);
        }

    }
	
	if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

template<typename _Tp>
void shrink(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int in_ny, int in_nx,int out_ny,int out_nx){
    std::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    int s = (in_ny / out_ny) < (in_nx / out_nx) ? (in_ny / out_ny) : (in_nx / out_nx);
    for (int h = 0;h < out_ny;h++){
        for (int w = 0;w < out_nx;w++){
            tmp[w + h * out_nx] = src[w * s + h * s * in_nx];
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

template<typename _Tp>
void expand(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int in_ny, int in_nx,int out_ny,int out_nx){
    std::unique_ptr<_Tp[]> tmp;
	if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    int s = (out_ny / in_ny) < (out_nx / in_nx) ? (out_ny / in_ny) : (out_nx / in_nx);
	for (int n = 0;n < in_ny;n++){
        for (int m = 0;m < in_nx;m++){
            for(int local_h = 0;local_h < s;local_h++){
                for(int local_w = 0;local_w < s;local_w++){
                    tmp[m * s + local_w + (n * s + local_h) * out_nx] = src[m + n * in_nx];
                }
            }
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

// 面積平均法
template<typename _Tp>
void AreaAverage(std::unique_ptr<_Tp[]>& src, std::unique_ptr<_Tp[]>& dst,int64_t in_ny, int64_t in_nx,int64_t out_ny,int64_t out_nx){
    std::unique_ptr<_Tp[]> tmp;
    if (src == dst || (void*)dst.get() == NULL ){
		tmp = std::make_unique<_Tp[]>(out_ny * out_nx);
	}
	else{
		tmp = std::move(dst);
	}
    float xscale = (float)out_nx / in_nx;
    float yscale = (float)out_ny / in_ny; 
    for(int n = 0;n < out_ny;n++){
        for (int m = 0;m < out_nx;m++){
            float dstx = m / xscale; 
            float dsty = n / yscale;
            int64_t ms = round(dstx); int64_t ns = round(dsty);
            if(0 <= ms && ms < in_nx && 0 <= ns && ns < in_ny){
                tmp[m + n * out_nx] =  src[ms + ns * in_nx];
            }
            else{
                tmp = 0;
            }
        }
    }
    if (src == dst){
		src.reset();
	}
    dst = std::move(tmp);
}

}
#endif