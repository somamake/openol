#ifndef LARGE_H
#define LARGE_H
#include "golutils.h"

namespace ol{
// u is device data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gAsmProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);
	// int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	//開口面a(x,y)のフーリエ変換
	cudaDeviceSynchronize();
	gzeropadding<<< grid, block >>>(u.get(),buf1.get(),ny,nx);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	//H(u,v)を算出
    gAsmTransferF<<< grid, block >>>(buf2.get(),ny2,nx2,dv,du,lambda,d);
	gfftshift(buf2,ny2,nx2);
	cudaDeviceSynchronize();
	//複素乗算
	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
	//逆FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	cudaDeviceSynchronize();
	gdel_zero<<< grid, block >>>(buf1.get(),u.get(),ny2,nx2);
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	mul_scalar<<<grid,block>>>(u.get(),(COMPLEX_T)(1.0f / (ny2 * nx2)),ny,nx);
	cudaDeviceSynchronize();
	//開放
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
}
}

#endif