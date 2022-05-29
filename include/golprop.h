#ifndef GOLPROP_H
#define GOLPROP_H
#include "golutils.h"

namespace ol{

template<typename COMPLEX_T,typename PREC_T>
__global__ void gFresnelResponseFFTShift(thrust::complex<COMPLEX_T>* u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
    if ( (w < nx) && (h < ny) ){
		int hnx = nx / 2;
		int hny = ny / 2;
		int hshift,wshift;
		if ( w <  hnx && h < hny){
			hshift = h + hny;
			wshift = w + hnx;
		}
		else if (w >=  hnx && h < hny){
			hshift = h + hny;
			wshift = w - hnx;
		}
		else if(w < hnx && h >= hny){
			hshift = h - hny;
			wshift = w + hnx;
		}
		else{
			hshift = h - hny;
			wshift = w - hnx;
		}
        PREC_T x = ((double)( (wshift - nx/2)  )) * dx;
        PREC_T y = ((double)( (hshift - ny/2)  )) * dy;
        PREC_T tmp= 1.0 * M_PI* ( x*x + y*y  )  / (lambda * d);
		u[idx] = thrust::complex<COMPLEX_T>(cos(tmp),sin(tmp));
    }
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gFresnelResponse(thrust::complex<COMPLEX_T>* u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
    if ( (w < nx) && (h < ny) ){
		int hnx = nx / 2;
		int hny = ny / 2;
		u[idx] = w;
		PREC_T x = (PREC_T) (w - hnx)  * dx;
        PREC_T y = (PREC_T) (h - hny) * dy;
        PREC_T tmp= 1.0 * M_PI* ( x*x + y*y  )  / (lambda * d);
		u[idx] = thrust::complex<COMPLEX_T>(cos(tmp),sin(tmp));
    }
}

template<typename PREC_T, typename COMPLEX_T>
void gFresnelResponse(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z)
{
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gFresnelResponse<<<grid,block>>>(H.get(),ny,nx,dv,du,lambda,z);
	cudaDeviceSynchronize();
}


// u is host data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gFresnelProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
	auto h_buf = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);

	// cufftComplex* d_buf1, *d_buf2, *d_buf3;
	thrust::complex<COMPLEX_T>* d_buf1, *d_buf2, *d_buf3; 
	cudaMalloc( (void**) &d_buf1, mem_size);
	cudaMalloc( (void**) &d_buf2, mem_size);
	cudaMalloc( (void**) &d_buf3, mem_size);
	ol::zeropadding(u,h_buf,ny,nx);
    ol::fftshift(h_buf,ny2,nx2);

    cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  

	//開口面a(x,y)のフーリエ変換
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)d_buf1, (cufftComplex*)d_buf2, CUFFT_FORWARD);

	//p(x,y)を算出
    ol::FresnelResponse(h_buf,ny2,nx2,dy,dx,lambda,d);
	ol::fftshift(h_buf, ny2, nx2);  //象限の交換
	//p(x,y)のFFT
    cudaMemcpy( d_buf3, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  
	cufftExecC2C(fftplan, (cufftComplex*)d_buf3, (cufftComplex*)d_buf1, CUFFT_FORWARD);

	//複素乗算
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	// KernelMult<<< grid, block>>>((cufftComplex*)d_buf2, (cufftComplex*)d_buf1, (cufftComplex*)d_buf3, nx2, ny2);
	KernelMult<<< grid, block>>>(d_buf2, d_buf1, d_buf3, ny2, nx2);

	//逆FFT
	cufftExecC2C(fftplan, (cufftComplex*)d_buf3,(cufftComplex*) d_buf1, CUFFT_INVERSE);

	//計算結果を転送
	cudaMemcpy( h_buf.get(), d_buf1, mem_size,cudaMemcpyDeviceToHost);
	//結果をビットマップファイルとして保存
    ol::fftshift(h_buf,ny2,nx2);

	ol::del_zero(h_buf,u,ny2,nx2);
	//開放
	cufftDestroy(fftplan);
	cudaFree(d_buf1);
	cudaFree(d_buf2);
	cudaFree(d_buf3);
	h_buf.reset();
}

// u is device data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	// int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	//開口面a(x,y)のフーリエ変換
	// gzeropadding<<< grid, block >>>(u.get(),buf1.get(),ny,nx);
	gzeropadding(u,buf1,ny,nx,ny2,nx2);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	//p(x,y)を算出
    // gFresnelResponseFFTShift<<< grid, block >>>(buf2.get(),ny2,nx2,dy,dx,lambda,d);
	gFresnelResponse<<< grid, block >>>(buf2.get(),ny2,nx2,dy,dx,lambda,d);
	gfftshift(buf2,ny2,nx2);
	cudaDeviceSynchronize();
	//p(x,y)のFFT
	cufftExecC2C(fftplan, (cufftComplex*)buf2.get(), (cufftComplex*)buf2.get(), CUFFT_FORWARD);
	mul_scalar<<<grid,block>>>(buf2.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	//複素乗算
	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
	//逆FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	
	gdel_zero<<< grid, block >>>(buf1.get(),u.get(),ny2,nx2);
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	// mul_scalar<<<grid,block>>>(u.get(),(COMPLEX_T) (1.0f / (ny2 * nx2 * d)),ny,nx);
	cudaDeviceSynchronize();
	//開放
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
	cudaDeviceSynchronize();
}


/*角スペクトル法のH計算関数*/
template<typename PREC_T, typename COMPLEX_T>
__global__ void gAsmTransferF(thrust::complex<COMPLEX_T>* H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (w < nx) && (h < ny) ){
		int idx = h * nx + w;
		int hnx = nx / 2;
		int hny = ny / 2;
		PREC_T u = (w - hnx) * du;
		PREC_T v = (h - hny) * dv;
		PREC_T w = sqrt(1/(lambda * lambda) - u * u - v * v);
		PREC_T phase = 2 * M_PI *  w * z;
		H[idx] = thrust::complex<COMPLEX_T>(cos(phase),sin(phase));
	}
}

template<typename PREC_T, typename COMPLEX_T>
void gAsmTransferF(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z)
{
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gAsmTransferF<<<grid,block>>>(H.get(),ny,nx,dv,du,lambda,z);
	cudaDeviceSynchronize();
}

/*角スペクトル法のH計算関数*/
template<typename PREC_T, typename COMPLEX_T>
__global__ void gshiftedAsmTransferF(thrust::complex<COMPLEX_T>* H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z,PREC_T oy, PREC_T ox,PREC_T v0,PREC_T vw,PREC_T u0,PREC_T uw)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	
    if ( (w < nx) && (h < ny) ){
		int idx = h * nx + w;
		int hnx = nx / 2;
		int hny = ny / 2;
		PREC_T u = (w - hnx) * du;
		PREC_T v = (h - hny) * dv;
		// if ( (abs(u - u0) < 0.5 * uw) && (abs(v - v0) < 0.5 * vw) ){
			PREC_T w = sqrt(1/(lambda * lambda) - u * u - v * v);
			PREC_T phase = 2 * M_PI *  (w * z + ox * u + oy * v);
			H[idx] = thrust::complex<COMPLEX_T>(cos(phase),sin(phase));
		// }
		// else{
			// H[idx] = 0;
		// }
		
	}
}

template<typename PREC_T, typename COMPLEX_T>
void gshiftedAsmTransferF(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z,PREC_T oy,PREC_T ox)
{
	PREC_T sx = 0.5 / du;
	PREC_T sy = 0.5 / dv;
	PREC_T u0, uw, ulm, ulp, v0, vw ,vlm, vlp;
	ulm = 1 / sqrt( std::pow((ox - sx), -2) * z * z + 1.0f) / lambda;
	ulp = 1 / sqrt( std::pow((ox + sx), -2) * z * z + 1.0f) / lambda;

	vlm = 1 / sqrt( std::pow((oy - sy), -2) * z * z + 1.0f) / lambda;
	vlp = 1 / sqrt( std::pow((oy + sy), -2) * z * z + 1.0f) / lambda;
	if (sx < ox){
		u0 = 0.5f * (ulp + ulm);
		uw = ulp - ulm;
	}
	else if (ox <= -sx){
		u0 = -0.5f * (ulp + ulm);
		uw = ulm - ulp;
	}
	else if (-sx <= ox && ox < sx){
		u0 = 0.5f * (ulp - ulm);
		uw = ulp + ulm;
	}
	
	else {
		printf("bugx\n");
	}

	if (sy < oy){
		v0 = 0.5f * (vlp + vlm);
		vw = vlp - vlm;
	}

	else if (oy <= -sy){
		v0 = -0.5f * (vlp + vlm);
		vw = vlm - vlp;
	}
	else{
		v0 = 0.5f * (vlp - vlm);
		vw = vlp + vlm;
	}
	vw = abs(vw);
	uw = abs(uw);
	// else if (-sy <= oy && oy < sy){
	// 	v0 = 0.5f * (vlp - vlm);
	// 	vw = vlp + vlm;
	// }
	
	// else{
	// 	printf("bugy\n");
		
	// }
	// printf("%lf\n",sy);
	// 	printf("%lf\n",oy);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gshiftedAsmTransferF<<<grid,block>>>(H.get(),ny,nx,dv,du,lambda,z,oy,ox,v0,vw,u0,uw);
	cudaDeviceSynchronize();
}

template<typename PREC_T, typename COMPLEX_T>
__global__ void gAsmTransferFFTShift(thrust::complex<COMPLEX_T>* H,int ny, int nx, 
                    PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (w < nx) && (h < ny) ){
		int idx = h * nx + w;
		int hnx = nx / 2;
		int hny = ny / 2;
		int hshift,wshift;
		if ( w <  hnx && h < hny){
			hshift = h + hny;
			wshift = w + hnx;
		}
		else if (w >=  hnx && h < hny){
			hshift = h + hny;
			wshift = w - hnx;
		}
		else if(w < hnx && h >= hny){
			hshift = h - hny;
			wshift = w + hnx;
		}
		else{
			hshift = h - hny;
			wshift = w - hnx;
		}
		PREC_T u = (wshift - hnx) * du;
		PREC_T v = (hshift - hny) * dv;
		PREC_T w = sqrt(1/(lambda * lambda) - u * u - v * v);
		PREC_T phase = 2 * M_PI *  w * z;
		H[idx] = thrust::complex<COMPLEX_T>(cos(phase),sin(phase));
	}
}

// u is host data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gAsmProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int ny,int nx, PREC_T dy,PREC_T dx,PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
    PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);
	auto h_buf = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(32, 32, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);

	cufftComplex* d_buf1, *d_buf2, *d_buf3; 
	cudaMalloc( (void**) &d_buf1, mem_size);
	cudaMalloc( (void**) &d_buf2, mem_size);
	cudaMalloc( (void**) &d_buf3, mem_size);
	ol::zeropadding(u,h_buf,ny,nx);
    // ol::fftshift(h_buf,ny2,nx2);

	//開口面a(x,y)のフーリエ変換
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  
	cufftExecC2C(fftplan, d_buf1, d_buf2, CUFFT_FORWARD);

    ol::AsmTransferF(h_buf,dv,du,ny2,nx2,lambda,d);
    ol::fftshift(h_buf,ny2,nx2);
	// auto imgtmp = std::make_unique<uint8_t[]>(ny2 * nx2);
    // ol::complex2img(h_buf,imgtmp,ny2,nx2,true);
    // bmpwrite(PROJECT_ROOT "/out/tmp.bmp",imgtmp,ny2,nx2);
    // imgtmp.reset();

    cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  

	//複素乗算
	KernelMult<<< grid, block>>>(d_buf2, d_buf1, d_buf3, ny2, nx2);

	//逆FFT
	cufftExecC2C(fftplan, d_buf3, d_buf1, CUFFT_INVERSE);

	//計算結果を転送
	cudaMemcpy( h_buf.get(), d_buf1, mem_size,cudaMemcpyDeviceToHost);
	//結果をビットマップファイルとして保存
    // ol::fftshift(h_buf,ny2,nx2);

	ol::del_zero(h_buf,u,ny2,nx2);
	//開放
	cufftDestroy(fftplan);
	cudaFree(d_buf1);
	cudaFree(d_buf2);
	cudaFree(d_buf3);
	h_buf.reset();
}

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
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	//開口面a(x,y)のフーリエ変換
	cudaDeviceSynchronize();
	gzeropadding<<< grid, block >>>(u.get(),buf1.get(),ny,nx);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	// mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

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

// u is device data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gshiftedAsmProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& usrc,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& udst,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d,PREC_T oy,PREC_T ox){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);
	// int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	//開口面a(x,y)のフーリエ変換
	cudaDeviceSynchronize();
	gzeropadding<<< grid, block >>>(usrc.get(),buf1.get(),ny,nx);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	// mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	//H(u,v)を算出
    gshiftedAsmTransferF(buf2,ny2,nx2,dv,du,lambda,d,oy,ox);
	gfftshift(buf2,ny2,nx2);
	cudaDeviceSynchronize();
	//複素乗算
	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
	//逆FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	cudaDeviceSynchronize();
	gdel_zero<<< grid, block >>>(buf1.get(),udst.get(),ny2,nx2);
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	mul_scalar<<<grid,block>>>(udst.get(),(COMPLEX_T)(1.0f / (ny2 * nx2)),ny,nx);
	cudaDeviceSynchronize();
	//開放
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
}

template<typename COMPLEX_T,typename PREC_T>
void Prop(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d,PROPMODE propmode = ol::PROPMODE::AUTO){
	if (propmode == PROPMODE::FRESNEL){
		printf("FresnelProp\n");
        gFresnelProp(u,ny,nx,dy,dx,lambda,d);
    }
    else if (propmode == PROPMODE::ASM){
		printf("AsmProp\n");
        gAsmProp(u,ny,nx,dy,dx,lambda,d);
    }
    else if (propmode == PROPMODE::AUTO){
        if (AsmPropCheck(ny,nx,dy,dx,lambda,d) == true){
            printf("AsmProp\n");
            gAsmProp(u,ny,nx,dy,dx,lambda,d);
        }
        else{
            if (FresnelPropCheck(ny,nx,dy,dx,lambda,d) == false){
                printf("warning!!\neither AsmProp or FresnelProp does not meet the condition.\n");
				gAsmProp(u,ny,nx,dy,dx,lambda,d);
				return;
            }
            printf("FresnelProp\n");
            gFresnelProp(u,ny,nx,dy,dx,lambda,d);
        }
    }
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gshiftedFresnelProp_u(thrust::complex<COMPLEX_T>* usrc,thrust::complex<COMPLEX_T>* udst,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	
	if ( (w < nx) && (h < ny) ){
		int hny = ny / 2; int hnx = nx / 2;
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y1 = (h - hny) * dy;
		PREC_T phiu_y = (s * s - s) * y1 * y1 - 2 * s * oy * y1;
		PREC_T x1 = (w - hnx) * dx;
		PREC_T phiu = (s * s - s) * x1 * x1 - 2 * s * ox * x1;
		phiu += phiu_y;
		phiu *= tmp;
		PREC_T real = cos(phiu);
		PREC_T imag = sin(phiu);
		udst[idx] = usrc[idx] * thrust::complex<COMPLEX_T>(real,imag);
	}
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gshiftedFresnelProp_hFFTShift(thrust::complex<COMPLEX_T>* dst,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	if ( (w < nx) && (h < ny) ){
		int hny = ny / 2; int hnx = nx / 2;
		int hshift,wshift;
		if ( w <  hnx && h < hny){
			hshift = h + hny;
			wshift = w + hnx;
		}
		else if (w >=  hnx && h < hny){
			hshift = h + hny;
			wshift = w - hnx;
		}
		else if(w < hnx && h >= hny){
			hshift = h - hny;
			wshift = w + hnx;
		}
		else{
			hshift = h - hny;
			wshift = w - hnx;
		}
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y1 = (hshift - hny) * dy;
		PREC_T phih_y = s * y1 * y1;
		
		PREC_T x1 = (wshift - hnx) * dx;
		PREC_T phih = s * x1 * x1 + phih_y;
		phih *= tmp;
		dst[idx] = thrust::complex<COMPLEX_T>(cos(phih),sin(phih));
	}
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gshiftedFresnelProp_h(thrust::complex<COMPLEX_T>* dst,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	if ( (w < nx) && (h < ny) ){
		int hny = ny / 2; int hnx = nx / 2;
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y1 = (h - hny) * dy;
		PREC_T phih_y = s * y1 * y1;
		
		PREC_T x1 = (w - hnx) * dx;
		PREC_T phih = s * x1 * x1 + phih_y;
		phih *= tmp;
		dst[idx] = thrust::complex<COMPLEX_T>(cos(phih),sin(phih));
	}
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gshiftedFresnelProp_Cz(thrust::complex<COMPLEX_T>* u,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	if ( (w < nx) && (h < ny) ){
		int hny = ny / 2; int hnx = nx / 2;
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y2 = (h - hny) * dy;
		PREC_T phic_y = ( (1 - s) * y2 * y2 + 2 * oy * y2 + oy * oy);
		PREC_T x2 = (w - hnx) * dx;
		PREC_T phic = ( (1 - s) * x2 * x2 + 2 * ox * x2 + ox * ox);
		phic += phic_y;
		phic *= tmp;
		PREC_T real = cos(phic);
		PREC_T imag = sin(phic);
		u[idx] = u[idx] * thrust::complex<COMPLEX_T>(real,imag);
	}
}


template<typename PREC_T=float,typename COMPLEX_T>
void gshiftedFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& usrc,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& udst,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z, PREC_T s,PREC_T oy,PREC_T ox){
	int ny2 = ny * 2;
	int nx2 = nx * 2;

	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);


	gshiftedFresnelProp_u<<<grid,block>>>(usrc.get(),udst.get(),ny,nx,dy,dx,lambda,z,s,oy,ox);
	cudaDeviceSynchronize();
	
	grid = dim3(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
    gzeropadding(udst,buf1,ny,nx,ny2,nx2);
	cudaDeviceSynchronize();

	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	cufftDestroy(fftplan);
	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	gshiftedFresnelProp_h<<<grid,block>>>(buf2.get(),ny2,nx2,dy,dx,lambda,z,s);
	// gFresnelResponse<<< grid, block >>>(buf2.get(),ny2,nx2,dy,dx,lambda,z);
	gfftshift(buf2,ny2,nx2);
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
    cufftExecC2C(fftplan, (cufftComplex*)buf2.get(), (cufftComplex*)buf2.get(), CUFFT_FORWARD);
	cufftDestroy(fftplan);
	mul_scalar<<<grid,block>>>(buf2.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);

	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
    
    gdel_zero<<< grid, block>>>(buf1.get(),udst.get(),ny2,nx2);
	
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	// mul_scalar<<<grid,block>>>(u.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny,nx);
	gshiftedFresnelProp_Cz<<<grid,block>>>(udst.get(),ny,nx,dy,dx,lambda,z,s,oy,ox);
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
	cudaDeviceSynchronize();
}

template<typename COMPLEX_T,typename PREC_T>
void shiftedProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& usrc,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& udst,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d, PREC_T oy,PREC_T ox,PROPMODE propmode){
	if (propmode == ol::ASM){
		gshiftedAsmProp(usrc,udst,ny,nx,dy,dx,lambda,d,oy,ox);
	}
	else if (propmode == ol::FRESNEL){
		gshiftedFresnelProp(usrc,udst,ny,nx,dy,dx,lambda,d,(PREC_T)1.0,oy,ox);
	}
	else{
		printf("error\n");
		exit(1);
	}
}


// split asm propagation
template<typename PREC_T=float,typename COMPLEX_T>
void gsplitProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int64_t height, int64_t width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z,ol::PROPMODE propmode,int blocknum_y = 2,int blocknum_x = 2){
    
	int blockwidth = width / blocknum_x;
    int blockheight = height / blocknum_y;
    auto udst = std::make_unique<std::complex<COMPLEX_T>[]>(height * width);
    auto u_block_tmp = std::make_unique<std::complex<COMPLEX_T>[]>(blockheight * blockwidth);
    // auto u_block_dst = std::make_unique<std::complex<COMPLEX_T>[]>(blockheight * blockwidth);
	auto du_block_src = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(blockheight * blockwidth);
    auto du_block_dst = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(blockheight * blockwidth);
    for (int global_n = 0; global_n < blocknum_y;global_n++){
        for (int global_m = 0;global_m < blocknum_x;global_m++){
            // copy from u to u_block
            for (int local_n = 0; local_n < blockheight;local_n++){
                for (int local_m = 0; local_m < blockwidth; local_m++){
                    int n = global_n * blockheight + local_n;
                    int m = global_m * blockwidth + local_m;
                    u_block_tmp[local_n * blockwidth + local_m] = u[n * width + m];
                }
            }
			ol::cpu2cuda(u_block_tmp,du_block_src,blockheight * blockwidth);
			
            for (int dst_n = 0; dst_n < blocknum_y;dst_n++){
                for (int dst_m = 0;dst_m < blocknum_x;dst_m++){
                    float oy = (global_n - dst_n) * blockheight * dy;
                    float ox = (global_m - dst_m) * blockwidth * dx;
                    shiftedProp(du_block_src,du_block_dst,blockheight,blockwidth,dy,dx,lambda,z,oy,ox,propmode);
                    // udst + ublock
					ol::cuda2cpu(du_block_dst,u_block_tmp,blockheight * blockwidth);
                    for (int local_n = 0; local_n < blockheight;local_n++){
                        for (int local_m = 0; local_m < blockwidth; local_m++){
                            int n = dst_n * blockheight + local_n;
                            int m = dst_m * blockwidth + local_m;
                            udst[n * width + m] += u_block_tmp[local_n * blockwidth + local_m];
                        }
                    }
                }
            }
        }
    }
    u = std::move(udst);
}

}

#endif