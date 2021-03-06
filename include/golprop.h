#ifndef GOLPROP_H
#define GOLPROP_H
#include "golutils.h"
#include "oldefine.h"

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

	cufftComplex* d_buf1, *d_buf2;
	// thrust::complex<COMPLEX_T>* d_buf1, *d_buf2, *d_buf3; 
	cudaMalloc( (void**) &d_buf1, mem_size);
	cudaMalloc( (void**) &d_buf2, mem_size);
	// cudaMalloc( (void**) &d_buf3, mem_size);
	ol::zeropadding(u,h_buf,ny,nx);
    ol::fftshift(h_buf,ny2,nx2);

    cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  

	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, d_buf1, d_buf1, CUFFT_FORWARD);

    ol::FresnelResponse(h_buf,ny2,nx2,dy,dx,lambda,d);
	ol::fftshift(h_buf, ny2, nx2);
    cudaMemcpy( d_buf2, h_buf.get(), mem_size,cudaMemcpyHostToDevice);

	// ol::NearestNeighborInterpolation(h_buf,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp1.bmp",u,ny,nx,ol::PHASE);
	// printf("save\n");

	// auto h_buf = std::make_unique<std::complex<COMPLEX_T>[]>(ny * nx);
	// cudaMemcpy( h_buf.get(), d_buf3, mem_size,cudaMemcpyDeviceToHost);
	// ol::NearestNeighborInterpolation(h_buf,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp.bmp",u,ny,nx,ol::PHASE);
	// printf("save\n");


	cufftExecC2C(fftplan, d_buf2, d_buf2, CUFFT_FORWARD);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	KernelMult<<< grid, block>>>(d_buf1, d_buf2, d_buf1, ny2, nx2);

	cufftExecC2C(fftplan, d_buf1, d_buf1, CUFFT_INVERSE);

	cudaMemcpy( h_buf.get(), d_buf1, mem_size,cudaMemcpyDeviceToHost);
    ol::fftshift(h_buf,ny2,nx2);

	ol::del_zero(h_buf,u,ny2,nx2);
	//??????
	cufftDestroy(fftplan);
	cudaFree(d_buf1);
	cudaFree(d_buf2);
	h_buf.reset();
}

// u is device data before zero padding
// template<typename COMPLEX_T,typename PREC_T>
// void gFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
//                 int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
// 	int ny2 = ny * 2;
// 	int nx2 = nx * 2;
// 	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
// 	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

// 	dim3 block(16, 16, 1);
// 	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
// 	//?????????a(x,y)?????????????????????
// 	gzeropadding(u,buf1,ny,nx,ny2,nx2);
// 	cudaDeviceSynchronize();
// 	cufftHandle fftplan;
// 	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
// 	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
// 	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
// 	//p(x,y)?????????
// 	gFresnelResponse<<< grid, block >>>(buf2.get(),ny2,nx2,dy,dx,lambda,d);
// 	gfftshift(buf2,ny2,nx2);
// 	cudaDeviceSynchronize();
// 	//p(x,y)???FFT
// 	cufftExecC2C(fftplan, (cufftComplex*)buf2.get(), (cufftComplex*)buf2.get(), CUFFT_FORWARD);
// 	mul_scalar<<<grid,block>>>(buf2.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
// 	//????????????
// 	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
// 	//???FFT
// 	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	
// 	gdel_zero<<< grid, block >>>(buf1.get(),u.get(),ny2,nx2);
// 	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
// 	// mul_scalar<<<grid,block>>>(u.get(),(COMPLEX_T) (1.0f / (ny2 * nx2 * d)),ny,nx);
// 	cudaDeviceSynchronize();
// 	//??????
// 	cufftDestroy(fftplan);
// 	buf1.reset();
// 	buf2.reset();
// 	cudaDeviceSynchronize();
// }

template<typename COMPLEX_T,typename PREC_T>
void gFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	ol::gFFT fft(ny2,nx2);

	gzeropadding(u,buf1,ny,nx,ny2,nx2);

	fft.fft(buf1,buf1);
	mul_scalar(buf1,(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	// ol::NearestNeighborInterpolation(buf1,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp.bmp",u,ny,nx,ol::AMP);
	// printf("save\n");

	gFresnelResponse(buf2,ny2,nx2,dy,dx,lambda,d);
	gfftshift(buf2,ny2,nx2);
	fft.fft(buf2,buf2);
	mul_scalar(buf2,(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	// ol::NearestNeighborInterpolation(buf2,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp.bmp",u,ny,nx,ol::PHASE);
	// printf("save\n");

	mult(buf1, buf2, buf1, ny2, nx2);

	fft.ifft(buf1,buf1);
	gdel_zero(buf1,u,ny2,nx2);
	buf1.reset();
	buf2.reset();
}


template<typename COMPLEX_T,typename PREC_T>
__global__ void gFresnelResponseBandLimit(thrust::complex<COMPLEX_T>* u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d,int64_t lim){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
    if ( (w < nx) && (h < ny) ){
		int hnx = nx / 2;
		int hny = ny / 2;
		u[idx] = w;
		if (abs(w - hnx) < lim && abs(h - hny) < lim){
			PREC_T x = (PREC_T) (w - hnx)  * dx;
			PREC_T y = (PREC_T) (h - hny) * dy;
			PREC_T tmp= 1.0 * M_PI* ( x*x + y*y  )  / (lambda * d);
			u[idx] = thrust::complex<COMPLEX_T>(cos(tmp),sin(tmp));
		}
		else{
			u[idx] = 0;
		}
    }
}

template<typename PREC_T,typename COMPLEX_T>
void gFresnelResponseBandLimit(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& h,size_t ny, size_t nx,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z)
{
    PREC_T xlim;
    int wlim;
    xlim = abs(lambda * z * 0.5 / dx);
    // ylim = abs(lambda * z * 0.5 / dy);
    wlim = floor (xlim / dx);
    // hlim = floor (ylim / dy);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gFresnelResponseBandLimit<<<grid,block>>>(h.get(),ny,nx,dy,dx,lambda,z,wlim);
	cudaDeviceSynchronize();
    
}

template<typename PREC_T,typename COMPLEX_T>
void gFresnelResponseBandLimit(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& h,size_t ny, size_t nx,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T p_limit)
{
    PREC_T xlim;
    int wlim;
    xlim = abs(lambda * z * 0.5 / p_limit);
    // ylim = abs(lambda * z * 0.5 / dy);
    wlim = floor (xlim / dx);
    // hlim = floor (ylim / dy);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gFresnelResponseBandLimit<<<grid,block>>>(h.get(),ny,nx,dy,dx,lambda,z,wlim);
	cudaDeviceSynchronize();
    
}


// u is device data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gFresnelPropBandLimit(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int ny2 = ny * 2;
	int nx2 = nx * 2;
	// int mem_size = sizeof(cufftComplex) * ny2 * nx2; 
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
	//?????????a(x,y)?????????????????????
	// gzeropadding<<< grid, block >>>(u.get(),buf1.get(),ny,nx);
	gzeropadding(u,buf1,ny,nx,ny2,nx2);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	//p(x,y)?????????
    // gFresnelResponseFFTShift<<< grid, block >>>(buf2.get(),ny2,nx2,dy,dx,lambda,d);
	gFresnelResponseBandLimit(buf2,ny2,nx2,dy,dx,lambda,d);
	gfftshift(buf2,ny2,nx2);
	cudaDeviceSynchronize();
	//p(x,y)???FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf2.get(), (cufftComplex*)buf2.get(), CUFFT_FORWARD);
	mul_scalar<<<grid,block>>>(buf2.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);
	//????????????
	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
	// gfftshift(buf1,ny2,nx2);
	//???FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	// gfftshift(buf1,ny2,nx2);
	
	gdel_zero<<< grid, block >>>(buf1.get(),u.get(),ny2,nx2);
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	// mul_scalar<<<grid,block>>>(u.get(),(COMPLEX_T) (1.0f / (ny2 * nx2 * d)),ny,nx);
	cudaDeviceSynchronize();
	//??????
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
	cudaDeviceSynchronize();
}


/*????????????????????????H????????????*/
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

/*????????????????????????H????????????*/
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

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);

	cufftComplex* d_buf1, *d_buf2; 
	cudaMalloc( (void**) &d_buf1, mem_size);
	cudaMalloc( (void**) &d_buf2, mem_size);
	// cudaMalloc( (void**) &d_buf3, mem_size);
	ol::zeropadding(u,h_buf,ny,nx);

	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  
	cufftExecC2C(fftplan, d_buf1, d_buf2, CUFFT_FORWARD);

    ol::AsmTransferF(h_buf,ny2,nx2,dv,du,lambda,d);
    ol::fftshift(h_buf,ny2,nx2);

    cudaMemcpy( d_buf1, h_buf.get(), mem_size,cudaMemcpyHostToDevice);  

	KernelMult<<< grid, block>>>(d_buf2, d_buf1, d_buf2, ny2, nx2);

	cudaMemcpy( h_buf.get(), d_buf2, mem_size,cudaMemcpyDeviceToHost);
	// ol::NearestNeighborInterpolation(h_buf,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp.bmp",u,ny,nx,ol::AMP);
	// printf("save\n");


	cufftExecC2C(fftplan, d_buf2, d_buf1, CUFFT_INVERSE);

	cudaMemcpy( h_buf.get(), d_buf1, mem_size,cudaMemcpyDeviceToHost);

	ol::del_zero(h_buf,u,ny2,nx2);
	cufftDestroy(fftplan);
	cudaFree(d_buf1);
	cudaFree(d_buf2);
	// cudaFree(d_buf3);
	h_buf.reset();
}

// u is device data before zero padding
template<typename COMPLEX_T,typename PREC_T>
void gAsmProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d){
	int64_t ny2 = ny * 2;
	int64_t nx2 = nx * 2;
	PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);
	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	ol::gFFT fft(ny2,nx2);
	gzeropadding(u,buf1,ny,nx,ny2,nx2);

	// ol::NearestNeighborInterpolation(buf2,u,ny2,nx2,ny,nx);
	// ol::Save(PROJECT_ROOT "/out/tmp.bmp",u,ny,nx,ol::AMP);
	// printf("save\n");

	fft.fft(buf1,buf1);
	mul_scalar(buf1,(COMPLEX_T)(1.0f / (ny2 * nx2)),ny2,nx2);
	

    gAsmTransferF(buf2,ny2,nx2,dv,du,lambda,d);
	gfftshift(buf2,ny2,nx2);
	
	mult(buf1, buf2, buf1, ny2, nx2);
	
	fft.ifft(buf1,buf1);
	gcut(buf1,u,ny2,nx2,(ny2 - ny)/2,(nx2 - nx)/2,ny,nx);
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
	//?????????a(x,y)?????????????????????
	cudaDeviceSynchronize();
	gzeropadding<<< grid, block >>>(usrc.get(),buf1.get(),ny,nx);
	cudaDeviceSynchronize();
	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	// mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	//H(u,v)?????????
    gshiftedAsmTransferF(buf2,ny2,nx2,dv,du,lambda,d,oy,ox);
	gfftshift(buf2,ny2,nx2);
	cudaDeviceSynchronize();
	//????????????
	KernelMult<<< grid, block>>>(buf1.get(), buf2.get(), buf1.get(), ny2, nx2);
	//???FFT
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_INVERSE);
	cudaDeviceSynchronize();
	gdel_zero<<< grid, block >>>(buf1.get(),udst.get(),ny2,nx2);
	grid = dim3(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	mul_scalar<<<grid,block>>>(udst.get(),(COMPLEX_T)(1.0f / (ny2 * nx2)),ny,nx);
	cudaDeviceSynchronize();
	//??????
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
				gFresnelPropBandLimit(u,ny,nx,dy,dx,lambda,d);
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


template<typename COMPLEX_T,typename PREC_T>
__global__ void gARSSFresnelProp_u(thrust::complex<COMPLEX_T>* src,thrust::complex<COMPLEX_T>* dst,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox,int hlim,int wlim){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	
	if ( (w >= nx) || (h >= ny) ){
		return;
	}
	int hny = ny / 2; int hnx = nx / 2;
	if (abs(w - hnx) >= wlim || abs(h - hny) >= hlim){
		dst[idx] = 0;
	}
	else{
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y1 = (h - hny) * dy;
		PREC_T phiu_y = (s * s - s) * y1 * y1 - 2 * s * oy * y1;
		PREC_T x1 = (w - hnx) * dx;
		PREC_T phiu = (s * s - s) * x1 * x1 - 2 * s * ox * x1;
		phiu += phiu_y;
		phiu *= tmp;
		PREC_T real = cos(phiu);
		PREC_T imag = sin(phiu);
		dst[idx] = src[idx] * thrust::complex<COMPLEX_T>(real,imag);
	}
	
}

template<typename PREC_T,typename COMPLEX_T>
void gARSSFresnelProp_u(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& src,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& dst,size_t ny, size_t nx,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox)
{
    PREC_T xlim,ylim;
    int wlim,hlim;
    xlim = lambda * abs(z) * 0.5 / (s * abs(s - 1) * dx);
    wlim = floor (xlim / dx);
	ylim = lambda * abs(z) * 0.5 / (s * abs(s - 1) * dy);
    hlim = floor (ylim / dy);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gARSSFresnelProp_u<<<grid,block>>>(src.get(),dst.get(),ny,nx,dy,dx,lambda,z,s,oy,ox,hlim,wlim);
	cudaDeviceSynchronize();
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gARSSFresnelProp_h(thrust::complex<COMPLEX_T>* dst,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,int64_t lim){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	if ( (w >= nx) || (h >= ny) ){
		return;
	}
	int hny = ny / 2; int hnx = nx / 2;
	
	if (abs(w - hnx) >= lim || abs(h - hny) >= lim){
		dst[idx] = 0;
	}
	else{
		PREC_T tmp = M_PI / (lambda * z);
		PREC_T y1 = (h - hny) * dy;
		PREC_T x1 = (w - hnx) * dx;
		PREC_T phih_y = s * y1 * y1;
		PREC_T phih = s * x1 * x1 + phih_y;
		phih *= tmp;
		dst[idx] = thrust::complex<COMPLEX_T>(cos(phih),sin(phih));
	}
}

template<typename PREC_T,typename COMPLEX_T>
void gARSSFresnelProp_h(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& h,size_t ny, size_t nx,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s)
{
    PREC_T xlim;
    int wlim;
    xlim = abs(lambda * z * 0.5 / (dx * s));
    wlim = floor (xlim / dx);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gARSSFresnelProp_h<<<grid,block>>>(h.get(),ny,nx,dy,dx,lambda,z,s,wlim);
	cudaDeviceSynchronize();
}

template<typename COMPLEX_T,typename PREC_T>
__global__ void gARSSFresnelProp_Cz(thrust::complex<COMPLEX_T>* u,int ny, int nx, 
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox,int hlim, int wlim){
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = h * nx + w;
	if ( (w >= nx) || (h >= ny) ){
		return;
	}
	int hny = ny / 2; int hnx = nx / 2;
	if (abs(w - hnx) < wlim && abs(h - hny) < hlim){
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

template<typename PREC_T,typename COMPLEX_T>
void gARSSFresnelProp_Cz(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& h,size_t ny, size_t nx,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T s,PREC_T oy, PREC_T ox)
{
    PREC_T xlim,ylim;
    int wlim,hlim;
    xlim = lambda * abs(z) * 0.5 / (abs(1 - s) * dx);
    wlim = floor (xlim / dx);
	ylim = lambda * abs(z) * 0.5 / (abs(1 - s) * dy);
    hlim = floor (ylim / dy);
	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
	gARSSFresnelProp_Cz<<<grid,block>>>(h.get(),ny,nx,dy,dx,lambda,z,s,oy,ox,hlim,wlim);
	cudaDeviceSynchronize();
}


template<typename PREC_T=float,typename COMPLEX_T>
void ARSSFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& usrc,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& udst,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z, PREC_T s,PREC_T oy,PREC_T ox){
	int ny2 = ny * 2;
	int nx2 = nx * 2;

	auto buf1 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);
	auto buf2 = cuda::make_unique<thrust::complex<COMPLEX_T>[]>(ny2 * nx2);

	dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);


	gARSSFresnelProp_u(usrc,udst,ny,nx,dy,dx,lambda,z,s,oy,ox);
	cudaDeviceSynchronize();
	
	grid = dim3(ceil((float)nx2 / block.x), ceil((float)ny2 / block.y), 1);
    gzeropadding(udst,buf1,ny,nx,ny2,nx2);
	cudaDeviceSynchronize();

	cufftHandle fftplan;
	cufftPlan2d(&fftplan, ny2, nx2, CUFFT_C2C);
	cufftExecC2C(fftplan, (cufftComplex*)buf1.get(), (cufftComplex*)buf1.get(), CUFFT_FORWARD);
	cufftDestroy(fftplan);
	mul_scalar<<<grid,block>>>(buf1.get(),(COMPLEX_T) (1.0f / (ny2 * nx2)),ny2,nx2);

	gARSSFresnelProp_h(buf2,ny2,nx2,dy,dx,lambda,z,s);
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
	gARSSFresnelProp_Cz(udst,ny,nx,dy,dx,lambda,z,s,oy,ox);
	cufftDestroy(fftplan);
	buf1.reset();
	buf2.reset();
	cudaDeviceSynchronize();
}

template<typename PREC_T=float,typename COMPLEX_T>
void ARSSFresnelProp(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& usrc,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& udst,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z, PREC_T s){
	ARSSFresnelProp(usrc,udst,ny,nx,dy,dx,lambda,z,s,0.0f,0.0f);
}
}

#endif