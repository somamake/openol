#include "golclass.h"
#include "thrust/complex.h"
#include "oldefine.h"
namespace ol{
template<typename _Tp,typename PREC_T, typename CGH_T>
__global__ void gCgh(_Tp* points,CGH_T* phase,int N,
                 int ny, int nx,PREC_T p, PREC_T lambda){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;
    thrust::complex<CGH_T> U = 0;

    if ( (w < nx) && (h < ny) ){
        for (int j = 0;j < N;j++){
            CGH_T dx = p * w - points[j].x;
            CGH_T dy = p * h - points[j].y;
            CGH_T dz = points[j].z;
            CGH_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
            thrust::complex<CGH_T> Utmp = thrust::complex<CGH_T>(cos(theta),sin(theta));
            U += Utmp;          
        }
        phase[idx] = atan2(U.imag(),U.real());
    }
}

template<typename _Tp,typename PREC_T, typename CGH_T>
__global__ void gCgh(_Tp* points,CGH_T* phase,int N,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;
    thrust::complex<CGH_T> U = 0;

    if ( (w < nx) && (h < ny) ){
        for (int j = 0;j < N;j++){
            CGH_T dx = px * w - points[j].x;
            CGH_T dy = py * h - points[j].y;
            CGH_T dz = points[j].z;
            CGH_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
            thrust::complex<CGH_T> Utmp = thrust::complex<CGH_T>(cos(theta),sin(theta));
            U += Utmp;          
        }
        phase[idx] = atan2(U.imag(),U.real());
    }
}

template<typename _Tp,typename PREC_T, typename COMPLEX_T>
__global__ void gCghShared(_Tp* points,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;
    const int blocksize = 256;
    __shared__ _Tp pointsbuf[blocksize];
    if ( (w < nx) && (h < ny) ){
        thrust::complex<COMPLEX_T> Utmp = 0;
        for (int block_j = 0;block_j < N;block_j += blocksize){
            if(threadIdx.x == 0 && threadIdx.y == 0){
                for (int local_j = 0;local_j < blocksize;local_j++){
                    int j = block_j + local_j;
                    if ( j < N){
                        pointsbuf[local_j].x = points[j].x;
                        pointsbuf[local_j].y = points[j].y;
                        pointsbuf[local_j].z = points[j].z;
                    }
                    else{
                        break;
                    }
                }
            }
            __syncthreads();
            for (int local_j = 0;local_j < blocksize;local_j++){
                int j = block_j + local_j;
                if (j < N){
                    PREC_T dx = px * w - pointsbuf[local_j].x;
                    PREC_T dy = py * h - pointsbuf[local_j].y;
                    PREC_T dz = pointsbuf[local_j].z;
                    PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
                    Utmp += thrust::complex<COMPLEX_T>(cos(theta),sin(theta));;
                }
                else{
                    break;
                }
            }
        }
        U[idx] = Utmp;
    }
}
template<typename _Tp,typename PREC_T, typename COMPLEX_T>
__global__ void _gCgh(_Tp* points,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;
    if ( (w < nx) && (h < ny) ){
        thrust::complex<COMPLEX_T> Utmp = 0;
        for (int j = 0;j < N;j++){
            PREC_T dx = px * w - points[j].x;
            PREC_T dy = py * h - points[j].y;
            PREC_T dz = points[j].z;
            PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
            Utmp += thrust::complex<COMPLEX_T>(cos(theta),sin(theta));;
        }
        U[idx] = Utmp;
    }
}

template<typename _Tp,typename PREC_T, typename COMPLEX_T>
__global__ void gCgh(_Tp* points,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;
    if ( (w < nx) && (h < ny) ){
        thrust::complex<COMPLEX_T> Utmp = 0;
        for (int j = 0;j < N;j++){
            PREC_T dx = px * w - points[j].x;
            PREC_T dy = py * h - points[j].y;
            PREC_T dz = points[j].z;
            PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
            Utmp += thrust::complex<COMPLEX_T>(cos(theta),sin(theta));
        }
        U[idx] = Utmp;
    }
}

// _Tp is thrust::complex or float or double
template<typename PREC_T, typename _Tp>
void gCgh(gObject<PREC_T>& gobj,cuda::unique_ptr<_Tp[]>& U,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
    gCgh<<<grid,block>>>(gobj.points.get(),U.get(),gobj.size,ny,nx,py,px,lambda);
    cudaDeviceSynchronize();
}

// _Tp is thrust::complex or float or double
template<typename PREC_T, typename _Tp>
void gCgh(gObject<PREC_T>& gobj,cuda::unique_ptr<_Tp[]>& U,
                 int ny, int nx,PREC_T p, PREC_T lambda){
    dim3 block(16, 16, 1);
	dim3 grid(ceil((float)nx / block.x), ceil((float)ny / block.y), 1);
    gCghShared<<<grid,block>>>(gobj.points.get(),U.get(),gobj.size,ny,nx,p,p,lambda);
    cudaDeviceSynchronize();
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
__global__ void gCghReduction(_Tp* points,thrust::complex<COMPLEX_T>* U,int N,int ny,int nx,
                 PREC_T p, PREC_T lambda){
    int j = threadIdx.x;
    int w = blockIdx.x;
    int h = blockIdx.y;
    extern __shared__ thrust::complex<COMPLEX_T> Ubuf[];
    if (j >= N){
        return;
    }
    PREC_T dx = p * w - points[j].x;
    PREC_T dy = p * h - points[j].y;
    PREC_T dz = points[j].z;
    PREC_T Wj = dz * lambda * 0.5f / p;
    PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
    Ubuf[j] = thrust::complex<PREC_T>(cos(theta),sin(theta));
    __syncthreads();
    // 共有メモリを使い、半々にリダクション
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if (j < s ) {
            if ( j + s < N){
                Ubuf[j] += Ubuf[j + s];
            }
        }
        __syncthreads();
    }
    // 残り１要素になったら結果をグローバルメモリへ書き出して終了
    if (j == 0) U[w + h * nx] += Ubuf[0]; 
    __syncthreads();
}
template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void gCghReduction(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda){
    int threadnum = 256;
    dim3 block(threadnum,1);
    dim3 grid(nx,ny);
    // mul_scalar(U,0.0f,ny,nx);
    for(int i = 0;i < obj.size;i += threadnum){
        int N;
        if (i + threadnum < obj.size){
            N = threadnum;
        }
        else{
            N = obj.size - i;
            // break;
        }
        gCghReduction<<<grid,block,sizeof(thrust::complex<COMPLEX_T>) * threadnum>>>(obj.points.get() + i,U.get(),N,ny,nx,p,lambda);
        cudaDeviceSynchronize();
    }
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
__global__ void _WRPStep1(_Tp* points,PREC_T W ,PREC_T refd,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T p, PREC_T lambda,PREC_T plane_z){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;

    
    
    if ( (w >= nx) || (h >= ny) ){
        return;
    }
    U[idx] = 0;
    // for (int j = 0;j < N;j++){
    //     PREC_T dx = p * w - points[j].x;
    //     PREC_T dy = p * h - points[j].y;
    //     PREC_T dz = points[j].z - plane_z;
    //     PREC_T tmp = dx * dx + dy * dy;
    //     PREC_T Wj = W * abs(dz) / refd;
    //     if (tmp < Wj * Wj){
    //         PREC_T theta = 2 * F32_PI * (tmp)/(2 * dz * lambda);
    //         // PREC_T theta = 2 * M_PI * sqrt(dx * dx + dy * dy + dz * dz)/(lambda);
    //         thrust::complex<PREC_T> Utmp = thrust::complex<PREC_T>(cos(theta),sin(theta))/abs(dz);
    //         U[idx] += Utmp;
    //     }
    // }
    const int blocksize = 64;
    __shared__ _Tp pointsbuf[blocksize];
    thrust::complex<COMPLEX_T> Utmp = 0;
    for (int block_j = 0;block_j < N;block_j += blocksize){
        if(threadIdx.x == 0 && threadIdx.y == 0){
            for (int local_j = 0;local_j < blocksize;local_j++){
                int j = block_j + local_j;
                if ( j < N){
                    pointsbuf[local_j].x = points[j].x;
                    pointsbuf[local_j].y = points[j].y;
                    pointsbuf[local_j].z = points[j].z;
                }
                else{
                    break;
                }
            }
        }
        __syncthreads();
        for (int local_j = 0;local_j < blocksize;local_j++){
            int j = block_j + local_j;
            if (j < N){
                PREC_T dx = p * w - pointsbuf[local_j].x;
                PREC_T dy = p * h - pointsbuf[local_j].y;
                PREC_T dz = pointsbuf[local_j].z - plane_z;
                PREC_T tmp = dx * dx + dy * dy;
                PREC_T Wj = W * abs(dz) / refd;
                if ( tmp < Wj * Wj ){
                    PREC_T theta = 2 * F32_PI * tmp/(2 * dz * lambda);
                    Utmp += thrust::complex<PREC_T>(cos(theta),sin(theta))/dz;
                }
            }
            else{
                break;
            }
        }
    }
    U[idx] = Utmp;
}

template<typename _Tp,typename PREC_T>
__global__ void calcW(PREC_T* W,PREC_T* refd,_Tp* points,PREC_T p, PREC_T lambda,PREC_T plane_z){
    *refd = abs(points[0].z - plane_z);
    *W = 1 / (sqrt(4 * std::pow((p / lambda),2) - 1 )) * (*refd);
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void _WRPStep1(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T z){
    dim3 block(16,16,1);
    dim3 grid(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    // PREC_T refd = abs(obj[obj.size-1].z - z);
    
    // PREC_T W = calcW(obj[0].z,p,lambda,z);
    
    PREC_T W;
    PREC_T refd;
    // dim3 block1(1,1,1);
    // dim3 grid1(1,1,1);
    // calcW<<<grid1,block1>>>(&W,&refd,obj.points.get(),p,lambda,z);
    // printf("W = %f, refd = %f\n",W,refd);
    W = 5e-3;
    refd = 1;
    
    _WRPStep1<<<grid,block>>>(obj.points.get(),W,refd,U.get(),obj.size,ny,nx,p,lambda,z);
    cudaDeviceSynchronize();
}



template<typename _Tp,typename PREC_T>
__global__ void WRPStep1Pre(_Tp* points,PREC_T* W,int N,
                 PREC_T p, PREC_T lambda,PREC_T z){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( j >= N){
        return;
    }
    W[j] = 1 / (sqrt(4 * std::pow((p / lambda),2) - 1 )) * abs(points[j].z - z);
}



template<typename _Tp,typename COMPLEX_T,typename PREC_T>
__global__ void WRPStep1(_Tp* points,PREC_T* W ,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T p, PREC_T lambda,PREC_T plane_z){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;

    
    
    if ( (w >= nx) || (h >= ny) ){
        return;
    }
    U[idx] = 0;
    for (int j = 0;j < N;j++){
        PREC_T dx = p * w - points[j].x;
        PREC_T dy = p * h - points[j].y;
        PREC_T dz = points[j].z - plane_z;
        PREC_T tmp = dx * dx + dy * dy;
        if (tmp < W[j] * W[j]){
            PREC_T theta = 2 * F32_PI * (tmp)/(2 * dz * lambda);
            // PREC_T theta = 2 * M_PI * sqrt(dx * dx + dy * dy + dz * dz)/(lambda);
            thrust::complex<PREC_T> Utmp = thrust::complex<PREC_T>(cos(theta),sin(theta))/abs(dz);
            // thrust::complex<PREC_T> Utmp = thrust::complex<PREC_T>(cos(theta),sin(theta));
            U[idx] += Utmp;
        }
    }
    // const int blocksize = 64;
    // __shared__ _Tp pointsbuf[blocksize];
    // __shared__ PREC_T Wbuf[blocksize];
    // thrust::complex<COMPLEX_T> Utmp = 0;
    // for (int block_j = 0;block_j < N;block_j += blocksize){
    //     if(threadIdx.x == 0 && threadIdx.y == 0){
    //         for (int local_j = 0;local_j < blocksize;local_j++){
    //             int j = block_j + local_j;
    //             if ( j < N){
    //                 pointsbuf[local_j].x = points[j].x;
    //                 pointsbuf[local_j].y = points[j].y;
    //                 pointsbuf[local_j].z = points[j].z;
    //                 Wbuf[local_j] = W[j];
    //             }
    //             else{
    //                 break;
    //             }
    //         }
    //     }
    //     __syncthreads();
    //     for (int local_j = 0;local_j < blocksize;local_j++){
    //         int j = block_j + local_j;
    //         if (j < N){
    //             PREC_T dx = p * w - pointsbuf[local_j].x;
    //             PREC_T dy = p * h - pointsbuf[local_j].y;
    //             PREC_T dz = pointsbuf[local_j].z - plane_z;
    //             PREC_T tmp = dx * dx + dy * dy;
    //             if ( tmp < Wbuf[local_j] ){
    //                 PREC_T theta = 2 * F32_PI * tmp/(2 * dz * lambda);
    //                 Utmp += thrust::complex<PREC_T>(cos(theta),sin(theta))/dz;
    //             }
    //         }
    //         else{
    //             break;
    //         }
    //     }
    // }
    // U[idx] = Utmp;
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPStep1(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d){
    dim3 block(256,1,1);
    dim3 grid(ceil((float) obj.size / block.x),1, 1);
    auto W = cuda::make_unique<PREC_T[]>(obj.size);
    WRPStep1Pre<<<grid,block>>>(obj.points.get(),W.get(),obj.size,p,lambda,d);
    block = dim3(16,16,1); grid = dim3(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    WRPStep1<<<grid,block>>>(obj.points.get(),W.get(),U.get(),obj.size,ny,nx,p,lambda,d);
    cudaDeviceSynchronize();
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPStep1(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d,PREC_T plimit){
    dim3 block(256,1,1);
    dim3 grid(ceil((float) obj.size / block.x),1, 1);
    auto W = cuda::make_unique<PREC_T[]>(obj.size);
    WRPStep1Pre<<<grid,block>>>(obj.points.get(),W.get(),obj.size,plimit,lambda,d);
    block = dim3(16,16,1); grid = dim3(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    WRPStep1<<<grid,block>>>(obj.points.get(),W.get(),U.get(),obj.size,ny,nx,p,lambda,d);
    cudaDeviceSynchronize();
}

// D is square
template<typename _Tp,typename PREC_T>
__global__ void WRPStep1Pre_D(_Tp* points,PREC_T* D,int N,
                 PREC_T p, PREC_T lambda,PREC_T z){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( j >= N){
        return;
    }
    PREC_T W = 1 / (sqrt(4 * std::pow((p / lambda),2) - 1 )) * abs(points[j].z - z);
    D[j] = W * sqrt(2.0f);
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
__global__ void WRPStep1_D(_Tp* points,PREC_T* D,thrust::complex<COMPLEX_T>* U,int N,
                 int ny, int nx,PREC_T p, PREC_T lambda,PREC_T plane_z){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = h * nx + w;

    
    if ( (w >= nx) || (h >= ny) ){
        return;
    }
    
    thrust::complex<PREC_T> Utmp = 0;
    for (int j = 0;j < N;j++){
        PREC_T dx = p * w - points[j].x;
        PREC_T dy = p * h - points[j].y;
        PREC_T dz = points[j].z - plane_z;
        if ( (abs(dx) * 2 <= D[j]) && (abs(dy) * 2 <= D[j]) ){
            PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
            // PREC_T theta = 2 * M_PI * sqrt(dx * dx + dy * dy + dz * dz)/(lambda);
            Utmp += thrust::complex<PREC_T>(cos(theta),sin(theta))/abs(dz);
            // Utmp += thrust::complex<PREC_T>(cos(theta),sin(theta));
        }
    }
    U[idx] = Utmp;
    
    
    // const int blocksize = 64;
    // __shared__ _Tp pointsbuf[blocksize];
    // __shared__ PREC_T Dbuf[blocksize];
    // thrust::complex<COMPLEX_T> Utmp = 0;
    // for (int block_j = 0;block_j < N;block_j += blocksize){
    //     if(threadIdx.x == 0 && threadIdx.y == 0){
    //         for (int local_j = 0;local_j < blocksize;local_j++){
    //             int j = block_j + local_j;
    //             if ( j < N){
    //                 pointsbuf[local_j].x = points[j].x;
    //                 pointsbuf[local_j].y = points[j].y;
    //                 pointsbuf[local_j].z = points[j].z;
    //                 Dbuf[local_j] = D[j];
    //             }
    //             else{
    //                 break;
    //             }
    //         }
    //     }
    //     __syncthreads();
    //     for (int local_j = 0;local_j < blocksize;local_j++){
    //         int j = block_j + local_j;
    //         if (j < N){
    //             PREC_T dx = p * w - pointsbuf[local_j].x;
    //             PREC_T dy = p * h - pointsbuf[local_j].y;
    //             PREC_T dz = pointsbuf[local_j].z - plane_z;
    //             PREC_T Dtmp = Dbuf[local_j];
    //             if ( (abs(dx) * 2 <= Dtmp) && (abs(dy) * 2 <= Dtmp) ){
    //                 PREC_T theta = 2 * F32_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
    //                 Utmp += thrust::complex<PREC_T>(cos(theta),sin(theta))/dz;
    //             }
    //         }
    //         else{
    //             break;
    //         }
    //     }
    // }
    // U[idx] = Utmp;
}


template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPStep1_D(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d){
    dim3 block(256,1,1);
    dim3 grid(ceil((float) obj.size / block.x),1, 1);
    auto D = cuda::make_unique<PREC_T[]>(obj.size);
    WRPStep1Pre_D<<<grid,block>>>(obj.points.get(),D.get(),obj.size,p,lambda,d);
    block = dim3(16,16,1); grid = dim3(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    WRPStep1_D<<<grid,block>>>(obj.points.get(),D.get(),U.get(),obj.size,ny,nx,p,lambda,d);
    cudaDeviceSynchronize();
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPStep1_D(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d,PREC_T plimit){
    dim3 block(256,1,1);
    dim3 grid(ceil((float) obj.size / block.x),1, 1);
    auto D = cuda::make_unique<PREC_T[]>(obj.size);
    WRPStep1Pre_D<<<grid,block>>>(obj.points.get(),D.get(),obj.size,plimit,lambda,d);
    block = dim3(16,16,1); grid = dim3(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    WRPStep1_D<<<grid,block>>>(obj.points.get(),D.get(),U.get(),obj.size,ny,nx,p,lambda,d);
    cudaDeviceSynchronize();
}


template<typename COMPLEX_T,typename PREC_T>
void WRPStep2(cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d,PROPMODE propmode=PROPMODE::AUTO){
    if (propmode == PROPMODE::FRESNEL){
        gFresnelProp(U,ny,nx,p,p,lambda,d);
    }
    else if (propmode == PROPMODE::ASM){
        gAsmProp(U,ny,nx,p,p,lambda,d);
    }
    else if (propmode == PROPMODE::AUTO){
        if (AsmPropCheck(ny,nx,p,p,lambda,d) == true){
            printf("WRP Step2 is AsmProp\n");
            gAsmProp(U,ny,nx,p,p,lambda,d);
        }
        else{
            if (FresnelPropCheck(ny,nx,p,p,lambda,d) == false){
                printf("warning!!\neither AsmProp or FresnelProp does not meet the condition.\n");
                gAsmProp(U,ny,nx,p,p,lambda,d);
                return;
            }
            printf("WRP Step2 is FresnelProp\n");
            gFresnelProp(U,ny,nx,p,p,lambda,d);
        }
    }
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPMethod_D(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T zmin,float s=0.9,PROPMODE propmode=PROPMODE::AUTO){
    // cudaMemset(U.get(),0,sizeof(thrust::complex<COMPLEX_T>) * ny *nx);
    dim3 block(16,16,1);
    dim3 grid(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    float plane_z;
    if (s < 0){
        float D = nx * p * 0.05;
        float d =  (sqrt(4 * std::pow((p / lambda),2) - 1 )) / sqrt(2) * D;
        // PREC_T W = 1 / (sqrt(4 * std::pow((p / lambda),2) - 1 )) * (points[j].z - z)
        plane_z =  zmin - d;
    }
    else{
        plane_z = zmin * s;
    }
    printf("plane z is %f, ratio is %f\n",plane_z, plane_z / zmin);
    WRPStep1_D(obj,U,ny,nx,p,lambda,plane_z);
    // Save(PROJECT_ROOT "/out/tmp.bmp",U,ny,nx);
    WRPStep2(U,ny,nx,p,lambda,plane_z,propmode);
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPMethod(gObject<_Tp>& obj,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& U,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T zmin,float s=0.9,PROPMODE propmode=PROPMODE::AUTO){
    // cudaMemset(U.get(),0,sizeof(thrust::complex<COMPLEX_T>) * ny *nx);
    dim3 block(16,16,1);
    dim3 grid(ceil((float) nx / block.x),ceil((float) ny / block.y), 1);
    float plane_z;
    if (s < 0){
        float Wpow2 = nx * ny * p * p / M_PI * 1/1024.f;
        float d =  sqrt( (4 * std::pow((p / lambda),2) - 1 ) * Wpow2 );
        // PREC_T W = 1 / (sqrt(4 * std::pow((p / lambda),2) - 1 )) * (points[j].z - z)
        plane_z =  zmin - d;
    }
    else{
        plane_z = zmin * s;
    }
    printf("plane z is %f, ratio is %f\n",plane_z, plane_z / zmin);
    // mul_scalar<<<block,grid>>>(U.get(),0.0f,ny,nx);
    WRPStep1(obj,U,ny,nx,p,lambda,plane_z);
    // Save(PROJECT_ROOT "/out/tmp.bmp",U,ny,nx);
    WRPStep2(U,ny,nx,p,lambda,plane_z,propmode);
}

}