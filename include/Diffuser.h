#ifndef DIFFUSER_H
#define DIFFUSER_H
#ifdef __NVCC__
#include "thrust/complex.h"
#include "golclass.h"
#include "curand.h"
#endif
#include "memory"
#include <random>
#include "oldefine.h"

namespace ol{
#ifdef __NVCC__
template<typename _Tp = float>
__global__ void KernelMultPhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
        float phase = (b0to1[adr] - 0.5f) * scalar;
        // float phase = 0;
		c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
	}	
}

template<typename _Tp = float>
__global__ void KernelMultLinePhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
        float phase = (b0to1[x] - 0.5f) * scalar;
        // float phase = 0;
		c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
	}	
}

template<typename _Tp = float>
__global__ void KernelMultLineYPhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
        float phase = (b0to1[y] - 0.5f) * scalar;
        // float phase = 0;
		c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
	}	
}

template<typename _Tp = float>
__global__ void KernelbitRandomPhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*width);
	if ( x < width && y < height){
        // b0to1[adr] = round(b0to1[adr] * 1) / 1.0f;
        float phase = (b0to1[adr] > 0.5f) ? F32_PI : 0;
        // float phase = (b0to1[adr] - 0.5f) * scalar;
		c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
	}	
}

template<typename _Tp = float>
__global__ void KernelBlockPhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width,int blocksize)
{
	//スレッド・ブロック番号を元にアドレス計算
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int rootadr = (w+h*width/blocksize);
    w = w * blocksize;
    h = h * blocksize;
	if ( w < width && h < height){
        for (int local_h = 0;local_h < blocksize;local_h++){
            for (int local_w = 0;local_w < blocksize;local_w++){
                unsigned int adr = (w + local_w + (h + local_h) *width);
                float phase = (b0to1[rootadr] - 0.5f) * scalar;
		        c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
            }
        }
        
	}	
}

template<typename _Tp = float>
__global__ void KernelBlockStepPhase( thrust::complex<_Tp>* a, _Tp*b0to1,float scalar ,thrust::complex<_Tp>*c, int height, int width,int blocksize)
{
	//スレッド・ブロック番号を元にアドレス計算
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int rootadr = (w+h*width/blocksize);
    w = w * blocksize;
    h = h * blocksize;
	if ( w < width && h < height){
        for (int local_h = 0;local_h < blocksize;local_h++){
            for (int local_w = 0;local_w < blocksize;local_w++){
                unsigned int adr = (w + local_w + (h + local_h) *width);
                float step = ((float)local_w + (float)local_h) / (float)(blocksize);
                float phase = (b0to1[rootadr] + step - 0.5f) * scalar;
		        c[adr] = a[adr] * thrust::complex<_Tp>(cos(phase),sin(phase));
            }
        }
        
	}	
}

template<typename _Tp = float>
__global__ void Kernelbitphase( thrust::complex<_Tp>* in, int height, int width)
{
	//スレッド・ブロック番号を元にアドレス計算
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if ( w >= width && h >= height){
        return;
    }
    if ((w % 2) == (h % 2)){
        in[w + h * width] = thrust::complex<_Tp>(cos(F32_PI),sin(F32_PI)) * in[w + h * width];
    }
    else{
        // in[w + h * width] = thrust::complex<_Tp>(cos(F32_PI),sin(F32_PI)) * in[w + h * width];
    }
}

template<typename _Tp = float>
__global__ void shift( _Tp* in,  _Tp* out,int height, int width,int oy,int ox)
{
	//スレッド・ブロック番号を元にアドレス計算
	int w = blockIdx.x*blockDim.x + threadIdx.x;
    int h = blockIdx.y*blockDim.y + threadIdx.y;
	if ( w >= width&& h >= height){
        return;
    }
    int _h = (h + oy) % height;
    int _w = (w + ox) % width;
    _h = (_h >= 0) ? _h : _h + height;
    _w = (_w >= 0) ? _w : _w + width;
    int _idx = _h * width + _w;
    int idx = w + h * width;
    // _Tp tmp = in[idx];
    // in[idx] = in[_idx];
    out[_idx] = in[idx];
}

template<typename _Tp = float>
void shift( cuda::unique_ptr<_Tp[]>& in, int height, int width,int oy,int ox){
    auto tmp = cuda::make_unique<_Tp[]>(height * width);
    dim3 block(16,16,1);
    dim3 grid(ceil((float) (width / 2) / block.x), ceil((float) (height / 2) / block.y),1);
    shift<<<grid,block>>>(in.get(),tmp.get(),height,width,oy,ox);
    cudaDeviceSynchronize();
    in = std::move(tmp);
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)

#endif

// enum HOLOGRAMSTEP{
//     RECORD,PLAY
// };
template<typename _Tp=float>
class Diffuser{
    public:
        int pitch;
        int width;
        int height;
        std::unique_ptr<std::complex<_Tp>[]> u;
        Diffuser(int height, int width, int pitch){
            this->pitch = pitch;
            this->height = height;
            this->width = width;
        }
        Diffuser(int height, int width){
            this->height = height;
            this->width = width;
        }
        ~Diffuser(){
        }

        template<typename PREC_T>
        void gen_complex(){

        }

        // メモリ消費少ないバージョン
        template<typename PREC_T>
        void random_phase(std::unique_ptr<std::complex<PREC_T>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI,int oy = 0,int ox = 0){
            // auto seedgen = [](){ return 1;};
            // std::mt19937 engine(seedgen());
            // std::uniform_real_distribution<float> dist(-1,1);
            std::mt19937 e2(seed);
            for ( int h = 0; h < height;h++){
                for (int w = 0; w < width;w++){
                    int _h = (h + oy) % height;
                    int _w = (w + ox) % width;
                    _h = (_h >= 0) ? _h : _h + height;
                    _w = (_w >= 0) ? _w : _w + width;
                    int idx = _h * width + _w;
                    float phase;
                    int a = e2() >> 5;
                    int b = e2() >> 6;
                    double value = (a * 67108864.0 + b) / 9007199254740992.0;
                    if (step == PLAY){
                        phase = range * (value -0.5f);
                    }
                    else if (step == RECORD){
                        phase = -range * (value - 0.5f);
                    }
                    u[idx] *= std::complex<PREC_T>(cos(phase),sin(phase));
                }
            }
        }
        #ifdef __NVCC__
        template<typename PREC_T>
        void random_phase(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI,int oy = 0, int ox = 0){
            curandGenerator_t gen;
            cuda::unique_ptr<PREC_T[]> devData = cuda::make_unique<float[]>(height * width);

            CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MT19937));
            curandSetPseudoRandomGeneratorSeed(gen,seed);
            curandGenerateUniform(gen,devData.get(),height * width);
            shift(devData,this->height,this->width,oy,ox);
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y),1);
            if (step == PLAY){
                KernelMultPhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width);
                // KernelMyPhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width);
            }
            else if (step == RECORD){
                KernelMultPhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width);
                // KernelMyPhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width);
            }
            curandDestroyGenerator(gen);
            devData.reset();
        }

        template<typename PREC_T>
        void random_xphase(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI){
            curandGenerator_t gen;
            cuda::unique_ptr<PREC_T[]> devData = cuda::make_unique<float[]>(width);

            CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MT19937));
            curandSetPseudoRandomGeneratorSeed(gen,seed);
            curandGenerateUniform(gen,devData.get(),width);
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y),1);
            if (step == PLAY){
                KernelMultLinePhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width);
                // KernelMyPhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width);
            }
            else if (step == RECORD){
                KernelMultLinePhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width);
                // KernelMyPhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width);
            }
            curandDestroyGenerator(gen);
            devData.reset();
        }

        template<typename PREC_T>
        void random_blockphase(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI,int blocksize = 1){
            curandGenerator_t gen;
            cuda::unique_ptr<PREC_T[]> devData = cuda::make_unique<float[]>(width * height /(blocksize*blocksize));

            CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MT19937));
            curandSetPseudoRandomGeneratorSeed(gen,seed);
            curandGenerateUniform(gen,devData.get(),width * height /(blocksize*blocksize));
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width /blocksize / block.x), ceil((float)height /blocksize / block.y),1);
            if (step == PLAY){
                KernelBlockPhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width,blocksize);
            }
            else if (step == RECORD){
                KernelBlockPhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width,blocksize);
            }
            curandDestroyGenerator(gen);
            devData.reset();
        }

        template<typename PREC_T>
        void random_blockstepphase(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI,int blocksize = 4){
            curandGenerator_t gen;
            cuda::unique_ptr<PREC_T[]> devData = cuda::make_unique<float[]>(width * height /(blocksize*blocksize));

            CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MT19937));
            curandSetPseudoRandomGeneratorSeed(gen,seed);
            curandGenerateUniform(gen,devData.get(),width * height /(blocksize*blocksize));
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width /blocksize / block.x), ceil((float)height /blocksize / block.y),1);
            if (step == PLAY){
                KernelBlockStepPhase<<<grid,block>>>(u.get(),devData.get(),range,u.get(),this->height,this->width,blocksize);
            }
            else if (step == RECORD){
                KernelBlockStepPhase<<<grid,block>>>(u.get(),devData.get(),-range,u.get(),this->height,this->width,blocksize);
            }
            curandDestroyGenerator(gen);
            devData.reset();
        }

        

        template<typename PREC_T>
        void bitphase(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u){
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y),1);
            Kernelbitphase<<<grid,block>>>(u.get(),this->height,this->width);
        }
        template<typename PREC_T>
        void phasefunc(cuda::unique_ptr<thrust::complex<PREC_T>[]>& u){
            dim3 block(16,16,1);
            dim3 grid(ceil((float)width / block.x), ceil((float)height / block.y),1);
            Kernelbitphase<<<grid,block>>>(u.get(),this->height,this->width);
        }
        #endif
};
}

#endif
