#ifndef GOLCLASS_H
#define GOLCLASS_H
#ifdef __NVCC__

#include <memory>
#include <type_traits>
#include <cuda_runtime_api.h>
#include "olclass.h"
#define CHECK_CUDA_ERROR(e) (cuda::check_error(e, __FILE__, __LINE__))
namespace cuda
{
    template<typename F, typename N>
	void check_error(const ::cudaError_t e, F&& f, N&& n)
	{
		if(e != ::cudaSuccess)
		{
			std::stringstream s;
			s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
			throw std::runtime_error{s.str()};
		}
	}
    
	struct deleter
	{
		void operator()(void* p) const
		{
			CHECK_CUDA_ERROR(::cudaFree(p));
			// ::cudaFree(p);
		}
	};
	template<typename T>
	using unique_ptr = std::unique_ptr<T, deleter>;

	// auto array = cuda::make_unique<float[]>(n);
	// ::cudaMemcpy(array.get(), src_array, sizeof(float)*n, ::cudaMemcpyHostToDevice);
	template<typename T>
	typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(const std::size_t n)
	{
		using U = typename std::remove_extent<T>::type;
		U* p;
		CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
		return cuda::unique_ptr<T>{p};
	}

	// auto value = cuda::make_unique<my_class>();
	// ::cudaMemcpy(value.get(), src_value, sizeof(my_class), ::cudaMemcpyHostToDevice);
	template<typename T>
	cuda::unique_ptr<T> make_unique()
	{
		T* p;
		CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)));
		return cuda::unique_ptr<T>{p};
	}
}


namespace ol{
template<typename _Tp>
struct gObject{
    private:
    public:
        int size;
        cuda::unique_ptr<ol::Vect3<_Tp>[]> points;
        gObject(){};
        gObject(int N){
            this->points = cuda::make_unique<ol::Vect3<_Tp>[]>(N);
            this->size = N;
        }
        // __device__ ol::Vect3<_Tp> & operator [](int n) { return points[n]; }
        ol::Vect3<_Tp> & operator [](int n) { return points[n]; }
};

template<typename _Tp>
gObject<_Tp> Object2cuda(ol::Object<_Tp>& obj){
    ol::gObject<_Tp> gobj(obj.size);
    cudaMemcpy(gobj.points.get(),obj.points.get(),sizeof(obj.points[0])*obj.size,cudaMemcpyHostToDevice);
    return gobj;
}

template<typename _Tp>
ol::Object<_Tp> Object2cpu(ol::gObject<_Tp>& gobj){
    ol::Object<_Tp> obj(gobj.size);
    cudaMemcpy(obj.points.get(),gobj.points.get(),sizeof(gobj.points[0]) * gobj.size,cudaMemcpyDeviceToHost);
    return obj;
}

template<typename _Tp>
cuda::unique_ptr<_Tp[]> cpu2cuda(std::unique_ptr<_Tp[]>& src,int size){
	auto src_p = src.get();
	cuda::unique_ptr<_Tp[]> dst = cuda::make_unique<_Tp[]>(size);
	cudaMemcpy(dst.get(),src_p,size * sizeof(_Tp),cudaMemcpyHostToDevice);
	return dst;
}

template<typename _Tp>
std::unique_ptr<_Tp[]> cuda2cpu(cuda::unique_ptr<_Tp[]>& src,int size){
	auto src_p = src.get();
	std::unique_ptr<_Tp[]> dst = std::make_unique<_Tp[]>(size);
	cudaMemcpy(dst.get(),src_p,size * sizeof(_Tp),cudaMemcpyDeviceToHost);
	return dst;
}

template<typename _Tsrc, typename _Tdst>
void cpu2cuda(std::unique_ptr<_Tsrc[]>& src,cuda::unique_ptr<_Tdst[]>& dst, int size){
	auto src_p = src.get();
	if ((void*)dst.get() == NULL){
		dst = cuda::make_unique<_Tdst[]>(size);
		printf("null allocate\n");
	}
	cudaMemcpy(dst.get(),src_p,size * sizeof(_Tsrc),cudaMemcpyHostToDevice);
}

template<typename _Tsrc, typename _Tdst>
void cuda2cpu(cuda::unique_ptr<_Tsrc[]>& src,std::unique_ptr<_Tdst[]>& dst, int size){
	auto src_p = src.get();
	if ((void*)dst.get() == NULL){
		dst = std::make_unique<_Tdst[]>(size);
		printf("null allocate\n");
	}
	cudaMemcpy(dst.get(),src_p,size * sizeof(_Tsrc),cudaMemcpyDeviceToHost);
}

// template<typename _DT,typename _CT>
// cuda::unique_ptr<thrust_DT> complex2cuda(std::unique_ptr<_CT>){
// }


}

#endif

#endif