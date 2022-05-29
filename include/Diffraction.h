#ifndef DIFFRACTION_H
#define DIFFRACTION_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <complex>
#include "olclass.h"
#include "olutils.h"
#include <fftw3.h>
// #include "olutils.h"



namespace ol{
    // 伝搬元と伝搬先の情報保持して，伝搬と計算精度表示を行う
enum PropPrecition{
    DOUBLE,FLOAT
};
struct Diffraction{
    private:
    public:
        float distance;
        Plane src,dst;
        // virtual void prop(){};
        virtual void info(){};
        void setplane(Plane src,Plane dst);
        Diffraction(Plane src,Plane dst);
        Diffraction();
};

void Diffraction::setplane(Plane src,Plane dst){
            this->src = src; this->dst = dst;
        };
Diffraction::Diffraction(Plane src,Plane dst){
    setplane(src,dst);
}

struct FresnelDif : public Diffraction{
    private:
    public:
    using Diffraction::Diffraction;
        template<typename _Tp>
        void prop(Mat<_Tp>& src,Mat<_Tp>& dst);
        // void prop(_Tp u);
        void info() override;
};

void FresnelDif::info(){
        printf("Fresnelinfo\n");
}

template<typename _Tp>
void FresnelDif::prop(Mat<_Tp>& src,Mat<_Tp>& dst){
    // if constexpr (std::is_same_v<_Tp,std::complex<double>>){
    //     printf("fresnelprop double\n");
    //     fftw_complex* _usrc = (fftw_complex*)(src.data.get());
    //     fftw_complex* _udst = (fftw_complex*)(dst.data.get());
    // }
    // else if constexpr (std::is_same_v<_Tp,std::complex<float>>){
    //     printf("fresnelprop float\n");
    //     fftwf_complex* _usrc = (fftwf_complex*)(src.data.get());
    //     fftwf_complex* _udst = (fftwf_complex*)(dst.data.get());
    // }
    // else{
    //     fprintf(stderr,"error\n");
    // }
}


}
#endif