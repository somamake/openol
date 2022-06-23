#include "openol.h"

int main(){
    auto src = ol::Plane(1024,1024,6e-6,6e-6);
    // ol::Plane dst;
    // dst.set(1024,1024,6e-6,6e-6);

    ol::Mat<std::complex<float>> u;
    ol::FresnelDif diffraction1(ol::Plane(1024,1024,6e-6,6e-6),src);
    diffraction1.prop(u,u);
    diffraction1.info();
    return 0;
}