#include "Diffuser.h"

namespace ol{
    template void Diffuser::random_phase<float>(std::unique_ptr<std::complex<float>[]>& u,HOLOGRAMSTEP step,int seed = 1,float range= 2*M_PI);
    
}