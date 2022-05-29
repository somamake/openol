#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda.h>
#include "openol.h"

void cpurandom(){
    auto seedgen = [](){ return 1;};
    std::mt19937 engine(seedgen());
    std::uniform_real_distribution<float> dist(0,1);
    for(int i = 0;i < 10;i++){
        printf("%f\n",dist(engine));
    }
}
void myrand(){
    std::mt19937 e2(1);
    
    printf("myrand\n");
    for(int i = 0;i < 10;i++){
        int a = e2() >> 5;
        int b = e2() >> 6;
        double value = (a * 67108864.0 + b) / 9007199254740992.0;
        std::cout << std::fixed << std::setprecision(16) << value << std::endl;
    }
    
}

int main(){
    size_t n = 100;
    size_t i;
    curandGenerator_t gen;
    // std::unique_ptr<float[]> hostData = std::make_unique<float[]>(n);
    cuda::unique_ptr<float[]> devData = cuda::make_unique<float[]>(n);

    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,0ULL);
    curandGenerateUniform(gen,devData.get(),n);
    auto hostData = ol::cuda2cpu(devData,n);
    for (i = 0; i < n;i++){
        printf("%1.4f ",hostData[i]);
    }
    printf("\n");

    curandDestroyGenerator(gen);
    devData.reset();
    hostData.reset();
    cpurandom();
    myrand();
}