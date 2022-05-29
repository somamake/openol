#ifndef OLCGH_H
#define OLCGH_H
#include "olutils.h"
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>
#include <omp.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <algorithm>
#include "olclass.h"

namespace ol{

// template<typename X_T, typename Y_T, typename Z_T>
// void objread(const char* path, std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,int &N){
//     FILE *fp;
//     if ( (fp = fopen(path,"rb")) == NULL){
//         perror("fopen");
//         exit(1);
//     }
//     if ( fread(&N,4,1,fp) < 1){
//         perror("fread N");
//         exit(1);
//     }
//     x = std::make_unique<X_T[]>(N);
//     y = std::make_unique<Y_T[]>(N);
//     z = std::make_unique<Z_T[]>(N);

//     int32_t xint, yint,zint;
//     for (int n = 0; n < N;n++){
//         if ( fread(&xint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         if ( fread(&yint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         if ( fread(&zint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         x[n] = xint;
//         y[n] = yint;
//         z[n] = zint;
//     }
//     fclose(fp);
// }

// template<typename X_T, typename Y_T, typename Z_T, typename OFFSET_T, typename SCALE_T>
// void objread(const char* path, std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                 int &N, OFFSET_T *offset, SCALE_T scale){
//     FILE *fp;
//     if ( (fp = fopen(path,"rb")) == NULL){
//         perror("fopen");
//         exit(1);
//     }
//     if ( fread(&N,4,1,fp) < 1){
//         perror("fread N");
//         exit(1);
//     }
//     x = std::make_unique<X_T[]>(N);
//     y = std::make_unique<Y_T[]>(N);
//     z = std::make_unique<Z_T[]>(N);
//     int32_t xint, yint,zint;
//     for (int n = 0; n < N;n++){
//         if ( fread(&xint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         if ( fread(&yint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         if ( fread(&zint,sizeof(int32_t),1,fp) < 1){
//             perror("fread");
//             exit(1);
//         }
//         x[n] = xint * scale + offset[0];
//         y[n] = yint * scale + offset[1];
//         z[n] = zint * scale + offset[2];
//     }
//     fclose(fp);
// }
// template<typename X_T, typename Y_T, typename Z_T>
// void objinfo(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                 int N){
//     auto xminmax = std::minmax_element(x.get(),x.get() + N);
//     auto yminmax = std::minmax_element(y.get(),y.get() + N);
//     auto zminmax = std::minmax_element(z.get(),z.get() + N);
//     std::cout << "xmax: " << *xminmax.second << "  xmin: " << *xminmax.first << std::endl;
//     std::cout << "ymax: " << *yminmax.second << "  ymin: " << *yminmax.first << std::endl;
//     std::cout << "zmax: " << *zminmax.second << "  zmin: " << *zminmax.first << std::endl;
// }

// template<typename X_T, typename Y_T, typename Z_T,typename PREC_T>
// void cghinfo(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,int N,
//         int ny, int nx,PREC_T p, PREC_T lambda){
//     auto xminmax = std::minmax_element(x.get(),x.get() + N);
//     auto yminmax = std::minmax_element(y.get(),y.get() + N);
//     auto zminmax = std::minmax_element(z.get(),z.get() + N);
//     std::cout << "xmax: " << *xminmax.second << "  xmin: " << *xminmax.first << std::endl;
//     std::cout << "ymax: " << *yminmax.second << "  ymin: " << *yminmax.first << std::endl;
//     std::cout << "zmax: " << *zminmax.second << "  zmin: " << *zminmax.first << std::endl;
// }

// template<typename X_T,typename Y_T, typename Z_T,typename PHASE_T,typename PREC_T>
// void _CghFresnelNormalized(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                  int N,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
//     auto U = new std::complex<PHASE_T>[ny * nx];
//     for (int h = 0;h < ny;h++){
//         #pragma omp parallel for
//         for (int w = 0;w < nx;w++){
//             U[h * nx + w] = 0;
//             for (int j = 0;j < N;j++){
//                 PREC_T dx = p * (w - x[j]);
//                 PREC_T dy = p * (h - y[j]);
//                 PREC_T dz = p * z[j];
//                 PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
//                 std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
//                 U[h * nx + w] += Utmp;
//             }
//             phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
//         }
//     }
//     delete[] U;
// }

// template<typename X_T,typename Y_T, typename Z_T,typename PHASE_T,typename PREC_T>
// void _CghFresnel(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                  int N,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
//     auto U = new std::complex<PHASE_T>[ny * nx];
//     for (int h = 0;h < ny;h++){
//         #pragma omp parallel for
//         for (int w = 0;w < nx;w++){
//             U[h * nx + w] = 0;
//             for (int j = 0;j < N;j++){
//                 PREC_T dx = p * w - x[j];
//                 PREC_T dy = p * h - y[j];
//                 PREC_T dz = z[j];
//                 PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
//                 std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
//                 U[h * nx + w] += Utmp;
//             }
//             phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
//         }
//     }
//     delete[] U;
// }

// template<typename X_T,typename Y_T, typename Z_T, typename PHASE_T,typename PREC_T>
// void _CghDiff(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                  int N,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
//     auto U = new std::complex<PREC_T>[ny * nx];
    
//     for (int j = 0;j < N;j++){
//         // h方向に依存がないのでスレッド毎に計算可能
//         #pragma omp parallel for
//         for (int h = 0;h < ny;h++){
//             for (int w = 0;w < nx;w++){
//                 PHASE_T theta,delta;
//                 PHASE_T gam = p / (lambda * z[j]);
//                 if (w == 0){
//                     PHASE_T dx0 = w - x[j];
//                     PHASE_T dy0 = h - y[j];
//                     delta = gam / 2 * (2 * dx0 + 1);
//                     theta = p * z[j] / lambda + gam / 2 * (dx0 * dx0 + dy0 * dy0);
                    
//                 }
//                 else {
//                     theta = theta + delta;
//                     delta = delta + gam;
//                 }
//                 std::complex<PHASE_T> Utmp = std::complex<PHASE_T>(cos(2 * M_PI * theta),sin(2 * M_PI * theta));
//                 U[h * nx + w] += Utmp;          
//             }
//         }
//     }
//     #pragma omp parallel for
//     for (int h = 0;h < ny;h++){
//         for (int w = 0;w < nx;w++){
//             phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
//         }
//     }
//     delete[] U;
// }

// template<typename X_T,typename Y_T, typename Z_T, typename PHASE_T,typename PREC_T>
// void Cgh(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                  int N,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda,bool normalized = false){
//     if (normalized == true){
//         _CghFresnelNormalized(x, y, z, N, phase, ny, nx,p, lambda);
//     }
//     else{
//         _CghFresnel(x, y, z, N, phase, ny, nx,p, lambda);
//     }
// }

// template<typename X_T,typename Y_T, typename Z_T, typename PHASE_T,typename PREC_T>
// void Cgh(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
//                  int N,std::unique_ptr<std::complex<PHASE_T>[]>& phase,
//                  int ny, int nx,PREC_T p, PREC_T lambda){
//     _CghFresnel(x, y, z, N, phase, ny, nx,p, lambda);
// }





// object version

template<typename _Tp,typename PHASE_T,typename PREC_T>
void _CghFresnelNormalized(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
    auto U = new std::complex<PHASE_T>[ny * nx];
    for (int h = 0;h < ny;h++){
        #pragma omp parallel for
        for (int w = 0;w < nx;w++){
            U[h * nx + w] = 0;
            for (int j = 0;j < obj.size;j++){
                PREC_T dx = p * (w - obj[j].x);
                PREC_T dy = p * (h - obj[j].x);
                PREC_T dz = p * obj[j].z;
                PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
                std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
                U[h * nx + w] += Utmp;
            }
            phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
        }
    }
    delete[] U;
}

// template<typename _Tp,typename PHASE_T,typename PREC_T>
// void _CghFresnel(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
//     auto U = new std::complex<PHASE_T>[ny * nx];
//     for (int h = 0;h < ny;h++){
//         #pragma omp parallel for
//         for (int w = 0;w < nx;w++){
//             U[h * nx + w] = 0;
//             for (int j = 0;j < obj.size;j++){
//                 PREC_T dx = p * w - obj[j].x;
//                 PREC_T dy = p * h - obj[j].y;
//                 PREC_T dz = obj[j].z;
//                 PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
//                 std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
//                 U[h * nx + w] += Utmp;
//             }
//             phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
//         }
//     }
//     delete[] U;
// }

// template<typename _Tp,typename COMPLEX_T,typename PREC_T>
// void _CghFresnel(Object<_Tp>& obj,std::unique_ptr<std::complex<COMPLEX_T>[]>& U, int ny, int nx,PREC_T p, PREC_T lambda){
//     for (int h = 0;h < ny;h++){
//         #pragma omp parallel for
//         for (int w = 0;w < nx;w++){
//             U[h * nx + w] = 0;
//             for (int j = 0;j < obj.size;j++){
//                 PREC_T dx = p * w - obj[j].x;
//                 PREC_T dy = p * h - obj[j].y;
//                 PREC_T dz = obj[j].z;
//                 PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
//                 std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
//                 U[h * nx + w] += Utmp;
//             }
//         }
//     }
// }

// ピッチxy
template<typename _Tp,typename PHASE_T,typename PREC_T>
void _CghFresnel(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    auto U = new std::complex<PHASE_T>[ny * nx];
    for (int h = 0;h < ny;h++){
        #pragma omp parallel for
        for (int w = 0;w < nx;w++){
            U[h * nx + w] = 0;
            for (int j = 0;j < obj.size;j++){
                PREC_T dx = px * w - obj[j].x;
                PREC_T dy = py * h - obj[j].y;
                PREC_T dz = obj[j].z;
                PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
                std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
                U[h * nx + w] += Utmp;
            }
            phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
        }
    }
    delete[] U;
}
// ピッチxy
template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void _CghFresnel(Object<_Tp>& obj,std::unique_ptr<std::complex<COMPLEX_T>[]>& U, int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    for (int h = 0;h < ny;h++){
        #pragma omp parallel for
        for (int w = 0;w < nx;w++){
            U[h * nx + w] = 0;
            for (int j = 0;j < obj.size;j++){
                PREC_T dx = px * w - obj[j].x;
                PREC_T dy = py * h - obj[j].y;
                PREC_T dz = obj[j].z;
                PREC_T theta = 2 * M_PI* (dx * dx + dy * dy)/(2 * dz * lambda);
                std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
                U[h * nx + w] += Utmp;
            }
        }
    }
}

template<typename _Tp,typename PHASE_T,typename PREC_T>
void _CghFresnelDiff(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda){
    auto U = new std::complex<PREC_T>[ny * nx];
    
    for (int j = 0;j < obj.size;j++){
        // h方向に依存がないのでスレッド毎に計算可能
        #pragma omp parallel for
        for (int h = 0;h < ny;h++){
            for (int w = 0;w < nx;w++){
                PHASE_T theta,delta;
                PHASE_T gam = p / (lambda * obj[j].z);
                if (w == 0){
                    PHASE_T dx0 = w - obj[j].x;
                    PHASE_T dy0 = h - obj[j].y;
                    delta = gam / 2 * (2 * dx0 + 1);
                    theta = p * obj[j].z / lambda + gam / 2 * (dx0 * dx0 + dy0 * dy0);
                    
                }
                else {
                    theta = theta + delta;
                    delta = delta + gam;
                }
                std::complex<PHASE_T> Utmp = std::complex<PHASE_T>(cos(2 * M_PI * theta),sin(2 * M_PI * theta));
                U[h * nx + w] += Utmp;          
            }
        }
    }
    #pragma omp parallel for
    for (int h = 0;h < ny;h++){
        for (int w = 0;w < nx;w++){
            phase[h * nx + w] = atan2(U[h * nx + w].imag(),U[h * nx + w].real());
        }
    }
    delete[] U;
}

template<typename _Tp,typename PHASE_T,typename PREC_T>
void Cgh(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda,bool normalized = false){
    if (normalized == true){
        _CghFresnelNormalized(obj, phase, ny, nx,p,lambda);
    }
    else{
        _CghFresnel(obj, phase, ny, nx,p, p, lambda);
    }
}

// phaseを出力する
template<typename _Tp,typename PHASE_T,typename PREC_T>
void Cgh(Object<_Tp>& obj,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    _CghFresnel(obj, phase, ny, nx,py,px,lambda);
}

// 複素振幅を出力する
template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void Cgh(Object<_Tp>& obj,std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                 int ny, int nx,PREC_T p, PREC_T lambda){
    _CghFresnel(obj, u, ny, nx,p, p, lambda);
}

// 複素振幅を出力する ピッチxy
template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void Cgh(Object<_Tp>& obj,std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                 int ny, int nx,PREC_T py,PREC_T px, PREC_T lambda){
    _CghFresnel(obj, u, ny, nx,py,px, lambda);
}

template<typename _Tp,typename COMPLEX_T,typename PREC_T>
void WRPStep1(Object<_Tp>& obj,std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int ny, int nx,PREC_T p, PREC_T lambda,PREC_T d){
    PREC_T c = lambda * 0.5 / p;
    for (int h = 0;h < ny;h++){
        #pragma omp parallel for
        for (int w = 0;w < nx;w++){
            u[h * nx + w] = 0;
            for (int j = 0;j < obj.size;j++){
                // if 
                PREC_T dx = p * w - obj[j].x;
                PREC_T dy = p * h - obj[j].y;
                PREC_T dz = obj[j].z;
                PREC_T theta = 2 * M_PI * (dx * dx + dy * dy)/(2 * dz * lambda);
                std::complex<PREC_T> Utmp = std::complex<PREC_T>(cos(theta),sin(theta));
                u[h * nx + w] += Utmp;
            }
        }
    }
}
// void WRPStep2(){

// }

template<typename X_T,typename Y_T, typename Z_T, typename PHASE_T,typename PREC_T>
void WRPMethod(std::unique_ptr<X_T[]>& x, std::unique_ptr<Y_T[]>& y, std::unique_ptr<Z_T[]>& z,
                int N,std::unique_ptr<PHASE_T[]>& phase, int ny, int nx,PREC_T p, PREC_T lambda,double zmin){
    // WRPStep1();
}

}
#endif