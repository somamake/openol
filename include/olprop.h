#ifndef OLPROP_H
#define OLPROP_H
#include "olutils.h"
#include "oldefine.h"

namespace ol{
/*フレネル回折のh計算関数*/
template<typename PREC_T=double,typename COMPLEX_T>
void FresnelResponse(std::unique_ptr<std::complex<COMPLEX_T>[]>& h,size_t height, size_t width,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z)
{
    PREC_T tmp = 1 / (lambda * z);
    int64_t hh = height/2;
    int64_t hw = width/2;
    for(int64_t n = 0;n < (int64_t)height;n++){
        PREC_T y = ((PREC_T)( (n - hh) )) * dy;
        for(int64_t m = 0;m < (int64_t)width;m++){
            size_t idx = n * width + m;
            PREC_T x = ((PREC_T)( (m - hw) )) * dx;
            PREC_T phase =  M_PI* ( x * x + y * y ) * tmp ;
            h[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
        }
    }
}

template<typename PREC_T=double,typename COMPLEX_T>
void FresnelResponseBandLimit(std::unique_ptr<std::complex<COMPLEX_T>[]>& h,size_t height, size_t width,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z)
{
    PREC_T tmp = 1 / (lambda * z);
    int64_t hh = height/2;
    int64_t hw = width/2;
    PREC_T xlim,ylim;
    int wlim, hlim;
    xlim = abs(lambda * z * 0.5 / dx);
    ylim = abs(lambda * z * 0.5 / dy);
    wlim = floor (xlim / dx);
    hlim = floor (ylim / dy);
    for(int64_t n = 0;n < (int64_t)height;n++){
        PREC_T y = ((PREC_T)( (n - hh) )) * dy;
        for(int64_t m = 0;m < (int64_t)width;m++){
            size_t idx = n * width + m;
            if (abs(m - hw) < wlim && abs(n - hh) < hlim){
                PREC_T x = ((PREC_T)( (m - hw) )) * dx;
                PREC_T phase =  M_PI* ( x * x + y * y ) * tmp ;
                h[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
            }
            else{
                h[idx] = 0;
            }
            
        }
    }
}

template<typename PREC_T=double,typename COMPLEX_T>
void FresnelResponseBandLimit(std::unique_ptr<std::complex<COMPLEX_T>[]>& h,size_t height, size_t width,
                    PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z,PREC_T p_limit)
{
    PREC_T tmp = 1 / (lambda * z);
    int64_t hh = height/2;
    int64_t hw = width/2;
    PREC_T xlim,ylim;
    int wlim, hlim;
    xlim = abs(lambda * z * 0.5 / p_limit);
    ylim = abs(lambda * z * 0.5 / p_limit);
    wlim = floor (xlim / dx);
    hlim = floor (ylim / dy);
    for(int64_t n = 0;n < (int64_t)height;n++){
        PREC_T y = ((PREC_T)( (n - hh) )) * dy;
        for(int64_t m = 0;m < (int64_t)width;m++){
            size_t idx = n * width + m;
            if (abs(m - hw) < wlim && abs(n - hh) < hlim){
                PREC_T x = ((PREC_T)( (m - hw) )) * dx;
                PREC_T phase =  M_PI* ( x * x + y * y ) * tmp ;
                h[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
            }
            else{
                h[idx] = 0;
            }
            
        }
    }
}

template<typename PREC_T = double>
void FresnelResponseCheck(size_t height, size_t width,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    width *= 2;
    height *= 2;
    PREC_T xmax = dx * width / 2;
    PREC_T ymax = dy * height / 2;
    PREC_T zmin_x,zmin_y,xlim,ylim;
    int wlim, hlim;
    zmin_x = 2 * xmax * dx / lambda;
    zmin_y = 2 * ymax * dy / lambda;
    xlim = lambda * z * 0.5 / dx;
    ylim = lambda * z * 0.5 / dy;
    wlim = floor (xlim / dx) * 2;
    hlim = floor (ylim / dy) * 2;
    std::cout << "zmin (from x) = " <<std::setprecision(4) << zmin_x << std::endl;
    std::cout << "zmin (from y) = " <<std::setprecision(4) << zmin_y << std::endl;
    std::cout << "xlim = " <<std::setprecision(4) << xlim << std::endl;
    std::cout << "ylim = " <<std::setprecision(4) << ylim << std::endl;
    std::cout << "widthlim[px] = "  << wlim << std::endl;
    std::cout << "heightlim[px] = " << hlim << std::endl;
}

template<typename PREC_T = double>
bool FresnelPropCheck(size_t ny, size_t nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    size_t nx2 = 2 * nx;
    size_t ny2 = 2 * ny;
    PREC_T xmax = dx * nx2 / 2;
    PREC_T ymax = dy * ny2 / 2;
    PREC_T zmin_x,zmin_y;
    zmin_x = 2 * xmax * dx / lambda;
    zmin_y = 2 * ymax * dy / lambda;
    if (zmin_x > abs(z) || zmin_y > abs(z)){
        return false;
    }
    return true;
}

template<typename PREC_T = double, typename COMPLEX_T>
void FresnelProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,size_t height, size_t width,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    
    zeropadding(u,u,height,width);
    height *= 2; 
    width *= 2;
    
    auto h = std::make_unique<std::complex<COMPLEX_T>[]>(height * width);
    FresnelResponse(h, height, width, dy, dx,  lambda, z);
    fftshift(h,height,width);
    fft(u,u,height,width);
    fft(h,h,height,width);
    // fftshift(h,height,width);
    // fftshift(u,height,width);
    mul_scalar(u,1.0/(height*width),height,width);
    mul_scalar(h,1.0/(height*width),height,width);
    mul_complex(u,h,u,height,width);
    // fftshift(u,height,width);
    ifft(u,u,height,width);
    
    del_zero(u,u,height,width);
    // mul_scalar(u,1.0/(height*width),height/2,width/2);
}

template<typename PREC_T = double, typename COMPLEX_T>
void FresnelPropBandLimit(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,size_t height, size_t width,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    
    zeropadding(u,u,height,width);
    height *= 2; 
    width *= 2;
    
    auto h = std::make_unique<std::complex<COMPLEX_T>[]>(height * width);
    FresnelResponseBandLimit(h, height, width, dy, dx,  lambda, z);
    fftshift(h,height,width);
    fft(u,u,height,width);
    fft(h,h,height,width);
    mul_scalar(u,1.0/(height*width),height,width);
    mul_scalar(h,1.0/(height*width),height,width);
    mul_complex(u,h,u,height,width);
    ifft(u,u,height,width);
    
    del_zero(u,u,height,width);
    // mul_scalar(u,1.0/(height*width),height/2,width/2);
}

/*角スペクトル法のH計算関数*/
template<typename PREC_T = float, typename COMPLEX_T>
void AsmTransferF(std::unique_ptr<std::complex<COMPLEX_T>[]>& H, 
                    int64_t ny, int64_t nx,PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z)
{
    PREC_T tmp = 1/(lambda * lambda);
    #pragma omp parallel for
    for (int64_t n=0;n<ny;n++){
        PREC_T v = (n - ny/2) * dv;
        for (int64_t m=0;m<nx;m++){
            int64_t idx = m + n * nx;
            PREC_T u = (m - nx/2) * du;
            PREC_T w = sqrt(tmp - u * u - v * v);
            PREC_T phase = 2 * M_PI *  w * z;
            H[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
        }
    }
}

// width and height are before padding
template<typename PREC_T = double>
void AsmTransferFCheck(int64_t height, int64_t width,PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z)
{
    width *= 2;
    height *= 2;
    PREC_T du = 1 / ( dx * width);
    PREC_T dv = 1 / (dy * height);
    PREC_T umax, vmax,zmax_u,zmax_v,tmpsqrt;
    umax = du * width / 2;
    vmax = dv * height / 2;
    tmpsqrt = sqrt(1/(lambda * lambda) - umax * umax - vmax * vmax);
    zmax_u = tmpsqrt / (2 * umax * du);
    zmax_v = tmpsqrt / (2 * vmax * dv);
    std::cout << "zmax (from x) = " <<std::setprecision(4) << zmax_u << std::endl;
    std::cout << "zmax (from y) = " <<std::setprecision(4) << zmax_v << std::endl;
}

template<typename PREC_T = float>
bool AsmPropCheck(int64_t ny, int64_t nx,PREC_T dy,PREC_T dx, PREC_T lambda, PREC_T z){
    int64_t nx2 = 2 * nx;
    int64_t ny2 = 2 * ny;
    PREC_T du = 1 / ( dx * nx2);
    PREC_T dv = 1 / ( dy * ny2);
    PREC_T umax, vmax,zmax_u,zmax_v,tmpsqrt;
    umax = du * nx2/ 2;
    vmax = dv * ny2 / 2;
    tmpsqrt = sqrt(1/(lambda * lambda) - umax * umax - vmax * vmax);
    zmax_u = tmpsqrt / (2 * umax * du);
    zmax_v = tmpsqrt / (2 * vmax * dv);
    if ( zmax_u < abs(z) || zmax_v < abs(z)){
        return false;
    }
    return true;
}

template<typename PREC_T = double, typename COMPLEX_T>
void AsmProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int64_t height, int64_t width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z)
{
    // auto img = std::make_unique<uint8_t[]>(height * width * 4);
    int64_t height2 = height * 2;
    int64_t width2 = width * 2;
    PREC_T du = 1 / ( dx * width2);
    PREC_T dv = 1 / (dy * height2);
    auto H = std::make_unique<std::complex<COMPLEX_T>[]>(height2 * width2);
    zeropadding(u,u,height,width,height2,width2);
    AsmTransferF(H,height * 2, width * 2,dv,du,lambda, z);

    fftshift(H,height2,width2);
    fft(u,u,height2,width2);
    
    mul_complex(u,H,u,height2,width2);
    mul_scalar(u,1.0/(height2*width2),height2,width2);

    ifft(u,u,height2,width2);
    del_zero(u,u,height2,width2);
    // mul_scalar(u,1.0/(height2*width2),height,width); 
}

/*角スペクトル法のH計算関数*/
template<typename PREC_T = float, typename COMPLEX_T>
void shiftedAsmTransferF(std::unique_ptr<std::complex<COMPLEX_T>[]>& H, 
                    int64_t ny, int64_t nx,PREC_T dv,PREC_T du, PREC_T lambda, PREC_T z,PREC_T oy,PREC_T ox)
{
    PREC_T sx = 0.5f / du;
	PREC_T sy = 0.5f / dv;
	PREC_T u0, uw, ulm, ulp, v0, vw ,vlm, vlp;
	ulm = 1 / sqrt( std::pow((ox - 0.5f / du), -2) * z * z + 1.0f) / lambda;
	ulp = 1 / sqrt( std::pow((ox + 0.5f / du), -2) * z * z + 1.0f) / lambda;

	vlm = 1 / sqrt( std::pow((oy - 0.5f / dv), -2) * z * z + 1.0f) / lambda;
	vlp = 1 / sqrt( std::pow((oy + 0.5f / dv), -2) * z * z + 1.0f) / lambda;
	if (sx < ox){
		u0 = 0.5f * (ulp + ulm);
		uw = ulp - ulm;
	}
	else if (-sx <= ox && ox < sx){
		u0 = 0.5f * (ulp - ulm);
		uw = ulp + ulm;
	}
	else{
		u0 = -0.5f * (ulp + ulm);
		uw = ulm - ulp;
	}

	if (sy < oy){
		v0 = 0.5f * (vlp + vlm);
		vw = vlp - vlm;
	}
	else if (-sy <= oy && oy < sy){
		v0 = 0.5f * (vlp - vlm);
		vw = vlp + vlm;
	}
	else{
		v0 = -0.5f * (vlp + vlm);
		vw = vlm - vlp;
	}
    PREC_T tmp = 1/(lambda * lambda);
    #pragma omp parallel for
    for (int64_t n=0;n<ny;n++){
        PREC_T v = (n - ny/2) * dv;
        for (int64_t m=0;m<nx;m++){
            int64_t idx = m + n * nx;
            PREC_T u = (m - nx/2) * du;
            if ( (abs(u - u0) * 2 < uw) && (abs(v - v0) * 2 < vw) ){
            PREC_T w = sqrt(tmp - u * u - v * v);
            PREC_T phase = 2 * M_PI *  (w * z + ox * u + oy * v);
            H[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
            }
            else{
                H[idx] = 0;
            }
        }
    }
}

template<typename PREC_T = double, typename COMPLEX_T>
void shiftedAsmProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& usrc,std::unique_ptr<std::complex<COMPLEX_T>[]>& udst,int64_t height, int64_t width, 
                    PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z,PREC_T oy,PREC_T ox)
{
    // auto img = std::make_unique<uint8_t[]>(height * width * 4);
    int64_t height2 = height * 2;
    int64_t width2 = width * 2;
    PREC_T du = 1 / ( dx * width2);
    PREC_T dv = 1 / (dy * height2);
    auto H = std::make_unique<std::complex<COMPLEX_T>[]>(height2 * width2);
    auto utmp = std::make_unique<std::complex<COMPLEX_T>[]>(height2 * width2);
    zeropadding(usrc,utmp,height,width,height2,width2);
    shiftedAsmTransferF(H,height * 2, width * 2,dv,du,lambda, z,oy,ox);

    fftshift(H,height2,width2);
    fft(utmp,utmp,height2,width2);
    
    mul_complex(utmp,H,utmp,height2,width2);
    mul_scalar(utmp,1.0/(height2*width2),height2,width2);

    ifft(utmp,utmp,height2,width2);
    del_zero(utmp,udst,height2,width2);
    mul_scalar(udst,1.0/(height2*width2),height,width); 
}

template<typename PREC_T=float,typename COMPLEX_T>
void shiftedFresnelPropInfo(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                        int height, int width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z, PREC_T s,PREC_T oy,PREC_T ox){
    
}

template<typename PREC_T=float,typename COMPLEX_T>
void shiftedFresnelProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                        int64_t height, int64_t width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z, PREC_T s,PREC_T oy,PREC_T ox){
    PREC_T tmp = M_PI / (lambda * z);
    int64_t hh = height / 2;
    int64_t hw = width / 2;

    for (int64_t h = 0;h < height;h++){
        PREC_T y1 = (h - hh) * dy;
        PREC_T phiu_y = (s * s - s) * y1 * y1 - 2 * s * oy * y1;
        for (int64_t w = 0;w < width;w++){
            int64_t idx = h * width + w;
            PREC_T x1 = (w - hw) * dx;
            PREC_T phiu = (s * s - s) * x1 * x1 - 2 * s * ox * x1;
            phiu += phiu_y;
            phiu *= tmp;
            PREC_T real = cos(phiu);
            PREC_T imag = sin(phiu);
            u[idx] = u[idx] * std::complex<COMPLEX_T>(real,imag);
        }
    }
    zeropadding(u,u,height,width);
    hh *= 2;
    hw *= 2;
    height *= 2;
    width *= 2;

    // fftshift(u,height,width);
    fft(u,u,height,width);
    mul_scalar(u,1.0/(height*width),height,width);

    auto  h = std::make_unique<std::complex<COMPLEX_T>[]>(height * width);
    for (int64_t n = 0;n < height;n++){
        PREC_T y1 = (n - hh) * dy;
        PREC_T phih_y = s * y1 * y1;
        for (int64_t m = 0; m < width;m++){
            int64_t idx = n * width + m;
            PREC_T x1 = (m - hw) * dx;
            PREC_T phih = s * x1 * x1 + phih_y;
            phih *= tmp;
            h[idx] = std::complex<COMPLEX_T>(cos(phih),sin(phih));
        }
    }

    fftshift(h,height,width);
    fft(h,h,height,width);
    mul_scalar(u,1.0/(height*width),height,width);
    mul_complex(u,h,u,height,width);
    ifft(u,u,height,width); 
    
    del_zero(u,u,height,width);
    mul_scalar(u,1.0/(height*width),height/2,width/2);
    hh /= 2;
    hw /= 2;
    height /= 2;
    width /= 2;
    for (int64_t h = 0;h < height;h++){
        PREC_T y2 = (h - hh) * dy;
        PREC_T phic_y = ( (1 - s) * y2 * y2 + 2 * oy * y2 + oy * oy);
        for (int64_t w = 0;w < width;w++){
            int64_t idx = h * width + w;
            PREC_T x2 = (w - hw) * dx;
            PREC_T phic = ( (1 - s) * x2 * x2 + 2 * ox * x2 + ox * ox);
            phic += phic_y;
            phic *= tmp;
            PREC_T real = cos(phic);
            PREC_T imag = sin(phic);
            u[idx] = u[idx] * std::complex<COMPLEX_T>(real,imag);
        }
    }
}

enum INTERPOLATION {
    BILINEAR, BICUBIC
};

// angleはsrcから見たdst planeの回転角
template<typename PREC_T=float,typename COMPLEX_T>
void TiltedSourceProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z,PREC_T angle_src2dst,INTERPOLATION inp = BILINEAR,PROPMODE propmode = ASM){
    int64_t ny2 = ny * 2;
    int64_t nx2 = nx * 2;
    PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);

    PREC_T fsqr = 1 / (lambda * lambda);


    // FFT on tiilted coordinate
    zeropadding(u,u,ny,nx);
    fft(u,u,ny2,nx2);
    fftshift(u,ny2,nx2);
    mul_scalar<PREC_T>(u,(1.0/(ny2 * nx2)),ny2,nx2);

    // fp (parallel coordinate)からfs (source coordinate)に変換するときの回転行列
    PREC_T angle_dst2src = -angle_src2dst;
    PREC_T a11 = cos(angle_dst2src);
    PREC_T a12 = 0;
    PREC_T a13 = -sin(angle_dst2src);
    PREC_T a21 = 0;
    PREC_T a22 = 1;
    PREC_T a23 = 0;
    // PREC_T a31 = sin(angle_dst2src);
    // PREC_T a32 = 0;
    // PREC_T a33 = cos(angle_dst2src);
    // Save(PROJECT_ROOT "/out/tmp0.bmp",u,ny2,nx2,AMP);
    auto Up = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);
    PREC_T f0 = 1 / lambda;
    for (int64_t n = 0;n < ny2;n++){
        for (int64_t m = 0;m < nx2;m++){
            PREC_T fpx = (m - nx2 / 2) * du;
            PREC_T fpy = (n - ny2 / 2) * dv;
            PREC_T fpz = sqrt(fsqr - fpx * fpx - fpy * fpy);
            // 空間周波数の座標軸変換 (dst to src)
            PREC_T fsx = a11 * fpx + a12 * fpy + a13 * fpz - a13 * f0;
            PREC_T fsy = a21 * fpx + a22 * fpy + a23 * fpz;

            if (inp == BILINEAR){
                Up[m + n * nx2] = bilinear(u,fsy / dv + ny2 / 2,fsx / du + nx2 / 2,ny2,nx2);
            }
            else if (inp == BICUBIC){
                Up[m + n * nx2] = bicubic(u,fsy / dv + ny2 / 2,fsx / du + nx2 / 2,ny2,nx2);
            }
            PREC_T J;
            J = (a11 * a22 - a12 * a21) + (a12 * a23 - a13 * a22) * fpx / fpz + (a13 * a21 - a11 * a23) * fpy / fpz;
            Up[m + n * nx2] *= J;
        }
    }
    u.reset();
    // Save(PROJECT_ROOT "/out/tmp1.bmp",Up,ny2,nx2,AMP);

    auto H = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);
    if (propmode == ASM){
        // AsmTransferF(H,ny2,nx2,dv,du,lambda,z);
        PREC_T tmp = 1/(lambda * lambda);
        for (int64_t n=0;n<ny2;n++){
            PREC_T v = (n - ny2/2) * dv;
            for (int64_t m=0;m<nx2;m++){
                int64_t idx = m + n * nx2;
                PREC_T u = (m - nx2/2) * du + a13 * f0;
                PREC_T w = sqrt(tmp - u * u - v * v);
                PREC_T phase = 2 * M_PI *  w * z;
                H[idx] = std::complex<COMPLEX_T>(cos(phase),sin(phase));
            }
        }
    }
    else if (propmode == FRESNEL){
        FresnelResponse(H,ny2,nx2,dy,dx,lambda,z);
        // mul phase
        for (int64_t n=0;n<ny2;n++){
            for (int64_t m=0;m<nx2;m++){
                int64_t idx = m + n * nx2;
                PREC_T w = (m - nx2/2) * dx;
                PREC_T phase = -2 * M_PI *  w * a13 * f0;
                H[idx] *= std::complex<COMPLEX_T>(cos(phase),sin(phase));
            }
        }
        fftshift(H,ny2,nx2);
        fft(H,H,ny2,nx2);
        mul_scalar<PREC_T>(H,(1.0/(ny2 * nx2)),ny2,nx2);
        fftshift(H,ny2,nx2);
    }
    
    // for (int n = 0;n < ny2;n++){
    //     for (int m = 0;m < nx2;m++){
    //         int idx = n * nx2 + m;
    //         // AsmTransfer
    //         PREC_T fpx = (m - nx2 / 2) * du;
    //         PREC_T fpy = (n - ny2 / 2) * dv;
    //         PREC_T fpz = sqrt(fsqr - fpx * fpx - fpy * fpy);
    //         // PREC_T phi = 2 * M_PI * z * fpz;

    //         // Jakobian
    //         PREC_T J;
    //         J = (a11 * a22 - a12 * a21) + (a12 * a23 - a13 * a22) * fpx / fpz + (a13 * a21 - a11 * a23) * fpy / fpz;
    //         // H[idx] = abs(J) * std::complex<PREC_T>(cos(phi),sin(phi));
    //         H[idx] *= abs(J);
    //     }
    // }
    mul_complex(Up,H,Up,ny2,nx2);
    fftshift(Up,ny2,nx2);
    ifft(Up,Up,ny2,nx2);
    del_zero(Up,u,ny2,nx2);
    mul_scalar<PREC_T>(u,(1.0/(ny2 * nx2)),ny,nx);
}

template<typename PREC_T=float,typename COMPLEX_T>
void TiltedReferenceProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                        int64_t ny, int64_t nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z,PREC_T rotate,INTERPOLATION inp = BILINEAR,PROPMODE propmode = ASM){
    int64_t ny2 = ny * 2;
    int64_t nx2 = nx * 2;
    PREC_T du = 1 / (dx * nx2);
    PREC_T dv = 1 / (dy * ny2);

    PREC_T fsqr = 1 / (lambda * lambda);
    // FFT on source coordinate
    zeropadding(u,u,ny,nx);
    fft(u,u,ny2,nx2);
    fftshift(u,ny2,nx2);
    mul_scalar<PREC_T>(u,((PREC_T)1.0/(ny2 * nx2)),ny2,nx2);

    auto H = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);
    if (propmode == ASM){
        AsmTransferF(H,ny2,nx2,dv,du,lambda,z);
    }
    else if (propmode == FRESNEL){
        FresnelResponse(H,ny2,nx2,dy,dx,lambda,z);
        fftshift(H,ny2,nx2);
        fft(H,H,ny2,nx2);
        mul_scalar<PREC_T>(H,(1.0/(ny2 * nx2)),ny2,nx2);
        fftshift(H,ny2,nx2);
    }
    mul_complex(u,H,u,ny2,nx2);
    H.reset();


    

    // 回転行列
    PREC_T a11 = cos(rotate);
    PREC_T a12 = 0;
    PREC_T a13 = -sin(rotate);
    PREC_T a21 = 0;
    PREC_T a22 = 1;
    PREC_T a23 = 0;
    
    auto Up = std::make_unique<std::complex<COMPLEX_T>[]>(ny2 * nx2);
    // PREC_T f0 = 1 / lambda;
    for (int64_t n = 0;n < ny2;n++){
        for (int64_t m = 0;m < nx2;m++){
            PREC_T fpx = (m - nx2 / 2) * du;
            PREC_T fpy = (n - ny2 / 2) * dv;
            PREC_T fpz = sqrt(fsqr - fpx * fpx - fpy * fpy);
            // 空間周波数の座標軸変換 (src to reference)
            // PREC_T fsx = a11 * fpx + a12 * fpy + a13 * fpz - a13 * f0;
            PREC_T fsx = a11 * fpx + a12 * fpy + a13 * fpz;
            PREC_T fsy = a21 * fpx + a22 * fpy + a23 * fpz;

            if (inp == BILINEAR){
                Up[m + n * nx2] = bilinear(u,fsy / dv + ny2 / 2,fsx / du + nx2 / 2,ny2,nx2);
            }
            else if (inp == BICUBIC){
                Up[m + n * nx2] = bicubic(u,fsy / dv + ny2 / 2,fsx / du + nx2 / 2,ny2,nx2);
            }
            PREC_T J;
            J = (a11 * a22 - a12 * a21) + (a12 * a23 - a13 * a22) * fpx / fpz + (a13 * a21 - a11 * a23) * fpy / fpz;
            Up[m + n * nx2] *= J;
        }
    }
    u.reset();
    fftshift(Up,ny2,nx2);
    ifft(Up,Up,ny2,nx2);
    del_zero(Up,u,ny2,nx2);
    mul_scalar<PREC_T>(u,(1.0/(ny2 * nx2)),ny,nx);
    // for (int n = 0;n < ny;n++){
    //     for (int m = 0;m < nx;m++){
    //         PREC_T x = (m - nx / 2) * dx;
    //         PREC_T phase = 2 * M_PI * x * a13 * f0;
    //         u[n * nx + m] *= std::complex(cos(phase),sin(phase));
    //     }
    // }

}

template<typename COMPLEX_T,typename PREC_T>
void shiftedProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& usrc,std::unique_ptr<std::complex<COMPLEX_T>[]>& udst,
                        int ny, int nx, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d, PREC_T oy,PREC_T ox,PROPMODE propmode){
	if (propmode == ol::ASM){
		shiftedAsmProp(usrc,udst,ny,nx,dy,dx,lambda,d,oy,ox);
	}
	else if (propmode == ol::FRESNEL){
		shiftedFresnelProp(usrc,udst,ny,nx,dy,dx,lambda,d,(PREC_T)1.0,oy,ox);
	}
	else{
		printf("error\n");
		exit(1);
	}
}

// split asm propagation
template<typename PREC_T=float,typename COMPLEX_T>
void splitAsmProp(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int64_t height, int64_t width, PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T z,int blocknum_y = 2,int blocknum_x = 2){
    int blockwidth = width / blocknum_x;
    int blockheight = height / blocknum_y;
    auto udst = std::make_unique<std::complex<COMPLEX_T>[]>(height * width);
    auto u_block_src = std::make_unique<std::complex<COMPLEX_T>[]>(blockheight * blockwidth);
    auto u_block_dst = std::make_unique<std::complex<COMPLEX_T>[]>(blockheight * blockwidth);
    
    for (int global_n = 0; global_n < blocknum_y;global_n++){
        for (int global_m = 0;global_m < blocknum_x;global_m++){
            // copy from u to u_block
            for (int local_n = 0; local_n < blockheight;local_n++){
                for (int local_m = 0; local_m < blockwidth; local_m++){
                    int n = global_n * blockheight + local_n;
                    int m = global_m * blockwidth + local_m;
                    u_block_src[local_n * blockwidth + local_m] = u[n * width + m];
                }
            }
            for (int dst_n = 0; dst_n < blocknum_y;dst_n++){
                for (int dst_m = 0;dst_m < blocknum_x;dst_m++){
                    float oy = (global_n - dst_n) * blockheight * dy;
                    float ox = (global_m - dst_m) * blockwidth * dx;
                    shiftedAsmProp(u_block_src,u_block_dst,blockheight,blockwidth,dy,dx,lambda,z,oy,ox);
                    // udst + ublock
                    for (int local_n = 0; local_n < blockheight;local_n++){
                        for (int local_m = 0; local_m < blockwidth; local_m++){
                            int n = dst_n * blockheight + local_n;
                            int m = dst_m * blockwidth + local_m;
                            udst[n * width + m] += u_block_dst[local_n * blockwidth + local_m];
                        }
                    }
                }
            }
        }
    }
    u = std::move(udst);
}

template<typename COMPLEX_T,typename PREC_T>
void Prop(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,
                int ny, int nx,PREC_T dy, PREC_T dx, PREC_T lambda, PREC_T d,PROPMODE propmode = ol::PROPMODE::AUTO){
	if (propmode == PROPMODE::FRESNEL){
		printf("FresnelProp\n");
        FresnelProp(u,ny,nx,dy,dx,lambda,d);
    }
    else if (propmode == PROPMODE::ASM){
		printf("AsmProp\n");
        AsmProp(u,ny,nx,dy,dx,lambda,d);
    }
    else if (propmode == PROPMODE::AUTO){
        if (AsmPropCheck(ny,nx,dy,dx,lambda,d) == true){
            printf("AsmProp\n");
            AsmProp(u,ny,nx,dy,dx,lambda,d);
        }
        else{
            if (FresnelPropCheck(ny,nx,dy,dx,lambda,d) == false){
                printf("warning!!\neither AsmProp or FresnelProp does not meet the condition.\n");
				FresnelPropBandLimit(u,ny,nx,dy,dx,lambda,d);
				return;
            }
            printf("FresnelProp\n");
            FresnelProp(u,ny,nx,dy,dx,lambda,d);
        }
    }
}

}

#endif