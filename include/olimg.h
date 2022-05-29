#ifndef OLBMP_H
#define OLBMP_H

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cmath>
#include <stdint.h>
#include <memory>
#include <complex>
#include <opencv2/opencv.hpp>
#include "olutils.h"

#ifdef __NVCC__
#include "golclass.h"
#include "thrust/complex.h"
#endif

#pragma pack(1)
typedef struct tagBITMAPFILEHEADER
{
	unsigned short bfType; //ファイルタイプ
	unsigned int  bfSize; //ファイルサイズ
	unsigned short bfReserved1; //予約領域1
	unsigned short bfReserved2;  //予約領域2
	unsigned int  bfOffBits; //
}BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER 
{
	unsigned int   biSize;//ヘッダサイズ
	int		    biWidth;//画像の幅(px)
	int		biHeight;//画像の高さ(px)
	unsigned short  biPlanes;//プレーン数
	unsigned short  biBitCount;//1画素あたりデータサイズ
	unsigned int   biCompression;//圧縮形式
	unsigned int   biSizeImage;//画像データ部のサイズ 0で最大を意味する?
	int		biXPelsPerMeter;//横方向解像度(dot/m)
	int		biYPelsPerMeter;//縦方向解像度(dot/m)
	unsigned int   biClrUsed;//格納されているパレット数
	unsigned int   biClrImportant;//重要なパレットのインデックス 0で全て重要
}BITMAPINFOHEADER;

typedef struct tagRGBQUAD
{
	unsigned char  rgbBlue;//青0~255
	unsigned char  rgbGreen;//緑0~255
	unsigned char  rgbRed;//赤0~255
	unsigned char  rgbReserved;//予約領域
}RGBQUAD;
#pragma pack()

namespace ol{

// image read
void bmpread(const char *fname,uint8_t *img,int ny,int nx){
    int color[1024];
    FILE *fp;
	BITMAPFILEHEADER BmpFileHeader;
	BITMAPINFOHEADER BmpInfoHeader;
	// RGBQUAD			 RGBQuad[256]; //256*4=1024

	fp = fopen(fname,"rb");
	if ( fp == NULL){
        perror("fopen");
        exit(1);
    }
	if ( fread(&BmpFileHeader, sizeof(BITMAPFILEHEADER) , 1 , fp) < 1){
        perror("fread");
        exit(1);
    }
	if ( fread(&BmpInfoHeader, sizeof(BITMAPINFOHEADER) , 1 , fp) < 1){
        perror("fread");
        exit(1);
    }
	if ( fread(&color,1024,1,fp) < 1 ){
        perror("fread");
        exit(1);
    }
	for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
			if ( fread(&img[m+n*nx],1,1,fp) < 1){
                perror("fread");
                exit(1);
            }
		}
	}
}

void bmpread(const char *fname,std::unique_ptr<uint8_t[]>& img,int ny,int nx){
    bmpread(fname,img.get(),ny,nx);
}

// bmp write
void bmpwrite(const char *fname,uint8_t *img,int ny,int nx){
	BITMAPFILEHEADER BFH = {19778,14+40+1024+(unsigned int)nx*(unsigned int)ny,0,0,14+40+1024};
    BITMAPINFOHEADER BIH = {40,nx,ny,1,8,0,0,0,0,0,0};
    RGBQUAD RGB[256];
    
    for(int i=0;i<256;i++){
       RGB[i].rgbBlue=i;
       RGB[i].rgbGreen=i;
       RGB[i].rgbRed=i;
       RGB[i].rgbReserved=0;
	}
	FILE *fp;
    if ( (fp = fopen(fname,"wb")) == NULL){
        exit(1);
    };
    fwrite(&BFH,sizeof(BITMAPFILEHEADER),1,fp);
    fwrite(&BIH,sizeof(BITMAPINFOHEADER),1,fp);
    fwrite(RGB,sizeof(RGB),1,fp);
    fwrite(img,nx*ny,1,fp);
    fclose(fp);
}

void bmpwrite(const char *fname,std::unique_ptr<uint8_t[]>& img,int ny,int nx){
    bmpwrite(fname,img.get(),ny,nx);
}

void imwrite(const char *fname,std::unique_ptr<uint8_t[]>& img,int64_t ny,int64_t nx){
    cv::Mat mat(ny,nx,CV_8U,img.get());
    cv::flip(mat,mat,0);
    cv::imwrite(fname,mat);
}

// real number to 8bit
template<typename PREC_T>
void quant8bit(PREC_T* real, uint8_t *img,int height,int width){
    double max = real[0];
    double min = real[0];
    for (int h = 0; h < height; h++){
        for (int w = 0; w < width; w++){
            double tmp = real[h * width + w];
            if (tmp > max){
                max = tmp;
            }
            else if (tmp < min){
                min = tmp;
            }
        }
    }
    for (int h = 0; h < height; h++){
        for ( int w = 0; w < width; w++){
            img[h * width + w] = round( (real[h * width + w] - min) / (max - min) * 255 );
        }
    }
}

template<typename PREC_T>
void quant8bit(std::unique_ptr<PREC_T>& real, std::unique_ptr<uint8_t[]>& img,int height,int width){
    quant8bit(real.get(),img.get(),height,width);
}

// -pi < real < pi to 8bit
template<typename REAL_T>
void phase8bit(REAL_T* real, uint8_t *img,int height,int width){
    for (int h = 0; h < height; h++){
        for ( int w = 0; w < width; w++){
            img[h * width + w] = round( (real[h * width + w] - M_PI) / (2 * M_PI) * 255 );
        }
    }
}

template<typename PREC_T>
void phase8bit(std::unique_ptr<PREC_T>& real, std::unique_ptr<uint8_t[]>& img,int height,int width){
    phase8bit(real.get(),img.get(),height,width);
}


template<typename COMPLEX_T,typename PHASE_T>
void CaliculatePhase(std::unique_ptr<std::complex<COMPLEX_T>[]>& u,std::unique_ptr<PHASE_T[]>& phase,
                        int height, int width){
    for (int h = 0; h < height;h++){
        for (int w = 0;w < width;w++){
            int idx = h * width + w;
            phase[idx] = atan2(u[idx].imag(),u[idx].real());
        }
    }
}

/*画像charを数値処理してfftw_complexへ*/
template<typename PREC_T>
void img2complex(std::unique_ptr<uint8_t[]>& img, std::unique_ptr<std::complex<PREC_T>[]>& u,
                    int height, int width,bool phasemode = true)
{
    uint8_t max,min;
	min = img[0];
    max = 0;
	for(int n = 0; n < height;n++){
		for(int m = 0; m < width;m++){
			if(img[m + n * width] > max)
				max = img[m + n * width];
			if(img[m + n * width] < min)
				min=img[m + n * width];
		}
	}
	/*1に正規化しbufに格納
    complexに変換*/
    if (phasemode == true){
        for(int n = 0;n < height;n++){
		    for(int m = 0;m < width;m++){ 
                PREC_T buf = (PREC_T)(img[m + n * width] - min)/(max - min);
                u[m + n * width] = std::complex<PREC_T>(cos(2 * M_PI * buf),sin(2 * M_PI * buf));
		    }
	    }
    }
    else{
        for(int n = 0;n < height;n++){
		    for(int m = 0;m < width;m++){
                PREC_T buf = (PREC_T)(img[m + n * width] - min)/(max - min);
                u[m + n * width] = std::complex<PREC_T>(buf,0);
		    }
	    }
    }
}

enum MODE{PHASE,REAL,IMAG,AMP};
template<typename PREC_T>
void img2complex(std::unique_ptr<uint8_t[]>& img, std::unique_ptr<std::complex<PREC_T>[]>& u,
                    int height, int width,MODE mode)
{
    if (u.get() == NULL){
        u = std::make_unique<std::complex<PREC_T>[]>(height * width);
    }
    uint8_t max,min;
    if (mode == PHASE){
        min = 0;
        max = 255;
    }
    else{
        min = img[0];
        max = 0;
        for(int n = 0; n < height;n++){
            for(int m = 0; m < width;m++){
                if(img[m + n * width] > max)
                    max = img[m + n * width];
                if(img[m + n * width] < min)
                    min=img[m + n * width];
            }
        }
    }
	/*1に正規化しbufに格納
    complexに変換*/
    for(int n = 0;n < height;n++){
        for(int m = 0;m < width;m++){ 
            if (mode == PHASE){
                PREC_T buf = (PREC_T)(img[m + n * width] - min)/(max - min);
                u[m + n * width] = std::complex<PREC_T>(cos(2 * M_PI * buf),sin(2 * M_PI * buf));
            }
            else if (mode == REAL){
                PREC_T buf = (PREC_T)(img[m + n * width] - min)/(max - min);
                u[m + n * width] = std::complex<PREC_T>(buf,0);
            }
            else if (mode == IMAG){
                PREC_T buf = (PREC_T)(img[m + n * width] - min)/(max - min);
                u[m + n * width] = std::complex<PREC_T>(0,buf);
            }
        }
    }
}

/*fftw_complexから振幅計算してcharへ*/
template<typename PREC_T>
void complex2img(std::unique_ptr<std::complex<PREC_T>[]>& u,std::unique_ptr<uint8_t[]>& img,
                int ny,int nx,bool realmode = false)
{
    double *img_double = new double[ny*nx];
    for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
            if ( realmode == true){
                img_double[m+n*nx] = u[m+n*nx].real();
            }
            else{
                img_double[m+n*nx] = sqrt(u[m+n*nx].real() * u[m+n*nx].real() + u[m+n*nx].imag() * u[m+n*nx].imag());
            }
        }
    }
    double max,min;
	min = img_double[0];
    max = 0;
	for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
			if(img_double[m+n*nx]>max)
				max=img_double[m+n*nx];
			if(img_double[m+n*nx]<min)
				min=img_double[m+n*nx];
		}
	}
	for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
			img[m+n*nx] = std::round( ((255*(img_double[m+n*nx]-min))/(max-min)) );
		}
	}
    delete[] img_double;
}

template<typename PREC_T>
void complex2img(std::unique_ptr<std::complex<PREC_T>[]>& u,std::unique_ptr<uint8_t[]>& img,
                int ny,int nx,MODE mode)
{
    double *img_double = new double[ny*nx];
    for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
            if ( mode == REAL){
                img_double[m+n*nx] = u[m+n*nx].real();
            }
            else if ( mode == AMP){
                img_double[m+n*nx] = sqrt(u[m+n*nx].real() * u[m+n*nx].real() + u[m+n*nx].imag() * u[m+n*nx].imag());
            }
            else if ( mode == IMAG){
                img_double[m+n*nx] = u[m+n*nx].imag();
            }
            else if (mode == PHASE){
                img_double[m+n*nx] = atan2(u[m+n*nx].imag(),u[m+n*nx].real());
            }
        }
    }
    double max,min;
    if (mode == PHASE){
        min = -M_PI; max = M_PI;
    }
    else{
        min = img_double[0];
        max = img_double[0];
        for(int n=0;n<ny;n++){
            for(int m=0;m<nx;m++){
                if(img_double[m+n*nx]>max)
                    max=img_double[m+n*nx];
                if(img_double[m+n*nx]<min)
                    min=img_double[m+n*nx];
            }
        }
    }
	for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
			img[m+n*nx] = std::round( ((255*(img_double[m+n*nx]-min))/(max-min)) );
		}
	}
    delete[] img_double;
}

template<typename PREC_T>
void complex2real(std::unique_ptr<std::complex<PREC_T>[]>& u,std::unique_ptr<PREC_T[]>& real,
                int ny,int nx,MODE mode)
{
    for(int n=0;n<ny;n++){
		for(int m=0;m<nx;m++){
            if ( mode == REAL){
                real[m+n*nx] = u[m+n*nx].real();
            }
            else if ( mode == AMP){
                real[m+n*nx] = sqrt(u[m+n*nx].real() * u[m+n*nx].real() + u[m+n*nx].imag() * u[m+n*nx].imag());
            }
            else if ( mode == IMAG){
                real[m+n*nx] = u[m+n*nx].imag();
            }
            else if (mode == PHASE){
                real[m+n*nx] = atan2(u[m+n*nx].imag(),u[m+n*nx].real());
            }
        }
    }
}

template<typename PREC_T>
std::unique_ptr<PREC_T[]> complex2real(std::unique_ptr<std::complex<PREC_T>[]>& u,int ny,int nx,MODE mode){
    std::unique_ptr<PREC_T[]> real = std::make_unique<PREC_T[]>(ny * nx);
    ol::complex2real(u,real,ny,nx,mode);
    return real;
}

template<typename PREC_T>
std::unique_ptr<uint8_t[]> complex2img(std::unique_ptr<std::complex<PREC_T>[]>& u,
                int ny,int nx,MODE mode){
    auto imgtmp = std::make_unique<uint8_t[]>(ny * nx);
    ol::complex2img(u,imgtmp,ny,nx,mode);
    return imgtmp;
}

template<typename COMPLEX_T>
void Save(const char* path,std::unique_ptr<std::complex<COMPLEX_T>[]>& u,int height, int width,
        MODE savemode = MODE::PHASE){
    auto imgtmp = std::make_unique<uint8_t[]>(width * height);
    ol::complex2img(u,imgtmp,height,width,savemode);
    bmpwrite(path,imgtmp,height,width);
}

void Save(const char* path,std::unique_ptr<float[]>& real,int64_t ny, int64_t nx,
        MODE savemode = MODE::PHASE){
    auto img = std::make_unique<uint8_t[]>(ny * nx);
    ol::quant8bit(real,img,ny,nx);
    bmpwrite(path,img,ny,nx);
}

#ifdef __NVCC__
template<typename COMPLEX_T>
void Save(const char* path,cuda::unique_ptr<thrust::complex<COMPLEX_T>[]>& u,int height, int width,
        MODE savemode = MODE::PHASE){
    std::unique_ptr<std::complex<COMPLEX_T>[]> h_buf;
    ol::cuda2cpu(u,h_buf,height * width);
    Save(path,h_buf,height,width,savemode);
}
#endif

// opencv
template<typename T>
std::unique_ptr<T[]> cvMat2sptr(cv::Mat mat){
    std::unique_ptr<uint8_t[]> sptr;
    int height = mat.rows; int width = mat.cols;
    sptr = std::make_unique<uint8_t[]>(height * width);
    // if (mat.isContinuous()){
        for(int h = 0;h < height;h++){
            for (int w = 0;w < width;w++){
                sptr[h * width + w] = mat.at<T>(h,w);
            }
        }
        mat.release();
    // }
    // else{
    //     printf("cannot convert mat to smart ptr");
    //     exit(1);
    // }
    return sptr;
}

}
#endif