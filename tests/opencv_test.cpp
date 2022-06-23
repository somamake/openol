#include <opencv2/opencv.hpp>
#include "openol.h"
#ifndef PROJECT_ROOT
#define PROJECT_ROOT
#endif

namespace ol{

// template<typename T>
// std::unique_ptr<T[]> cvMat2sptr(cv::Mat mat){
//     std::unique_ptr<uint8_t[]> sptr;
//     int height = mat.rows; int width = mat.cols;
//     sptr = std::make_unique<uint8_t[]>(height * width);
//     if (mat.isContinuous()){
//         for(int h = 0;h < height;h++){
//             for (int w = 0;w < width;w++){
//                 sptr[h * width + w] = mat.at<T>(h,w);
//             }
//         }
//         mat.release();
//     }
//     else{
//         printf("cannot convert mat to smart ptr");
//         exit(1);
//     }
//     return sptr;
// }
// void imwrite(const char *fname,std::unique_ptr<uint8_t[]>& img,int64_t ny,int64_t nx){
//     cv::Mat mat(ny,nx,CV_8U,img.get());
//     cv::flip(mat,mat,0);
//     cv::imwrite(fname,mat);
// }
}
int main(){
    int width = 1920; int height = 1080;
    cv::Mat img = cv::imread(PROJECT_ROOT "/data/DIV2K/0824.png");
    cv::flip(img, img, 0);
    cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(), (float)width/img.cols ,(float)width/img.cols,cv::INTER_AREA);
    cv::Mat cut_img(img, cv::Rect(0, (img.rows - height)/2, width, height));
    cv::imwrite(PROJECT_ROOT "/out/tmp.png",cut_img);
    
    // std::unique_ptr<uint8_t[]> sptr;
    // sptr = std::make_unique<uint8_t[]>(height * width);
    // if (cut_img.isContinuous()){
    //     for(int h = 0;h < height;h++){
    //         for (int w = 0;w < width;w++){
    //             sptr[h * width + w] = cut_img.at<unsigned char>(h,w);
    //         }
    //     }
    //     cut_img.release();
    // }
    auto sptr = ol::cvMat2sptr<uint8_t>(cut_img);
    ol::imwrite(PROJECT_ROOT "/out/tmp.bmp",sptr,height,width);
}