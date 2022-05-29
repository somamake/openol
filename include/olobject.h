#ifndef OLOBJECT_H
#define OLOBJE_H
#include "olclass.h"

namespace ol{



template<typename _Tp>
ol::Object<_Tp> objread_3d(const char* path){
    FILE *fp;
    int N;
    if ( (fp = fopen(path,"rb")) == NULL){
        perror("fopen");
        exit(1);
    }
    if ( fread(&N,4,1,fp) < 1){
        perror("fread N");
        exit(1);
    }
    ol::Object<_Tp> obj(N);
    int32_t xint, yint,zint;
    for (int n = 0; n < N;n++){
        if ( fread(&xint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&yint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&zint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        obj[n].x = xint;
        obj[n].y = yint;
        obj[n].z = zint;
    }
    fclose(fp);
    return obj;
}

template<typename _Tp>
ol::Object<_Tp> objread_3df(const char* path){
    FILE *fp;
    int N;
    if ( (fp = fopen(path,"rb")) == NULL){
        perror("fopen");
        exit(1);
    }
    if ( fread(&N,4,1,fp) < 1){
        perror("fread N");
        exit(1);
    }
    ol::Object<_Tp> obj(N);
    float xint, yint,zint;
    for (int n = 0; n < N;n++){
        if ( fread(&xint,sizeof(float),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&yint,sizeof(float),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&zint,sizeof(float),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        obj[n].x = xint;
        obj[n].y = yint;
        obj[n].z = zint;
    }
    fclose(fp);
    return obj;
}

template<typename _Tp>
ol::Object<_Tp> objread(const char* path){
    const char *ext = strrchr(path, '.');
    if (strcmp(".3d", ext) == 0)
        return objread_3d<_Tp>(path);
    else if (strcmp(".3df", ext) == 0)
        return objread_3df<_Tp>(path);
    else{
        printf("error\n");
        exit(1);
    }
        
}

template<typename _Tp,typename OFFSET_T, typename SCALE_T>
ol::Object<_Tp> objread(const char* path,OFFSET_T *offset, SCALE_T scale){
    FILE *fp;
    int N;
    if ( (fp = fopen(path,"rb")) == NULL){
        perror("fopen");
        exit(1);
    }
    if ( fread(&N,4,1,fp) < 1){
        perror("fread N");
        exit(1);
    }
    ol::Object<_Tp> obj(N);
    int32_t xint, yint,zint;
    for (int n = 0; n < N;n++){
        if ( fread(&xint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&yint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&zint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        obj[n].x = xint * scale + offset[0];
        obj[n].y = yint * scale + offset[1];
        obj[n].z = zint * scale + offset[2];
    }
    fclose(fp);
    return obj;
}

template<typename _Tp,typename OFFSET_T, typename SCALE_T>
ol::Object<_Tp> objread(const char* path,Vect3<OFFSET_T> offset, SCALE_T scale){
    FILE *fp;
    int N;
    if ( (fp = fopen(path,"rb")) == NULL){
        perror("fopen");
        exit(1);
    }
    if ( fread(&N,4,1,fp) < 1){
        perror("fread N");
        exit(1);
    }
    ol::Object<_Tp> obj(N);
    int32_t xint, yint,zint;
    for (int n = 0; n < N;n++){
        if ( fread(&xint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&yint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        if ( fread(&zint,sizeof(int32_t),1,fp) < 1){
            perror("fread");
            exit(1);
        }
        obj[n].x = xint * scale + offset.x;
        obj[n].y = yint * scale + offset.y;
        obj[n].z = zint * scale + offset.z;
    }
    fclose(fp);
    return obj;
}

template<typename _Tp>
void objinfo(ol::Object<_Tp>& obj){
    auto xminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.x < r.x; });
    auto yminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.y < r.y; });
    auto zminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.z < r.z; });
    std::cout << "xmax: " << xminmax.second->x << "  xmin: " << xminmax.first->x << std::endl;
    std::cout << "ymax: " << yminmax.second->y << "  ymin: " << yminmax.first->y << std::endl;
    std::cout << "zmax: " << zminmax.second->z << "  zmin: " << zminmax.first->z << std::endl;
}

template<typename _Tp>
void objsort(ol::Object<_Tp>& obj){
    std::sort(obj.points.get(), obj.points.get() + obj.size, [](const auto& l, const auto& r) {
    return l.z > r.z;
  });
}

// objを良い位置に自動でセットする．
// 1. 最大最小求める
// 2. objのx and yの範囲を指定した範囲 (width * p ?) にスケーリング
//    x,yの幅の大きな方を指定した距離に合わせる
//    この計算も外部でやる
// 3. z の最小値を画像幅からいい感じに合わす．
//    いい感じの距離の計算はまた違う関数にやらせてその結果をうけとるのが良い．
// 4. x yのオフセット合わせる
// 2,3,4一つのfor文でまとめて行える．

enum OBJSET_MODE{
    ZMIN,ZMIDDLE
};
template<typename _Tp>
void objset(ol::Object<_Tp>& obj, float xy_range,Vect3<_Tp> offset,OBJSET_MODE mode = ZMIN){
    auto xminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.x < r.x; });
    auto yminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.y < r.y; });
    auto zminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.z < r.z; });
    auto range_x = xminmax.second->x - xminmax.first->x;
    auto range_y = yminmax.second->y - yminmax.first->y;
    auto range_z = zminmax.second->z - zminmax.first->z;
    auto scale = (range_x > range_y) ? range_x : range_y;
    scale = xy_range / scale; 
    std::cout << "xmax: " << xminmax.second->x << "  xmin: " << xminmax.first->x << "  range" << range_x << std::endl;
    std::cout << "ymax: " << yminmax.second->y << "  ymin: " << yminmax.first->y << "  range" << range_y <<std::endl;
    std::cout << "zmax: " << zminmax.second->z << "  zmin: " << zminmax.first->z << "  range" << range_z << std::endl;

    offset.x -= (xminmax.second->x + xminmax.first->x) / 2 * scale;
    offset.y -= (yminmax.second->y + yminmax.first->y) / 2 * scale;
    if (mode == ZMIN){
        offset.z -= zminmax.first->z * scale;
    }
    else if (mode == ZMIDDLE){
         offset.z -= (zminmax.second->z + zminmax.first->z) / 2 * scale;
    }
    
    
    for (int j = 0; j < obj.size; j++){
        obj[j].x = obj[j].x * scale + offset.x;
        obj[j].y = obj[j].y * scale + offset.y;
        obj[j].z = obj[j].z * scale + offset.z;
    }
}

// x方向とy方向の最大幅を別々に指定する
// xとyの倍率のうち小さい方を採用する(大きい方を採用するのは間違い)．
template<typename _Tp>
void objset(ol::Object<_Tp>& obj, float x_range,float y_range,Vect3<_Tp> offset,OBJSET_MODE mode = ZMIN){
    auto xminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.x < r.x; });
    auto yminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.y < r.y; });
    auto zminmax = std::minmax_element(obj.points.get(),obj.points.get() + obj.size,
            [](const auto& l, const auto& r) { return l.z < r.z; });
    auto range_x = xminmax.second->x - xminmax.first->x;
    auto range_y = yminmax.second->y - yminmax.first->y;
    // auto range_z = zminmax.second->z - zminmax.first->z;
    auto scale_x = x_range / range_x;
    auto scale_y = y_range / range_y; 
    auto scale = (scale_x < scale_y) ? scale_x : scale_y;
    // std::cout << "xmax: " << xminmax.second->x << "  xmin: " << xminmax.first->x << "  range:" << range_x << std::endl;
    // std::cout << "ymax: " << yminmax.second->y << "  ymin: " << yminmax.first->y << "  range: " << range_y <<std::endl;
    // std::cout << "zmax: " << zminmax.second->z << "  zmin: " << zminmax.first->z << "  range: " << range_z << std::endl;

    offset.x -= (xminmax.second->x + xminmax.first->x) / 2 * scale;
    offset.y -= (yminmax.second->y + yminmax.first->y) / 2 * scale;
    if (mode == ZMIN){
        offset.z -= zminmax.first->z * scale;
    }
    else if (mode == ZMIDDLE){
         offset.z -= (zminmax.second->z + zminmax.first->z) / 2 * scale;
    }
    
    for (int j = 0; j < obj.size; j++){
        obj[j].x = obj[j].x * scale + offset.x;
        obj[j].y = obj[j].y * scale + offset.y;
        obj[j].z = obj[j].z * scale + offset.z;
    }
}

}
#endif