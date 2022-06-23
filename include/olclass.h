#ifndef OLCLASS_H
#define OLCLASS_H
#include <memory>
#include <chrono>
#include <iostream>
namespace ol{

#define OL_32F 0
#define OL_64F 1
#define OL_32COMPLEX 2
#define OL_64COMPLEX 3
#define OL_8U 4
#define OL_8S 5

template<typename _Tp>
struct Mat{
    private:
    public:
        std::unique_ptr<_Tp[]> data;
        int height,width;
        int type;
        Mat(int height,int width,int type = OL_64COMPLEX){
            this->height = height; this->width = width; this->type = type;
        };
        Mat(){};
};
// template <typename _Tp = double>
struct Plane {
    private:
    public:
        int height,width;
        double dy,dx;
        void set(int height, int width, double dy, double dx){
            this->height = height; this->width = width; this->dy = dy; this->dx = dx;
        }
        Plane(int height, int width, double dy, double dx){
            set(height,width,dy,dx);
        };
        Plane(){};
};



template<typename _Tp>
struct Vect3{
    public:
        _Tp x,y,z;
        Vect3(_Tp x, _Tp y,_Tp z){
            this->x = x; this->y = y; this->z = z;
        }
        Vect3(){}
};

template<typename _Tp>
struct Object{
    private:
        // int idx = 0;
    public:
        int size;
        std::unique_ptr<Vect3<_Tp>[]> points;
        Object(){};
        Object(int N){
            this->points = std::make_unique<Vect3<_Tp>[]>(N);
            this->size = N;
        }
        Vect3<_Tp> & operator [](int n) { return points[n]; }
        // Vect3<_Tp>& operator *() {return &points;}
        // Object& operator +(int n) {idx += n; return this;}
        // Object& operator ++() {idx++; return points;}
}; 

class ProcessingTime{
    public:
        std::chrono::system_clock::time_point  start_time, end_time;
        bool startflag = false;
        void start(){
            if (startflag == false){
                start_time = std::chrono::system_clock::now();
            }
        }
        void end(){
            end_time = std::chrono::system_clock::now();
        }
        void print(){
            auto time = end_time - start_time;
            auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
            std::cout << msec << " msec" << std::endl;
        }
};


}

#endif