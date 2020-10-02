//
//  utils.cpp
//  Project
//
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Header.h"

using namespace std;
using namespace cv;


void Histogram(Mat input, Mat& hist);
void ShowHistogram(Mat hist);

//function that compute the cropped
Mat Crop(Mat input, Vec3f C){
   

    //creation of a mask corrisponding to the circle
    Mat mask = Mat::zeros(input.rows, input.cols, input.type());
    circle(mask,Point(C[0],C[1]),C[2],Scalar(255,255,255),-1,8,0);

    //apply the mask
    Mat crop;
    input.copyTo(crop, mask);

    //cut the rectangular region around the circle
    Rect cut = Rect(Point(C[0]-C[2],C[1]-C[2]),Point(C[0]+C[2],C[1]+C[2]));

    //condition on circles inside the eye region
    if(C[0]+C[2]<input.cols && C[1]+C[2]<input.rows && C[0]-C[2] > 0 && C[1]-C[2] > 0) {
        crop = crop(cut);
    }
    
    //if the circle is not inside the region is discarded
    else crop = Mat(Size(100,100), CV_8UC1, Scalar(255));
    
    return crop;
}


//choice of the circle
Vec3f ChoiceCircles1Eye(Mat input, vector<Vec3f> SingleEyeCircle){
    
    
    int N=SingleEyeCircle.size();

    //area weight
    vector<pair<int,int>> area;
    //white weight
    vector<pair<double,int>> white;

    //loooking for the max value
    for(int i=0; i<N; ++i){
        
        //computing the masked image
        Vec3f c = SingleEyeCircle[i];
        Mat crop=Crop(input, c);

        //to normalize area
        int P=crop.total();

        //computing the value of the area normalized
        area.push_back(make_pair(sum(crop)[0]/P, i));
        
        //computing frequency through hist
        Mat hist;
        Histogram(crop, hist);

        //max of the hist
        double min, max;
        minMaxLoc(crop, &min, &max);
        double freq = 0;
        //choice on the hist
        for(int j=0;j<2;++j){
            freq += hist.at<float>(max-j*15)/P*100;
        }
        white.push_back(make_pair(freq,i));
      
    }

    //sort the vectors
    sort(area.begin(), area.end());
    sort(white.begin(), white.end());
    
    //vector containing best area and best white frequency
    vector<pair<double,int>> Weight;
    
    //filling the vector
    for(int i=0; i<N; ++i){
        int j=0;
        while(white[j].second != area[i].second){j += 1;}
        if (white[j].first != 0) Weight.push_back(make_pair(j+i,area[i].second));
    }
    
    //sort
    sort(Weight.begin(), Weight.end());
    
    
    //best circle
    Vec3f iris;
    if(Weight.size()>0) {
        int best = Weight[0].second;
        iris = SingleEyeCircle[best];
    }
    else  {
        Vec3f u(0,0,0);
        iris = u;
    }
  
    return iris;
}

