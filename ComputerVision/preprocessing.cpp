//
//  preprocessing.cpp
//  Project
//


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Header.h"

using namespace std;
using namespace cv;



/*****SINGLE PIXEL OPERATIONS*****/

//function that computes the histogram of a grayscale image
void Histogram(Mat input, Mat& hist){
    
    //hist size
    const int histSize = 256;
    
    //ranges: array of array with min and max
    //min=0 max=255
    //array with max and min
    float r[2]={0, 255};
    //array of array with max and min (pointers)
    const float* ranges = {r};
    
    //computing histogram
    calcHist(&input, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);

    
    return;
}

//function that equalize the channel L of the image
void EqualizationHist(Mat input, Mat& equalized){
    
    Mat Lab;
    cvtColor(input,Lab,CV_BGR2Lab);
    
    //splitting the input image in its channels
    vector<Mat> chs;
    split(Lab,chs);
    
    //equalizing the channel
    equalizeHist(chs[0], chs[0]);
    
    //merging the equalized channel in the eq image
    Mat merged;
    merge(chs,merged);

    //convert to color space 
    cvtColor(merged,equalized,CV_Lab2BGR);
    
    return;
}

//change the range resolution

void RangeResolution(Mat& input, int N){
    
    //to keep the right domain
    int D = 256/N;
    
    for(int i=0; i<input.cols; ++i){
        for(int j=0; j<input.rows; ++j){
            input.at<uchar>(i,j) = D*floor(input.at<uchar>(i,j)/D);
        }
    }
    
    return;
}


/*****THRESHOLD COMPUTATION*****/

//function that compute the perc of black pixels
double Frequency(int P, Mat input){
    Mat hist;
    Histogram(input, hist);
    return hist.at<float>(0)/P*100;
}


//looking for the best treshold
double Threshold(Mat input, double perc){
    
    //number of pixels
    int P=input.total();
    
    //look for the minimum of the matrix
    double min, max;
    minMaxLoc(input, &min, &max);
    
    
    //define an initial value of the frequecy by thresholding the image with the maximum value of the image
    double t=max;
    //temporary matrix to compute the initial frequency
    Mat tempMax;
    //apply to the image the initial threshold
    threshold(input, tempMax, t, 255, THRESH_TRUNC);
    threshold(tempMax, tempMax, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //initial frequency
    double freq=Frequency(P, tempMax);
    
    //looking for the threshold
    while(freq>perc) {
        Mat temp;
        t -= 1;
        threshold(input, temp, t, 255, THRESH_TRUNC);
        threshold(temp, temp, 0, 255, THRESH_BINARY | THRESH_OTSU);
        freq = Frequency(P, temp);
    }

    return t+1;
}



/*****PREPROCESSING*****/

void Preprocessing(Mat input, Mat& preprocessed){
    
    /*****INITIAL PREPROCESSING*****/
    
    //the hist of the image is equalixed in L
    Mat eq;
    EqualizationHist(input, eq);
    
    //convert to grayscale
    cvtColor(eq, eq, CV_BGR2GRAY);
    
    //apply a contrast to the image
    Mat contrast;
    eq.convertTo(contrast, -1, 1.5, 30);
    
    //reducing the range resolution
    RangeResolution(contrast, 100);
    
    
    /*****THRESHOLDING*****/
    
    //computing the optimal threshold in order to perform segmentation
    double t=Threshold(contrast, 15);
    
    //apply a first threshold
    Mat thres;
    threshold(contrast, thres, t, 255, THRESH_TRUNC);

    //apply morphological operator
    morphologyEx(thres,thres,MORPH_ERODE,getStructuringElement(MORPH_ELLIPSE, Size(2,2)));
    
    //apply contrast
    Mat contrastbis = Mat::zeros(thres.size(),thres.type());
    thres.convertTo(contrastbis, -1, 2, 30);
    
    //reducing the range resolution
    RangeResolution(contrastbis, 15);
    
    //apply multiple thresholds
    double min, max;
    minMaxLoc(contrastbis, &min, &max);
    threshold(contrastbis, contrastbis, max-1, 255, THRESH_TRUNC);
    minMaxLoc(contrastbis, &min, &max);
    threshold(contrastbis, contrastbis, max-255/15-1, 255, THRESH_TRUNC);

    //image on which the circle are looked for
    preprocessed = contrastbis;

    return;
}
