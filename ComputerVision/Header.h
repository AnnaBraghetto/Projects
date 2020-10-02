//
//  Header.h
//  Project
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>


using namespace std;
using namespace cv;


#ifndef Header_h
#define Header_h


/*****PREPROCESSING*****/

//compute hist
void Histogram(Mat input, Mat& hist);
//equalize the hist in L
void EqualizationHist(Mat input, Mat& equalized);

//change range resolution
void RangeResolution(Mat& input, int N);
//compute the freq of balck pixels
double Frequency(int P, Mat input);
// best treshold
double Threshold(Mat input, double perc);

//perform preprocessing
void Preprocessing(Mat input, Mat& preprocessed);

/*****DETECTION*****/

//UTILS

//function that compute the cropped
Mat Crop(Mat input, Vec3f C);
//choice of the circle
Vec3f ChoiceCircles1Eye(Mat input, vector<Vec3f> SingleEyeCircle);

//EYE RECOGNITION

//detection of the eyes
void EyeDetection(Mat input, vector<Rect>& Faces, vector<vector<Rect>>& Eyes);
//defining a max number of eyes
void MaxEyes(vector<Rect>& Faces, vector<vector<Rect>>& Eyes);
//detection of the iris
void Iris(Mat input, vector<Rect>& Faces, vector<vector<Rect>>& Eyes, vector<vector<Vec3f>>& Circles);
//rotation of the eye region upon the inclination of the head
void Rotation(vector<Vec3f> Circles, double& Angle, Rect Face, vector<Rect> Eyes);
//extreme of the corners
void Extrema(Mat input, Vec3f Circle, pair<Point,Point>& coor);
//find the eye corners
void CornerEye(Mat input, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles, vector<vector<pair<Point,Point>>>& Corners);

//GAZE

//gaze detection
Mat Gaze(Mat input, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles, vector<vector<pair<Point,Point>>>& Corners);

/*****DRAW*****/
void Draw(Mat input, Mat output, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles,vector<vector<pair<Point,Point>>> Corners);

#endif /* Header_h */
