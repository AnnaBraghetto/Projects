//
//  detection.cpp
//  Project
//
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "Header.h"

using namespace std;
using namespace cv;


//detection of the eyes
void EyeDetection(Mat input, vector<Rect>& Faces, vector<vector<Rect>>& Eyes) {
    
    //directory where the petrained models are stored
    string Dir = "/usr/local/opt/opencv@3/share/OpenCV/haarcascades/";
    
    //cascade classifier to detect faces in the image
    CascadeClassifier FaceCasc;
    //cascade classifier to detect eyes in the faces
    CascadeClassifier EyeCasc;
    
    //load the petrained models
    
    //face model
    string FaceModel = Dir + "haarcascade_frontalface_alt.xml";
    FaceCasc.load(FaceModel);
    //eye model
    string EyeModel = Dir + "haarcascade_eye.xml";
    EyeCasc.load(EyeModel);
    
    
    /*****DETECTION*****/
    
    
    //detection of the faces in the image
    FaceCasc.detectMultiScale(input, Faces);
    
    
    //for each detected face the detection of the eyes is performed
    for (int i=0; i<Faces.size(); ++i) {
        //detected eyes
        vector<Rect> TempEyes;
        //detection of the eyes in the face
        EyeCasc.detectMultiScale(input(Faces[i]), TempEyes);
        Eyes.push_back(TempEyes);
    }

    return;
}


//defining a max number of eyes
 void MaxEyes(vector<Rect>& Faces, vector<vector<Rect>>& Eyes){
 
     //just keep the two biggest eyes
     for(int i=0; i<Faces.size(); ++i){
         vector<pair<int,int>> Dimension;
         vector<Rect> NewEyes;
         for(int j=0; j<Eyes[i].size(); ++j){
             //in case of two eyes detected continue the cycle
             if(Eyes[i].size() <= 2) continue;
             else {
                 Dimension.push_back(make_pair(Eyes[i][j].width+Eyes[i][j].height,j));
             }
         }
         
         if(Eyes[i].size() <= 2) continue;
         else {
             //sort
             sort(Dimension.begin(), Dimension.end());
             for(int j=1; j<3; ++j){
                 NewEyes.push_back(Eyes[i][Dimension[Dimension.size()-j].second]);
             }
             Eyes[i] = NewEyes;
         }
     }
 return;
 }


//detection of the iris
void Iris(Mat input, vector<Rect>& Faces, vector<vector<Rect>>& Eyes, vector<vector<Vec3f>>& Circles){
    
    //look for all the iris
    for(int i=0; i<Faces.size(); ++i){
        vector<Vec3f> SingleEyeCircle;
        for(int j=0; j<Eyes[i].size(); ++j){
            
            //for each eye, the image is cropped inside the region containing the eye
            Mat crop = input(Faces[i]);
            crop = crop(Eyes[i][j]);
            
            //preprocessing on the image
            Mat preprocessed;
            Preprocessing(crop, preprocessed);

            //looking for edges
            Mat edges;
            Canny(preprocessed, edges, 70, 100, 3);

            //look for circles for each eye
            vector<Vec3f> TempCircle;
            
            //10 = number of edge points are needed to computed the circle
            //                                                                  min r, max r
            HoughCircles(edges, TempCircle, HOUGH_GRADIENT, 2, 1, 1, 10, edges.rows/8, edges.rows/4);
            
            //choose the best circle
            SingleEyeCircle.push_back(ChoiceCircles1Eye(preprocessed, TempCircle));
        }
        
        //filling the circle vector
        Circles.push_back(SingleEyeCircle);
    }
    
    return;
}

//rotation of the eye region upon the inclination of the head
void Rotation(vector<Vec3f> Circles, double& Angle, Rect Face, vector<Rect> Eyes){
    
    //compute the inclination angle
    vector<Point> center;
    for(int i=0; i<Circles.size(); ++i) {
        Vec3f C = Circles[i];
        center.push_back(Point(C[0]+Face.tl().x+Eyes[i].tl().x,C[1]+Face.tl().y+Eyes[i].tl().y));
    }
    
    if(Circles.size()>1){
        double m;
        if(center[1].x>center[0].x){
            m=(float)(center[1].y-center[0].y)/(float)(center[1].x-center[2].x);
        }
        else m=(float)(center[0].y-center[1].y)/(float)(center[0].x-center[1].x);
        Angle = atan(m)*180/CV_PI;
    }

    else Angle=0;
    
    return;
}

//extreme of the corners
void Extrema(Mat input, Vec3f Circle, pair<Point,Point>& coor){
    
    //preprocessing the image to perform a good corner detection
    RangeResolution(input, 20);
    threshold(input, input, Threshold(input, 25), 255, THRESH_TRUNC);

    //perform Harris corners detection
    Mat corners;
    cornerHarris(input, corners, 5, 1, 0.05, BORDER_DEFAULT );
    
    /// Normalizing
    normalize( corners, corners, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( corners, corners );
    
    //to threshold
    corners.convertTo(corners, CV_32F);

    //threshold the image
    double M;
    Mat hist;
    double min, max;
    Point minLoc, maxLoc;
    //looking for thresholds
    M=mean(corners)[0];
    Histogram(corners, hist);
    minMaxLoc(hist, &min, &max, &minLoc, &maxLoc);
    //thresholding the matrix
    threshold(corners, corners, maxLoc.y, 0, THRESH_TOZERO);
    threshold(corners, corners, M, 0, THRESH_TOZERO);
    
    //extrema
    int X1=input.cols,X2=0,Y1=0,Y2=0;
    
    //lim
    double lim = (input.rows+input.cols)/12;


    for( int i=0; i<corners.cols; ++i ) {
        for( int j=0; j<corners.rows ; ++j ) {
            if( corners.at<float>(j,i) > 0  && corners.at<float>(j,i) > M  && corners.at<float>(j,i) > maxLoc.y){
                if(i<X1 && fabs(Circle[1]-j)<lim) {
                    X1=i;
                    Y1=j;
                }
                if(i>X2 && fabs(Circle[1]-j)<lim) {
                    X2=i;
                    Y2=j;
                }
            }
        }
    }
    
    //store the results
    if(X1<X2) coor=make_pair(Point(X1,Y1),Point(X2,Y2));
    else coor=make_pair(Point(X2,Y2),Point(X1,Y1));

    return;    
}


//find the eye corners
void CornerEye(Mat input, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles, vector<vector<pair<Point,Point>>>& Corners){

    Mat temp = input.clone();
    cvtColor(temp, temp, COLOR_BGR2GRAY);
    
    for(int i=0; i<Faces.size(); ++i){
        double Angle;
        vector<pair<Point,Point>> FaceCorners;

        Rotation(Circles[i], Angle, Faces[i], Eyes[i]);
        Mat rotation = getRotationMatrix2D(Point(0.0), Angle, 1.0);
        
        for(int j=0; j<Eyes[i].size(); ++j){
        
            pair<Point, Point> EyeCorners;
            
            //for each eye, the image is cropped inside the region containing the eye
            Mat crop = temp(Faces[i]);
            crop = crop(Eyes[i][j]);

            //rotation of the matrix
            if(fabs(Angle)>5) {
                //set the background
                Scalar color = Scalar(mean(crop)[0]);
                warpAffine(crop, crop, rotation, crop.size(),INTER_LINEAR,BORDER_CONSTANT,color);
            }
            
            Vec3f C=Circles[i][j];
            Extrema(crop, C, EyeCorners);
            
            //rotation of points
            if(fabs(Angle)>5) {
                Angle *= -1*CV_PI/180;
                Point L = EyeCorners.first;
                Point R = EyeCorners.second;
                double tempX = (L.x*cos(Angle) + L.y*sin(Angle));
                double tempY = ((-L.x)*sin(Angle) + L.y*cos(Angle));
                EyeCorners.first=Point(tempX,tempY);
                tempX = (R.x*cos(Angle) + R.y*sin(Angle));
                tempY = ((-R.x)*sin(Angle) + R.y*cos(Angle));
                EyeCorners.second=Point(tempX,tempY);
                Angle *= -1*180/CV_PI;
                }
            
            FaceCorners.push_back(EyeCorners);
        }
        
        Corners.push_back(FaceCorners);
    }
    
return;
}

//gaze detection
Mat Gaze(Mat input, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles, vector<vector<pair<Point,Point>>>& Corners) {
    
    string direction;
    Mat output=input.clone();
    double scale = (input.rows+input.cols)/400;
    
    //look for all the rest iris position
    for(int i=0; i<Faces.size(); ++i){
        //to determine direction
        double check=0.0;
        vector<double> G;
        for(int j=0; j<Eyes[i].size(); ++j){
            //to normalize
            int N = Eyes[i][j].width*Eyes[i][j].height;
            
            //looking for gaze direction
            Vec3f C=Circles[i][j];
            pair<Point,Point> EyeCorners = Corners[i][j];
            double L = norm(EyeCorners.first-Point(C[0],C[1]));
            double R = norm(EyeCorners.second-Point(C[0],C[1]));

            if(L>R) G.push_back(1*L);
            else G.push_back(-1*R);
            
            //check the direction
            check+=(fabs(L-R)/N*100)/C[2]*100;
            
        }
        //drawing the detected objects on the image
        Point left = Point(Faces[i].tl().x + Faces[i].width/4, Faces[i].tl().y +Faces[i].height/10);
        Point right = Point(Faces[i].tl().x + 3*Faces[i].width/4, Faces[i].tl().y +Faces[i].height/10);
        Point center = Point(Faces[i].tl().x + Faces[i].width/2, Faces[i].tl().y +Faces[i].height/10);
        check /= Eyes[i].size();
        double gazeDir=0.0;
        for(int k=0; k<G.size(); ++k){
            gazeDir += G[i];
        }
        gazeDir /= G.size();
        if(check>3 && gazeDir!=0){
            if(gazeDir>0){
                arrowedLine(output, left, right, Scalar(200,20,20),scale);
                direction="RIGHT";
            }
            if(gazeDir<0) {
                arrowedLine(output, right, left, Scalar(20,200,20),scale);
                direction="LEFT";
            }
        }
        else {
            circle(output, center, 2, Scalar(20,20,200),scale*10);
            direction="STRAIGHT";
        }
    }
    
    cout <<"The direction is: " <<direction <<endl;
    
    return output;
}
