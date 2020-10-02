//
//  draw.cpp
//  Project
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


void Draw(Mat input, Mat output, vector<Rect> Faces, vector<vector<Rect>> Eyes, vector<vector<Vec3f>> Circles,vector<vector<pair<Point,Point>>> Corners){
    
    /*****INPUT*****/
    namedWindow( "Input Image", CV_WINDOW_AUTOSIZE );
    imshow( "Input Image", input );
    waitKey(0);
    
    destroyAllWindows();
    
    /*****DETECTION*****/
    Mat detection = input.clone();
    //draw the region in which there are the detected objects
    for (int i=0; i<Faces.size(); ++i) {
        //drawing the detected objects on the image
        rectangle(detection, Faces[i].tl(), Faces[i].br(), Scalar( 255, 255, 0 ),3);
        for(int j=0; j<Eyes[i].size(); ++j){
            rectangle(detection, Faces[i].tl()+Eyes[i][j].tl(), Faces[i].tl()+Eyes[i][j].br(), Scalar( 255, 0, 0 ), 3);
        }
    }
    
    //show the detection
    namedWindow("Detected Face and Eyes", CV_WINDOW_AUTOSIZE );
    imshow("Detected Face and Eyes", detection );
    //to save the image with rect around face and eyes
    imwrite("FaceEye.jpg",detection);
    waitKey(0);
    
    destroyAllWindows();
    
    /*****IRIS*****/
    Mat iris = input.clone();
    //draw the circles of the iris on the image
    for (int i=0; i<Faces.size(); ++i) {
        for(int j=0; j<Eyes[i].size(); ++j){
            //shift in order to have the circles in the right position
            double trX = Faces[i].tl().x+Eyes[i][j].tl().x;
            double trY = Faces[i].tl().y+Eyes[i][j].tl().y;
            //plot the circle
            Point origin = Point(trX+Circles[i][j][0], trY+Circles[i][j][1]);
            int radius = Circles[i][j][2];
            if(radius > 0) circle(iris, origin, radius, Scalar(255,0,255),3);
            //plot the corner
            pair<Point,Point> Cor = Corners[i][j];
            Point L = Point(trX+Cor.first.x, trY+Cor.first.y);
            Point R = Point(trX+Cor.second.x, trY+Cor.second.y);
            circle(iris, L, 2, Scalar(255,255,0), 2);
            circle(iris, R, 2, Scalar(255,255,0), 2);
        }
    }
    
    //show the iris
    namedWindow("Detected Iris", CV_WINDOW_AUTOSIZE );
    imshow("Detected Iris", iris );
    //to save the image with rect around face and eyes
    imwrite("Iris.jpg",iris);
    waitKey(0);

    /*****OUTPUT*****/
    
    //show the detection
    namedWindow("Detected Gaze", CV_WINDOW_AUTOSIZE );
    imshow("Detected Gaze", output );
    //to save the image with rect around face and eyes
    imwrite("DetectedGaze.jpg",output);
    waitKey(0);
    destroyAllWindows();
    
}
