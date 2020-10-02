//
//  main.cpp
//

// include opencv and standard headers
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Header.h"

using namespace std;
using namespace cv;


#include <iostream>

int main(int argc, const char * argv[]) {

    // read image
    Mat input= imread( argv[1] );
    
    /*****DETECTION OF THE EYE*****/
    //vector containing the detected faces
    vector<Rect> Faces;
    //vector containing the detected eyes
    vector<vector<Rect>> Eyes;
    
    
    /*****DETECTION OF THE IRIS*****/
    vector<vector<Vec3f>> Circles;
        
    /*****DETECTION OF THE CORNERS*****/
    vector<vector<pair<Point,Point>>> Corners;

    
    /*****CALL THE FUNCTIONS*****/
    
    EyeDetection(input, Faces, Eyes);
    
    MaxEyes(Faces, Eyes);
        
    Iris(input, Faces, Eyes, Circles);

    CornerEye(input, Faces, Eyes, Circles, Corners);
 
    Mat output=Gaze(input, Faces, Eyes, Circles, Corners);
    
    Draw(input, output, Faces, Eyes, Circles, Corners);
    
    return 0;
}

