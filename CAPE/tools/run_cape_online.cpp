/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "CAPE/CAPE.h"
#include "RSCamera.h"

bool done = false;
float COS_ANGLE_MAX = cos(M_PI/12);
float MAX_MERGE_DIST = 50.0f;
bool cylinder_detection= true;
CAPE * plane_detector;
std::vector<cv::Vec3b> color_code;

// void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d){
//     cv::Scalar red(0, 0, 255);
//     cv::Scalar green(0,255,0);
//     cv::Scalar blue(255,0,0);
//     cv::Scalar black(0,0,0);

//     cv::Point2i origin = list_points2d[0];
//     cv::Point2i pointX = list_points2d[1];
//     cv::Point2i pointY = list_points2d[2];
//     cv::Point2i pointZ = list_points2d[3];

//     drawArrow(image, origin, pointX, red, 9, 2);
//     drawArrow(image, origin, pointY, green, 9, 2);
//     drawArrow(image, origin, pointZ, blue, 9, 2);
//     cv::circle(image, origin, radius/2, black, -1, lineType );
// }

bool loadCalibParameters(std::string filepath, cv:: Mat & intrinsics_rgb, cv::Mat & dist_coeffs_rgb, cv:: Mat & intrinsics_ir, cv::Mat & dist_coeffs_ir, cv::Mat & R, cv::Mat & T){

    cv::FileStorage fs(filepath,cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["RGB_intrinsic_params"]        >> intrinsics_rgb;
        fs["RGB_distortion_coefficients"] >> dist_coeffs_rgb;
        fs["IR_intrinsic_params"]         >> intrinsics_ir;
        fs["IR_distortion_coefficients"]  >> dist_coeffs_ir;
        fs["Rotation"]                    >> R;
        fs["Translation"]                 >> T;
        fs.release();
        return true;
    }else{
        std::cerr << "Calibration file missing" << std::endl;
        return false;
    }
}

int main(int argc, char ** argv){   

    RSCamera* camera = new RSCamera();
    if(!camera->init(0)){
        std::cout << "Error inicializating camera \n";
        return -1;
    }
    for(int i = 0 ; i < 10 ; i++)
        camera->grab();

    // Create window
    cv::namedWindow("Segmentation",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("mask",cv::WINDOW_AUTOSIZE);
    cvStartWindowThread();

    cv::Mat intrinsics, coeffs;
    camera->leftCalibration(intrinsics, coeffs);

    // Get intrinsics
    cv::Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
    std::stringstream calib_path;
    calib_path << "/home/grvc/programming/CAPE/Data/pipe/calib_params_2.xml";
    loadCalibParameters(calib_path.str(), K_rgb, dist_coeffs_rgb, K_ir, dist_coeffs_ir, R_stereo, t_stereo);
    float fx_ir  = K_ir.at<double>(0,0);  float fy_ir  = K_ir.at<double>(1,1);
    float cx_ir  = K_ir.at<double>(0,2);  float cy_ir  = K_ir.at<double>(1,2);
    float fx_rgb = K_rgb.at<double>(0,0); float fy_rgb = K_rgb.at<double>(1,1);
    float cx_rgb = K_rgb.at<double>(0,2); float cy_rgb = K_rgb.at<double>(1,2);

    cv::Mat left, rigth;
    if(!camera->rgb(left, rigth)){
        std::cout << "Error getting first color frame \n";
        return -1;
    }

    cv::Mat depth;
    if(!camera->depth(depth)){
        std::cout << "Error getting first depth frame \n";
        return -1;
    }

    int width, height;
    if(left.data){
        width = left.cols;
        height = left.rows;
    }else{
        std::cout << "Error loading image color dimensions \n";
        return -1;
    }
    int PATCH_SIZE = 20;
    int nr_horizontal_cells = width/PATCH_SIZE;
    int nr_vertical_cells = height/PATCH_SIZE;

    // Pre-computations for backprojection
    cv::Mat_<float> X_pre(height,width);
    cv::Mat_<float> Y_pre(height,width);
    cv::Mat_<float> U(height,width);
    cv::Mat_<float> V(height,width);
    for (int r=0;r<height; r++){
        for (int c=0;c<width; c++){
            // Not efficient but at this stage doesn t matter
            X_pre.at<float>(r,c) = (c-cx_ir)/fx_ir; 
            Y_pre.at<float>(r,c) = (r-cy_ir)/fy_ir;
        }
    }

    // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
    cv::Mat_<int> cell_map(height,width);

    for (int r=0;r<height; r++){
        int cell_r = r / PATCH_SIZE;
        int local_r = r % PATCH_SIZE;

        for (int c=0;c<width; c++){
            int cell_c = c/PATCH_SIZE;
            int local_c = c%PATCH_SIZE;
            cell_map.at<int>(r,c) = (cell_r*nr_horizontal_cells+cell_c)*PATCH_SIZE*PATCH_SIZE + local_r*PATCH_SIZE + local_c;
        }
    }

    cv::Mat_<float> X(height, width);
    cv::Mat_<float> Y(height, width);
    cv::Mat_<float> X_t(height, width);
    cv::Mat_<float> Y_t(height, width);
    Eigen::MatrixXf cloud_array(width * height, 3);
    Eigen::MatrixXf cloud_array_organized(width * height, 3);

    // Populate with random color codes
    for(int i=0; i<100;i++){
        cv::Vec3b color;
        color[0]=rand()%255;
        color[1]=rand()%255;
        color[2]=rand()%255;
        color_code.push_back(color);
    }

    // Add specific colors for planes
    color_code[0][0] = 0;   color_code[0][1] = 0;   color_code[0][2] = 255;
    color_code[1][0] = 255; color_code[1][1] = 0;   color_code[1][2] = 204;
    color_code[2][0] = 255; color_code[2][1] = 100; color_code[2][2] = 0;
    color_code[3][0] = 0;   color_code[3][1] = 153; color_code[3][2] = 255;
    // Add specific colors for cylinders
    color_code[50][0] = 178; color_code[50][1] = 255; color_code[50][2] = 0;
    color_code[51][0] = 255; color_code[51][1] = 0;   color_code[51][2] = 51;
    color_code[52][0] = 0;   color_code[52][1] = 255; color_code[52][2] = 51;
    color_code[53][0] = 153; color_code[53][1] = 0;   color_code[53][2] = 255;

    // Initialize CAPE
    plane_detector = new CAPE(height, width, PATCH_SIZE, PATCH_SIZE, cylinder_detection, COS_ANGLE_MAX, MAX_MERGE_DIST);

    while(1){
        camera->grab();

        cv::Mat left, rigth;
        if(!camera->rgb(left, rigth)){
            std::cout << "Error color frame \n";
            return -1;
        }
    
        cv::Mat depth;
        if(!camera->depth(depth)){
            std::cout << "Error depth frame \n";
            return -1;
        }
        depth.convertTo(depth, CV_32F);

        // Backproject to point cloud
        X = X_pre.mul(depth); Y = Y_pre.mul(depth);
        cloud_array.setZero();

        // The following transformation+projection is only necessary to visualize RGB with overlapped segments
        // Transform point cloud to color reference frame
        X_t = ((float)R_stereo.at<double>(0,0))*X + ((float)R_stereo.at<double>(0,1))*Y + ((float)R_stereo.at<double>(0,2))*depth + (float)t_stereo.at<double>(0);
        Y_t = ((float)R_stereo.at<double>(1,0))*X + ((float)R_stereo.at<double>(1,1))*Y + ((float)R_stereo.at<double>(1,2))*depth + (float)t_stereo.at<double>(1);
        depth = ((float)R_stereo.at<double>(2,0))*X + ((float)R_stereo.at<double>(2,1))*Y + ((float)R_stereo.at<double>(2,2))*depth + (float)t_stereo.at<double>(2);

        CAPE::projectPointCloud(X_t, Y_t, depth, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb, t_stereo.at<double>(2), cloud_array);

        cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(height,width,cv::Vec3b(0,0,0));
        cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height,width,uchar(0));
        cv::Mat_<uchar> seg_output_cylinder = cv::Mat_<uchar>(height,width,uchar(0));

        // Run CAPE
        int nr_planes, nr_cylinders;
        std::vector<PlaneSeg> plane_params;
        std::vector<CylinderSeg> cylinder_params;
        double t1 = cv::getTickCount();
        CAPE::organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);
        plane_detector->process(cloud_array_organized, nr_planes, nr_cylinders, seg_output, plane_params, cylinder_params);
        double t2 = cv::getTickCount();
        double time_elapsed = (t2-t1)/(double)cv::getTickFrequency();
        std::cout<<"Total time elapsed: " << time_elapsed << " sec" << std::endl;


        /* Uncomment this block to print model params
        for(int p_id=0; p_id<nr_planes;p_id++){
            cout<<"[Plane #"<<p_id<<"] with ";
            cout<<"normal: ("<<plane_params[p_id].normal[0]<<" "<<plane_params[p_id].normal[1]<<" "<<plane_params[p_id].normal[2]<<"), ";
            cout<<"d: "<<plane_params[p_id].d<<endl;
        }

        */
        for(int c_id=0; c_id<nr_cylinders;c_id++){
            std::cout << "[Cylinder #"<<c_id<<"] with ";
            std::cout << "axis: ("<<cylinder_params[c_id].axis[0]<<" "<<cylinder_params[c_id].axis[1]<<" "<<cylinder_params[c_id].axis[2]<<"), ";
            std::cout << "center: (" << cylinder_params[c_id].centers[0].transpose()<<"), ";
            std::cout << "radius: " << cylinder_params[c_id].radii[0] << std::endl;
        }

        // loop to extract cylinder mask
        for(int ii=0; ii < seg_output.rows; ii++){
            for(int jj=0; jj < seg_output.cols; jj++){
                cv::Scalar px = seg_output.at<uchar>(ii,jj);
                if (px[0] > 50){ // cylinder threshold
                    seg_output_cylinder.at<uchar>(ii,jj) = uchar(255); 
                }else{
                    seg_output_cylinder.at<uchar>(ii,jj) = uchar(0);
                }
            }
        }
        cv::imshow("mask", seg_output_cylinder);

        // Map segments with color codes and overlap segmented image w/ RGB
        uchar * sCode;
        uchar * dColor;
        uchar * srgb;
        int code;
        for(int r=0; r<  height; r++){
            dColor = seg_rz.ptr<uchar>(r);
            sCode  = seg_output.ptr<uchar>(r);
            srgb   = left.ptr<uchar>(r);
            for(int c=0; c< width; c++){
                code = *sCode;
                if (code>0){
                    dColor[c*3] =   color_code[code-1][0]/2 + srgb[0]/2;
                    dColor[c*3+1] = color_code[code-1][1]/2 + srgb[1]/2;
                    dColor[c*3+2] = color_code[code-1][2]/2 + srgb[2]/2;
                }else{
                    dColor[c*3]   = srgb[0];
                    dColor[c*3+1] = srgb[1];
                    dColor[c*3+2] = srgb[2];
                }
                sCode++; srgb++; srgb++; srgb++;
            }
        }

        // Show frame rate and labels
        cv::rectangle(seg_rz,  cv::Point(0,0),cv::Point(width,20), cv::Scalar(0,0,0),-1);
        std::stringstream fps;
        fps << (int)(1/time_elapsed+0.5) << " fps";
        cv::putText(seg_rz, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));
        std::cout << "Number of cylinders: " << nr_cylinders << std::endl;
        int cylinder_code_offset = 50;
        // show cylinder labels
        if (nr_cylinders>0){
            std::stringstream text;
            text<<"Cylinders: ";
            
            cv::putText(seg_rz, text.str(), cv::Point(width/2,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));
            for(int j=0;j<nr_cylinders;j++){
                cv::rectangle(seg_rz,  cv::Point(width/2 + 80+15*j,6),cv::Point(width/2 + 90+15*j,16), cv::Scalar(color_code[cylinder_code_offset+j][0],color_code[cylinder_code_offset+j][1],color_code[cylinder_code_offset+j][2]),-1);
            }
        }
        cv::imshow("Segmentation", seg_rz);
    }

    cv::destroyWindow("Segmentation");
    cv::destroyWindow("mask");
    
    return 0;
}

