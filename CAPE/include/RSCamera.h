// ---------------------------------------------------------------------------------------------------------------------
//   CAPE
// ---------------------------------------------------------------------------------------------------------------------
//   Copyright 2019 Marco A. Montes Grova (a.k.a. mgrova) marrcogrova@gmail.com
// ---------------------------------------------------------------------------------------------------------------------
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
//   and associated documentation files (the "Software"), to deal in the Software without restriction, 
//   including without limitation the rights to use, copy, modify, merge, publish, distribute, 
//   sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
//   furnished to do so, subject to the following conditions:
// 
//   The above copyright notice and this permission notice shall be included in all copies or substantial 
//   portions of the Software.
// 
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
//   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
//   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES 
//   OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
//   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ---------------------------------------------------------------------------------------------------------------------

#ifndef REALSENSE_CAMERA_H_
#define REALSENSE_CAMERA_H_

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <Eigen/Dense>

// Forward declarations
namespace rs {
	class context;
	class device;

	struct extrinsics;
	struct intrinsics;
}

class RSCamera{
public:
    RSCamera();
    ~RSCamera();

    bool init(int _deviceId);
    bool rgb(cv::Mat &_left, cv::Mat &_right);
    bool depth(cv::Mat &_depth);
    bool depthColorized(cv::Mat &_depth);
    bool grab();

    bool leftCalibration(cv::Mat &_intrinsic, cv::Mat &_coefficients);
    bool rightCalibration(cv::Mat &_intrinsic, cv::Mat &_coefficients);
    bool extrinsic(cv::Mat &_rotation, cv::Mat &_translation);
    bool extrinsic(Eigen::Matrix3f &_rotation, Eigen::Vector3f &_translation);
    bool disparityToDepthParam(double &_dispToDepth);

    bool colorPixelToPoint(const cv::Point2f &_pixel, cv::Point3f &_point);

private:
    cv::Point3f deproject(const cv::Point &_point, const float _depth);

    rs2::context mRsContext;
    rs2::device mRsDevice;

    rs2::pipeline mRsPipe;
    rs2::pipeline_profile mRsPipeProfile;
    rs2::align *mRsAlign;

    rs2_intrinsics mRsColorIntrinsic, mRsDepthIntrinsic;
    rs2_extrinsics mRsDepthToColor, mRsColorToDepth;

    cv::Mat mCvColorIntrinsic;
    cv::Mat mCvDepthIntrinsic;

    float mRsDepthScale;

    cv::Mat mExtrinsicColorToDepth; // 4x4 transformation between color and depth camera

    bool mSetDenseCloud = true;
    bool mHasRGB = false, mComputedDepth = false;
    cv::Mat mLastRGB, mLastDepthInColor, mLastDepthColorized;

    // declare filteres
    rs2::threshold_filter mThrFilter;	// Threshold - removes values outside recommended range
    rs2::colorizer mColorFilter;		// Colorize - convert from depth to RGB color
    rs2::disparity_transform mDepthToDisparity;	// Converting depth to disparity
    rs2::disparity_transform mDisparityToDepth;	// Converting disparity to depth

    bool mIsDisparity = true;
};


#endif