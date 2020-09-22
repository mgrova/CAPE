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

#include "RSCamera.h"

RSCamera::RSCamera(){

}

RSCamera::~RSCamera(){
    mRsPipe.stop();

    // mRsDevice.stop();
    // mRsDevice.close();
}

//-----------------------------------------------------------------------------------------------------------------
bool RSCamera::init(int _deviceId){

    auto list = mRsContext.query_devices();
    if (list.size() == 0) {
        std::cout << "[STEREOCAMERA][REALSENSE] There's no any compatible device connected." << std::endl;
        return false;
    }
    mRsDevice = list[_deviceId];

    // std::cout << "[STEREOCAMERA][REALSENSE] Camera hardware reset. 5 seconds sleeping ... \n";
    // mRsDevice.hardware_reset();
    // std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "[STEREOCAMERA][REALSENSE] Using device "<< _deviceId <<", an "<< mRsDevice.get_info(rs2_camera_info::RS2_CAMERA_INFO_NAME) << std::endl;
    std::cout << "[STEREOCAMERA][REALSENSE]     Serial number: " << mRsDevice.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
    std::cout << "[STEREOCAMERA][REALSENSE]     Firmware version: " << mRsDevice.get_info(rs2_camera_info::RS2_CAMERA_INFO_FIRMWARE_VERSION)<< std::endl;
    
    // Initialize streams of data.
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    mRsPipeProfile = mRsPipe.start(cfg);
    
    // Get intrinsics and extrinsics
    auto depth_stream = mRsPipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = mRsPipeProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    mRsAlign = new rs2::align(RS2_STREAM_COLOR);

    mRsDepthToColor = depth_stream.get_extrinsics_to(color_stream);
    mRsColorToDepth = color_stream.get_extrinsics_to(depth_stream);
    mRsDepthIntrinsic = depth_stream.get_intrinsics();
    mRsColorIntrinsic = color_stream.get_intrinsics();

    auto sensor = mRsPipeProfile.get_device().first<rs2::depth_sensor>();
    mRsDepthScale = sensor.get_depth_scale();

    // Projection matrix Depth
    mCvDepthIntrinsic = cv::Mat::eye(3,3,CV_32F);
    mCvDepthIntrinsic.at<float>(0,0) = mRsDepthIntrinsic.fx;
    mCvDepthIntrinsic.at<float>(1,1) = mRsDepthIntrinsic.fy;
    mCvDepthIntrinsic.at<float>(0,2) = mRsDepthIntrinsic.ppx;
    mCvDepthIntrinsic.at<float>(1,2) = mRsDepthIntrinsic.ppy;

    // Projection matrix Color
    mCvColorIntrinsic= cv::Mat::eye(3,3,CV_32F);
    mCvColorIntrinsic.at<float>(0,0) = mRsColorIntrinsic.fx;
    mCvColorIntrinsic.at<float>(1,1) = mRsColorIntrinsic.fy;
    mCvColorIntrinsic.at<float>(0,2) = mRsColorIntrinsic.ppx;
    mCvColorIntrinsic.at<float>(1,2) = mRsColorIntrinsic.ppy;

    mExtrinsicColorToDepth = cv::Mat::eye(4,4,CV_32F);
    cv::Mat(3,3,CV_32F, &mRsColorToDepth.rotation[0]).copyTo(mExtrinsicColorToDepth(cv::Rect(0,0,3,3)));
    mExtrinsicColorToDepth(cv::Rect(0,0,3,3)) = mExtrinsicColorToDepth(cv::Rect(0,0,3,3)).t(); // RS use color major instead of row mayor.
    cv::Mat(3,1,CV_32F, &mRsColorToDepth.translation[0]).copyTo(mExtrinsicColorToDepth(cv::Rect(3,0,1,3)));

    // 666 Distance filter and colorize
    float min_depth = 0.29f;
    float max_depth = 10.0f;

    mDepthToDisparity = rs2::disparity_transform(true);
    mDisparityToDepth = rs2::disparity_transform(false);

    // filter settings
    mThrFilter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);
    mThrFilter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);
    mColorFilter.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0);
    mColorFilter.set_option(RS2_OPTION_COLOR_SCHEME, 9.0f);		// Hue colorization
    mColorFilter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);
    mColorFilter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);

    return true;
}

//-----------------------------------------------------------------------------------------------------------------
bool RSCamera::rgb(cv::Mat & _left, cv::Mat & _right){
    mLastRGB.copyTo(_left);
    return mHasRGB;
}

//-----------------------------------------------------------------------------------------------------------------
bool RSCamera::depth(cv::Mat & _depth){
    mLastDepthInColor.copyTo(_depth);
    return mComputedDepth;
}

//-----------------------------------------------------------------------------------------------------------------
bool RSCamera::depthColorized(cv::Mat & _depth){
    mLastDepthColorized.copyTo(_depth);
    return mComputedDepth;
}

//-----------------------------------------------------------------------------------------------------------------
bool RSCamera::grab(){

    rs2::frameset frames = mRsPipe.wait_for_frames();

    auto processedFrames = mRsAlign->process(frames);

    rs2::frame frameRGB = processedFrames.first(RS2_STREAM_COLOR);
    rs2::frame frameDepth = processedFrames.first(RS2_STREAM_DEPTH);

    // Filter depth by distance and color it
    rs2::frame frameDepthColorized = mThrFilter.process(frameDepth);
    frameDepthColorized = mDepthToDisparity.process(frameDepthColorized);
    if (!mIsDisparity)
        frameDepthColorized = mDisparityToDepth.process(frameDepthColorized);
    frameDepthColorized = mColorFilter.process(frameDepthColorized);

    mLastRGB = cv::Mat(cv::Size(mRsColorIntrinsic.width, mRsColorIntrinsic.height), CV_8UC3, (void*)frameRGB.get_data(), cv::Mat::AUTO_STEP);
    mHasRGB = true;

    mLastDepthInColor = cv::Mat(cv::Size(mRsDepthIntrinsic.width, mRsDepthIntrinsic.height), CV_16U, (uchar*) frameDepth.get_data(), cv::Mat::AUTO_STEP);
    mLastDepthColorized = cv::Mat(cv::Size(mRsDepthIntrinsic.width, mRsDepthIntrinsic.height), CV_8UC3, (void*) frameDepthColorized.get_data(), cv::Mat::AUTO_STEP);
    mComputedDepth = true;

    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::leftCalibration(cv::Mat &_intrinsic, cv::Mat &_coefficients) {
    mCvColorIntrinsic.copyTo(_intrinsic);
    _coefficients = cv::Mat(1,5, CV_32F, mRsColorIntrinsic.coeffs);
    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::rightCalibration(cv::Mat &_intrinsic, cv::Mat &_coefficients) {
    mCvDepthIntrinsic.copyTo(_intrinsic);
    _coefficients = cv::Mat(1,5, CV_32F, mRsColorIntrinsic.coeffs);
    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::extrinsic(cv::Mat &_rotation, cv::Mat &_translation) {
    cv::Mat(3,3,CV_32F, &mRsDepthToColor.rotation[0]).copyTo(_rotation);
    cv::Mat(3,1,CV_32F, &mRsDepthToColor.translation[0]).copyTo(_translation);
    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::extrinsic(Eigen::Matrix3f &_rotation, Eigen::Vector3f &_translation) {
    _rotation = Eigen::Matrix3f(&mRsDepthToColor.rotation[0]);
    _translation = Eigen::Vector3f(&mRsDepthToColor.translation[0]);
    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::disparityToDepthParam(double &_dispToDepth){
    _dispToDepth = mRsDepthScale;
    return true;
}

//---------------------------------------------------------------------------------------------------------------------
bool RSCamera::colorPixelToPoint(const cv::Point2f &_pixel, cv::Point3f &_point){
    // // Retrieve the 16-bit depth value and map it into a depth in meters
    // uint16_t depth_value = mLastDepthInColor.at<uint16_t>(_pixel.y, _pixel.x);
    // float depth_in_meters = depth_value * mRsDepthScale;
    // // Set invalid pixels with a depth value of zero, which is used to indicate no data
    // pcl::PointXYZRGB point;
    // if (depth_value == 0) {
    //     return false;
    // } else {
    //     // Map from pixel coordinates in the depth image to pixel coordinates in the color image
    //     cv::Point2f depth_pixel(_pixel.x, _pixel.y);
    //     cv::Point3f depth_point = deproject(depth_pixel, depth_in_meters);

    //     _point.x = depth_point.x;
    //     _point.y = depth_point.y;
    //     _point.z = depth_point.z;
    //     return true;
    // }
    return false;
}

//---------------------------------------------------------------------------------------------------------------------
cv::Point3f RSCamera::deproject(const cv::Point &_point, const float _depth){
    float x = (_point.x - mRsDepthIntrinsic.ppx) / mRsDepthIntrinsic.fx;
    float y = (_point.y - mRsDepthIntrinsic.ppy) / mRsDepthIntrinsic.fy;

    cv::Point3f p(x*_depth, y*_depth, _depth);

    return p;
}
