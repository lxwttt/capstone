/**
 * @file stamp.hpp
 *
 * @brief Define All the data structure and utility functions used in the project
 *
 * @author Sora
 *
 */

#ifndef __NEW_ARMOR_DETECTOR__STAMP_HPP__
#define __NEW_ARMOR_DETECTOR__STAMP_HPP__

#include <memory>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <string>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

namespace rm_auto_aim
{
const int RED  = 0;
const int BLUE = 1;

enum class ArmorType
{
    SMALL,
    LARGE,
    INVALID
};
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct Light : public cv::Rect
{
    Light() = default;
    explicit Light(cv::Rect box, cv::Point2f top, cv::Point2f bottom, int area, float tilt_angle)
        : cv::Rect(box), top(top), bottom(bottom), tilt_angle(tilt_angle)
    {
        length  = cv::norm(top - bottom);
        width   = area / length;
        ratio   = width / length;
        center  = (top + bottom) / 2;
        incline = tilt_angle * CV_PI / 180;
    }

    int color;
    cv::Point2f top, bottom;
    cv::Point2f center;
    double length;
    double width;
    double ratio;
    float tilt_angle;
    float incline;
};

struct Lightbar
{
    std::vector<cv::Point> contour;
    cv::RotatedRect rect;
    double length;
    double angle;
    Lightbar() = default;
};

struct Armor
{
    Armor() = default;
    Armor(const Light &l1, const Light &l2)
    {
        if (l1.center.x < l2.center.x)
        {
            left_light = l1, right_light = l2;
        }
        else
        {
            left_light = l2, right_light = l1;
        }
        center = (left_light.center + right_light.center) / 2;
    }

    // Light pairs part
    Light left_light, right_light;
    cv::Point2f center;
    ArmorType type;

    // Number part
    cv::Mat number_img;
    std::string number;
    float confidence;
    std::string classfication_result;
};

struct Frame
{
    std_msgs::msg::Header header_;                      // ros header
    uint64_t id_;                                       // frame id
    std::chrono::system_clock::time_point time_stamp_;  // frame reach time

    // Image data
    std::shared_ptr<cv::Mat> raw_image_;          // origin image
    std::shared_ptr<cv::Mat> binary_image_;       // black & white image
    std::shared_ptr<cv::Mat> match_light_image_;  // debug use

    // Detected data
    std::vector<Light> lights_;        // store light/contour found in frame
    std::vector<Lightbar> lightbars_;  // store light bar found in frame
    std::vector<Armor> armors_;        // store armor found in frame

    // DEBUG INFO
    float image_latency_;
    float image_process_time_;

    Frame(std::shared_ptr<cv::Mat> &_raw_image, uint64_t _id)
        : raw_image_(_raw_image), id_(_id), binary_image_(std::make_shared<cv::Mat>()), match_light_image_(std::make_shared<cv::Mat>())
    {
    }
};

}  // namespace rm_auto_aim

#endif  // __NEW_ARMOR_DETECTOR__STAMP_HPP__