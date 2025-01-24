#ifndef ARMOR_DETECTOR__ARMOR_DETECTOR_HPP_
#define ARMOR_DETECTOR__ARMOR_DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <cmath>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace rm_auto_aim
{
class ArmorDetector
{
   public:
    struct LightParams
    {
        // width / height
        double min_ratio_;
        double max_ratio_;
        // vertical angle
        double max_angle_;
        // area condition
        double min_fill_ratio_;
    };

    struct ArmorParams
    {
        double min_light_ratio_;
        // light pairs distance
        double min_small_center_distance_;
        double max_small_center_distance_;
        double min_large_center_distance_;
        double max_large_center_distance_;
        // horizontal angle
        double max_angle_;
    };

    // parameters for armor detector
    bool detector_debug_;
    int binary_thres_;
    int detect_color_;
    LightParams l_;
    ArmorParams a_;

    // number classifier
    std::unique_ptr<NumberClassifier> classifier_;

    // Constructor for armor detector
    ArmorDetector(const int &_bin_thres, const int &_color, const LightParams &_l, const ArmorParams &_a);

    // main functions
    std::vector<Armor> detect(const cv::Mat &_raw_image);
    void findLights(const cv::Mat &_raw_image);
    void matchLights();
    void newFindLights(const cv::Mat &_raw_image);
    void newMatchLights();
    std::vector<cv::Point2f> getRectPoints(const cv::Rect &_rect);

    // For debug usage
    cv::Mat binary_img_;
    auto_aim_interfaces::msg::DebugLights debug_lights_;
    auto_aim_interfaces::msg::DebugArmors debug_armors_;
    cv::Mat getAllNumbersImage();

   private:
    // functions for filtering invalid lights and armors
    bool isLight(const Light &_possible_light);
    bool containLight(const Light &_light_1, const Light &_light_2);
    ArmorType isArmor(const Light &_light_1, const Light &_light_2);

    // variables for lights and armors
    std::vector<Light> lights_;
    std::vector<Armor> armors_;
    std::vector<Lightbar> lightbars_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_DETECTOR_HPP_