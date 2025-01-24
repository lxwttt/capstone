#ifndef __NEW_ARMOR_DETECTOR__ARMOR_DETECTOR_HPP__
#define __NEW_ARMOR_DETECTOR__ARMOR_DETECTOR_HPP__

// OPENCV
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// other
#include "new_armor_detector/number_classifier.hpp"
#include "new_armor_detector/stamp.hpp"

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
    ArmorDetector() = default;
    ArmorDetector(const int &_bin_thres, const int &_color, const LightParams &_l, const ArmorParams &_a);
    ~ArmorDetector();

    void detect(std::shared_ptr<rm_auto_aim::Frame> _frame);
    // std::vector<Armor> detect(const cv::Mat &_raw_image);
    void findLights(std::shared_ptr<rm_auto_aim::Frame> &_frame);
    void matchLights(std::shared_ptr<rm_auto_aim::Frame> &_frame);
    void newFindLights(const cv::Mat &_raw_image);
    void newMatchLights();
    std::vector<cv::Point2f> getRectPoints(const cv::Rect &_rect);

    // For debug usage
    // auto_aim_interfaces::msg::DebugLights debug_lights_;
    // auto_aim_interfaces::msg::DebugArmors debug_armors_;
    cv::Mat getAllNumbersImage();

    // cv::Mat binary_img_;

    // parameters for armor detector
    bool detector_debug_;
    int binary_thres_;
    int detect_color_;
    LightParams l_;
    ArmorParams a_;

    // number classifier
    // std::unique_ptr<NumberClassifier> classifier_;
    std::shared_ptr<NumberClassifier> classifier_;

   private:
    // functions for filtering invalid lights and armors
    bool containLight(const Light &_light_1, const Light &_light_2, std::shared_ptr<rm_auto_aim::Frame> &_frame);
    ArmorType isArmor(const Light &_light_1, const Light &_light_2);
    bool isLight(const Light &_possible_light);

    // variables for lights and armors
    // std::vector<Light> lights_;
    // std::vector<Armor> armors_;
    // std::vector<Lightbar> lightbars_;
};

}  // namespace rm_auto_aim

#endif  // __NEW_ARMOR_DETECTOR__ARMOR_DETECTOR_HPP__