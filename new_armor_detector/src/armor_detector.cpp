#include "new_armor_detector/armor_detector.hpp"

namespace rm_auto_aim
{

ArmorDetector::ArmorDetector(const int &_bin_thres, const int &_color, const LightParams &_l, const ArmorParams &_a)
    : binary_thres_(_bin_thres), detect_color_(_color), l_(_l), a_(_a)
{
}

ArmorDetector::~ArmorDetector() {}

void ArmorDetector::detect(std::shared_ptr<rm_auto_aim::Frame> _frame)
{
    // std::cout << "[Time: " << std::chrono::system_clock::now().time_since_epoch().count() << "] ";
    // std::cout << "Start detecting frame" << _frame->id_ << std::endl;
    // std::this_thread::sleep_for(std::chrono::milliseconds(_frame->rand_delay_));

    // 1. preprocessing raw images
    cv::Mat gray_img;
    cv::cvtColor(*_frame->raw_image_, gray_img, cv::COLOR_RGB2GRAY);
    cv::threshold(gray_img, *_frame->binary_image_, binary_thres_, 255, cv::THRESH_BINARY);

    // if (detector_debug_)
    // {
    //     cv::imshow("1. binary_img", *_frame->binary_image_);
    //     cv::waitKey(1);
    // }

    // 2. get the lights contours from binary_img
    findLights(_frame);

    // 3. match the lights to form armorplates
    matchLights(_frame);

    // 4. classsify the number for valid armorplates
    if (!_frame->armors_.empty())
    {
        classifier_->extractNumbers(*_frame->raw_image_, _frame->armors_);
        classifier_->classify(_frame->armors_);
        // std::cout<< "armor size: " << _frame->armors_.size() << std::endl;
    }
    // std::cout << "armor size: " << _frame->armors_.size() << std::endl;
    // std::cout << "armor is empty: " << _frame->armors_.empty() << std::endl;
    // std::cout << "[Frame: " << _frame->id_ << "] detect: armors_.size(): " << _frame->armors_.size() << std::endl;
    // std::cout << "[Frame: " << _frame->id_ << "] detect: is empty: " << _frame->armors_.empty() << std::endl;
    // std::cout << "[Time: " << std::chrono::system_clock::now().time_since_epoch().count() << "] ";
    // std::cout << "Finish detecting frame" << _frame->id_ << std::endl;

    if (detector_debug_)
    {
        cv::Mat match_lights = _frame->raw_image_->clone();
        cv::String labels[4] = {"0", "1", "2", "3"};
        for (const Light &light : _frame->lights_)
        {
            std::vector<cv::Point2f> vertices(4);
            vertices = this->getRectPoints(light);
            cv::putText(match_lights,
                        "ratio: " + std::to_string(light.ratio),
                        light.center + cv::Point2f(0, 50),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 0, 255),
                        2);
            for (int i = 0; i < 4; i++)
                cv::line(match_lights, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        for (const Armor &armor : _frame->armors_)
        {
            std::vector<cv::Point2f> vertices;
            vertices.push_back(armor.left_light.bottom);
            vertices.push_back(armor.left_light.top);
            vertices.push_back(armor.right_light.top);
            vertices.push_back(armor.right_light.bottom);
            for (int i = 0; i < 4; i++)
            {
                cv::circle(match_lights, vertices[i % 4], 4, cv::Scalar(255, 0, 255), -1);
                cv::putText(match_lights, labels[i], vertices[i % 4] + cv::Point2f(12, 15), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 3);
            }
            cv::line(match_lights, armor.left_light.top, armor.left_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(match_lights, armor.right_light.top, armor.right_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(match_lights, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(match_lights, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::putText(match_lights,
                        "left: " + std::to_string(armor.left_light.ratio),
                        vertices[1] - cv::Point2f(60, 18),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 0, 255),
                        2);
            cv::putText(match_lights,
                        "right: " + std::to_string(armor.right_light.ratio),
                        vertices[2] - cv::Point2f(60, 18),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 0, 0),
                        2);
            cv::putText(match_lights,
                        armor.classfication_result,
                        armor.left_light.bottom + cv::Point2f(-20, 38),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 255),
                        2);
        }
        cv::imshow("2. match_lights", match_lights);
        cv::waitKey(1);
    }
}

void ArmorDetector::findLights(std::shared_ptr<rm_auto_aim::Frame> &_frame)
{
    std::vector<std::vector<cv::Point>> contours;
    // std::vector<cv::Vec4i> hierarchy;
    // last_time = std::chrono::high_resolution_clock::now();
    cv::findContours(*_frame->binary_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // RCLCPP_INFO(
    // rclcpp::get_logger("rm_auto_aim"), "findContours time: %f, countor size: %ld",
    // std::chrono::duration_cast<std::chrono::microseconds>(
    //   std::chrono::high_resolution_clock::now() - last_time)
    //   .count() /
    //   1000.0, contours.size());

    // this->lights_.clear();
    // this->debug_lights_.data.clear();

    for (const auto &contour : contours)
    {
        if (contour.size() < 25)
            continue;
        if (contour.size() > 900)
            continue;

        cv::Rect b_rect        = cv::boundingRect(contour);
        cv::RotatedRect r_rect = cv::minAreaRect(contour);
        cv::Mat mask           = cv::Mat::zeros(b_rect.size(), CV_8UC1);
        std::vector<cv::Point> mask_contour;
        for (const auto &p : contour)
        {
            // RCLCPP_INFO(
            //   rclcpp::get_logger("rm_auto_aim"), "p: %d, %d", p.x, p.y);
            mask_contour.emplace_back(p - cv::Point(b_rect.x, b_rect.y));
        }
        // RCLCPP_INFO(
        //   rclcpp::get_logger("rm_auto_aim"), "END");
        cv::fillPoly(mask, {mask_contour}, 255);
        // cv::imshow("mask", mask);
        std::vector<cv::Point> points;
        cv::findNonZero(mask, points);
        // points / rotated rect area
        bool is_fill_rotated_rect = points.size() / (r_rect.size.width * r_rect.size.height) > l_.min_fill_ratio_;
        cv::Vec4f return_param;
        cv::fitLine(points, return_param, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Point2f top, bottom;
        double angle_k;
        if (int(return_param[0] * 100) == 100 || int(return_param[1] * 100) == 0)
        {
            top     = cv::Point2f(b_rect.x + b_rect.width / 2, b_rect.y);
            bottom  = cv::Point2f(b_rect.x + b_rect.width / 2, b_rect.y + b_rect.height);
            angle_k = 0;
        }
        else
        {
            auto k  = return_param[1] / return_param[0];
            auto b  = (return_param[3] + b_rect.y) - k * (return_param[2] + b_rect.x);
            top     = cv::Point2f((b_rect.y - b) / k, b_rect.y);
            bottom  = cv::Point2f((b_rect.y + b_rect.height - b) / k, b_rect.y + b_rect.height);
            angle_k = std::atan(k) / CV_PI * 180 - 90;
            if (angle_k > 90)
            {
                angle_k = 180 - angle_k;
            }
        }

        // int radius                    = (int)(0.05 * cv::norm(top - bottom));
        // cv::Point2f first_barycenter  = getBarycenter(binary_img, top, radius);
        // cv::Point2f second_barycenter = getBarycenter(binary_img, bottom, radius);
        // auto light = Light(b_rect, first_barycenter, second_barycenter, points.size(), angle_k);
        auto light = Light(b_rect, top, bottom, points.size(), angle_k);

        if (isLight(light) && is_fill_rotated_rect)
        {
            if (light.area() < 25)
                continue;
            if (  // Avoid assertion failed
                0 <= light.x && 0 <= light.width && light.x + light.width <= _frame->raw_image_->cols && 0 <= light.y && 0 <= light.height &&
                light.y + light.height <= _frame->raw_image_->rows)
            {
                int sum_r = 0, sum_b = 0;
                // auto roi = _raw_image(light);
                auto roi = _frame->raw_image_->operator()(light);
                // Iterate through the ROI
                int iterate_number = 5;
                for (int i = 0; i < iterate_number; i++)
                {
                    for (int j = 0; j < iterate_number; j++)
                    {
                        int i_ = i * (roi.rows / 5);
                        int j_ = j * (roi.cols / 5);
                        if (cv::pointPolygonTest(contour, cv::Point2f(j_ + light.x, i_ + light.y), false) >= 0)
                        {
                            // if point is inside contour
                            sum_r += roi.at<cv::Vec3b>(i_, j_)[2];
                            sum_b += roi.at<cv::Vec3b>(i_, j_)[0];
                        }
                    }
                }
                // Sum of red pixels > sum of blue pixels ?
                light.color = sum_r > sum_b ? RED : BLUE;
                // this->lights_.emplace_back(light);
                _frame->lights_.emplace_back(light);
            }
        }
    }

    // RCLCPP_INFO(
    //   rclcpp::get_logger("rm_auto_aim"), "findContours time: %f, countor size: %ld",
    //   std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - last_time)
    //     .count() /
    //     1000.0, contours.size());
    return;
}

void ArmorDetector::matchLights(std::shared_ptr<rm_auto_aim::Frame> &_frame)
{
    // this->armors_.clear();
    // this->debug_armors_.data.clear();

    // Loop all the pairing of lights
    // for (auto light_1 = this->lights_.begin(); light_1 != this->lights_.end(); light_1++)
    for (auto light_1 = _frame->lights_.begin(); light_1 != _frame->lights_.end(); light_1++)
    {
        // for (auto light_2 = light_1 + 1; light_2 != this->lights_.end(); light_2++)
        for (auto light_2 = light_1 + 1; light_2 != _frame->lights_.end(); light_2++)
        {
            if (light_1->color != detect_color_ || light_2->color != detect_color_)
                continue;

            if (containLight(*light_1, *light_2, _frame))
                continue;

            auto type = isArmor(*light_1, *light_2);
            if (type != ArmorType::INVALID)
            {
                auto armor = Armor(*light_1, *light_2);
                armor.type = type;
                // this->armors_.emplace_back(armor);
                _frame->armors_.emplace_back(armor);
            }
        }
    }
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool ArmorDetector::containLight(const Light &_light_1, const Light &_light_2, std::shared_ptr<rm_auto_aim::Frame> &_frame)
{
    auto points        = std::vector<cv::Point2f>{_light_1.top, _light_1.bottom, _light_2.top, _light_2.bottom};
    auto bounding_rect = cv::boundingRect(points);

    for (const auto &test_light : _frame->lights_)
    {
        if (test_light.center == _light_1.center || test_light.center == _light_2.center)
            continue;

        if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) || bounding_rect.contains(test_light.center))
        {
            return true;
        }
    }

    return false;
}

ArmorType ArmorDetector::isArmor(const Light &_light_1, const Light &_light_2)
{
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = _light_1.length < _light_2.length ? _light_1.length / _light_2.length : _light_2.length / _light_1.length;
    bool light_ratio_ok      = light_length_ratio > a_.min_light_ratio_;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length  = (_light_1.length + _light_2.length) / 2;
    float center_distance   = cv::norm(_light_1.center - _light_2.center) / avg_light_length;
    bool center_distance_ok = (a_.min_small_center_distance_ <= center_distance && center_distance < a_.max_small_center_distance_) ||
                              (a_.min_large_center_distance_ <= center_distance && center_distance < a_.max_large_center_distance_);

    // Angle of light center connection
    cv::Point2f diff = _light_1.center - _light_2.center;
    float angle      = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok    = angle < a_.max_angle_;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    ArmorType type;
    if (is_armor)
    {
        type = center_distance > a_.min_large_center_distance_ ? ArmorType::LARGE : ArmorType::SMALL;
    }
    else
    {
        type = ArmorType::INVALID;
    }

    // Fill in debug information
#if DEBUG_FOXGLOVE
    auto_aim_interfaces::msg::DebugArmor armor_data;
    armor_data.type            = ARMOR_TYPE_STR[static_cast<int>(type)];
    armor_data.center_x        = (light_1.center.x + light_2.center.x) / 2;
    armor_data.light_ratio     = light_length_ratio;
    armor_data.center_distance = center_distance;
    armor_data.angle           = angle;
    this->debug_armors.data.emplace_back(armor_data);
#endif

    return type;
}

bool ArmorDetector::isLight(const Light &_light)
{
    // The ratio of light (short side / long side)
    bool ratio_ok = l_.min_ratio_ < _light.ratio && _light.ratio < l_.max_ratio_;

    bool angle_ok = _light.tilt_angle < l_.max_angle_;

    bool is_light = ratio_ok && angle_ok;

    // Fill in debug information
#if DEBUG_FOXGLOVE
    auto_aim_interfaces::msg::DebugLight light_data;
    light_data.center_x = light.center.x;
    light_data.ratio    = light.ratio;
    light_data.angle    = light.tilt_angle;
    light_data.is_light = is_light;
    this->debug_lights.data.emplace_back(light_data);
#endif

    return is_light;
}

std::vector<cv::Point2f> ArmorDetector::getRectPoints(const cv::Rect &_rect)
{
    std::vector<cv::Point2f> points(4);
    points[0] = _rect.tl();  // Top-left corner
    points[1] = cv::Point(_rect.x + _rect.width, _rect.y);
    points[2] = _rect.br();
    points[3] = cv::Point(_rect.x, _rect.y + _rect.height);  // Bottom-left corner
    return points;
}

}  // namespace rm_auto_aim