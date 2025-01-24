#ifndef __NEW_ARMOR_TRACKER__GARAGE__CAR_HPP__
#define __NEW_ARMOR_TRACKER__GARAGE__CAR_HPP__

#include "new_armor_tracker/garage/IRobot.hpp"
#include "new_armor_tracker/kalman/interface/antitopV3.h"
#include "new_armor_tracker/kalman/interface/trackqueueV4.h"
#include "new_armor_tracker/utils/tf.hpp"

class Car : public IRobot
{
   public:
    Car(uint8_t robot_id, std::vector<std::vector<double>> &_ekf_params);
    ~Car() = default;

    void push(const auto_aim_interfaces::msg::Armor &armor, rclcpp::Time time) override;

    void update() override;

    double getOmega() override { return antitop_4->getOmega(); }

    double getYaw() override;

    void getStatus(auto_aim_interfaces::msg::NewTarget &_msg) override;

   public:
    rm::TrackQueueV4 track_queue;
    rm::AntitopV3 *antitop_2;
    rm::AntitopV3 *antitop_4;

    bool is_big         = false;
    int big_armor_cnt_  = 0;  // 装甲板尺寸计数
    int curr_armor_num_ = 0;  // 当前一次更新内观测的装甲板数量
};

#endif  // __NEW_ARMOR_TRACKER__GARAGE__CAR_HPP__