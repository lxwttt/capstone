#ifndef __NEW_ARMOR_TRACKER__GARAGE__TOWER_HPP__
#define __NEW_ARMOR_TRACKER__GARAGE__TOWER_HPP__

#include "new_armor_tracker/garage/IRobot.hpp"
#include "new_armor_tracker/kalman/interface/outpostV2.h"
#include "new_armor_tracker/kalman/interface/trackqueueV3.h"

class Tower : public IRobot
{
   public:
    Tower(uint8_t robot_id, std::vector<std::vector<double>> &_ekf_params);
    ~Tower() = default;

    void push(const auto_aim_interfaces::msg::Armor &armor, rclcpp::Time time) override;

    void update() override;

    double getOmega() override { return outpost.getOmega(); }

    double getYaw() override { return 1.0f; }

    void getStatus(auto_aim_interfaces::msg::NewTarget &_msg) override;

   public:
    rm::TrackQueueV3 track_queue;
    rm::OutpostV2 outpost;
};

#endif  // __NEW_ARMOR_TRACKER__GARAGE__TOWER_HPP__
