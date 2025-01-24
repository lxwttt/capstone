#ifndef _NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP_
#define _NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP_

#include "auto_aim_interfaces/msg/armor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "auto_aim_interfaces/msg/new_target.hpp"

class IRobot
{
   public:
    IRobot() = delete;
    IRobot(uint8_t robot_id) : robot_id_(robot_id){};
    ~IRobot() = default;

    virtual void push(const auto_aim_interfaces::msg::Armor &armor, rclcpp::Time time) = 0;

    virtual void update() = 0;

    virtual double getOmega() = 0;

    virtual double getYaw() = 0;

    virtual void getStatus(auto_aim_interfaces::msg::NewTarget &_msg) = 0;

    /**
     * @brief Set the Armor Size object
     * @param size, 1 for small armor, 2 for large armor, 0 for unknown
     */
    void setArmorSize(uint8_t size) { armor_size = size; }

   public:
    rclcpp::Time last_t_;
    uint8_t armor_size = 0;
    uint8_t robot_id_  = 0;
};

using RobotPtr = std::shared_ptr<IRobot>;

#endif  // __NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP__
