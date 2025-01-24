
#ifndef __NEW_ARMOR_TRACKER_UTILS_TIMER_HPP__
#define __NEW_ARMOR_TRACKER_UTILS_TIMER_HPP__

#include "rclcpp/rclcpp.hpp"

namespace rm
{

static inline rclcpp::Time getTime() { return rclcpp::Clock(RCL_ROS_TIME).now(); }

static inline double getDuration(rclcpp::Time start, rclcpp::Time end) { return (end - start).seconds(); }

static inline double getDoubleOfS(rclcpp::Time start, rclcpp::Time end) { return (end - start).seconds(); }

}  // namespace rm
#endif  // __NEW_ARMOR_TRACKER_UTILS_TIMER_HPP__