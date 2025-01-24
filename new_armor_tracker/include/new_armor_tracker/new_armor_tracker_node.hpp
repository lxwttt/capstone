#ifndef __NEW_ARMOR_TRACKER__NEW_ARMOR_TRACKER_NODE_HPP__
#define __NEW_ARMOR_TRACKER__NEW_ARMOR_TRACKER_NODE_HPP__

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>

// msgs
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/new_target.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "garage/garage.hpp"

// VISUALIZATION
#include <visualization_msgs/msg/marker_array.hpp>

namespace rm_auto_aim
{

using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;

class NewArmorTrackerNode : public rclcpp::Node
{
   public:
    NewArmorTrackerNode(const rclcpp::NodeOptions &options);

    ~NewArmorTrackerNode();

   private:
    // sub callback
    void armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr msg);

    // walltimer
    rclcpp::TimerBase::SharedPtr timer_;

    // publisher
    // rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::NewTarget>::SharedPtr target_pub_;

    // robot id mapping
    std::map<std::string, int> id_table = {{"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"outpost", 6}, {"sentry", 7}};

    // garage
    std::shared_ptr<Garage> garage;

    // tf2
    std::string target_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;

    // msg
    auto_aim_interfaces::msg::NewTarget armor_target_msg_;
    auto_aim_interfaces::msg::NewTarget blank_target_msg_;

    // Visualization Marker Publisher
    // Visualization marker publisher
    visualization_msgs::msg::Marker position_marker_;
    visualization_msgs::msg::Marker linear_v_marker_;
    visualization_msgs::msg::Marker angular_v_marker_;
    visualization_msgs::msg::Marker armor_marker_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // tracker node variable
    uint8_t target_id_;
    rclcpp::Time t_;

    enum class State
    {
        TRACKING,
        LOST
    } state_,
        last_state_;
};

}  // namespace rm_auto_aim

#endif  // __NEW_ARMOR_TRACKER__NEW_ARMOR_TRACKER_NODE_HPP__