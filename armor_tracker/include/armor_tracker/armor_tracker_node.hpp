#ifndef ARMOR_TRACKER__ARMOR_TRACKER_NODE_HPP_
#define ARMOR_TRACKER__ARMOR_TRACKER_NODE_HPP_

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "armor_tracker/armor_tracker.hpp"
#include "armor_tracker/line_tracker.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"

namespace rm_auto_aim
{

enum TrackerState
{
  NORMAL_DETECT = 0,
  NORMAL_TRACK,
  LINE_DETECT,
  LINE_TRACKER, 
};

using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;

class ArmorTrackerNode : public rclcpp::Node
{
public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions & options);

private:
  void armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_ptr);
  void publishMarkers(const auto_aim_interfaces::msg::Target & target_msg);

  // Maximum allowable armor distance in the XOY plane
  double max_armor_distance_;

  // The time when the last message was received
  rclcpp::Time last_time_;
  double dt_;

  // Armor tracker
  double s2qxyz_, s2qyaw_, s2qr_;
  double line_s2qxyz_, line_s2qyaw_, line_s2qr_, line_s2qa_;
  double r_xyz_factor, r_yaw;
  double lost_time_thres_;
  std::unique_ptr<ArmorTracker> tracker_;

  // Reset tracker service
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_tracker_srv_;

  // Subscriber with tf2 message_filter
  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;
  std::shared_ptr<tf2_filter> tf2_filter_;

  // Tracker info publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::TrackerInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker position_marker_;
  visualization_msgs::msg::Marker linear_v_marker_;
  visualization_msgs::msg::Marker angular_v_marker_;
  visualization_msgs::msg::Marker armor_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_TRACKER__ARMOR_TRACKER_NODE_HPP_
