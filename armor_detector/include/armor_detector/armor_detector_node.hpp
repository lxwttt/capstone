#ifndef ARMOR_DETECTOR__ARMOR_DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR__ARMOR_DETECTOR_NODE_HPP_

// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "std_msgs/msg/float32.hpp"

// TF
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// STD
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor_detector.hpp"
#include "armor_detector/number_classifier.hpp"
#include "armor_detector/pnp_solver.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/cvmode.hpp"
#include "shm_bridge/fill_image.hpp"
#include "shm_bridge/opencv_conversions.hpp"

namespace rm_auto_aim
{

class ArmorDetectorNode : public rclcpp::Node
{
   public:
    ArmorDetectorNode(const rclcpp::NodeOptions &_options);

   private:
    void imageCallback(const shm_msgs::msg::Image2m::SharedPtr _img_msg);
    std::unique_ptr<ArmorDetector> initArmorDetector();
    std::vector<Armor> detectArmors(const shm_msgs::msg::Image2m::SharedPtr &_img_msg);

    void createDebugPublishers();
    void destroyDebugPublishers();

    void publishMarkers();

    // tf boardcast
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Armor Detector
    std::unique_ptr<ArmorDetector> armor_detector_;
    cv::Mat raw_image_;
    float fps_     = 0;
    int fps_count_ = 0;
    rclcpp::Time last_time_;
    std::map<std::string, int> id_table_ = {{"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"outpost", 6}, {"sentry", 7}, {"base", 8}};
    double append_yaw_ = 0;

    // Detected armors publisher
    auto_aim_interfaces::msg::Armors armors_msg_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;

    // Visualization marker publisher
    visualization_msgs::msg::Marker armor_marker_;
    visualization_msgs::msg::Marker text_marker_;
    visualization_msgs::msg::MarkerArray marker_array_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr armor_yaw_pub_;

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    std::unique_ptr<PnPSolver> pnp_solver_;

    // Image subscrpition
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<shm_msgs::msg::Image2m>::SharedPtr img_sub_;
    shm_msgs::CvImageConstPtr img_mat_msg_;

    // Subscription of cv mode
    rclcpp::Subscription<auto_aim_interfaces::msg::Cvmode>::SharedPtr cvmode_sub_;
    bool cv_mode_;  // false:armor, true:rune

    // Debug information
    bool debug_;
    bool detector_node_debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugLights>::SharedPtr lights_data_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugArmors>::SharedPtr armors_data_pub_;
    image_transport::Publisher binary_img_pub_;
    image_transport::Publisher number_img_pub_;
    image_transport::Publisher result_img_pub_;

    // tf
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

    // Target Locking
    /// 1. Define a dot as the canter of range circle & it radius
    /// 2. Target is locked when the dot is in the range circle with right click
    /// 3. Target will be locked as long as the right click is not released
    /// 4. Target will be unlocked when the right click is released
    /// 5. If right click is pressed but no target is in the lock circle, it will lock the target nearest to the center of lock circle
    /// 6. If right click is pressed and no target is detected in whole image, it will do nothing
    bool target_locked_ = false;
    int target_id_;
    cv::Point2f target_center_;
    float radius_ = 0.0;
};

}  // namespace rm_auto_aim

std::vector<double> orientationToRPY(const geometry_msgs::msg::Quaternion &_q)
{
    // Get armor yaw
    tf2::Quaternion tf_q;
    tf2::fromMsg(_q, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    std::vector<double> rpy = {roll, pitch, yaw};
    return rpy;
}

double getYawDiff(double _yaw1, double _yaw2)
{
    double diff = _yaw1 - _yaw2;
    if (diff > M_PI)
    {
        diff -= 2 * M_PI;
    }
    else if (diff < -M_PI)
    {
        diff += 2 * M_PI;
    }
    return diff;
}

#endif  // ARMOR_DETECTOR__ARMOR_DETECTOR_NODE_HPP_