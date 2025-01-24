#ifndef ARMOR_TRACKER__ARMOR_TRACKER_HPP_
#define ARMOR_TRACKER__ARMOR_TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// ROS
#include <angles/angles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>
#include <vector>

#include "armor_tracker/extended_kalman_filter.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{

class ArmorTracker
{
public:
  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  ArmorTracker(
    double max_match_distance, double max_match_yaw_diff, double s2qxyz, double s2qyaw, double s2qr,
    double line_s2qxyz, double line_s2qyaw, double line_s2qr, double line_s2qa, double r_xyz_factor,
    double r_yaw);
  using Armors = auto_aim_interfaces::msg::Armors;
  using Armor = auto_aim_interfaces::msg::Armor;

  void init(const Armors::SharedPtr & armors_msg);
  void update(const Armors::SharedPtr & armors_msg);

  ExtendedKalmanFilter ekf;
  ExtendedKalmanFilter line_ekf;

  int tracking_thres;
  int lost_thres;

  std::string tracked_id;
  Armor tracked_armor;
  ArmorsNum tracked_armors_num;

  double info_position_diff;
  double info_yaw_diff;

  double line_min_position_diff;
  double line_yaw_diff;

  Eigen::VectorXd measurement;
  Eigen::VectorXd target_state;
  Eigen::VectorXd line_target_state;
  Eigen::VectorXd final_state;

  // The time when the last message was received
  double dt_;
  // To store another pair of armors message
  double dz, another_r;

private:
  void initEKFMatrix();
  void initLineEKFMatrix();

  void initEKFs(const Armor & a);

  void updateArmorsNum(const Armor & a);

  void handleArmorJump(const Armor & a);

  double orientationToYaw(const geometry_msgs::msg::Quaternion & q);

  Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd & x);
  Eigen::Vector3d getArmorPositionFromLineState(const Eigen::VectorXd & x);

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;

  double last_yaw_;

  // EKF matrix
  double s2qxyz_, s2qyaw_, s2qr_;
  double line_s2qxyz_, line_s2qyaw_, line_s2qr_, line_s2qa_;
  double r_xyz_factor_, r_yaw_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_TRACKER__ARMOR_TRACKER_HPP_
