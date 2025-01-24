#ifndef __NEW_ARMOR_TRACKER_UTILS_TF_HPP__
#define __NEW_ARMOR_TRACKER_UTILS_TF_HPP__

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


static inline double orientationToYawWrap(const geometry_msgs::msg::Quaternion & q){
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
//   // Make yaw change continuous (-pi~pi to -inf~inf)
//   yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
//   last_yaw_ = yaw;
  return yaw;

}

#endif  // __NEW_ARMOR_TRACKER_UTILS_TF_HPP__