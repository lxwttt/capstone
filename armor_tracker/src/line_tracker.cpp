#include "armor_tracker/line_tracker.hpp"

namespace rm_auto_aim
{
    // EKF for line (xa = x_armor, xc = x_robot_center)
    //
    //        0    1     2     3   4     5     6   7   8    9  
    // state: xc, v_xc, a_xc, yc, v_yc, a_yc,  z,  vz, yaw, r 
    // measurement: xa, ya, za, yaw
LineTracker::LineTracker(double max_match_distance, double max_match_yaw_diff,

 double s2qxyz, double s2qyaw, double s2qr, double s2qa, double r_xyz_factor, double r_yaw)
: tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
  target_state(Eigen::VectorXd::Zero(9)),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff),
  s2qxyz_(s2qxyz),
  s2qyaw_(s2qyaw),
  s2qr_(s2qr),
  s2qa_(s2qa),
  r_xyz_factor_(r_xyz_factor),
  r_yaw_(r_yaw)
{
  initMatrix();
}

void LineTracker::init(const Armors::SharedPtr & armors_msg)
{
  if (armors_msg->armors.empty()) 
  {
    return;
  }

  // Simply choose the armor that is closest to image center
  double min_distance = DBL_MAX;
  tracked_armor = armors_msg->armors[0];
  for (const auto & armor : armors_msg->armors) 
  {
    if (armor.distance_to_image_center < min_distance)
    {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  initEKF(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init Line Tracker EKF!");

  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;

  updateArmorsNum(tracked_armor);
}

void LineTracker::update(const Armors::SharedPtr & armors_msg)
{
  // KF predict
  Eigen::VectorXd ekf_prediction = ekf.predict();
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Line tracker EKF predict");

  bool matched = false;
  // Use KF prediction as default target state if no matched armor is found
  target_state = ekf_prediction;

  if (!armors_msg->armors.empty()) 
  {
    // Find the closest armor with the same id
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;
    for (const auto & armor : armors_msg->armors) 
    {
      // Only consider armors with the same id
      if (armor.number == tracked_id) 
      {
        same_id_armor = armor;
        same_id_armors_count++;
        // Calculate the difference between the predicted position and the current armor position
        auto p = armor.pose.position;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
        if (position_diff < min_position_diff) 
        {
          // Find the closest armor
          min_position_diff = position_diff;
          yaw_diff = abs(orientationToYaw(armor.pose.orientation) - ekf_prediction(8));
          tracked_armor = armor;
        }
      }
    }

    // Store tracker info
    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;
    
    // Check if the distance and yaw difference of closest armor are within the threshold
    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) 
    {
      // Matched armor found
      matched = true;
      auto p = tracked_armor.pose.position;
      // Update EKF
      double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
      RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Line tracker EKF update");
    } 
    else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) 
    {
      handleArmorJump(same_id_armor);
    } 
    else 
    {
      // Matched armor found
      matched = true;
      auto p = tracked_armor.pose.position;
      // Update EKF
      double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
      RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Line tracker no matched armor found! min_position_diff: %f, yaw_diff: %f",min_position_diff,yaw_diff);
    }
  }

  // Prevent radius from spreading
  if (target_state(9) < 0.12) {
    target_state(9) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(9) > 0.4) {
    target_state(9) = 0.4;
    ekf.setState(target_state);
  }

  // Tracking state machine
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

void LineTracker::initEKF(const Armor & a)
{
  double xa = a.pose.position.x;
  double ya = a.pose.position.y;
  double za = a.pose.position.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.pose.orientation);

  // Set initial position at 0.2m behind the target
  target_state = Eigen::VectorXd::Zero(10);
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0, another_r = r;
  target_state << xc, 0, 0, yc, 0, 0, za, 0, yaw, r;

  ekf.setState(target_state);
  tracker_state = DETECTING;
}

void LineTracker::updateArmorsNum(const Armor & armor)
{
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void LineTracker::handleArmorJump(const Armor & current_armor)
{
  double yaw = orientationToYaw(current_armor.pose.orientation);
  target_state(8) = yaw;
  updateArmorsNum(current_armor);
  // Only 4 armors has 2 radius and height
  if (tracked_armors_num == ArmorsNum::NORMAL_4) {
    dz = target_state(6) - current_armor.pose.position.z;
    target_state(6) = current_armor.pose.position.z;
    std::swap(target_state(9), another_r);
  }
  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Line tracker Armor jump!");

  // If position difference is larger than max_match_distance_,
  // take this case as the ekf diverged, reset the state
  auto p = current_armor.pose.position;
  Eigen::Vector3d current_p(p.x, p.y, p.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_) {
    double r = target_state(9);
    target_state(0) = p.x + r * cos(yaw);  // xc
    target_state(1) = 0;                   // vxc
    target_state(2) = 0;                   // axc
    target_state(3) = p.y + r * sin(yaw);  // yc
    target_state(4) = 0;                   // vyc
    target_state(5) = 0;                   // ayc
    target_state(6) = p.z;                 // za
    target_state(7) = 0;                   // vza
    RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Line tracker Reset State!");
  }

  ekf.setState(target_state);
}

double LineTracker::orientationToYaw(const geometry_msgs::msg::Quaternion & q)
{
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  // Make yaw change continuous (-pi~pi to -inf~inf)
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

Eigen::Vector3d LineTracker::getArmorPositionFromState(const Eigen::VectorXd & x)
{
  // Calculate predicted position of the current armor
  double xc = x(0), yc = x(3), za = x(6);
  double yaw = x(8), r = x(9);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

void LineTracker::initMatrix()
{
    // f_line - Process function for line
    auto f = [this](const Eigen::VectorXd & x) 
    {
      Eigen::VectorXd x_new = x;
      x_new(0) += x(1) * dt_ + 0.5*x(2)*dt_*dt_;
      x_new(1) += x(2) * dt_;
      x_new(3) += x(4) * dt_ + 0.5*x(5)*dt_*dt_;
      x_new(4) += x(5) * dt_;
      x_new(6) += x(7) * dt_;
      return x_new;
    };

    // J_f_line - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &) 
    {
      Eigen::MatrixXd f(10, 10);
      // clang-format off
      //      x   vx    ax          y   vy    ay          z   vz    yaw r
      f <<    1,  dt_,  dt_*dt_/2,  0,  0,    0,          0,  0,    0,  0,
              0,  1,    dt_,        0,  0,    0,          0,  0,    0,  0,
              0,  0,    1,          0,  0,    0,          0,  0,    0,  0,
              0,  0,    0,          1,  dt_,  dt_*dt_/2,  0,  0,    0,  0,
              0,  0,    0,          0,  1,    dt_,        0,  0,    0,  0,
              0,  0,    0,          0,  0,    1,          0,  0,    0,  0,
              0,  0,    0,          0,  0,    0,          1,  dt_,  0,  0,
              0,  0,    0,          0,  0,    0,          0,  1,    0,  0,
              0,  0,    0,          0,  0,    0,          0,  0,    1,  0,
              0,  0,    0,          0,  0,    0,          0,  0,    0,  1;
      return f;
    };

    // h_line - Observation function
    auto h = [this](const Eigen::VectorXd & x) 
    {
      Eigen::VectorXd z(4);
      double xc = x(0), yc = x(3), yaw = x(8), r = x(9);
      z(0) = xc - r * cos(yaw);  // xa
      z(1) = yc - r * sin(yaw);  // ya
      z(2) = x(6);               // za
      z(3) = x(8);               // yaw
      return z;
    };

    // J_h_line - Jacobian of observation function
    auto j_h = [this](const Eigen::VectorXd & x) 
    {
        Eigen::MatrixXd h(4, 10);
        double yaw = x(8), r = x(9);
        // clang-format off
        //      x   vx    ax   y   vy   ay    z   vz      yaw           r
        h <<    1,   0,   0,   0,   0,   0,   0,   0,  r*sin(yaw),  -cos(yaw),
                0,   0,   0,   1,   0,   0,   0,   0,  -r*cos(yaw), -sin(yaw),
                0,   0,   0,   0,   0,   0,   1,   0,      0,           0,    
                0,   0,   0,   0,   0,   0,   0,   0,      1,           0;
        // clang-format on
        return h;
    };

    // update_Q_line - process noise covariance matrix
        auto u_q = [this]() 
    {
        Eigen::MatrixXd q(10, 10);
        // double 
        double x = s2qxyz_, yaw = s2qyaw_, r = s2qr_;
        double t = dt_;
        // double q_x_ax = pow(t,5)/5 * x;
        // double q_vx_ax = pow(t,4)/4 * x;
        // double q_ax_ax = pow(t,4)/4 * a;
        // double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        // double q_yaw = pow(t, 4) / 4 * yaw;
        // double q_r   = pow(t, 4) / 4 * r;
        // double q_a   = pow(t, 4) / 4 * a;


        // double q_x_x = x;
        // double q_ax_ax = a;
        // double q_vx_vx = x;
        // double q_x_vx = x;
        // double q_x_ax = x;
        // double q_vx_ax = x;
        // double q_yaw = yaw;
        // double q_r = r;

        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * yaw;
        double q_r = pow(t, 4) / 4 * r;
        double q_x_ax =0;
        double q_vx_ax =0;
        double q_ax_ax = s2qa_;

        // clang-format off
        //          x       vx        ax         y       vy         ay        z        vz       yaw      r
        q    << q_x_x,    q_x_vx,     q_x_ax,    0,       0,        0,        0,       0,        0,      0,
                q_x_vx,   q_vx_vx,    q_vx_ax,   0,       0,        0,        0,       0,        0,      0,
                q_x_ax,   q_vx_ax,    q_ax_ax,   0,       0,        0,        0,       0,        0,      0,  
                0,        0,          0,         q_x_x,   q_x_vx,   q_x_ax,   0,       0,        0,      0,
                0,        0,          0,         q_x_vx,  q_vx_vx,  q_vx_ax,  0,       0,        0,      0,
                0,        0,          0,         q_x_ax,  q_vx_ax,  q_ax_ax,  0,       0,        0,      0,
                0,        0,          0,         0,        0,       0,        q_x_x,   q_x_vx,   0,      0,
                0,        0,          0,         0,        0,       0,        q_x_vx,  q_vx_vx,  0,      0,
                0,        0,          0,         0,        0,       0,        0,       0,        q_y_y,  0,
                0,        0,          0,         0,        0,       0,        0,       0,        0,      q_r;        
        // clang-format on
        return q;
    };
    // auto u_q = [this]() 
    // {
    //     Eigen::MatrixXd q(10, 10);
    //     double t = dt_, x = s2qxyz_, yaw = s2qyaw_, r = s2qr_, a = s2qa_ ;
    //     double q_x_ax = pow(t,5)/5 * x;
    //     double q_vx_ax = pow(t,4)/4 * x;
    //     double q_ax_ax = pow(t,4)/4 * a;
    //     double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    //     double q_yaw = pow(t, 4) / 4 * yaw;
    //     double q_r   = pow(t, 4) / 4 * r;
    //     // double q_a   = pow(t, 4) / 4 * a/
    //     // clang-format off
    //     //          x       vx        ax         y       vy         ay        z        vz       yaw      r
    //     q    << q_x_x,    q_x_vx,     q_x_ax,    0,       0,        0,        0,       0,        0,      0,
    //             q_x_vx,   q_vx_vx,    q_vx_ax,   0,       0,        0,        0,       0,        0,      0,
    //             q_x_ax,   q_vx_ax,    q_ax_ax,   0,       0,        0,        0,       0,        0,      0,  
    //             0,        0,          0,         q_x_x,   q_x_vx,   q_x_ax,   0,       0,        0,      0,
    //             0,        0,          0,         q_x_vx,  q_vx_vx,  q_vx_ax,  0,       0,        0,      0,
    //             0,        0,          0,         q_x_ax,  q_vx_ax,  q_ax_ax,  0,       0,        0,      0,
    //             0,        0,          0,         0,        0,       0,        q_x_x,   q_x_vx,   0,      0,
    //             0,        0,          0,         0,        0,       0,        q_x_vx,  q_vx_vx,  0,      0,
    //             0,        0,          0,         0,        0,       0,        0,       0,        q_yaw,  0,
    //             0,        0,          0,         0,        0,       0,        0,       0,        0,      q_r;        
    //     // clang-format on
    //     return q;
    // };


  // update_R - measurement noise covariance matrix
  auto u_r = [this](const Eigen::VectorXd & z) {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = r_xyz_factor_;
    r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw_;
    return r;
  };
  
    // P_line - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 10> p0;
    p0.setIdentity();
    ekf = ExtendedKalmanFilter{f,h,j_f,j_h,u_q,u_r,p0};
  return ;
}
}  // namespace rm_auto_aim