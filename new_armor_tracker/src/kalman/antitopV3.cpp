#include "new_armor_tracker/kalman/interface/antitopV3.h"
// #include "utils/print.h"
// #include "uniterm/uniterm.h"
#include <cmath>

#include "rclcpp/rclcpp.hpp"

// model
// [x, y, z, theta, vx, vy, vz, omega, r]    [x, y, z, theta]
// [0, 1, 2,   3,   4,  5,  6,    7,   8]    [0, 1, 2,   3  ]

// center model
// [ x, y, vx, vy ]    [ x, y ]
// [ 0, 1, 2,  3  ]    [ 0, 1 ]

// Omega model
// [ theta, omega, beta ]    [ theta ]
// [   0,     1,    2   ]    [   0   ]

namespace rm
{

AntitopV3::AntitopV3()
{
    t_ = rm::getTime();
    setMatrixQ(0.01, 0.01, 0.01, 0.02, 0.05, 0.05, 0.0001, 0.04, 0.001);
    setMatrixR(0.1, 0.1, 0.1, 0.2);
    setCenterMatrixQ(0.001, 0.001, 0.01, 0.01);
    setCenterMatrixR(1, 1);
    setOmegaMatrixQ(1, 1, 1);
    setOmegaMatrixR(1);
    weighted_z_ = new SlideWeightedAvg<double>[2] { 500, 500 };
}

AntitopV3::AntitopV3(double r_min, double r_max, int armor_num, bool enable_weighted)
    : r_min_(r_min), r_max_(r_max), armor_num_(armor_num), enable_weighted_(enable_weighted), model_()
{
    t_ = rm::getTime();
    setMatrixQ(0.01, 0.01, 0.01, 0.02, 0.05, 0.05, 0.0001, 0.04, 0.001);
    setMatrixR(0.1, 0.1, 0.1, 0.2);
    setCenterMatrixQ(0.01, 0.01, 0.05, 0.05);
    setCenterMatrixR(1, 1);
    setOmegaMatrixQ(1, 1, 1);
    setOmegaMatrixR(1);
    weighted_z_ = new SlideWeightedAvg<double>[2] { 500, 500 };
}

void AntitopV3::push(const Eigen::Matrix<double, 4, 1> &pose, TimePoint t)
{
    // get the dt between last time and incoming pose time
    double dt = getDoubleOfS(t_, t);
    if (dt > fire_delay_)
    {
        // if the delay is larger than fire_delay_, restart the model
        update_num_ = 0;
        model_.restart();
        RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "model restart");
    }
    update_num_++;
    t_ = t;

    if (armor_num_ == 2)
    {
        toggle_ = 0;
        if (dt > 0.05)
        {
            omega_model_.estimate_X[0] = pose[3];
            model_.estimate_X[3]       = pose[3];
            return;
        }
    }
    else
    {
        // armor jump handler
        toggle_ = getToggle(pose[3], omega_model_.estimate_X[0]);
        // RCLCPP_INFO(rclcpp::get_logger("new_tracker"), "Toggle: %d", toggle_);
        if (isAngleTrans(pose[3], omega_model_.estimate_X[0] + omega_model_.estimate_X[1] * dt))
        {
            RCLCPP_WARN(rclcpp::get_logger("new_tracker"), "armor jump");
            omega_model_.estimate_X[0] = pose[3];
            model_.estimate_X[3]       = pose[3];
            return;
        }
    }

    Eigen::Matrix<double, 1, 1> pose_theta(pose[3]);
    omega_model_.estimate_X[0] = getAngleTrans(pose[3], omega_model_.estimate_X[0], omega_model_.estimate_X[0] + omega_model_.estimate_X[1] * dt);
    omega_funcA_.dt            = dt;
    omega_model_.predict(omega_funcA_);
    omega_model_.update(omega_funcH_, pose_theta);

    model_.estimate_X[3] = getAngleTrans(pose[3], model_.estimate_X[3], model_.estimate_X[3] + model_.estimate_X[7] * dt);
    model_.estimate_X[2] = z_[toggle_];
    model_.estimate_X[8] = r_[toggle_];

    funcA_.dt = dt;
    model_.predict(funcA_);
    model_.update(funcH_, pose);
    model_.estimate_X[3] = omega_model_.estimate_X[0];
    model_.estimate_X[7] = omega_model_.estimate_X[1];
    model_.estimate_X[8] = std::min(std::max(model_.estimate_X[8], r_min_), r_max_);

    z_[toggle_] = model_.estimate_X[2];
    r_[toggle_] = model_.estimate_X[8];

    weighted_z_[toggle_].push(pose[2], getWeightByTheta(pose[3]));

    Eigen::Matrix<double, 2, 1> pose_center(model_.estimate_X[0], model_.estimate_X[1]);
    center_funcA_.dt = dt;
    center_model_.predict(center_funcA_);
    center_model_.update(center_funcH_, pose_center);

    // rm::message("antitop count", (int)update_num_);
    // rm::message("antitop toggle", toggle_);
}

Eigen::Matrix<double, 4, 1> AntitopV3::getPose(double append_delay)
{
    auto now         = rm::getTime();
    double sys_delay = getDoubleOfS(t_, now);

    if (sys_delay > fire_delay_)
    {
        return Eigen::Matrix<double, 4, 1>::Zero();
    }
    double dt = sys_delay + append_delay;

    // get the x y position of the center of robot at time t
    double x_center = model_.estimate_X[0] + model_.estimate_X[4] * dt;
    double y_center = model_.estimate_X[1] + model_.estimate_X[5] * dt;

    // double ekf_theta = model_.estimate_X[3] + model_.estimate_X[7] * dt;
    double kf_theta = omega_model_.estimate_X[0] + omega_model_.estimate_X[1] * dt;

    double theta = getAngleMin(kf_theta, x_center, y_center);

    double z;
    if (enable_weighted_)
    {
        z = weighted_z_[getToggle(theta, kf_theta)].getAvg();
    }
    else
    {
        z = z_[getToggle(theta, kf_theta)];
    }
    double r = r_[getToggle(theta, kf_theta)];

    double x = x_center - r * cos(theta);
    double y = y_center - r * sin(theta);

    Eigen::Matrix<double, 4, 1> pose(x, y, z, theta);
    return pose;
}

Eigen::Matrix<double, 4, 1> AntitopV3::getCenter(double append_delay)
{
    auto now = rm::getTime();
    // get the delay between last time modle update and now
    double sys_delay = getDoubleOfS(t_, now);

    // if the delay is larger than fire_delay_, return zero pose
    if (sys_delay > fire_delay_)
    {
        return Eigen::Matrix<double, 4, 1>::Zero();
    }
    double dt = sys_delay + append_delay;

    // estimate x y position of center of robot at time t
    double x_center = center_model_.estimate_X[0] + center_model_.estimate_X[2] * dt;
    double y_center = center_model_.estimate_X[1] + center_model_.estimate_X[3] * dt;

    // get the theta estimation of the armor plate from two modle
    // double ekf_theta = model_.estimate_X[3] + model_.estimate_X[7] * dt;
    double kf_theta = omega_model_.estimate_X[0] + omega_model_.estimate_X[1] * dt;

    //
    double theta = getAngleMin(kf_theta, x_center, y_center);

    double z;
    if (enable_weighted_)
    {
        z = weighted_z_[getToggle(theta, kf_theta)].getAvg();
    }
    else
    {
        z = z_[getToggle(theta, kf_theta)];
    }
    double r = r_[getToggle(theta, kf_theta)];

    double target_yaw = atan2(y_center, x_center);
    double x          = x_center - r * cos(target_yaw);
    double y          = y_center - r * sin(target_yaw);

    Eigen::Matrix<double, 4, 1> pose(x, y, z, theta);
    return pose;
}

double AntitopV3::getSafeSub(const double angle1, const double angle2)
{
    double angle = angle1 - angle2;
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;
    return angle;
}

// center angle, armor angle
double AntitopV3::getAngleTrans(const double target_angle, const double src_angle)
{
    double dst_angle = src_angle;

    while (getSafeSub(dst_angle, target_angle) > (M_PI / armor_num_))
        dst_angle -= (2 * M_PI) / armor_num_;
    while (getSafeSub(target_angle, dst_angle) > (M_PI / armor_num_))
        dst_angle += (2 * M_PI) / armor_num_;

    while (dst_angle > M_PI)
        dst_angle -= 2 * M_PI;
    while (dst_angle < -M_PI)
        dst_angle += 2 * M_PI;

    if (dst_angle * target_angle >= 0)
        return dst_angle;
    if (dst_angle > (M_PI / 2))
        dst_angle -= 2 * M_PI;
    else if (dst_angle < (-M_PI / 2))
        dst_angle += 2 * M_PI;
    return dst_angle;
}

double AntitopV3::getAngleTrans(const double target_angle, const double src_angle, double refer_angle)
{
    double dst_angle = src_angle;

    while (getSafeSub(refer_angle, target_angle) > (M_PI / armor_num_))
    {
        refer_angle -= (2 * M_PI) / armor_num_;
        dst_angle -= (2 * M_PI) / armor_num_;
    }
    while (getSafeSub(target_angle, refer_angle) > (M_PI / armor_num_))
    {
        refer_angle += (2 * M_PI) / armor_num_;
        dst_angle += (2 * M_PI) / armor_num_;
    }

    while (dst_angle > M_PI)
        dst_angle -= 2 * M_PI;
    while (dst_angle < -M_PI)
        dst_angle += 2 * M_PI;

    if (dst_angle * target_angle >= 0)
        return dst_angle;
    if (dst_angle > (M_PI / 2))
        dst_angle -= 2 * M_PI;
    else if (dst_angle < (-M_PI / 2))
        dst_angle += 2 * M_PI;
    return dst_angle;
}

// check if two angle diff is larger than M_PI / armor_num_
bool AntitopV3::isAngleTrans(const double target_angle, const double src_angle)
{
    double differ_angle = fabs(getSafeSub(target_angle, src_angle));
    if (differ_angle > (M_PI / armor_num_))
    {
        return true;
    }

    return false;
}

double AntitopV3::getAngleMin(double armor_angle, const double x, const double y)
{
    // get the angle between the center of robot and my center
    double center_angle = atan2(y, x);
    return getAngleTrans(center_angle, armor_angle);
}

int AntitopV3::getToggle(const double target_angle, const double src_angle)
{
    if (armor_num_ < 4)
        return 0;
    double differ_angle = fabs(getSafeSub(target_angle, src_angle));
    int differ_toggle   = static_cast<int>(round(2 * differ_angle / M_PI)) % 2;
    return (differ_toggle ^ toggle_);
}

double AntitopV3::getWeightByTheta(const double theta) { return exp(-pow(theta, 2) * 400); }

// void AntitopV3::getStateStr(std::vector<std::string> &str)
// {
//     str.push_back("AntitopV3");
//     str.push_back("  toggle: " + to_string(toggle_));
//     str.push_back("  update num: " + to_string(update_num_));
//     str.push_back("  ekf theta: " + to_string(model_.estimate_X[3] * 180 /
//     M_PI)); str.push_back("  ekf omega: " + to_string(model_.estimate_X[7]));
//     str.push_back("  kf theta: " + to_string(omega_model_.estimate_X[0] * 180
//     / M_PI)); str.push_back("  kf omega: " +
//     to_string(omega_model_.estimate_X[1])); str.push_back(" ");
// }

bool AntitopV3::getFireArmor(const Eigen::Matrix<double, 4, 1> &pose)
{
    double angle = getSafeSub(atan2(pose[1], pose[0]), pose[3]);
    if ((fabs(angle) < fire_armor_angle_) && (update_num_ > fire_update_))
        return true;
    return false;
}

bool AntitopV3::getFireCenter(const Eigen::Matrix<double, 4, 1> &pose)
{
    double angle = getSafeSub(atan2(pose[1], pose[0]), pose[3]);
    if ((fabs(angle) < fire_center_angle_) && (update_num_ > fire_update_))
        return true;
    return false;
}

void AntitopV3::getStatus(auto_aim_interfaces::msg::NewTarget &_msg)
{
    // auto now = rm::getTime();
    // double sys_delay = getDoubleOfS(t_, now);
    // if (sys_delay > fire_delay_)
    // {
    //     msg.status = 0;
    //     return;
    // }
    // double dt = sys_delay;

    // double x_center = model_.estimate_X[0] + model_.estimate_X[4] * dt;
    // double y_center = model_.estimate_X[1] + model_.estimate_X[5] * dt;

    // double kf_theta = omega_model_.estimate_X[0] + omega_model_.estimate_X[1] *
    // dt;

    // double theta = getAngleMin(kf_theta, x_center, y_center);

    // double z;
    // if (enable_weighted_)
    // {
    //     z = weighted_z_[getToggle(theta, kf_theta)].getAvg();
    // }
    // else
    // {
    //     z = z_[getToggle(theta, kf_theta)];
    // }
    // double r = r_[getToggle(theta, kf_theta)];

    // double target_yaw = atan2(y_center, x_center);
    // double x          = x_center - r * cos(target_yaw);
    // double y          = y_center - r * sin(target_yaw);

    // msg.status = 1;
    // msg.pose.position.x = x;
    // msg.pose.position.y = y;
    // msg.pose.position.z = z;
    // msg.pose.orientation.z = theta;
    // _msg.header.stamp     = this->t_;
    _msg.antitop_t        = this->t_;
    _msg.armor_num        = armor_num_;
    _msg.model_position.x = model_.estimate_X[0];
    _msg.model_position.y = model_.estimate_X[1];
    _msg.model_position.z = model_.estimate_X[2];
    _msg.model_velocity.x = model_.estimate_X[4];
    _msg.model_velocity.y = model_.estimate_X[5];
    _msg.model_velocity.z = model_.estimate_X[6];

    _msg.center_model_position.x = center_model_.estimate_X[0];
    _msg.center_model_position.y = center_model_.estimate_X[1];
    _msg.center_model_velocity.x = center_model_.estimate_X[2];
    _msg.center_model_velocity.y = center_model_.estimate_X[3];

    // _msg.z1 = weighted_z_[this->toggle_].getAvg();
    // _msg.z2 = weighted_z_[!this->toggle_].getAvg();
    if (enable_weighted_)
    {
        _msg.z1 = weighted_z_[0].getAvg();
        _msg.z2 = weighted_z_[1].getAvg();
    }
    else
    {
        _msg.z1 = z_[0];
        _msg.z2 = z_[1];
    }

    // _msg.r1 = r_[this->toggle_];
    // _msg.r2 = r_[!this->toggle_];
    _msg.r1 = r_[0];
    _msg.r2 = r_[1];

    _msg.toggle = toggle_;

    _msg.yaw   = omega_model_.estimate_X[0];
    _msg.omega_v_yaw = omega_model_.estimate_X[1];
    _msg.model_v_yaw = model_.estimate_X[7];
}

void AntitopV3::setMatrixQ(double q0, double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8)
{
    model_.Q << q0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q1, 0, 0, 0, 0, 0, 0, 0, 0, 0, q2, 0, 0, 0, 0, 0, 0, 0, 0, 0, q3, 0, 0, 0, 0, 0, 0, 0, 0, 0, q4, 0, 0,
        0, 0, 0, 0, 0, 0, 0, q5, 0, 0, 0, 0, 0, 0, 0, 0, 0, q6, 0, 0, 0, 0, 0, 0, 0, 0, 0, q7, 0, 0, 0, 0, 0, 0, 0, 0, 0, q8;
}

void AntitopV3::setMatrixR(double r0, double r1, double r2, double r3) { model_.R << r0, 0, 0, 0, 0, r1, 0, 0, 0, 0, r2, 0, 0, 0, 0, r3; }

void AntitopV3::setCenterMatrixQ(double q0, double q1, double q2, double q3)
{
    center_model_.Q << q0, 0, 0, 0, 0, q1, 0, 0, 0, 0, q2, 0, 0, 0, 0, q3;
}
void AntitopV3::setCenterMatrixR(double r0, double r1) { center_model_.R << r0, 0, 0, r1; }

void AntitopV3::setOmegaMatrixQ(double q0, double q1, double q2) { omega_model_.Q << q0, 0, 0, 0, q1, 0, 0, 0, q2; }

void AntitopV3::setOmegaMatrixR(double r0) { omega_model_.R << r0; }

}  // namespace rm