#ifndef __NEW_ARMOR_TRACKER__INTERFACE__TRACKQUEUEV4_H__
#define __NEW_ARMOR_TRACKER__INTERFACE__TRACKQUEUEV4_H__
#include <memory>
#include <vector>
// #include <utils/timer.h>
// #include <kalman/filter/ekf.h>
#include "new_armor_tracker/utils/timer.hpp"
#include "new_armor_tracker/kalman/filter/ekf.h"
#include "auto_aim_interfaces/msg/new_target.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include <structure/slidestd.hpp>

// [ x, y, z, v, vz, angle, w, a ]  [ x, y, z ]
// [ 0, 1, 2, 3, 4,    5,   6, 7 ]  [ 0, 1, 2 ]

using TimePoint = rclcpp::Time;

namespace rm {

struct TrackQueueV4_FuncA {
    template<class T>
    void operator()(const T x0[8], T x1[8]) {
        x1[0] = x0[0] + dt * x0[3] * ceres::cos(x0[5]) + 0.5 * dt * dt * x0[7] * ceres::cos(x0[5]);
        x1[1] = x0[1] + dt * x0[3] * ceres::sin(x0[5]) + 0.5 * dt * dt * x0[7] * ceres::sin(x0[5]);
        x1[2] = x0[2] + dt * x0[4];
        x1[3] = x0[3] + dt * x0[7];
        x1[4] = x0[4];
        x1[5] = x0[5] + dt * x0[6];
        x1[6] = x0[6];
        x1[7] = x0[7];
    }
    double dt;
};

struct TrackQueueV4_FuncH {
    template<typename T>
    void operator()(const T x[8], T y[3]) {
        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
    }
};

class TQstateV4 {
public:
    // TimePoint last_t;                       // 目标上一次的时间
    rclcpp::Time last_t;
    Eigen::Matrix<double, 4, 1> last_pose;  // 目标上一次的位置
    EKF<8, 3> *model;                      // 目标运动模型
    int count;                              // 此目标更新计数
    int keep;                               // 此目标保持计数
    bool available;                         // 此目标是否可用

    TQstateV4() : count(0), keep(50), available(false) {
        last_t = rm::getTime();
        model = new EKF<8, 3>();
    }

    void refresh(const Eigen::Matrix<double, 4, 1>& pose, TimePoint t) {
        this->last_t = t;
        this->last_pose = pose;
        this->count += 1;
        this->keep = 50;
        this->available = true;
    }
};

class TrackQueueV4 {
public:
    TrackQueueV4() {}
    TrackQueueV4(int count, double distance, double delay);
    ~TrackQueueV4() {}

    void push(Eigen::Matrix<double, 4, 1>& pose, TimePoint t);                       // 推入单次目标信息
    void update();                                                                   // 每帧更新一次

public:
    void setCount(int c) { this->count_ = c; }                                       // 设置认为模型可用的最小更新次数
    void setDistance(double d) { this->distance_ = d; }                              // 设置认为是同一个目标的最大移动距离
    void setDelay(double d) { this->delay_ = d; }                                    // 设置模型不重置的最大延迟
    void setMatrixQ(double, double, double, double, double, double, double, double);
    void setMatrixR(double, double, double);

    Eigen::Matrix<double, 4, 1> getPose(double append_delay);                        // 获取根据模型预测的位姿
    bool getPose(Eigen::Matrix<double, 4, 1>& pose, TimePoint& t);                    // 获取
    void getStatus(auto_aim_interfaces::msg::NewTarget& _msg);                       // 获取目标信息
    
    void getStateStr(std::vector<std::string>& str);                                 // 获取目标状态信息字符串
    bool getFireFlag();                                                              // 判断是否满足开火条件


private:
    double getDistance(
        const Eigen::Matrix<double, 4, 1>& this_pose,
        const Eigen::Matrix<double, 4, 1>& last_pose);                               // 两目标之间的距离
    double getDistance(
        const Eigen::Matrix<double, 3, 1>& this_pose,
        const Eigen::Matrix<double, 3, 1>& last_pose);                               // 两目标之间的距离
    
private:
    int    count_        = 10;              // 可维持状态稳定的最小更新次数
    double distance_     = 0.15;            // 可认为是同一个目标的最大移动距离
    double delay_        = 0.5;             // 可认为是同一个目标的最大更新延迟

    TQstateV4* last_state_   = nullptr;     // 上一次的状态

    TrackQueueV4_FuncA funcA_;              // 运动模型的状态转移函数
    TrackQueueV4_FuncH funcH_;              // 运动模型的观测函数

    Eigen::Matrix<double, 8, 8> matrixQ_;   // 运动模型的过程噪声协方差矩阵
    Eigen::Matrix<double, 3, 3> matrixR_;   // 运动模型的观测噪声协方差矩阵

public:
    std::vector<TQstateV4*> list_;          // 目标状态列表
};

}

#endif