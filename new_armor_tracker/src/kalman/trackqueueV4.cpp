#include "new_armor_tracker/kalman/interface/trackqueueV4.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"

// #include "utils/print.h"
// #include "uniterm/uniterm.h"

namespace rm
{

// [ x, y, z, v, vz, angle, w, a ]  [ x, y, z ]
// [ 0, 1, 2, 3, 4,    5,   6, 7 ]  [ 0, 1, 2 ]

static std::mutex mtx_;

TrackQueueV4::TrackQueueV4(int count, double distance, double delay) : count_(count), distance_(distance), delay_(delay)
{
    setMatrixQ(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    setMatrixR(0.1, 0.1, 0.1);
}

void TrackQueueV4::push(Eigen::Matrix<double, 4, 1> &input_pose, TimePoint t)
{
    std::unique_lock<std::mutex> lock(mtx_);
    Eigen::Matrix<double, 3, 1> pose;
    pose << input_pose[0], input_pose[1], input_pose[2];

    double min_distance   = 1e4;
    TQstateV4 *best_state = nullptr;

    // if queue is not empty, iterate through the list
    for (auto it = list_.begin(); it != list_.end();)
    {
        // get the state of current iteration
        TQstateV4 *state = *it;
        // get the dt between the time when this state was pushed and the current time
        double dt = rm::getDoubleOfS(state->last_t, t);

        // if dt is larger than delay_ or keep is less than 0, delete this state
        if ((dt > delay_) || (state->keep <= 0))
        {
            if (last_state_ == state)
                last_state_ = nullptr;
            delete *it;
            // erase the state from the list, then remained state will move forward by one index, now it points to the next state
            it = list_.erase(it);
        }
        else
        {
            // counter ++, weird to put it here, can ignore
            ++it;
            // get the predicted pose of the state from the last time it was pushed to the current time
            double predict_x = state->model->estimate_X[0] + dt * state->model->estimate_X[3] * cos(state->model->estimate_X[5]);
            double predict_y = state->model->estimate_X[1] + dt * state->model->estimate_X[3] * sin(state->model->estimate_X[5]);
            double predict_z = state->model->estimate_X[2];

            Eigen::Matrix<double, 3, 1> predict_pose;
            predict_pose << predict_x, predict_y, predict_z;

            // get the distance between the predicted pose and the input pose
            double distance = getDistance(pose, predict_pose);
            // get the minimum distance and the state with the minimum distance between the predicted pose and the input pose
            if (distance < min_distance)
            {
                min_distance = distance;
                best_state   = state;
            }
        }
    }
    // if thes best state is not available or the minimum distance is larger than distance_, create a new state
    if (best_state == nullptr || min_distance > distance_)
    {
        best_state           = new TQstateV4();
        best_state->model->Q = matrixQ_;
        best_state->model->R = matrixR_;
        best_state->refresh(input_pose, t);

        // update the modle of the new state
        funcA_.dt = 0;
        best_state->model->predict(funcA_);
        best_state->model->update(funcH_, pose);

        list_.push_back(best_state);
    }
    else
    {
        // if the best state is available and the minimum distance is less than distance_, update the model of the best state
        funcA_.dt = rm::getDoubleOfS(best_state->last_t, t);
        best_state->refresh(input_pose, t);
        best_state->model->predict(funcA_);
        best_state->model->update(funcH_, pose);
    }
}

void TrackQueueV4::update()
{
    std::unique_lock<std::mutex> lock(mtx_);
    for (auto it = list_.begin(); it != list_.end(); ++it)
    {
        // decrease keep counter for every time update
        (*it)->keep -= 1;
    }
}

void TrackQueueV4::setMatrixQ(double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8)
{
    matrixQ_ << q1, 0, 0, 0, 0, 0, 0, 0, 0, q2, 0, 0, 0, 0, 0, 0, 0, 0, q3, 0, 0, 0, 0, 0, 0, 0, 0, q4, 0, 0, 0, 0, 0, 0, 0, 0, q5, 0, 0, 0, 0, 0, 0,
        0, 0, q6, 0, 0, 0, 0, 0, 0, 0, 0, q7, 0, 0, 0, 0, 0, 0, 0, 0, q8;
}

void TrackQueueV4::setMatrixR(double r1, double r2, double r3) { matrixR_ << r1, 0, 0, 0, r2, 0, 0, 0, r3; }

// void TrackQueueV4::getStateStr(std::vector<std::string>& str) {
//     str.push_back("TrackQueueV4:");
//     str.push_back(" ");
//     for(size_t i = 0; i < list_.size(); i++) {
//         str.push_back("Track " + to_string(i) + ":");
//         str.push_back(" count: " + to_string(list_[i]->count));
//         str.push_back(" keep: " + to_string(list_[i]->keep));
//         str.push_back(" ");
//     }
// }

Eigen::Matrix<double, 4, 1> TrackQueueV4::getPose(double append_delay)
{
    std::unique_lock<std::mutex> lock(mtx_);

    TQstateV4 *state = nullptr;
    // if last state is available, if the delay is less than delay_ and keep > 0, use the last state
    if (last_state_ != nullptr)
    {
        double dt = rm::getDoubleOfS(last_state_->last_t, rm::getTime());
        if ((dt < delay_) && (last_state_->keep >= 0))
        {
            state = last_state_;
        }
        else
        {
            last_state_ = nullptr;
        }
    }

    // else, find the state with the largest count
    if (last_state_ == nullptr)
    {
        int max_count = -1;
        for (auto it = list_.begin(); it != list_.end(); ++it)
        {
            double dt = rm::getDoubleOfS((*it)->last_t, rm::getTime());
            if ((dt > delay_) || ((*it)->keep <= 0))
                continue;

            if ((*it)->count > max_count)
            {
                max_count = (*it)->count;
                state     = *it;
            }
        }
    }

    if (state != nullptr)
    {
        last_state_ = state;

        // get the estimated pose on next time
        double sys_delay = rm::getDoubleOfS(state->last_t, rm::getTime());
        double dt        = sys_delay + append_delay;
        double x         = state->model->estimate_X[0] + dt * state->model->estimate_X[3] * cos(state->model->estimate_X[5]);
        double y         = state->model->estimate_X[1] + dt * state->model->estimate_X[3] * sin(state->model->estimate_X[5]);
        double z         = state->model->estimate_X[2] + dt * state->model->estimate_X[4];

        return Eigen::Matrix<double, 4, 1>(x, y, z, 0);
    }
    else
    {
        return Eigen::Matrix<double, 4, 1>::Zero();
    }
}

bool TrackQueueV4::getPose(Eigen::Matrix<double, 4, 1> &pose, TimePoint &t)
{
    std::unique_lock<std::mutex> lock(mtx_);

    std::vector<TQstateV4 *> available_state;
    // get available state, drop the state which is too old or keep <= 0
    for (auto it = list_.begin(); it != list_.end(); ++it)
    {
        double dt = rm::getDoubleOfS((*it)->last_t, rm::getTime());
        if ((dt > delay_) || ((*it)->keep <= 0))
            continue;

        if ((*it)->available)
        {
            (*it)->available = false;
            // if the state is available for 2 times, push it into available_state
            if ((*it)->count > 2)
                available_state.push_back(*it);
        }
    }
    // possible sinaario for no available state: 1. the target is too new
    if (available_state.size() == 0)
    {
        pose = Eigen::Matrix<double, 4, 1>::Zero();
        t    = rm::getTime();
        return false;
    }
    else if (available_state.size() == 1)
    {
        pose = available_state[0]->last_pose;
        t    = available_state[0]->last_t;
        return true;
    }
    // sort count in descending order, the first one is the latest one
    std::sort(available_state.begin(), available_state.end(), [](TQstateV4 *a, TQstateV4 *b) { return a->count > b->count; });
    // get the latest one
    pose = available_state[0]->last_pose;
    t    = available_state[0]->last_t;

    return true;
}

void TrackQueueV4::getStatus(auto_aim_interfaces::msg::NewTarget &_msg)
{
    std::unique_lock<std::mutex> lock(mtx_);

    TQstateV4 *state = nullptr;
    // if last state is available, if the delay is less than delay_ and keep > 0, use the last state
    if (last_state_ != nullptr)
    {
        double dt = getDoubleOfS(last_state_->last_t, getTime());
        if ((dt < delay_) && (last_state_->keep >= 0))
        {
            state = last_state_;
        }
        else
        {
            last_state_ = nullptr;
        }
    }

    // else, find the state with the largest count
    if (last_state_ == nullptr)
    {
        int max_count = -1;
        // double dt = sys_delay + append_delay;
        // double x = state->model->estimate_X[0] + dt * state->model->estimate_X[3] * cos(state->model->estimate_X[5]);
        // double y = state->model->estimate_X[1] + dt * state->model->estimate_X[3] * sin(state->model->estimate_X[5]);
        // double z = state->model->estimate_X[2] + dt * state->model->estimate_X[4];
        for (auto it = list_.begin(); it != list_.end(); ++it)
        {
            double dt = getDoubleOfS((*it)->last_t, getTime());
            if ((dt > delay_) || ((*it)->keep <= 0))
                continue;

            if ((*it)->count > max_count)
            {
                max_count = (*it)->count;
                state     = *it;
            }
        }
    }

    if (state != nullptr)
    {
        last_state_ = state;

        // get the estimated pose on next time
        // double sys_delay = getDoubleOfS(state->last_t, getTime());
        // double dt = sys_delay + append_delay;
        // double x = state->model->estimate_X[0] + dt * state->model->estimate_X[3] * cos(state->model->estimate_X[5]);
        // double y = state->model->estimate_X[1] + dt * state->model->estimate_X[3] * sin(state->model->estimate_X[5]);
        // double z = state->model->estimate_X[2] + dt * state->model->estimate_X[4];

        _msg.trackqueue_t = state->last_t;

        _msg.armor_position.x = state->model->estimate_X[0];
        _msg.armor_position.y = state->model->estimate_X[1];
        _msg.armor_position.z = state->model->estimate_X[2];

        _msg.armor_velocity.x = state->model->estimate_X[3] * cos(state->model->estimate_X[5]);
        _msg.armor_velocity.y = state->model->estimate_X[3] * sin(state->model->estimate_X[5]);
        _msg.armor_velocity.z = state->model->estimate_X[4];

        _msg.armor_yaw   = state->model->estimate_X[5];
        _msg.armor_v_yaw = state->model->estimate_X[6];
    }
    else
    {
        _msg.tracking         = false;
        _msg.armor_position.x = 0;
        _msg.armor_position.y = 0;
        _msg.armor_position.z = 0;

        _msg.armor_velocity.x = 0;
        _msg.armor_velocity.y = 0;
        _msg.armor_velocity.z = 0;
    }
}

double TrackQueueV4::getDistance(const Eigen::Matrix<double, 4, 1> &this_pose, const Eigen::Matrix<double, 4, 1> &last_pose)
{
    double dx = this_pose(0) - last_pose(0);
    double dy = this_pose(1) - last_pose(1);
    double dz = this_pose(2) - last_pose(2);
    double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    return d;
}

double TrackQueueV4::getDistance(const Eigen::Matrix<double, 3, 1> &this_pose, const Eigen::Matrix<double, 3, 1> &last_pose)
{
    double dx = this_pose(0) - last_pose(0);
    double dy = this_pose(1) - last_pose(1);
    double dz = this_pose(2) - last_pose(2);
    double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    return d;
}

bool TrackQueueV4::getFireFlag()
{
    if (last_state_ == nullptr)
        return false;
    double dt = rm::getDoubleOfS(last_state_->last_t, rm::getTime());
    if ((last_state_->count > count_) && (dt < delay_))
        return true;
    else
        return false;
}

}  // namespace rm