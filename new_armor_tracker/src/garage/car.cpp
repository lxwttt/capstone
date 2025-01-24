#include "new_armor_tracker/garage/car.hpp"

#include "new_armor_tracker/utils/tf.hpp"
#include "rclcpp/rclcpp.hpp"

// double orientationToYawWrap(const geometry_msgs::msg::Quaternion &q);

Car::Car(uint8_t robot_id, std::vector<std::vector<double>> &_ekf_params) : IRobot(robot_id)
{
    robot_id_   = robot_id;
    track_queue = rm::TrackQueueV4(10, 0.20, 0.5);
    antitop_4   = new rm::AntitopV3(0.15, 0.4, 4);
    antitop_2   = new rm::AntitopV3(0.15, 0.4, 2);

    // clang-format off
    // [ x, y, z, v, vz, angle, w, a ]  [ x, y, z ]
    track_queue.setMatrixQ(
        _ekf_params[0][0], _ekf_params[0][1], _ekf_params[0][2],
        _ekf_params[0][3], _ekf_params[0][4], _ekf_params[0][5], 
        _ekf_params[0][6], _ekf_params[0][7]
    );
    track_queue.setMatrixR(_ekf_params[1][0], _ekf_params[1][1], _ekf_params[1][2]);
    // clang-format on

    antitop_4->setMatrixQ(
        //  x,   y,   z, theta,  vx,  vy,  vz, omega,   r
        _ekf_params[2][0],
        _ekf_params[2][1],
        _ekf_params[2][2],
        _ekf_params[2][3],
        _ekf_params[2][4],
        _ekf_params[2][5],
        _ekf_params[2][6],
        _ekf_params[2][7],
        _ekf_params[2][8]);
    // [ x, y, z, theta]
    antitop_4->setMatrixR(_ekf_params[3][0], _ekf_params[3][1], _ekf_params[3][2], _ekf_params[3][3]);

    // x    y    z    theta
    antitop_4->setCenterMatrixQ(_ekf_params[4][0], _ekf_params[4][1], _ekf_params[4][2], _ekf_params[4][3]);

    antitop_4->setCenterMatrixR(_ekf_params[5][0], _ekf_params[5][1]);
    // [ theta, omega, beta ]
    antitop_4->setOmegaMatrixQ(_ekf_params[6][0], _ekf_params[6][1], _ekf_params[6][2]);
    antitop_4->setOmegaMatrixR(_ekf_params[7][0]);

    antitop_2->setMatrixQ(
        //  x,   y,   z, theta,  vx,  vy,  vz, omega,   r
        0.01,
        0.01,
        0.01,
        0.02,
        0.05,
        0.05,
        0.0001,
        0.04,
        0.001);
    antitop_2->setMatrixR(0.01, 0.01, 0.01, 0.02);

    // x    y    z    theta
    antitop_2->setCenterMatrixQ(0.1, 0.1, 0.01, 0.01);

    antitop_2->setCenterMatrixR(1, 1);

    antitop_2->setOmegaMatrixQ(1, 2, 5);
    antitop_2->setOmegaMatrixR(1);
}

void Car::push(const auto_aim_interfaces::msg::Armor &armor, rclcpp::Time time)
{
    Eigen::Vector4d pose(armor.pose.position.x, armor.pose.position.y, armor.pose.position.z, orientationToYawWrap(armor.pose.orientation));

    // auto pose = rm::getPose(armor.pose);

    track_queue.push(pose, time);

    curr_armor_num_++;
    if (armor.type == "LARGE")
    {
        big_armor_cnt_++;
    }
}

void Car::update()
{
    track_queue.update();

    if (curr_armor_num_ > 1)
    {
        big_armor_cnt_ -= 1;
    }
    curr_armor_num_ = 0;
    Eigen::Vector4d pose;
    TimePoint t;
    if (!track_queue.getPose(pose, t))
    {
        return;
    }

    rm::AntitopV3 *antitop = nullptr;
    if (robot_id_ == 3 || robot_id_ == 4 || robot_id_ == 5)
    {
        if (big_armor_cnt_ > 0)
        {
            antitop = antitop_2;
        }
        else
        {
            antitop = antitop_4;
        }
    }
    else
    {
        antitop = antitop_4;
    }

    antitop->push(pose, t);
}

double Car::getYaw() { return this->antitop_4->getTheta(); }

void Car::getStatus(auto_aim_interfaces::msg::NewTarget &_msg)
{
    track_queue.getStatus(_msg);
    antitop_4->getStatus(_msg);
}
