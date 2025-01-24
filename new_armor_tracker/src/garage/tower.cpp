#include "new_armor_tracker/garage/tower.hpp"

#include "new_armor_tracker/utils/tf.hpp"

Tower::Tower(uint8_t robot_id, std::vector<std::vector<double>> &_ekf_params) : IRobot(robot_id)
{
    robot_id_ = robot_id;

    track_queue = rm::TrackQueueV3(10, 0.20, 0.5);

    track_queue.setMatrixQ(
        _ekf_params[8][0], _ekf_params[8][1], _ekf_params[8][2],
        _ekf_params[8][3], _ekf_params[8][4], _ekf_params[8][5],
        _ekf_params[8][6], _ekf_params[8][7], _ekf_params[8][8], _ekf_params[8][9], _ekf_params[8][10]
    );
    track_queue.setMatrixR(_ekf_params[9][0], _ekf_params[9][1], _ekf_params[9][2], _ekf_params[9][3]);

    outpost = rm::OutpostV2();

    outpost.setMatrixQ(
        _ekf_params[10][0], _ekf_params[10][1], _ekf_params[10][2],
        _ekf_params[10][3], _ekf_params[10][4], _ekf_params[10][5],
        _ekf_params[10][6], _ekf_params[10][7]
    );
    outpost.setMatrixR(_ekf_params[11][0], _ekf_params[11][1], _ekf_params[11][2], _ekf_params[11][3]);
    outpost.setMatrixOmegaQ(_ekf_params[12][0], _ekf_params[12][1]);
    outpost.setMatrixOmegaR(_ekf_params[13][0]);
}

void Tower::push(const auto_aim_interfaces::msg::Armor &armor, rclcpp::Time time)
{
    // Eigen::Vector4d pose(armor.pose.position.x, armor.pose.position.y, armor.pose.position.z, orientationToYawWrap(armor.pose.orientation));
    // track_queue.push(pose, time);
}

void Tower::update()
{
    track_queue.update();

    Eigen::Vector4d pose;
    TimePoint t;
    if (!track_queue.getPose(pose, t))
    {
        return;
    }
    outpost.push(pose, t);
}

void Tower::getStatus(auto_aim_interfaces::msg::NewTarget &_msg)
{
    // track_queue.getStatus(_msg);
    outpost.getStatus(_msg);
}
