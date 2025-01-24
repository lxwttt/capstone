#include "new_armor_tracker/garage/garage.hpp"

#include "new_armor_tracker/garage/car.hpp"
// #include "new_armor_tracker/garage/tower.hpp"

//
static uint8_t robotmap[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// Garage::Garage()
// {
//     robots_    = std::vector<RobotPtr>(8);
//     robots_[0] = std::make_shared<Car>(1);    // Hero
//     robots_[1] = std::make_shared<Car>(2);    // Eng
//     robots_[2] = std::make_shared<Car>(3);    // Inf 3
//     robots_[3] = std::make_shared<Car>(4);    // Inf 4
//     robots_[4] = std::make_shared<Car>(5);    // Inf 5
//     robots_[5] = std::make_shared<Tower>(6);  // Outpost
//     robots_[6] = std::make_shared<Car>(7);    // Sentry
//     robots_[7] = std::make_shared<Car>(8);    // Base
// }

Garage::Garage(std::vector<std::vector<double>> &ekf_params)
{
    robots_    = std::vector<RobotPtr>(8);
    robots_[0] = std::make_shared<Car>(1, ekf_params);    // Hero
    robots_[1] = std::make_shared<Car>(2, ekf_params);    // Eng
    robots_[2] = std::make_shared<Car>(3, ekf_params);    // Inf 3
    robots_[3] = std::make_shared<Car>(4, ekf_params);    // Inf 4
    robots_[4] = std::make_shared<Car>(5, ekf_params);    // Inf 5
    robots_[5] = std::make_shared<Tower>(6, ekf_params);  // Outpost
    robots_[6] = std::make_shared<Car>(7, ekf_params);    // Sentry
    robots_[7] = std::make_shared<Car>(8, ekf_params);    // Base
}

RobotPtr Garage::getRobot(uint8_t robot_id) { return this->robots_[robotmap[robot_id - 1]]; }
