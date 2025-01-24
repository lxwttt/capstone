#ifndef __NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP__
#define __NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP__

#include "new_armor_tracker/garage/IRobot.hpp"
#include "new_armor_tracker/garage/car.hpp"
#include "new_armor_tracker/garage/tower.hpp"
#include "new_armor_tracker/utils/tf.hpp"

/**
 * @brief The Garage class, hold all type of robot except rune
 */
class Garage
{
   public:
    // static std::shared_ptr<Garage> getInstance()
    // {
    //     static std::shared_ptr<Garage> instance(new Garage());
    //     return instance;
    // }

    static std::shared_ptr<Garage> getInstance(std::vector<std::vector<double>> &ekf_params)
    {
        static std::shared_ptr<Garage> instance(new Garage(ekf_params));
        return instance;
    }

    // static std::shared_ptr<Garage> getInstance()

    /**
     * @brief Get the robot object from the garage
     * @param robot_id, the id of the robot
     */
    RobotPtr getRobot(uint8_t robot_id);

   private:
    Garage() = delete;
    Garage(std::vector<std::vector<double>> &ekf_params);
    Garage(const Garage &)            = delete;
    Garage &operator=(const Garage &) = delete;

    bool param_init_ = false;

   public:
    std::vector<RobotPtr> robots_;
};

#endif  // __NEW_ARMOR_TRACKER__GARAGE__GARAGE_HPP__
