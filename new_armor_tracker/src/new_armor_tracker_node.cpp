#include "new_armor_tracker/new_armor_tracker_node.hpp"

namespace rm_auto_aim
{

NewArmorTrackerNode::NewArmorTrackerNode(const rclcpp::NodeOptions &options) : Node("new_armor_tracker", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting NewArmorTrackerNode!");

    // Tower

    // Armors Subscriber & tf2 Filter
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // Publisher
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::NewTarget>("/new/armor_tracker/target", rclcpp::SensorDataQoS());
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/new/armor_tracker/marker", 10);

    // subscriber and filter
    armors_sub_.subscribe(this, "armor_detector/armors", rmw_qos_profile_sensor_data);
    target_frame_ = this->declare_parameter("target_frame", "gimbal_odom");
    tf2_filter_   = std::make_shared<tf2_filter>(armors_sub_,
                                               *tf2_buffer_,
                                               target_frame_,
                                               10,
                                               this->get_node_logging_interface(),
                                               this->get_node_clock_interface(),
                                               std::chrono::duration<int>(1));
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_->registerCallback(&NewArmorTrackerNode::armorsCallback, this);

    // EKF params
    auto car_tq_q = this->declare_parameter("Car.TrackQueue_Q", std::vector<double>{1, 2, 3});
    auto car_tq_r = this->declare_parameter("Car.TrackQueue_R", std::vector<double>{1, 2, 3});
    auto car_at_q = this->declare_parameter("Car.Antitop_Q", std::vector<double>{1, 2, 3});
    auto car_at_r = this->declare_parameter("Car.Antitop_R", std::vector<double>{1, 2, 3});
    auto car_c_q  = this->declare_parameter("Car.Center_Q", std::vector<double>{1, 2, 3});
    auto car_c_r  = this->declare_parameter("Car.Center_R", std::vector<double>{1, 2, 3});
    auto car_o_q  = this->declare_parameter("Car.Omega_Q", std::vector<double>{1, 2, 3});
    auto car_o_r  = this->declare_parameter("Car.Omega_R", std::vector<double>{1, 2, 3});
    auto tow_tq_q = this->declare_parameter("Tower.TrackQueue_Q", std::vector<double>{1, 2, 3});
    auto tow_tq_r = this->declare_parameter("Tower.TrackQueue_R", std::vector<double>{1, 2, 3});
    auto tow_at_q = this->declare_parameter("Tower.Antitop_Q", std::vector<double>{1, 2, 3});
    auto tow_at_r = this->declare_parameter("Tower.Antitop_R", std::vector<double>{1, 2, 3});
    auto tow_o_q  = this->declare_parameter("Tower.Omega_Q", std::vector<double>{1, 2, 3});
    auto tow_o_r  = this->declare_parameter("Tower.Omega_R", std::vector<double>{1, 2, 3});

    std::vector<std::vector<double>> ekf_params = {
        car_tq_q, car_tq_r, car_at_q, car_at_r, car_c_q, car_c_r, car_o_q, car_o_r, tow_tq_q, tow_tq_r, tow_at_q, tow_at_r, tow_o_q, tow_o_r};

    // init variable
    target_id_  = 0;
    t_          = now();
    state_      = State::LOST;
    last_state_ = State::LOST;

    // init msg
    blank_target_msg_.tracking              = false;
    blank_target_msg_.armor_num             = 0;
    blank_target_msg_.armor_position        = geometry_msgs::msg::Point();
    blank_target_msg_.armor_velocity        = geometry_msgs::msg::Vector3();
    blank_target_msg_.model_position        = geometry_msgs::msg::Point();
    blank_target_msg_.model_velocity        = geometry_msgs::msg::Vector3();
    blank_target_msg_.center_model_position = geometry_msgs::msg::Point();
    blank_target_msg_.center_model_velocity = geometry_msgs::msg::Vector3();
    blank_target_msg_.r1                    = 0;
    blank_target_msg_.r2                    = 0;
    blank_target_msg_.z1                    = 0;
    blank_target_msg_.z2                    = 0;
    blank_target_msg_.yaw                   = 0;
    blank_target_msg_.omega_v_yaw           = 0;

    position_marker_.ns      = "position";
    position_marker_.type    = visualization_msgs::msg::Marker::SPHERE;
    position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
    position_marker_.color.a                                                       = 1.0;
    position_marker_.color.g                                                       = 1.0;

    armor_marker_.ns      = "armors";
    armor_marker_.type    = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x = 0.03;
    armor_marker_.scale.z = 0.125;
    armor_marker_.color.a = 1.0;
    armor_marker_.color.r = 1.0;

    // Timer

    // Garage
    garage = Garage::getInstance(ekf_params);
    RCLCPP_INFO(this->get_logger(), "NewArmorTrackerNode started!");
}

NewArmorTrackerNode::~NewArmorTrackerNode() { RCLCPP_INFO(this->get_logger(), "Destroying NewArmorTrackerNode!"); }

void NewArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr _armors_msg)
{
    for (auto &armor : _armors_msg->armors)
    {
        uint8_t robot_id = id_table[armor.number];
        RobotPtr robot   = garage->getRobot(robot_id);

        robot->push(armor, _armors_msg->header.stamp);
    }

    for (auto &robot : garage->robots_)
    {
        robot->update();
    }

    armor_target_msg_.header.stamp    = _armors_msg->header.stamp;
    armor_target_msg_.header.frame_id = "gimbal_odom";

    if (_armors_msg->target_id != 0)
    {
        target_id_ = _armors_msg->target_id;
        t_         = now();
        state_     = State::TRACKING;
    }
    else
    {
        if ((now() - t_) > rclcpp::Duration(0, 500000000))
        {
            if (last_state_ == State::TRACKING)
            {
                RCLCPP_WARN(this->get_logger(), "Target Lost.");
            }
            armor_target_msg_ = blank_target_msg_;
            state_            = State::LOST;
        }
    }
    if (state_ == State::TRACKING)
    {
        armor_target_msg_.id = target_id_;
        garage->getRobot(target_id_)->getStatus(armor_target_msg_);
        armor_target_msg_.tracking = true;
        if (armor_target_msg_.armor_position.x == 0)
        {
            armor_target_msg_.tracking = false;
            armor_target_msg_          = blank_target_msg_;
            RCLCPP_WARN(this->get_logger(), "Target Lost.");
            state_ = State::LOST;
        }
    }

    // Visualization
    position_marker_.header = _armors_msg->header;
    armor_marker_.header    = _armors_msg->header;
    visualization_msgs::msg::MarkerArray marker_array;
    if (state_ == State::TRACKING)
    {
        double yaw = armor_target_msg_.yaw, r1 = armor_target_msg_.r1, r2 = armor_target_msg_.r2;
        double xc = armor_target_msg_.center_model_position.x, yc = armor_target_msg_.center_model_position.y;
        double z1 = armor_target_msg_.z1, z2 = armor_target_msg_.z2;

        position_marker_.pose.position   = armor_target_msg_.center_model_position;
        position_marker_.pose.position.z = (z1 + z2) / 2;

        armor_marker_.action  = visualization_msgs::msg::Marker::ADD;
        armor_marker_.scale.y = 0.135;

        double armor_num = 4;
        geometry_msgs::msg::Point armor_position;
        double r   = 0;
        int toggle = armor_target_msg_.toggle;
        for (int i = 0; i < armor_num; i++)
        {
            double d_yaw = yaw + (double)i * (2.0 * M_PI / armor_num);

            r                = toggle ? r2 : r1;
            armor_position.z = toggle ? z2 : z1;
            toggle           = !toggle;

            armor_position.x = xc - r * cos(d_yaw);
            armor_position.y = yc - r * sin(d_yaw);

            if (i == 0)
            {
                armor_marker_.color.b = 1.0;
                armor_marker_.color.g = 0.0;
                armor_marker_.color.r = 0.0;
            }
            else if (i == 1)
            {
                armor_marker_.color.b = 0.0;
                armor_marker_.color.g = 1.0;
                armor_marker_.color.r = 0.0;
            }
            else if (i == 2)
            {
                armor_marker_.color.b = 0.0;
                armor_marker_.color.g = 0.0;
                armor_marker_.color.r = 1.0;
            }
            else
            {
                armor_marker_.color.b = 1.0;
                armor_marker_.color.g = 1.0;
                armor_marker_.color.r = 1.0;
            }

            armor_marker_.id            = i;
            armor_marker_.pose.position = armor_position;
            tf2::Quaternion q;
            q.setRPY(0, 0.26, d_yaw);
            armor_marker_.pose.orientation = tf2::toMsg(q);
            marker_array.markers.emplace_back(armor_marker_);
        }
    }
    else
    {
        position_marker_.action  = visualization_msgs::msg::Marker::DELETE;
        linear_v_marker_.action  = visualization_msgs::msg::Marker::DELETE;
        angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;

        armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.emplace_back(armor_marker_);
    }

    marker_array.markers.emplace_back(position_marker_);
    marker_pub_->publish(marker_array);

    target_pub_->publish(armor_target_msg_);
    last_state_ = state_;
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::NewArmorTrackerNode)