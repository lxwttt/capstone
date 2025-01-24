#include "armor_tracker/armor_tracker_node.hpp"

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions &options) : Node("armor_tracker", options)
{
    RCLCPP_ERROR(this->get_logger(), "Starting ArmorTrackerNode!");

    // Maximum allowable armor distance in the XOY plane
    double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
    double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
    this->max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);
    this->s2qxyz_             = this->declare_parameter("ekf.sigma2_q_xyz", 0.05);
    this->s2qyaw_             = this->declare_parameter("ekf.sigma2_q_yaw", 50.0);
    this->s2qr_               = this->declare_parameter("ekf.sigma2_q_r", 80.0);

    this->line_s2qxyz_ = this->declare_parameter("ekf.line_sigma2_q_xyz", 0.5);
    this->line_s2qyaw_ = this->declare_parameter("ekf.line_sigma2_q_yaw", 5.0);
    this->line_s2qr_   = this->declare_parameter("ekf.line_sigma2_q_r", 8.0);
    this->line_s2qa_   = this->declare_parameter("ekf.line_sigma2_q_a", 0.5);

    this->r_xyz_factor     = this->declare_parameter("ekf.r_xyz_factor", 0.05);
    this->r_yaw            = this->declare_parameter("ekf.r_yaw", 0.02);
    this->lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 1.0);

    // Tracker
    tracker_ = std::make_unique<ArmorTracker>(
        max_match_distance, max_match_yaw_diff, s2qxyz_, s2qyaw_, s2qr_, line_s2qxyz_, line_s2qyaw_, line_s2qr_, line_s2qa_, r_xyz_factor, r_yaw);

    tracker_->tracking_thres = this->declare_parameter("tracker.tracking_thres", 5);

    // Reset tracker service
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;

    reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/tracker/reset",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr response)
        {
            tracker_->tracker_state = ArmorTracker::LOST;
            response->success       = true;
            RCLCPP_INFO(this->get_logger(), "Armor Tracker reset!");
            return;
        });

    // Subscriber with tf2 message_filter
    // tf2 relevant
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // subscriber and filter
    armors_sub_.subscribe(this, "/armor_detector/armors", rmw_qos_profile_sensor_data);
    target_frame_ = this->declare_parameter("target_frame", "odom");
    tf2_filter_   = std::make_shared<tf2_filter>(armors_sub_,
                                               *tf2_buffer_,
                                               target_frame_,
                                               10,
                                               this->get_node_logging_interface(),
                                               this->get_node_clock_interface(),
                                               std::chrono::duration<int>(1));
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

    // publisher
    info_pub_   = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/armor_tracker/info", 10);
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>("/armor_tracker/target", rclcpp::SensorDataQoS());

    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    position_marker_.ns      = "position";
    position_marker_.type    = visualization_msgs::msg::Marker::SPHERE;
    position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
    position_marker_.color.a                                                       = 1.0;
    position_marker_.color.g                                                       = 1.0;
    linear_v_marker_.type                                                          = visualization_msgs::msg::Marker::ARROW;
    linear_v_marker_.ns                                                            = "linear_v";
    linear_v_marker_.scale.x                                                       = 0.03;
    linear_v_marker_.scale.y                                                       = 0.05;
    linear_v_marker_.color.a                                                       = 1.0;
    linear_v_marker_.color.r                                                       = 1.0;
    linear_v_marker_.color.g                                                       = 1.0;
    angular_v_marker_.type                                                         = visualization_msgs::msg::Marker::ARROW;
    angular_v_marker_.ns                                                           = "angular_v";
    angular_v_marker_.scale.x                                                      = 0.03;
    angular_v_marker_.scale.y                                                      = 0.05;
    angular_v_marker_.color.a                                                      = 1.0;
    angular_v_marker_.color.b                                                      = 1.0;
    angular_v_marker_.color.g                                                      = 1.0;
    armor_marker_.ns                                                               = "armors";
    armor_marker_.type                                                             = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x                                                          = 0.03;
    armor_marker_.scale.z                                                          = 0.125;
    armor_marker_.color.a                                                          = 1.0;
    armor_marker_.color.r                                                          = 1.0;
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/armor_tracker/marker", 10);
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
    // Tranform armor position from image frame to world coordinate
    for (auto &armor : armors_msg->armors)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = armors_msg->header;
        ps.pose   = armor.pose;
        try
        {
            armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
        }
        catch (const tf2::ExtrapolationException &ex)
        {
            RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
            return;
        }
    }

    // Filter abnormal armors
    armors_msg->armors.erase(std::remove_if(armors_msg->armors.begin(),
                                            armors_msg->armors.end(),
                                            [this](const auto_aim_interfaces::msg::Armor &armor) {
                                                return abs(armor.pose.position.z) > 1.3 ||
                                                       Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() > max_armor_distance_;
                                            }),
                             armors_msg->armors.end());

    // Init message
    auto_aim_interfaces::msg::TrackerInfo info_msg;
    auto_aim_interfaces::msg::Target target_msg;
    rclcpp::Time time          = armors_msg->header.stamp;
    target_msg.header.stamp    = time;
    target_msg.header.frame_id = target_frame_;

    // Update tracker
    if (tracker_->tracker_state == ArmorTracker::LOST)
    {
        tracker_->init(armors_msg);
        target_msg.tracking = false;
    }
    else
    {
        dt_                  = (time - last_time_).seconds();
        tracker_->dt_        = dt_;
        tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
        if (tracker_->tracked_armors_num == rm_auto_aim::ArmorsNum::OUTPOST_3)
            tracker_->lost_thres = tracker_->lost_thres * 5;
        tracker_->update(armors_msg);

        // Publish Info
        info_msg.tracker_state = tracker_->tracker_state;
        info_msg.position_diff = tracker_->info_position_diff;
        info_msg.yaw_diff      = tracker_->info_yaw_diff;
        info_msg.position.x    = tracker_->measurement(0);
        info_msg.position.y    = tracker_->measurement(1);
        info_msg.position.z    = tracker_->measurement(2);
        info_msg.yaw           = tracker_->measurement(3) * 180 / M_PI;
        info_pub_->publish(info_msg);

        if (tracker_->tracker_state == ArmorTracker::DETECTING)
        {
            target_msg.tracking = false;
        }
        else if (tracker_->tracker_state == ArmorTracker::TRACKING || tracker_->tracker_state == ArmorTracker::TEMP_LOST)
        {
            target_msg.tracking = true;
            // Fill target message
            const auto &state     = tracker_->final_state;
            target_msg.id         = tracker_->tracked_id;
            target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
            target_msg.position.x = state(0);
            target_msg.velocity.x = state(1);
            target_msg.position.y = state(2);
            target_msg.velocity.y = state(3);
            target_msg.position.z = state(4);
            target_msg.velocity.z = state(5);
            target_msg.a_x        = state(9);
            target_msg.a_y        = state(10);
            target_msg.yaw        = state(6);
            target_msg.v_yaw      = state(7);
            target_msg.radius_1   = state(8);
            target_msg.radius_2   = tracker_->another_r;
            target_msg.dz         = tracker_->dz;
        }
    }

    last_time_ = time;

    target_pub_->publish(target_msg);

    publishMarkers(target_msg);
}

void ArmorTrackerNode::publishMarkers(const auto_aim_interfaces::msg::Target &target_msg)
{
    position_marker_.header  = target_msg.header;
    linear_v_marker_.header  = target_msg.header;
    angular_v_marker_.header = target_msg.header;
    armor_marker_.header     = target_msg.header;

    visualization_msgs::msg::MarkerArray marker_array;
    if (target_msg.tracking)
    {
        double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
        double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
        double vx = target_msg.velocity.x, vy = target_msg.velocity.y, vz = target_msg.velocity.z;
        double dz = target_msg.dz;

        position_marker_.action          = visualization_msgs::msg::Marker::ADD;
        position_marker_.pose.position.x = xc;
        position_marker_.pose.position.y = yc;
        position_marker_.pose.position.z = za + dz / 2;

        linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        linear_v_marker_.points.clear();
        linear_v_marker_.points.emplace_back(position_marker_.pose.position);
        geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
        arrow_end.x += vx;
        arrow_end.y += vy;
        arrow_end.z += vz;
        linear_v_marker_.points.emplace_back(arrow_end);

        angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        angular_v_marker_.points.clear();
        angular_v_marker_.points.emplace_back(position_marker_.pose.position);
        arrow_end = position_marker_.pose.position;
        arrow_end.z += target_msg.v_yaw / M_PI;
        angular_v_marker_.points.emplace_back(arrow_end);

        armor_marker_.action  = visualization_msgs::msg::Marker::ADD;
        armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
        bool is_current_pair  = true;
        size_t a_n            = target_msg.armors_num;
        geometry_msgs::msg::Point p_a;
        double r = 0;
        for (size_t i = 0; i < a_n; i++)
        {
            double tmp_yaw = yaw + i * (2 * M_PI / a_n);
            // Only 4 armors has 2 radius and height
            if (a_n == 4)
            {
                r               = is_current_pair ? r1 : r2;
                p_a.z           = za + (is_current_pair ? 0 : dz);
                is_current_pair = !is_current_pair;
            }
            else
            {
                r     = r1;
                p_a.z = za;
            }
            p_a.x = xc - r * cos(tmp_yaw);
            p_a.y = yc - r * sin(tmp_yaw);

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
            armor_marker_.pose.position = p_a;
            tf2::Quaternion q;
            q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
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
    marker_array.markers.emplace_back(linear_v_marker_);
    marker_array.markers.emplace_back(angular_v_marker_);
    marker_pub_->publish(marker_array);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
