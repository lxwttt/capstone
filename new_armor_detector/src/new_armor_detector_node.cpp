#include "new_armor_detector/new_armor_detector_node.hpp"

// ROS
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "new_armor_detector/armor_detector.hpp"

namespace rm_auto_aim
{

NewArmorDetectorNode::NewArmorDetectorNode(const rclcpp::NodeOptions &_options)
    : Node("armor_detector", _options),
      armor_detector_(initArmorDetector()),
      armor_detector_thread_pool_(2, std::bind(&NewArmorDetectorNode::detectArmors, this, std::placeholders::_1)),
      armor_solver_thread_pool_(1, std::bind(&NewArmorDetectorNode::solveArmors, this, std::placeholders::_1))
{
    RCLCPP_INFO(this->get_logger(), "Starting NewArmorDetectorNode!");

    // Armors Publisher
    armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>("armor_detector/armors", rclcpp::SensorDataQoS());

    armors_msg_frame_id_ = this->declare_parameter("armors_msg_frame_id", "gimbal_odom");

    // tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    armor_marker_.ns       = "armors";
    armor_marker_.action   = visualization_msgs::msg::Marker::ADD;
    armor_marker_.type     = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x  = 0.05;
    armor_marker_.scale.z  = 0.125;
    armor_marker_.color.a  = 1.0;
    armor_marker_.color.g  = 0.5;
    armor_marker_.color.b  = 1.0;
    armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    text_marker_.ns       = "classification";
    text_marker_.action   = visualization_msgs::msg::Marker::ADD;
    text_marker_.type     = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker_.scale.z  = 0.1;
    text_marker_.color.a  = 1.0;
    text_marker_.color.r  = 1.0;
    text_marker_.color.g  = 1.0;
    text_marker_.color.b  = 1.0;
    text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("armor_detector/marker", 10);

    // Debug Publishers
    debug_ = this->declare_parameter("debug", false);
    if (debug_)
    {
        createDebugPublishers();
    }

    detector_node_debug_ = this->declare_parameter("detector_node_debug", 0);

    // Debug param change moniter
    // debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    // debug_cb_handle_ = debug_param_sub_->add_parameter_callback("debug",
    //                                                             [this](const rclcpp::Parameter &_p)
    //                                                             {
    //                                                                 debug_ = _p.as_bool();
    //                                                                 debug_ ? createDebugPublishers() : destroyDebugPublishers();
    //                                                             });

    cam_info_sub_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>("camera/camera_info",
                                                                rclcpp::SensorDataQoS(),
                                                                [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr _camera_info)
                                                                {
                                                                    cam_center_ = cv::Point2f(_camera_info->k[2], _camera_info->k[5]);
                                                                    cam_info_   = std::make_shared<sensor_msgs::msg::CameraInfo>(*_camera_info);
                                                                    pnp_solver_ = std::make_unique<PnPSolver>(_camera_info->k, _camera_info->d);
                                                                    cam_info_sub_.reset();
                                                                });

    img_sub_ = this->create_subscription<shm_msgs::msg::Image2m>(
        "camera/image_raw", rclcpp::SensorDataQoS(), std::bind(&NewArmorDetectorNode::imageCallback, this, std::placeholders::_1));

    cvmode_sub_ = create_subscription<auto_aim_interfaces::msg::Cvmode>("serial_driver/cv_mode",
                                                                        rclcpp::SensorDataQoS(),
                                                                        [this](auto_aim_interfaces::msg::Cvmode::SharedPtr _msg)
                                                                        {
                                                                            cv_mode_ = _msg->cur_cv_mode ? true : false;
                                                                            // target_locked_ = _msg->target_locked;
                                                                        });

    last_time_ = now();

    armor_yaw_pub_ = this->create_publisher<std_msgs::msg::Float32>("armor_detector/armor_yaw", 10);

    tf2_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // thread_pool_ = ThreadPool(4);
}
static uint64_t img_cnt = 1;
void NewArmorDetectorNode::imageCallback(const shm_msgs::msg::Image2m::SharedPtr _img_msg)
{
    // If not in ArmorDetector mode, return
    if (cv_mode_)
        return;

    // count the latency from capturing image to now
    rclcpp::Time reach_time = this->now();
    double latency          = (reach_time - _img_msg->header.stamp).seconds() * 1000;

    // Start armor detection
    auto img                                  = shm_msgs::toCvShare(_img_msg, "bgr8");
    std::shared_ptr<cv::Mat> img_ptr          = std::make_shared<cv::Mat>(img->image);
    std::shared_ptr<rm_auto_aim::Frame> frame = std::make_shared<rm_auto_aim::Frame>(img_ptr, img_cnt);
    frame->header_                            = shm_msgs::get_header(_img_msg->header);
    frame->time_stamp_                        = std::chrono::system_clock::now();
    frame->image_latency_                     = latency;
    armor_detector_thread_pool_.enqueueObj2BeProcess(frame);
    img_cnt++;
};

void NewArmorDetectorNode::detectArmors(std::shared_ptr<rm_auto_aim::Frame> _frame)
{
    // Update params
    armor_detector_->detect_color_ = get_parameter("detect_color").as_int();
    if (armor_detector_->detect_color_ == RED)
        armor_detector_->binary_thres_ = get_parameter("red_binary_thres").as_int();
    else
        armor_detector_->binary_thres_ = get_parameter("blue_binary_thres").as_int();
    armor_detector_->classifier_->threshold = get_parameter("classifier_threshold").as_double();

    armor_detector_->detect(_frame);

    // Print process time in ms
    // RCLCPP_INFO(this->get_logger(),
    //             "Finish detecting frame %ld, time: %f",
    //             _frame->id_,
    //             (std::chrono::system_clock::now() - _frame->time_stamp_).count() / 1000000.0);
    armor_solver_thread_pool_.enqueueObj2BeProcess(_frame);
}

void NewArmorDetectorNode::solveArmors(std::shared_ptr<rm_auto_aim::Frame> _frame)
{
    // do something
    // RCLCPP_WARN(this->get_logger(), "Start solving frame %ld", _frame->id_);

    if (_frame->id_ < this->expected_id_)
    {
        RCLCPP_WARN(this->get_logger(), "Frame %ld is too old, drop it", _frame->id_);
        return;
    }
    this->expected_id_ = _frame->id_ + 1;

    float min_ratios[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < int(_frame->armors_.size()); i++)
    {
        float armor_min_ratio = std::min(_frame->armors_[i].left_light.ratio, _frame->armors_[i].right_light.ratio);
        if (armor_min_ratio > min_ratios[id_table_[_frame->armors_[i].number]])
            min_ratios[id_table_[_frame->armors_[i].number]] = armor_min_ratio;
    }
    std::vector<int> valid_ids;
    int i                        = 0;
    int nearest_armor_index      = -1;
    float nearest_armor_distance = 10000;

    if (this->pnp_solver_ != nullptr)
    {
        armors_msg_.header = armor_marker_.header = _frame->header_;
        armors_msg_.header.frame_id = armor_marker_.header.frame_id = armors_msg_frame_id_;

        armors_msg_.armors.clear();
        marker_array_.markers.clear();
        armor_marker_.id = 0;
        text_marker_.id  = 0;

        auto_aim_interfaces::msg::Armor armor_msg;

        if ((now() - last_time_).seconds() > 0.5f)
        {
            fps_       = (float)fps_count_ / (now() - last_time_).seconds();
            last_time_ = now();
            fps_count_ = 0;
        }
        fps_count_++;

        for (const auto &armor : _frame->armors_)
        {
            float armor_min_ratio = std::min(armor.left_light.ratio, armor.right_light.ratio);
            if (armor_min_ratio < min_ratios[id_table_[_frame->armors_[i].number]] - 0.002)
            {
                i++;
                continue;
            }

            cv::Mat rvec, tvec;
            bool pnp_success = pnp_solver_->solvePnP(armor, rvec, tvec);

            // transform solved armor to world frame
            geometry_msgs::msg::PoseStamped rvec_world;
            cv::Mat rotation_matrix;
            cv::Rodrigues(rvec, rotation_matrix);
            tf2::Matrix3x3 tf2_rotation_matrix(rotation_matrix.at<double>(0, 0),
                                               rotation_matrix.at<double>(0, 1),
                                               rotation_matrix.at<double>(0, 2),
                                               rotation_matrix.at<double>(1, 0),
                                               rotation_matrix.at<double>(1, 1),
                                               rotation_matrix.at<double>(1, 2),
                                               rotation_matrix.at<double>(2, 0),
                                               rotation_matrix.at<double>(2, 1),
                                               rotation_matrix.at<double>(2, 2));
            tf2::Quaternion tf2_q;
            tf2_rotation_matrix.getRotation(tf2_q);
            rvec_world.header.stamp     = armors_msg_.header.stamp;
            rvec_world.header.frame_id  = "camera_optical_frame";
            rvec_world.pose.orientation = tf2::toMsg(tf2_q);
            rvec_world.pose.position.x  = tvec.at<double>(0, 0);
            rvec_world.pose.position.y  = tvec.at<double>(1, 0);
            rvec_world.pose.position.z  = tvec.at<double>(2, 0);
            geometry_msgs::msg::TransformStamped transform_stamped;
            geometry_msgs::msg::TransformStamped gimbal_link_to_odom;

            // Start transform & get TF
            try
            {
                rvec_world = tf2_buffer_->transform(rvec_world, "gimbal_odom");
                // get TF from gimbal_odom to camera_optical_frame within 10ms
                transform_stamped =
                    tf2_buffer_->lookupTransform("camera_optical_frame", "gimbal_odom", tf2::TimePointZero, tf2::durationFromSec(0.05));
                // get TF from gimbal_link to gimbal_odom
                gimbal_link_to_odom = tf2_buffer_->lookupTransform("gimbal_link", "gimbal_odom", tf2::TimePointZero, tf2::durationFromSec(0.01));
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
                continue;
            }

            /** Setup params for solving yaw with gradiant descent **/

            Eigen::Quaterniond q(transform_stamped.transform.rotation.w,
                                 transform_stamped.transform.rotation.x,
                                 transform_stamped.transform.rotation.y,
                                 transform_stamped.transform.rotation.z);
            Eigen::Matrix3d rotation = q.toRotationMatrix();

            auto q2 = tf2::Quaternion(gimbal_link_to_odom.transform.rotation.x,
                                      gimbal_link_to_odom.transform.rotation.y,
                                      gimbal_link_to_odom.transform.rotation.z,
                                      gimbal_link_to_odom.transform.rotation.w);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q2).getRPY(roll, pitch, yaw);
            pnp_solver_->sys_yaw = -yaw;
            // RCLCPP_INFO(this->get_logger(), "sys_yaw: %f", pnp_solver_->sys_yaw);

            Eigen::Matrix4d transform_matrix   = Eigen::Matrix4d::Identity();
            transform_matrix.block<3, 3>(0, 0) = rotation;
            transform_matrix(0, 3)             = transform_stamped.transform.translation.x;
            transform_matrix(1, 3)             = transform_stamped.transform.translation.y;
            transform_matrix(2, 3)             = transform_stamped.transform.translation.z;
            // quat to rpy, in 360 degree
            auto rpy = orientationToRPY(rvec_world.pose.orientation);

            double armor_pitch_pnp = rpy[1];
            double armor_yaw_pnp   = rpy[2];
            Eigen::Vector4d tpose;
            tpose << rvec_world.pose.position.x, rvec_world.pose.position.y, rvec_world.pose.position.z, 1;
            pnp_solver_->setPose(tpose);
            pnp_solver_->setRP(rpy[0], rpy[1]);
            pnp_solver_->setArmorType(armor.type == rm_auto_aim::ArmorType::SMALL);
            pnp_solver_->setT_Odom_to_Camera(transform_matrix);
            pnp_solver_->setElevation(armor_pitch_pnp);
            pnp_solver_->setIncline(armor_yaw_pnp);

            double range_left, range_right;
            range_left  = -M_PI / 2;
            range_right = M_PI / 2;

            append_yaw_ = pnp_solver_->getYawByJiaoCost(range_left, range_right, 0.03);

            if (pnp_success)
            {
                // Fill armor basic info
                armor_msg.type   = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
                armor_msg.number = armor.number;

                // Fill pose
                armor_msg.pose.position.x = rvec_world.pose.position.x;
                armor_msg.pose.position.y = rvec_world.pose.position.y;
                armor_msg.pose.position.z = rvec_world.pose.position.z;

                tf2::Quaternion q;
                q.setRPY(rpy[0], rpy[1], append_yaw_ + -yaw);
                armor_msg.pose.orientation = tf2::toMsg(q);

                // Fill the distance to image center
                armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
                if (armor_msg.distance_to_image_center < nearest_armor_distance)
                {
                    nearest_armor_distance = armor_msg.distance_to_image_center;
                    nearest_armor_index    = i;
                }

                armors_msg_.armors.emplace_back(armor_msg);
                valid_ids.emplace_back(i++);

                if (detector_node_debug_)
                {
                    // Fill the markers
                    armor_marker_.id++;
                    armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
                    armor_marker_.pose    = armor_msg.pose;
                    marker_array_.markers.emplace_back(armor_marker_);
                    text_marker_.id++;
                    text_marker_.pose.position = armor_msg.pose.position;
                    text_marker_.pose.position.y -= 0.1;
                    text_marker_.text = armor.classfication_result;
                    marker_array_.markers.emplace_back(text_marker_);
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "PnP failed ");
            }
        }
        _frame->image_process_time_ = (std::chrono::system_clock::now() - _frame->time_stamp_).count() / 1000000.0;

        // Determine the target armor by the distance to image center and target locking

        // If target_locked = 1, the target is locked, id close to the center of the image will be selected until the target_locked is = 0
        // If target_locked = 1, and no armor is detected, it will automatically select the nearest armor to the center of the image once armor is
        // If target_locked = 0, but armor is detected, the target will be the nearest armor to the center of the image
        // detected
        if (target_locked_)
        {
            if (nearest_armor_index != -1)
            {
                armors_msg_.target_id = id_table_[_frame->armors_[nearest_armor_index].number];
            }
        }
        else
        {
            if (int(armors_msg_.armors.size()) > 0)
            {
                armors_msg_.target_id = id_table_[_frame->armors_[nearest_armor_index].number];
            }
            else
            {
                armors_msg_.target_id = 0;
            }
        }

        // Publish armors
        armors_pub_->publish(armors_msg_);

        if (detector_node_debug_)
        {
            publishMarkers();
        }
    }

    if (1)
    {
        // cv::Mat final = img_mat_msg_->image.clone();
        cv::Mat final = _frame->raw_image_->clone();
        cv::putText(final, "FPS: " + std::to_string(fps_), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
        cv::String labels[4] = {"0", "1", "2", "3"};

        std::stringstream latency_ss;
        latency_ss << "Latency: " << std::fixed << std::setprecision(2) << _frame->image_latency_ << "ms";
        auto latency_s = latency_ss.str();
        cv::putText(final, latency_s, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        std::stringstream image_process_time_ss;
        image_process_time_ss << "IPT: " << std::fixed << std::setprecision(2) << _frame->image_process_time_ << "ms";
        auto image_process_time_s = image_process_time_ss.str();
        cv::putText(final, image_process_time_s, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        int number = valid_ids.size();
        for (int i = 0; i < number; i++)
        {
            auto armor = _frame->armors_[valid_ids[i]];

            std::vector<cv::Point2f> temp_l(4);
            temp_l = armor_detector_->getRectPoints(armor.left_light);
            std::vector<cv::Point2f> temp_r(4);
            temp_r = armor_detector_->getRectPoints(armor.right_light);

            std::vector<cv::Point2f> vertices;
            vertices.push_back(armor.left_light.bottom);
            vertices.push_back(armor.left_light.top);
            vertices.push_back(armor.right_light.top);
            vertices.push_back(armor.right_light.bottom);
            for (int i = 0; i < 4; i++)
            {
                cv::line(final, temp_l[i % 4], temp_l[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
                cv::line(final, temp_r[i % 4], temp_r[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
                cv::circle(final, vertices[i % 4], 4, cv::Scalar(255, 0, 255), -1);
                cv::putText(final, labels[i], vertices[i % 4] + cv::Point2f(12, 15), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 3);
            }
            cv::line(final, armor.left_light.top, armor.left_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(final, armor.right_light.top, armor.right_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(final, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 0, 200), 2, cv::LINE_8);
            cv::line(final, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 0, 200), 2, cv::LINE_8);

            cv::putText(final,
                        armor.classfication_result,
                        armor.left_light.bottom + cv::Point2f(-20, 38),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 255),
                        2);
            cv::putText(final,
                        "x: " + std::to_string(armors_msg_.armors[i].pose.position.x),
                        armor.center - cv::Point2f(5, 65),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "y: " + std::to_string(armors_msg_.armors[i].pose.position.y),
                        armor.center - cv::Point2f(5, 40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(0, 255, 255),
                        2);
            cv::putText(final,
                        "z: " + std::to_string(armors_msg_.armors[i].pose.position.z),
                        armor.center - cv::Point2f(5, 15),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 0, 255),
                        2);
            std::vector rpy      = orientationToRPY(armors_msg_.armors[i].pose.orientation);
            std::vector rpy_test = orientationToRPY(armors_msg_.armors[i].pose_pnp.orientation);
            // cv::putText(final,
            //             "yaw_ours: " + std::to_string(rpy[2] * 180 / CV_PI),
            //             armor.center - cv::Point2f(5, -10),
            //             cv::FONT_HERSHEY_SIMPLEX,
            //             0.9,
            //             cv::Scalar(255, 255, 0),
            //             2);
            // cv::putText(final,
            //             "yaw_pnp: " + std::to_string(rpy_test[2] * 180 / CV_PI),
            //             armor.center - cv::Point2f(5, -35),
            //             cv::FONT_HERSHEY_SIMPLEX,
            //             0.9,
            //             cv::Scalar(255, 255, 0),
            //             2);
            cv::putText(final,
                        "roll: " + std::to_string(rpy[0] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -60),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(0, 255, 255),
                        2);
            cv::putText(final,
                        "pitch: " + std::to_string(rpy[1] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -85),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 0, 255),
                        2);
            cv::putText(final,
                        "IPPE_yaw: " + std::to_string(rpy[2] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -185),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "Jiao_yaw: " + std::to_string(append_yaw_ * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -285),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);

            // cv::putText(
            //   final, "x1: " + std::to_string(tvec1.at<double>(2, 0)), armor.center - cv::Point2f(-250, 65),
            //   cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 0), 2);
            // cv::putText(
            //   final, "y1: " + std::to_string(-tvec1.at<double>(0, 0)), armor.center - cv::Point2f(-250, 40),
            //   cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
            // cv::putText(
            //   final, "z1: " + std::to_string(-tvec1.at<double>(1, 0)), armor.center - cv::Point2f(-250, 15),
            //   cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 255), 2);

            // cv::putText(
            //   final, "yaw1: " + std::to_string(rvec1.at<double>(1, 0) * 180 / CV_PI),
            //   armor.center - cv::Point2f(-250, -10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 0),
            //   2);
            // cv::putText(
            //   final, "roll1: " + std::to_string(rvec1.at<double>(0, 0) * 180 / CV_PI),
            //   armor.center - cv::Point2f(-250, -35), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255),
            //   2);
            // cv::putText(
            //   final, "pitch1: " + std::to_string(rvec1.at<double>(2, 0) * 180 / CV_PI),
            //   armor.center - cv::Point2f(-250, -60), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 255),
            //   2);
        }
        // RCLCPP_ERROR(this->get_logger(), "FPS: %f", fps);
        // cv::imshow("raw_image", raw_image);
        cv::imshow("armor_final", final);
        cv::waitKey(1);
    }
}

std::unique_ptr<ArmorDetector> NewArmorDetectorNode::initArmorDetector()
{
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.integer_range.resize(1);
    param_desc.description                 = "0-RED, 1-BLUE";
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value   = 1;
    auto detect_color                      = declare_parameter("detect_color", BLUE, param_desc);

    param_desc.integer_range[0].step       = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value   = 255;
    int blue_binary_thres                  = declare_parameter("blue_binary_thres", 120, param_desc);
    int red_binary_thres                   = declare_parameter("red_binary_thres", 60, param_desc);

    ArmorDetector::LightParams l_params = {.min_ratio_      = declare_parameter("light.min_ratio", 0.07),
                                           .max_ratio_      = declare_parameter("light.max_ratio", 0.7),
                                           .max_angle_      = declare_parameter("light.max_angle", 75.0),
                                           .min_fill_ratio_ = declare_parameter("light.min_fill_ratio", 0.8)};

    ArmorDetector::ArmorParams a_params = {.min_light_ratio_           = declare_parameter("armor.min_light_ratio", 0.75),
                                           .min_small_center_distance_ = declare_parameter("armor.min_small_center_distance", 0.8),
                                           .max_small_center_distance_ = declare_parameter("armor.max_small_center_distance", 3.2),
                                           .min_large_center_distance_ = declare_parameter("armor.min_large_center_distance", 3.2),
                                           .max_large_center_distance_ = declare_parameter("armor.max_large_center_distance", 5.5),
                                           .max_angle_                 = declare_parameter("armor.max_angle", 75.0)};

    // Init classifier
    auto pkg_path                           = ament_index_cpp::get_package_share_directory("new_armor_detector");
    auto model_path                         = pkg_path + "/model/mlp.onnx";
    auto label_path                         = pkg_path + "/model/label.txt";
    double threshold                        = this->declare_parameter("classifier_threshold", 0.7);
    std::vector<std::string> ignore_classes = this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
    auto armor_detector =
        std::make_unique<ArmorDetector>(detect_color == RED ? red_binary_thres : blue_binary_thres, detect_color, l_params, a_params);
    // armor_detector->classifier_             = std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);
    armor_detector->classifier_ = std::make_shared<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    // // Init debug mode
    int detector_debug_num          = this->declare_parameter("detector_debug", 0);
    armor_detector->detector_debug_ = detector_debug_num ? true : false;

    return armor_detector;
}

void NewArmorDetectorNode::publishMarkers()
{
    using Marker         = visualization_msgs::msg::Marker;
    armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

std::vector<double> NewArmorDetectorNode::orientationToRPY(const geometry_msgs::msg::Quaternion &_q)
{
    // Get armor yaw
    tf2::Quaternion tf_q;
    tf2::fromMsg(_q, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    std::vector<double> rpy = {roll, pitch, yaw};
    return rpy;
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::NewArmorDetectorNode)