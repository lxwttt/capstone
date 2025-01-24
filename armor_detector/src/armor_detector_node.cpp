#include "armor_detector/armor_detector_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>

#include "armor_detector/armor.hpp"

std::chrono::high_resolution_clock::time_point start_time, end_time;
std::chrono::microseconds duration;

namespace rm_auto_aim
{

ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions &_options) : Node("armor_detector", _options)
{
    RCLCPP_INFO(this->get_logger(), "Starting ArmorDetectorNode!");

    // ArmorDetector
    armor_detector_ = initArmorDetector();

    // Armors Publisher
    armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>("armor_detector/armors", rclcpp::SensorDataQoS());

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

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
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ = debug_param_sub_->add_parameter_callback("debug",
                                                                [this](const rclcpp::Parameter &_p)
                                                                {
                                                                    debug_ = _p.as_bool();
                                                                    debug_ ? createDebugPublishers() : destroyDebugPublishers();
                                                                });

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
        "camera/image_raw", rclcpp::SensorDataQoS(), std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));

    cvmode_sub_ = create_subscription<auto_aim_interfaces::msg::Cvmode>("serial_driver/cv_mode",
                                                                        rclcpp::SensorDataQoS(),
                                                                        [this](auto_aim_interfaces::msg::Cvmode::SharedPtr _msg)
                                                                        {
                                                                            cv_mode_       = _msg->cur_cv_mode ? true : false;
                                                                            target_locked_ = _msg->target_locked;
                                                                        });

    last_time_ = now();

    armor_yaw_pub_ = this->create_publisher<std_msgs::msg::Float32>("armor_detector/armor_yaw", 10);

    tf2_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
}
static long image_count;
static bool first = true;
static std::chrono::duration<double> elapsed;
static std::chrono::high_resolution_clock::time_point curr_img_time, last_img_time;
static double last_yaw = 0;

void ArmorDetectorNode::imageCallback(const shm_msgs::msg::Image2m::SharedPtr _img_msg)
{
    // If not in ArmorDetector mode, return
    if (cv_mode_)
        return;

    // count the latency from capturing image to now
    rclcpp::Time reach_time = this->now();
    double latency          = (reach_time - _img_msg->header.stamp).seconds() * 1000;
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

    if (first)
    {
        first         = false;
        last_img_time = std::chrono::high_resolution_clock::now();
    }
    else
    {
        image_count++;
        elapsed = std::chrono::high_resolution_clock::now() - last_img_time;
        if (elapsed.count() > 0.25)
        {
            // RCLCPP_INFO(this->get_logger(), "FPS: %f", image_count / elapsed.count());
            image_count   = 0;
            last_img_time = std::chrono::high_resolution_clock::now();
        }
    }

    // Start armor detection
    std::vector<rm_auto_aim::Armor> armors = detectArmors(_img_msg);

    float min_ratios[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < int(armors.size()); i++)
    {
        float armor_min_ratio = std::min(armors[i].left_light.ratio, armors[i].right_light.ratio);
        if (armor_min_ratio > min_ratios[id_table_[armors[i].number]])
            min_ratios[id_table_[armors[i].number]] = armor_min_ratio;
    }
    std::vector<int> valid_ids;
    int i                        = 0;
    int nearest_armor_index      = -1;
    float nearest_armor_distance = 10000;
    double yaws[6];
    double pnp_yaws[6];
    if (pnp_solver_ != nullptr)
    {
        armors_msg_.header = armor_marker_.header = text_marker_.header = shm_msgs::get_header(_img_msg->header);
        armors_msg_.header.frame_id = armor_marker_.header.frame_id = text_marker_.header.frame_id = "gimbal_odom";

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

#if DEBUG
// pnp_solver_->passDebugImg(img_mat_msg_->image.clone());
#endif

        for (const auto &armor : armors)
        {
            float armor_min_ratio = std::min(armor.left_light.ratio, armor.right_light.ratio);
            if (armor_min_ratio < min_ratios[id_table_[armors[i].number]] - 0.002)
            {
                i++;
                continue;
            }
            // print 4 point
            // RCLCPP_INFO(this->get_logger(),
            //             "bottom_l: %5.5f %5.5f, top_l: %5.5f %5.5f, top_r: %5.5f %5.5f, bottom_r: %5.5f %5.5f",
            //             armor.left_light.bottom.x,
            //             armor.left_light.bottom.y,
            //             armor.left_light.top.x,
            //             armor.left_light.top.y,
            //             armor.right_light.top.x,
            //             armor.right_light.top.y,
            //             armor.right_light.bottom.x,
            //             armor.right_light.bottom.y);

            // cv::Mat rvec_ba, tvec_ba;
            // bool success_l = pnp_solver_->solvePnP_BA(armor, rvec_ba, tvec_ba);
            // cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            // rvec.at<double>(0, 0) = rvec_ba.at<double>(2, 0);
            // rvec.at<double>(1, 0) = rvec_ba.at<double>(1, 0);
            // rvec.at<double>(2, 0) = rvec_ba.at<double>(0, 0);
            cv::Mat rvec, tvec;
            bool success = pnp_solver_->solvePnP(armor, rvec, tvec);

            // TEST
            geometry_msgs::msg::PoseStamped rvec_world;
            cv ::Mat rotation_matrix;
            cv::Rodrigues(rvec, rotation_matrix);
            // rotation matrix to quaternion
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
            {
                double roll, pitch, yaw;
                tf2_rotation_matrix.getRPY(roll, pitch, yaw);
                // pnp_yaws[3] = yaw;
                // RCLCPP_INFO(this->get_logger(), "rpy: %f %f %f", roll * 180 / M_PI, pitch * 180 / M_PI, yaw * 180 / M_PI);
            }
            rvec_world.header.stamp     = armors_msg_.header.stamp;
            rvec_world.header.frame_id  = "camera_optical_frame";
            rvec_world.pose.orientation = tf2::toMsg(tf2_q);
            rvec_world.pose.position.x  = tvec.at<double>(0, 0);
            rvec_world.pose.position.y  = tvec.at<double>(1, 0);
            rvec_world.pose.position.z  = tvec.at<double>(2, 0);
            geometry_msgs::msg::TransformStamped transform_stamped;
            geometry_msgs::msg::TransformStamped gimbal_link_to_odom;

            // TEST TF BOARDCAST
            // geometry_msgs::msg::TransformStamped tf_msg;
            // tf_msg.header.stamp            = armors_msg_.header.stamp;
            // tf_msg.header.frame_id         = "camera_optical_frame";
            // tf_msg.child_frame_id          = "armor";
            // tf_msg.transform.translation.x = tvec.at<double>(0, 0);
            // tf_msg.transform.translation.y = tvec.at<double>(1, 0);
            // tf_msg.transform.translation.z = tvec.at<double>(2, 0);
            // tf_msg.transform.rotation      = tf2::toMsg(tf2_q);
            // tf_broadcaster_->sendTransform(tf_msg);

            try
            {
                rvec_world = tf2_buffer_->transform(rvec_world, "gimbal_odom");
                // get TF from gimbal_odom to camera_optical_frame within 10ms
                transform_stamped = tf2_buffer_->lookupTransform("camera_optical_frame", "gimbal_odom", tf2::TimePointZero, tf2::durationFromSec(0.05));
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
            // RCLCPP_INFO(this->get_logger(), "r incline: %f, l incline: %f, avg incline: %f",
            //             armor.right_light.incline, armor.left_light.incline, (armor.right_light.incline + armor.left_light.incline) / 2);

            double range_left, range_right;
            range_left  = -M_PI / 2;
            range_right = M_PI / 2;

            append_yaw_ = pnp_solver_->getYawByJiaoCost(range_left, range_right, 0.03);
            // RCLCPP_INFO(this->get_logger(),
            //             "rpy world: %8.5f %8.5f %8.5f  re-projected y: %.5f, re-projected y raw: %.5f, angle_yaw: %.5f, pixel_yaw: %.5f",
            //             rpy[0] * 180 / M_PI,
            //             rpy[1] * 180 / M_PI,
            //             rpy[2] * 180 / M_PI,
            //             (append_yaw + -yaw) * 180 / M_PI,
            //             append_yaw * 180 / M_PI,
            //             angle_yaw * 180 / M_PI,
            //             pixel_yaw * 180 / M_PI);

            if (success)
            {
                // Fill basic info
                armor_msg.type   = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
                armor_msg.number = armor.number;

                // Fill pose

                armor_msg.pose.position.x = rvec_world.pose.position.x;
                armor_msg.pose.position.y = rvec_world.pose.position.y;
                armor_msg.pose.position.z = rvec_world.pose.position.z;

                // armor_msg.pose.orientation = rvec_world.pose.orientation;
                // if (armor_msg.pose.position.z > 3 && armor_msg.pose.position.z < 4) {
                //   if (armor_min_ratio < 0.115) {
                //     i++;
                //     continue;
                //   }
                // } else if (armor_msg.pose.position.z > 4 && armor_msg.pose.position.z < 5) {
                //   if (armor_min_ratio < 0.155) {
                //     i++;
                //     continue;
                //   }
                // } else if (armor_msg.pose.position.z > 5) {
                //   if (armor_min_ratio < 0.165) {
                //     i++;
                //     continue;
                //   }
                // }

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
                RCLCPP_WARN(this->get_logger(), "PnP failed!");
            }
        }

        // Determine the target armor by the distance to image center and target locking

        // If target_locked = 1, the target is locked, id close to the center of the image will be selected until the target_locked is = 0
        // If target_locked = 1, and no armor is detected, it will automatically select the nearest armor to the center of the image once armor is
        // If target_locked = 0, but armor is detected, the target will be the nearest armor to the center of the image
        // detected
        if (target_locked_)
        {
            if (nearest_armor_index != -1)
            {
                armors_msg_.target_id = id_table_[armors[nearest_armor_index].number];
            }
        }
        else
        {
            if (int(armors_msg_.armors.size()) > 0)
            {
                armors_msg_.target_id = id_table_[armors[nearest_armor_index].number];
            }
            else
            {
                armors_msg_.target_id = 0;
            }
        }

        // Publishing detected armors
        armors_pub_->publish(armors_msg_);

        // FOR DEBUG
        // if (int(armors_msg_.armors.size()) > 0)
        // {
        //     // Publish armor yaw
        //     double roll, pitch, yaw;
        //     std_msgs::msg::Float32 msg;
        //     auto rpy = orientationToRPY(armors_msg_.armors[0].pose.orientation);
        //     msg.data = rpy[2];
        //     armor_yaw_pub_->publish(msg);
        //     if (int(armors_msg_.armors.size()) > 1)
        //         RCLCPP_WARN(this->get_logger(), "Published %d armors", int(armors_msg_.armors.size()));
        // }
        if (detector_node_debug_)
        {
            // Publishing marker
            publishMarkers();
        }
    }

    if (detector_node_debug_)
    {
        cv::Mat final = img_mat_msg_->image.clone();
        cv::putText(final, "FPS: " + std::to_string(fps_), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
        cv::String labels[4] = {"0", "1", "2", "3"};

        std::stringstream latency_ss;
        latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
        auto latency_s = latency_ss.str();
        cv::putText(final, latency_s, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        int number = valid_ids.size();
        for (int i = 0; i < number; i++)
        {
            auto armor = armors[valid_ids[i]];

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
                        "yaw_angle: " + std::to_string((yaws[0] - pnp_solver_->sys_yaw) * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -110),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "yaw_pixel: " + std::to_string((yaws[1] - pnp_solver_->sys_yaw) * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -135),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "yaw_ratio: " + std::to_string((yaws[2] - pnp_solver_->sys_yaw) * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -160),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "IPPE_yaw: " + std::to_string(rpy[2] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -185),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "iter_yaw: " + std::to_string(pnp_yaws[0] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -210),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "SQ_yaw: " + std::to_string(pnp_yaws[1] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -235),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "Ransac_IPPE_yaw: " + std::to_string(pnp_yaws[2] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -260),
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
            cv::putText(final,
                        "Ransac_SQ_yaw: " + std::to_string(pnp_yaws[4] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -310),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.9,
                        cv::Scalar(255, 255, 0),
                        2);
            cv::putText(final,
                        "TJ_yaw: " + std::to_string(yaws[3] * 180 / CV_PI),
                        armor.center - cv::Point2f(5, -335),
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
        if (debug_)
            result_img_pub_.publish(cv_bridge::CvImage(armors_msg_.header, "bgr8", final).toImageMsg());
    }
}

std::unique_ptr<ArmorDetector> ArmorDetectorNode::initArmorDetector()
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
    int blue_binary_thres                  = declare_parameter("blue_binary_thres", 50, param_desc);
    int red_binary_thres                   = declare_parameter("red_binary_thres", 100, param_desc);

    ArmorDetector::LightParams l_params = {.min_ratio_      = declare_parameter("light.min_ratio", 0.1),
                                           .max_ratio_      = declare_parameter("light.max_ratio", 0.4),
                                           .max_angle_      = declare_parameter("light.max_angle", 75.0),
                                           .min_fill_ratio_ = declare_parameter("light.min_fill_ratio", 0.8)};

    ArmorDetector::ArmorParams a_params = {.min_light_ratio_           = declare_parameter("armor.min_light_ratio", 0.75),
                                           .min_small_center_distance_ = declare_parameter("armor.min_small_center_distance", 0.8),
                                           .max_small_center_distance_ = declare_parameter("armor.max_small_center_distance", 3.2),
                                           .min_large_center_distance_ = declare_parameter("armor.min_large_center_distance", 3.2),
                                           .max_large_center_distance_ = declare_parameter("armor.max_large_center_distance", 5.5),
                                           .max_angle_                 = declare_parameter("armor.max_angle", 75.0)};

    auto armor_detector =
        std::make_unique<ArmorDetector>(detect_color == RED ? red_binary_thres : blue_binary_thres, detect_color, l_params, a_params);

    // Init classifier
    auto pkg_path                           = ament_index_cpp::get_package_share_directory("armor_detector");
    auto model_path                         = pkg_path + "/model/mlp.onnx";
    auto label_path                         = pkg_path + "/model/label.txt";
    double threshold                        = this->declare_parameter("classifier_threshold", 0.7);
    std::vector<std::string> ignore_classes = this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
    armor_detector->classifier_             = std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    // Init debug mode
    int detector_debug_num_         = this->declare_parameter("detector_debug", 0);
    armor_detector->detector_debug_ = detector_debug_num_ ? true : false;

    return armor_detector;
}

std::vector<Armor> ArmorDetectorNode::detectArmors(const shm_msgs::msg::Image2m::SharedPtr &_img_msg)
{
    // Convert ROS img to cv::Mat
    this->img_mat_msg_ = shm_msgs::toCvShare(_img_msg, "bgr8");

    // Update params
    armor_detector_->detect_color_ = get_parameter("detect_color").as_int();
    if (armor_detector_->detect_color_ == RED)
        armor_detector_->binary_thres_ = get_parameter("red_binary_thres").as_int();
    else
        armor_detector_->binary_thres_ = get_parameter("blue_binary_thres").as_int();
    armor_detector_->classifier_->threshold = get_parameter("classifier_threshold").as_double();

    std::vector<rm_auto_aim::Armor> armors = armor_detector_->detect(this->img_mat_msg_->image);

    // Publish debug info
    if (debug_)
    {
        binary_img_pub_.publish(cv_bridge::CvImage(armors_msg_.header, "mono8", armor_detector_->binary_img_).toImageMsg());

        // Sort lights and armors data by x coordinate
        std::sort(armor_detector_->debug_lights_.data.begin(),
                  armor_detector_->debug_lights_.data.end(),
                  [](const auto &l1, const auto &l2) { return l1.center_x < l2.center_x; });
        std::sort(armor_detector_->debug_armors_.data.begin(),
                  armor_detector_->debug_armors_.data.end(),
                  [](const auto &a1, const auto &a2) { return a1.center_x < a2.center_x; });

        lights_data_pub_->publish(armor_detector_->debug_lights_);
        armors_data_pub_->publish(armor_detector_->debug_armors_);

        if (!armors.empty())
        {
            auto all_num_img = armor_detector_->getAllNumbersImage();
            number_img_pub_.publish(*cv_bridge::CvImage(armors_msg_.header, "mono8", all_num_img).toImageMsg());
        }
    }

    return armors;
}

void ArmorDetectorNode::createDebugPublishers()
{
    lights_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugLights>("/armor_detector/debug_lights", 10);
    armors_data_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugArmors>("/armor_detector/debug_armors", 10);

    binary_img_pub_ = image_transport::create_publisher(this, "/armor_detector/binary_img");
    number_img_pub_ = image_transport::create_publisher(this, "/armor_detector/number_img");
    result_img_pub_ = image_transport::create_publisher(this, "/armor_detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
    lights_data_pub_.reset();
    armors_data_pub_.reset();

    binary_img_pub_.shutdown();
    number_img_pub_.shutdown();
    result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers()
{
    using Marker         = visualization_msgs::msg::Marker;
    armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)