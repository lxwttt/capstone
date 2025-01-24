#include "armor_detector/pnp_solver.hpp"

#include <cstring>
#include <opencv2/calib3d.hpp>
#include <vector>

#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"

namespace rm_auto_aim
{
PnPSolver::PnPSolver(const std::array<double, 9> &camera_matrix, const std::vector<double> &dist_coeffs)
    : camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
      dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
    // Start from bottom left in clockwise order
    // Model coordinate: x forward, y left, z up
    small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT));   // bottom left
    small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT));    // top left
    small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT));   // top right
    small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT));  // bottom right

    // test_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0));
    // test_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0));
    // test_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0));
    // test_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0));
    // test_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0));  // bottom right
    // test_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0));   // bottom left
    // test_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0));   // top right
    // test_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0));    // top left

    // small_armor_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0)); // top left
    // small_armor_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0)); // top right
    // small_armor_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0)); // bottom left
    // small_armor_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0)); // bottom right

    // small_armor_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT,0 )); // bottom left
    // small_armor_points_.emplace_back(cv::Point3f(SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0)); // top left
    // small_armor_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0)); // top right
    // small_armor_points_.emplace_back(cv::Point3f(-SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0)); // bottom right

    for (const auto &point : small_armor_points_)
    {
        // P_world.push_back(Eigen::Vector4d(0, -point.x, -point.y, 1));
        // P_world.push_back(Eigen::Vector4d(-point.z, 0, -point.y, 1));
        // P_world.push_back(Eigen::Vector4d(0, point.y, point.z, 1));
        P_world_small.push_back(Eigen::Vector4d(0, point.y, point.z, 1));
    }

    // P_world.push_back(Eigen::Vector4d(SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0, 1));
    // P_world.push_back(Eigen::Vector4d(SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0, 1));
    // P_world.push_back(Eigen::Vector4d(-SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT, 0, 1));
    // P_world.push_back(Eigen::Vector4d(-SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT, 0, 1));

    // small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT)); // bottom right
    // small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT)); // bottom left
    // small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT)); // top right
    // small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT)); // top left

    large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_WEIGHT, -LARGE_HALF_HEIGHT));
    large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_WEIGHT, LARGE_HALF_HEIGHT));
    large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_WEIGHT, LARGE_HALF_HEIGHT));
    large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_WEIGHT, -LARGE_HALF_HEIGHT));

    for (const auto &point : large_armor_points_)
    {
        P_world_big.push_back(Eigen::Vector4d(0, point.y, point.z, 1));
    }

    // initialize ceres solver
    camera_coeffs[0] = camera_matrix_.at<double>(0, 0);
    camera_coeffs[1] = camera_matrix_.at<double>(1, 1);
    camera_coeffs[2] = camera_matrix_.at<double>(0, 2);
    camera_coeffs[3] = camera_matrix_.at<double>(1, 2);

    ceres::CostFunction *cost_function = ReprojectionError_AutoDiff::Create(observations, camera_coeffs);
    problem.AddResidualBlock(cost_function, nullptr /* squared loss */, vecs);

    problem.SetParameterLowerBound(vecs, 0, -10);
    problem.SetParameterUpperBound(vecs, 0, 10);
    problem.SetParameterLowerBound(vecs, 1, -10);
    problem.SetParameterUpperBound(vecs, 1, 10);
    problem.SetParameterLowerBound(vecs, 2, -10);
    problem.SetParameterUpperBound(vecs, 2, 10);

    options.linear_solver_type           = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.logging_type                 = ceres::SILENT;
    options.use_nonmonotonic_steps       = true;

    camera_matrix_eigen << camera_matrix[0], 0, camera_matrix[2], 0, camera_matrix[4], camera_matrix[5], 0, 0, 1;
}

bool PnPSolver::solvePnP_BA(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    cur_armor_type = armor.type == ArmorType::SMALL ? 0 : 1;

    rvec.create(3, 1, CV_64F);
    tvec.create(3, 1, CV_64F);

    // Solve pnp
    observations[0] = armor.left_light.bottom.x;
    observations[1] = armor.left_light.bottom.y;
    observations[2] = armor.left_light.top.x;
    observations[3] = armor.left_light.top.y;
    observations[4] = armor.right_light.top.x;
    observations[5] = armor.right_light.top.y;
    observations[6] = armor.right_light.bottom.x;
    observations[7] = armor.right_light.bottom.y;

    // print observations
    // std::cout << "observations: ";
    // for (int i = 0; i < 8; i++) {
    //   std::cout << observations[i] << ", ";
    // }
    // std::cout << std::endl;

    memset(vecs, 0, sizeof(vecs));
    //   vecs[0] = - 15 * CV_PI / 180;
    vecs[5] = 2.0;

    ceres::Solve(options, &problem, &summary);
    //   std::cout << summary.FullReport() << "\n";

    rvec.at<double>(0, 0) = vecs[0];
    rvec.at<double>(1, 0) = vecs[1];
    rvec.at<double>(2, 0) = vecs[2];
    tvec.at<double>(0, 0) = vecs[3];
    tvec.at<double>(1, 0) = vecs[4];
    tvec.at<double>(2, 0) = vecs[5];

    // std::cout << "tvec: " << tvec << std::endl;
    // std::cout << "rvec: " << rvec << std::endl;
    // std::cout << "rvec: " << vecs[0] * 180 / CV_PI << " " << vecs[1] * 180 / CV_PI << " "
    //           << vecs[2] * 180 / CV_PI << " tvec: " << tvec << " cost: " << summary.final_cost
    //           << std::endl;
    return true;
}

bool PnPSolver::solvePnP(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;

    // Fill in image points
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    // image_armor_points.emplace_back(armor.left_light.top);
    // image_armor_points.emplace_back(armor.right_light.top);
    // image_armor_points.emplace_back(armor.left_light.bottom);
    // image_armor_points.emplace_back(armor.right_light.bottom);

    // image_armor_points.emplace_back(armor.right_light.bottom);
    // image_armor_points.emplace_back(armor.left_light.bottom);
    // image_armor_points.emplace_back(armor.right_light.top);
    // image_armor_points.emplace_back(armor.left_light.top);

    P_pixel.clear();
    for (const auto &point : image_armor_points)
    {
        P_pixel.push_back(Eigen::Vector2d(point.x, point.y));
    }

    // P_world.push_back(Eigen::)

    // Solve pnp
    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    // auto object_points = test_points_;
    return cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

bool PnPSolver::solvePnP_Iterative(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    return cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
}

bool PnPSolver::solvePnPSQ(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    return cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_SQPNP);
}

bool PnPSolver::solvePnPRansac_IPPE(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    return cv::solvePnPRansac(
        object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, 500, 2.0, 0.99, cv::noArray(), cv::SOLVEPNP_IPPE);
}

bool PnPSolver::sovlePnPRansac_Iterative(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    return cv::solvePnPRansac(
        object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, 500, 2.0, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);
}

bool PnPSolver::solvePnPRansac_SQ(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
{
    std::vector<cv::Point2f> image_armor_points;
    image_armor_points.emplace_back(armor.left_light.bottom);
    image_armor_points.emplace_back(armor.left_light.top);
    image_armor_points.emplace_back(armor.right_light.top);
    image_armor_points.emplace_back(armor.right_light.bottom);

    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    return cv::solvePnPRansac(
        object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, 500, 2.0, 0.99, cv::noArray(), cv::SOLVEPNP_SQPNP);
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f &image_point)
{
    float cx = camera_matrix_.at<double>(0, 2);
    float cy = camera_matrix_.at<double>(1, 2);
    return cv::norm(image_point - cv::Point2f(cx, cy));
}

/****************TEST****************/

// bool SolveYawPnp(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec)
// {
//     std::vector<cv::Point2f> image_armor_points;
//     // Fill in image points
//     image_armor_points.emplace_back(armor.left_light.bottom);
//     image_armor_points.emplace_back(armor.left_light.top);
//     image_armor_points.emplace_back(armor.right_light.top);
//     image_armor_points.emplace_back(armor.right_light.bottom);
//     // setWorldPoints(object_points);
//     // setImagePoints(image_armor_points);

//     // Solve pnp
//     auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
//     bool result        = cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);

//     return result;
// }

ArmorElevation PnPSolver::setElevation(double pitch)
{
    ArmorElevation elevation;
    if (pitch > ANGLE_BOUNDARY_UP)
    {
        elevation = ARMOR_ELEVATION_UP_75;
    }
    else if (pitch < ANGLE_BOUNDARY_DOWN)
    {
        elevation = ARMOR_ELEVATION_DOWN_15;
    }
    else
    {
        elevation = ARMOR_ELEVATION_UP_15;
    }
    this->elevation = elevation;

    return elevation;
}

std::vector<Eigen::Vector4d> PnPSolver::getMapping(double append_yaw) const
{
    Eigen::Matrix4d M;
    std::vector<Eigen::Vector4d> P_mapping;

    double yaw = sys_yaw + append_yaw;
    double pitch;
    switch (elevation)
    {
    case ARMOR_ELEVATION_UP_15:
        pitch = ANGLE_UP_15;
        break;
    case ARMOR_ELEVATION_UP_75:
        pitch = ANGLE_UP_75;
        break;
    case ARMOR_ELEVATION_DOWN_15:
        pitch = ANGLE_DOWN_15;
        break;
    default:
        pitch = 0;
        break;
    }

    pitch = -pitch;
    // yaw   = -yaw;
    // clang-format off
    M << cos(yaw) * cos(pitch), -sin(yaw), -sin(pitch) * cos(yaw), pose(0),
         sin(yaw) * cos(pitch),  cos(yaw), -sin(pitch) * sin(yaw), pose(1),
                    sin(pitch),         0,             cos(pitch), pose(2),
                             0,         0,                      0,       1;
    // clang-format on

    // create another M matrix that use armor_roll, armor_pitch, yaw, pose to calculate
    // double cos_roll  = cos(0);
    // double sin_roll  = sin(0);
    // double cos_pitch = cos(armor_pitch);
    // double sin_pitch = sin(armor_pitch);
    // // double cos_pitch = cos(pitch);
    // // double sin_pitch = sin(pitch);
    // double cos_yaw   = cos(yaw);
    // double sin_yaw   = sin(yaw);
    // // clang-format off
    // M << cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll, pose(0),
    //      sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll, pose(1),
    //               -sin_pitch,                                cos_pitch * sin_roll,                                cos_pitch * cos_roll, pose(2),
    //                        0,                                                   0,                                                   0,       1;

 
    // P_world is reference point of armor plate
    for (const auto &p : *P_world)
    {
        P_mapping.push_back(M * p);
    }
    // for (const auto& p : P_mapping)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "P_mapping: %f %f %f %f", p(0), p(1), p(2), p(3));
    // }
    return P_mapping;
}

std::vector<Eigen::Vector2d> PnPSolver::getProject(const std::vector<Eigen::Vector4d> &P_world) const
{
    // project point to camera plane
    std::vector<Eigen::Vector2d> P_project;
    for (const auto &p : P_world)
    {
        // TODO: Transform p to camera coordinate
        Eigen::Vector3d p_camera  = (T_odom_to_camera * p).head(3);
        Eigen::Vector3d p_project = camera_matrix_eigen * p_camera;
        P_project.push_back(p_project.head(2) / p_camera(2));
        // P_project.push_back(p_project.segment(1, 2) / p_camera(0));
    }
    // for (const auto& p : P_project)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "P_project: %f %f", p(0), p(1));
    // }
    return P_project;
}
static int count  = 0;
static int count2 = 0;
double PnPSolver::getPixelCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const
{
    if (P_pixel.size() != P_project.size() || P_project.size() < 4)
        return 0.0;

    int map[4] = {0, 1, 3, 2};

    double cost = 0.0;
    for (int i = 0; i < 4; i++)
    {
        int index_this               = map[i];
        int index_next               = map[(i + 1) % 4];
        Eigen::Vector2d pixel_line   = P_pixel[index_next] - P_pixel[index_this];
        Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];
        // cv::line(debug_img,
        //          cv::Point(P_pixel[index_this](0), P_pixel[index_this](1)),
        //          cv::Point(P_pixel[index_next](0), P_pixel[index_next](1)),
        //          cv::Scalar(0, 50 * (i + 1), 0),
        //          2);
        // cv::Mat debug_img_pixel_line = debug_img.clone();
        // cv::line(debug_img_pixel_line,
        //          cv::Point(P_project[index_this](0), P_project[index_this](1)),
        //          cv::Point(P_project[index_next](0), P_project[index_next](1)),
        //          cv::Scalar((count - 1) * 255, (9 - count) * 50, 50 * count),
        //          2);
        // cv::circle(debug_img,
        //            cv::Point(P_project[index_this](0), P_project[index_this](1)),
        //            5,
        //            cv::Scalar((count - 1) * 255, (9 - count) * 50, 50 * count),
        //            -1);
        double this_dist = (P_pixel[index_this] - P_project[index_this]).norm();
        double next_dist = (P_pixel[index_next] - P_project[index_next]).norm();
        double line_dist = fabs(pixel_line.norm() - project_line.norm());

        double pixel_dist = (0.5 * (this_dist + next_dist) + line_dist) / pixel_line.norm();
        cost += pixel_dist;

        // cv::imshow("debug_img", debug_img);
        // cv::imshow("debug_img_pixel_line", debug_img_pixel_line);
        // cv::waitKey(1);
    }

    // Eigen::Vector2d left_P_pixel   = (P_pixel[0] + P_pixel[1]) / 2;
    // Eigen::Vector2d right_P_pixel  = (P_pixel[2] + P_pixel[3]) / 2;
    // Eigen::Vector2d left_P_project = (P_project[0] + P_project[1]) / 2;
    // Eigen::Vector2d right_P_project = (P_project[2] + P_project[3]) / 2;

    // double left_dist  = (left_P_pixel - left_P_project).norm();
    // double right_dist = (right_P_pixel - right_P_project).norm();

    // cost += left_dist + right_dist;

    return cost;
}

double PnPSolver::getAngleCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const
{
    if (P_pixel.size() != P_project.size() || P_project.size() < 4)
    {
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "P_pixel.size(): %d, P_project.size(): %d", P_pixel.size(), P_project.size());
        return 0.0;
    }
    int map[4] = {0, 1, 2, 3};
    // cv::Mat debug_img(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    double cost = 0.0;
    for (int i = 0; i < 4; i++)
    {
        int index_this = map[i];
        int index_next = map[(i + 1) % 4];
        // get diff between this and next point
        Eigen::Vector2d pixel_line   = P_pixel[index_next] - P_pixel[index_this];
        Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];
        // cv::line(debug_img_angle,
        //          cv::Point(P_pixel[index_this](0), P_pixel[index_this](1)),
        //          cv::Point(P_pixel[index_next](0), P_pixel[index_next](1)),
        //          cv::Scalar(0, 50 * (i + 1), 0),
        //          2);
        // cv::circle(debug_img_angle,
        //            cv::Point(P_project[index_this](0), P_project[index_this](1)),
        //            5,
        //            cv::Scalar((count2 - 1) * 255, (9 - count2) * 50, 50 * count2),
        //            -1);
        double cos_angle  = pixel_line.dot(project_line) / (pixel_line.norm() * project_line.norm());
        double angle_dist = fabs(acos(cos_angle));

        // auto lambda = [this](double x) { 
        //    return x * tanh(x);
        //     };
        // double sqr = angle_dist * angle_dist;

        // cost += lambda(angle_dist);
        // cost += sqr;
        cost += angle_dist;

        // cv::imshow("debug_img_angle", debug_img_angle);
    }
    
    return cost;
}

double PnPSolver::getRatioCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const
{
    if (P_pixel.size() != P_project.size() || P_project.size() < 4)
    {
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "P_pixel.size(): %d, P_project.size(): %d", P_pixel.size(), P_project.size());
        return 0.0;
    }
    int map[4] = {0, 1, 3, 2};
    // cv::Mat debug_img(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    double cost = 0.0;

    double origin_left  = (P_pixel[1] - P_pixel[0]).norm();
    double origin_right = (P_pixel[2] - P_pixel[3]).norm();
    double map_left     = (P_project[1] - P_project[0]).norm();
    double map_right    = (P_project[2] - P_project[3]).norm();
    double ratio_lr     = (map_left / origin_left) * (origin_right / map_right);

    double residual_lr = (ratio_lr > 1 ? ratio_lr : 1 / ratio_lr);
    cost += 1.0 * (residual_lr - 1);

    // double residual_lr_ratio_diff = origin_left / origin_right - map_left / map_right;
    // cost += fabs(residual_lr_ratio_diff);


    double origin_up   = (P_pixel[1] - P_pixel[2]).norm();
    double origin_down = (P_pixel[0] - P_pixel[3]).norm();
    double map_up      = (P_project[1] - P_project[2]).norm();
    double map_down    = (P_project[0] - P_project[3]).norm();
    double ratio_ud    = (map_up / origin_up) * (origin_down / map_down);

    double residual_ud = (ratio_ud > 1 ? ratio_ud : 1 / ratio_ud);
    cost += 1.0 * (residual_ud - 1);

    // double residual_ud_ratio_diff = origin_up / origin_down - map_up / map_down;
    // cost += fabs(residual_ud_ratio_diff);
    // cost = cost * 1e2;

    // Angle cost
    // for (int i = 0; i < 4; i++)
    // {
    //     int index_this = map[i];
    //     int index_next = map[(i + 1) % 4];
    //     // get diff between this and next point
    //     Eigen::Vector2d pixel_line   = P_pixel[index_next] - P_pixel[index_this];
    //     Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];
    //     // cv::line(debug_img_angle,
    //     //          cv::Point(P_pixel[index_this](0), P_pixel[index_this](1)),
    //     //          cv::Point(P_pixel[index_next](0), P_pixel[index_next](1)),
    //     //          cv::Scalar(0, 50 * (i + 1), 0),
    //     //          2);
    //     // cv::circle(debug_img_angle,
    //     //            cv::Point(P_project[index_this](0), P_project[index_this](1)),
    //     //            5,
    //     //            cv::Scalar((count2 - 1) * 255, (9 - count2) * 50, 50 * count2),
    //     //            -1);
    //     double cos_angle  = pixel_line.dot(project_line) / (pixel_line.norm() * project_line.norm());
    //     double angle_dist = fabs(acos(cos_angle));

    //     cost += angle_dist;

    //     // cv::imshow("debug_img_angle", debug_img_angle);
    // }

    // pixel cost
    // for (int i = 0; i < 4; i++)
    // {
    //     int index_this               = map[i];
    //     int index_next               = map[(i + 1) % 4];
    //     Eigen::Vector2d pixel_line   = P_pixel[index_next] - P_pixel[index_this];
    //     Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];
    //     // cv::line(debug_img,
    //     //          cv::Point(P_pixel[index_this](0), P_pixel[index_this](1)),
    //     //          cv::Point(P_pixel[index_next](0), P_pixel[index_next](1)),
    //     //          cv::Scalar(0, 50 * (i + 1), 0),
    //     //          2);
    //     // cv::Mat debug_img_pixel_line = debug_img.clone();
    //     // cv::line(debug_img_pixel_line,
    //     //          cv::Point(P_project[index_this](0), P_project[index_this](1)),
    //     //          cv::Point(P_project[index_next](0), P_project[index_next](1)),
    //     //          cv::Scalar((count - 1) * 255, (9 - count) * 50, 50 * count),
    //     //          2);
    //     // cv::circle(debug_img,
    //     //            cv::Point(P_project[index_this](0), P_project[index_this](1)),
    //     //            5,
    //     //            cv::Scalar((count - 1) * 255, (9 - count) * 50, 50 * count),
    //     //            -1);
    //     double this_dist = (P_pixel[index_this] - P_project[index_this]).norm();
    //     double next_dist = (P_pixel[index_next] - P_project[index_next]).norm();
    //     double line_dist = fabs(pixel_line.norm() - project_line.norm());

    //     double pixel_dist = (0.5 * (this_dist + next_dist) + line_dist) / pixel_line.norm();
    //     cost += pixel_dist;

    //     // cv::imshow("debug_img", debug_img);
    //     // cv::imshow("debug_img_pixel_line", debug_img_pixel_line);
    //     // cv::waitKey(1);
    // }

    return cost;
}

double PnPSolver::getJiaoCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const
{
    if (P_pixel.size() != P_project.size() || P_project.size() < 4)
    {
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "P_pixel.size(): %d, P_project.size(): %d", P_pixel.size(), P_project.size());
        return 0.0;
    }
    int map[4] = {0, 1, 3, 2};
    // cv::Mat debug_img(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    double cost = 0.0;
    for (int i = 0; i < 4; i++)
    {
        int index_this = map[i];
        int index_next = map[(i + 1) % 4];
        // get diff between this and next point
        Eigen::Vector2d pixel_line   = P_pixel[index_next] - P_pixel[index_this];
        Eigen::Vector2d project_line = P_project[index_next] - P_project[index_this];
        double this_dist = (P_pixel[index_this] - P_project[index_this]).norm();
        double next_dist = (P_pixel[index_next] - P_project[index_next]).norm();
        double line_dist = fabs(pixel_line.norm() - project_line.norm());
        
        double pixel_dist = (0.5 * (this_dist + next_dist) + line_dist) / pixel_line.norm();
        double cos_angle  = pixel_line.dot(project_line) / (pixel_line.norm() * project_line.norm());
        double angle_dist = fabs(acos(cos_angle));
        double cost_i     =  pow(pixel_dist * std::sin(this->lightbar_avg_incline),2) + pow(angle_dist * std::cos(this->lightbar_avg_incline),2) * 5.0;
        cost += std::sqrt(cost_i);
        // cost += angle_dist + pixel_dist;
    }
    return cost;
}

    

double PnPSolver::getPixelCost(double append_yaw) const
{
    // RCLCPP_WARN(rclcpp::get_logger("rm_auto_aim"), " PIXEL append_yaw: %f", append_yaw);
    auto P_mapping = getMapping(append_yaw);
    // for (int i = 0; i < 4; i++)
    // {
    //     RCLCPP_WARN(rclcpp::get_logger("rm_auto_aim"),
    //                 "PIXEL P_mapping %d: %f %f %f %f",
    //                 i + 1,
    //                 P_mapping[i](0),
    //                 P_mapping[i](1),
    //                 P_mapping[i](2),
    //                 P_mapping[i](3));
    // }
    auto P_project = getProject(P_mapping);
    return getPixelCost(P_project, append_yaw);
}

double PnPSolver::getAngleCost(double append_yaw) const
{
    // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE append_yaw: %f", append_yaw);
    auto P_mapping = getMapping(append_yaw);
    // for (int i = 0; i < 4; i++)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"),
    //                 "ANGLE P_mapping %d: %f %f %f %f",
    //                 i + 1,
    //                 P_mapping[i](0),
    //                 P_mapping[i](1),
    //                 P_mapping[i](2),
    //                 P_mapping[i](3));
    // }
    auto P_project = getProject(P_mapping);
    return getAngleCost(P_project, append_yaw);
}

double PnPSolver::getRatioCost(double append_yaw) const
{
    // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE append_yaw: %f", append_yaw);
    auto P_mapping = getMapping(append_yaw);
    // for (int i = 0; i < 4; i++)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"),
    //                 "ANGLE P_mapping %d: %f %f %f %f",
    //                 i + 1,
    //                 P_mapping[i](0),
    //                 P_mapping[i](1),
    //                 P_mapping[i](2),
    //                 P_mapping[i](3));
    // }
    auto P_project = getProject(P_mapping);
    return getRatioCost(P_project, append_yaw);
}

double PnPSolver::getJiaoCost(double append_yaw) const
{
    auto P_mapping = getMapping(append_yaw);
    auto P_project = getProject(P_mapping);
    return getJiaoCost(P_project, append_yaw);
}

double PnPSolver::getYawByPixelCost(double left, double right, double epsilon) const
{
    count = 0;
    while (right - left > epsilon)
    {
        double mid1  = left + (right - left) / 3;
        double mid2  = right - (right - left) / 3;
        double cost1 = getPixelCost(mid1);
        double cost2 = getPixelCost(mid2);
        // RCLCPP_WARN(rclcpp::get_logger("rm_auto_aim"), "PIXEL: mid1: %f, mid2: %f, f1: %f, f2: %f", mid1, mid2, cost1, cost2);
        if (cost1 < cost2)
        {
            right = mid2;
        }
        else
        {
            left = mid1;
        }
        // RCLCPP_WARN(rclcpp::get_logger("rm_auto_aim"), "PIXEL: left: %f, right: %f", left, right);
        count++;
    }
    // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "PIXEL: count: %d", count);
    return (left + right) / 2;
}

double PnPSolver::getYawByAngleCost(double left, double right, double epsilon) const
{
    count2 = 0;
    while (right - left > epsilon)
    {
        count2++;
        double mid1  = left + (right - left) / 3;
        double mid2  = right - (right - left) / 3;
        double cost1 = getAngleCost(mid1);
        double cost2 = getAngleCost(mid2);
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "mid1: %f, mid2: %f, f1: %f, f2: %f", mid1, mid2, cost1, cost2);
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE: mid1: %f, mid2: %f, f1: %f, f2: %f", mid1, mid2, cost1, cost2);
        if (cost1 < cost2)
        {
            right = mid2;
        }
        else
        {
            left = mid1;
        }
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE: left: %f, right: %f", left, right);
    }
    // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "END");
    return (left + right) / 2;
}

double PnPSolver::getYawByRatioCost(double left, double right, double epsilon) const
{
    count2 = 0;
    while (right - left > epsilon)
    {
        count2++;
        double mid1  = left + (right - left) / 3;
        double mid2  = right - (right - left) / 3;
        double cost1 = getRatioCost(mid1);
        double cost2 = getRatioCost(mid2);
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "mid1: %f, mid2: %f, f1: %f, f2: %f", mid1, mid2, cost1, cost2);
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE: mid1: %f, mid2: %f, f1: %f, f2: %f", mid1, mid2, cost1, cost2);
        if (cost1 < cost2)
        {
            right = mid2;
        }
        else
        {
            left = mid1;
        }
        // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "ANGLE: left: %f, right: %f", left, right);
    }
    // RCLCPP_INFO(rclcpp::get_logger("rm_auto_aim"), "END");
    return (left + right) / 2;
}

double PnPSolver::getYawByMix(double pixel_yaw, double angle_yaw) const
{
    double mid = 0.3;
    double len = 0.1;

    double ratio = 0.5 + 0.5 * sin(M_PI * (fabs(pixel_yaw) - mid) / len);
    double append_yaw;

    if ((fabs(pixel_yaw) > (mid - len / 2)) && (fabs(pixel_yaw) < (mid + len / 2)))
    {
        append_yaw = ratio * pixel_yaw + (1 - ratio) * angle_yaw;
    }
    else if (fabs(pixel_yaw) <= (mid - len / 2))
    {
        append_yaw = angle_yaw;
    }
    else
    {
        append_yaw = pixel_yaw;
    }
    return append_yaw;
}

double PnPSolver::getYawByJiaoCost(double left, double right, double epsilon) const
{

    while (right - left > epsilon)
    {
   
        double mid1  = left + (right - left) / 3;
        double mid2  = right - (right - left) / 3;
        double cost1 = getJiaoCost(mid1);
        double cost2 = getJiaoCost(mid2);
    
        if (cost1 < cost2)
        {
            right = mid2;
        }
        else
        {
            left = mid1;
        }
       
    }

    return (left + right) / 2;
}

void PnPSolver::setPose(const Eigen::Vector4d pose_) { pose = pose_; }

void PnPSolver::setRP(const double &roll, const double &pitch)
{
    armor_roll  = roll;
    armor_pitch = pitch;
}

void PnPSolver::setT_Odom_to_Camera(const Eigen::Matrix4d &T_odom_to_camera_) { this->T_odom_to_camera = T_odom_to_camera_; }
void PnPSolver::passDebugImg(cv::Mat debug_img_)
{
    debug_img       = debug_img_;
    debug_img_angle = debug_img_.clone();
}

void PnPSolver::displayYawPnP()
{
    cv::Mat img_cost(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));

    double step = M_PI / 250;
    double max = -1.0, min = 1e10;
    std::vector<double> pixel_cost_list(250);
    std::vector<double> angle_cost_list(250);
    std::vector<double> ratio_cost_list(250);
    std::vector<double> jiao_cost_list(250);
    std::vector<double> cost_list(250);

    for (int i = 0; i < 250; i++)
    {
        double app_yaw = i * step - M_PI / 2;

        pixel_cost_list[i] = getPixelCost(app_yaw);
        angle_cost_list[i] = getAngleCost(app_yaw);
        ratio_cost_list[i] = getRatioCost(app_yaw);
        jiao_cost_list[i] = getJiaoCost(app_yaw);

        max = std::max(max, pixel_cost_list[i]);
        min = std::min(min, pixel_cost_list[i]);

        max = std::max(max, angle_cost_list[i]);
        min = std::min(min, angle_cost_list[i]);

        max = std::max(max, ratio_cost_list[i]);
        min = std::min(min, ratio_cost_list[i]);
        max = 10;
    }

    for (int i = 0; i < 250; i++)
    {
        int pixel_show_cost = (pixel_cost_list[i] - min) / (max - min) * 500;
        int angle_show_cost = (angle_cost_list[i] - min) / (max - min) * 500;
        int ratio_show_cost = (ratio_cost_list[i] - min) / (max - min) * 500;
        int jiao_show_cost = (jiao_cost_list[i] - min) / (max - min) * 500;
        int cost_show       = (cost_list[i] - min) / (max - min) * 500;

        cv::circle(img_cost, cv::Point(i * 2, 499 - pixel_show_cost), 1, cv::Scalar(0, 255, 255), 2);
        cv::circle(img_cost, cv::Point(i * 2, 499 - angle_show_cost), 1, cv::Scalar(255, 255, 0), 2);
        cv::circle(img_cost, cv::Point(i * 2, 499 - ratio_show_cost), 1, cv::Scalar(255, 0, 255), 2);
        cv::circle(img_cost, cv::Point(i * 2, 499 - jiao_show_cost), 1, cv::Scalar(0, 0, 255), 2);
    }

    double angle_yaw  = getYawByAngleCost(-(M_PI / 2), (M_PI / 2), 0.03);
    double pixel_yaw  = getYawByPixelCost(-(M_PI / 2), (M_PI / 2), 0.03);
    double ratio_yaw  = getYawByRatioCost(-(M_PI / 2), (M_PI / 2), 0.03);
    double jiao_yaw = getYawByJiaoCost(-(M_PI / 2), (M_PI / 2), 0.03);
    double append_yaw = getYawByMix(pixel_yaw, angle_yaw);

    cv::line(img_cost,
             cv::Point((append_yaw + M_PI / 2) / M_PI * 500, 0),
             cv::Point((append_yaw + M_PI / 2) / M_PI * 500, 500),
             cv::Scalar(255, 255, 255),
             2);

    cv::line(img_cost,
             cv::Point((ratio_yaw + M_PI / 2) / M_PI * 500, 0),
                cv::Point((ratio_yaw + M_PI / 2) / M_PI * 500, 500),
                cv::Scalar(255, 0, 255),
                2);
    
    cv::line(img_cost,
                cv::Point((jiao_yaw + M_PI / 2) / M_PI * 500, 0),
                    cv::Point((jiao_yaw + M_PI / 2) / M_PI * 500, 500),
                    cv::Scalar(0, 0, 255),
                    2);



    switch (elevation)
    {
    case ARMOR_ELEVATION_UP_15:
        cv::putText(img_cost, "UP15", cv::Point(20, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 150, 0), 1);
        break;
    case ARMOR_ELEVATION_UP_75:
        cv::putText(img_cost, "UP75", cv::Point(20, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 150, 0), 1);
        break;
    case ARMOR_ELEVATION_DOWN_15:
        cv::putText(img_cost, "DW15", cv::Point(20, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 150, 0), 1);
        break;
    }

    for (int i = 1; i <= 7; i++)
    {
        int left_x  = 250 - 250 * 0.2 * i / M_PI;
        int right_x = 250 + 250 * 0.2 * i / M_PI;
        cv::line(img_cost, cv::Point(left_x, 486), cv::Point(left_x, 499), cv::Scalar(255, 255, 255), 1);
        cv::line(img_cost, cv::Point(right_x, 486), cv::Point(right_x, 499), cv::Scalar(255, 255, 255), 1);
    }
    cv::line(img_cost, cv::Point(250, 483), cv::Point(250, 499), cv::Scalar(255, 255, 255), 2);

    cv::imshow("cost", img_cost);
    cv::waitKey(1);
}

void PnPSolver::setArmorType(bool is_small_armor)
{
    if (is_small_armor)
    {
        P_world = &P_world_small;
    }
    else
    {
        P_world = &P_world_big;
    }
}

void PnPSolver::setIncline(double incldine)
{
    this->lightbar_avg_incline = incldine;
}

double PnPSolver::calculateLoss(const cv::Mat &rvec, const cv::Mat &tvec, const cv::Mat &tvec_world)
{
    std::vector<cv::Point2f> projected_points;
    auto object_points = small_armor_points_;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, projected_points);

    double armor_loss = 0;
    for (int i = 0; i < 4; i++)
    {
        // auto point = armor.
        auto point           = P_pixel[i];
        cv::Point2f cv_point = cv::Point2f(point(0), point(1));
        // double loss          = euclideanDistance(cv_point, projected_points[i] / std::abs(tvec_world.at<double>(2)));
        // double loss = cv::norm(cv_point, projected_points[i], cv::NORM_L2);
        double loss = cv::norm(cv_point - projected_points[i]) / std::abs(tvec_world.at<double>(2));
        armor_loss += loss;
    }
    return armor_loss;
}

void fixArmorYaw(const cv::Mat &rvec, const cv::Mat &tvec, const cv::Mat &tvec_world) {}

// bool optDLS_mrp_seqrot(const Eigen::MatrixXd& p3, 
//                        const Eigen::MatrixXd& p2, 
//                        Eigen::Matrix3d& R, 
//                        Eigen::Vector3d& t, 
//                        double& minrepr, 
//                        bool chk_singular = false) {
//     // Check input dimensions
//     if (p3.rows() != 3 || p2.rows() != 2 || p3.cols() != p2.cols()) {
//         std::cerr << "Input dimensions are incorrect." << std::endl;
//         return false;
//     }

//     int npts = p3.cols();

//     // Initialize accumulators for AtA and BtA
//     Eigen::Matrix3d Z = Eigen::Matrix3d::Zero();
//     Eigen::Matrix3d Zu = Eigen::Matrix3d::Zero();
//     Eigen::Matrix3d Zv = Eigen::Matrix3d::Zero();
//     Eigen::Matrix3d Zuv = Eigen::Matrix3d::Zero();

//     Eigen::Vector2d sump2 = Eigen::Vector2d::Zero();
//     double sump2sq = 0.0;

//     Eigen::Vector3d up3 = Eigen::Vector3d::Zero();
//     Eigen::Vector3d vp3 = Eigen::Vector3d::Zero();
//     Eigen::Vector3d uvp3 = Eigen::Vector3d::Zero();
//     Eigen::Vector3d sump3 = Eigen::Vector3d::Zero();

//     for (int i = 0; i < npts; ++i) {
//         Eigen::Vector2d p2_ = p2.col(i);
//         Eigen::Vector3d p3_ = p3.col(i);

//         double p2_sq = p2_.squaredNorm();

//         // A'*A accumulation
//         Eigen::Matrix3d p3p3t = p3_ * p3_.transpose();
//         Z += p3p3t;
//         Zu -= p2_(0) * p3p3t;
//         Zv -= p2_(1) * p3p3t;
//         Zuv += p2_sq * p3p3t;

//         // B'*B accumulation
//         sump2 += p2_;
//         sump2sq += p2_sq;

//         // B'*A accumulation
//         sump3 += p3_;
//         up3 -= p2_(0) * p3_;
//         vp3 -= p2_(1) * p3_;
//         uvp3 += p2_sq * p3_;
//     }

//     // Assemble AtA matrix
//     Eigen::MatrixXd AtA(9, 9);
//     AtA.setZero();
//     AtA.block<3,3>(0,0) = Z;
//     AtA.block<3,3>(0,6) = Zu;
//     AtA.block<3,3>(3,3) = Z;
//     AtA.block<3,3>(3,6) = Zv;
//     AtA.block<3,3>(6,6) = Zuv;
//     // Since AtA is symmetric, fill the symmetric blocks
//     AtA.block<3,3>(6,0) = Zu.transpose();
//     AtA.block<3,3>(6,3) = Zv.transpose();

//     // Assemble BtA matrix
//     Eigen::MatrixXd BtA(3, 9);
//     BtA.setZero();
//     BtA(0,0) = sump3(0); BtA(0,1) = sump3(1); BtA(0,2) = sump3(2);raphic coordinates of 
//     BtA(0,6) = up3(0);     BtA(0,7) = up3(1);     BtA(0,8) = up3(2);
//     BtA(1,0) = sump3(0); BtA(1,1) = sump3(1); BtA(1,2) = sump3(2);
//     BtA(1,6) = vp3(0);     BtA(1,7) = vp3(1);     BtA(1,8) = vp3(2);
//     BtA(2,6) = uvp3(0);    BtA(2,7) = uvp3(1);    BtA(2,8) = uvp3(2);

//     // Assemble BtB matrix
//     Eigen::Matrix3d BtB;
//     BtB.setZero();
//     BtB(0,0) = npts;
//     BtB(0,2) = -sump2(0);
//     BtB(1,1) = npts;
//     BtB(1,2) = -sump2(1);
//     BtB(2,0) = -sump2(0);
//     BtB(2,1) = -sump2(1);
//     BtB(2,2) = sump2sq;

//     // Compute tMat = inv(BtB) * BtA
//     Eigen::MatrixXd BtB_inv;
//     bool invertible;
//     double condition_number;
//     // Check if BtB is invertible
//     Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(BtB);
//     invertible = lu_decomp.isInvertible();
//     if (!invertible) {
//         if (chk_singular) {
//             return false;
//         } else {
//             // Use pseudo-inverse or other methods
//             BtB_inv = BtB.completeOrthogonalDecomposition().pseudoInverse();
//         }
//     } else {
//         BtB_inv = BtB.inverse();
//     }

//     Eigen::MatrixXd tMat = BtB_inv * BtA;

//     // Compute M = AtA - BtA.transpose() * tMat
//     Eigen::MatrixXd M = AtA - BtA.transpose() * tMat;

//     // Define R0 rotations (Identity, rotation by pi about x, y, z)
//     std::vector<Eigen::Matrix3d> R0_list;
//     R0_list.emplace_back(Eigen::Matrix3d::Identity());

//     Eigen::Matrix3d rrotx_pi;
//     rrotx_pi << 1, 0, 0,
//                0, -1, 0,
//                0, 0, -1;
//     R0_list.emplace_back(rrotx_pi);

//     Eigen::Matrix3d rroty_pi;
//     rroty_pi << -1, 0, 0,
//                 0, 1, 0,
//                 0, 0, -1;
//     R0_list.emplace_back(rroty_pi);

//     Eigen::Matrix3d rrotz_pi;
//     rrotz_pi << -1, 0, 0,
//                 0, -1, 0,
//                 0, 0, 1;
//     R0_list.emplace_back(rrotz_pi);

//     // Initialize minimum error and corresponding variables
//     double minerr = std::numeric_limits<double>::infinity();
//     Eigen::Vector3d minmrp = Eigen::Vector3d::Zero();
//     Eigen::Matrix3d minR = Eigen::Matrix3d::Zero();
//     Eigen::Vector3d mint = Eigen::Vector3d::Zero();

//     // Iterate over all R0 rotations
//     for (size_t j = 0; j < R0_list.size(); ++j) {
//         const Eigen::Matrix3d& R0j = R0_list[j];
//         // Construct G matrix
//         Eigen::MatrixXd G(9,9);
//         G.setZero();
//         G.block<3,3>(0,0) = R0j;
//         G.block<3,3>(3,3) = R0j;
//         G.block<3,3>(6,6) = R0j;

//         // Compute M0 = G * M * G.transpose()
//         Eigen::MatrixXd M0 = G * M * G.transpose();

//         // Solve M0 * psi = 0
//         std::vector<Eigen::Vector3d> solutions;
//         bool solved = solver_optDLS_mrp(M0, solutions, chk_singular);
//         if (!solved || solutions.empty()) {
//             continue;
//         }

//         // Iterate through all solutions
//         for (const auto& psi : solutions) {
//             // Check magnitude
//             double magsq = psi.squaredNorm();
//             if (magsq > 1.0) {
//                 continue; // Optionally handle shadow MRP here
//             }

//             // Convert MRP to quaternion
//             Eigen::Vector4d q;
//             q(0) = (1.0 - magsq) / (1.0 + magsq);
//             q.segment<3>(1) = (2.0 * psi) / (1.0 + magsq);

//             // Ensure the quaternion is normalized
//             q.normalize();

//             // Convert quaternion to rotation matrix
//             Eigen::Matrix3d R_est = quatToRotMatrix(q);

//             // Undo pre-rotation
//             R_est = R_est * R0j;

//             // Compute t = -tMat * r, where r is the reshaped R_est (column-major)
//             Eigen::VectorXd r = Eigen::Map<Eigen::Vector<double, 9, 1>>(R_est.data());
//             Eigen::Vector3d t_est = -tMat * r;

//             // Compute reprojection error
//             // Transform p3 points
//             Eigen::MatrixXd xp3 = (R_est * p3).colwise() + t_est;

//             // Check if all points are in front of the camera
//             if ((xp3.row(2).array() < 0).any()) {
//                 continue;
//             }

//             // Project points
//             Eigen::MatrixXd proj = xp3.topRows(2).cwiseQuotient(xp3.row(2).replicate(2,1));

//             // Compute error
//             Eigen::MatrixXd e = proj - p2;
//             double err = e.array().square().sum();

//             // Update minimum error and corresponding R, t if necessary
//             if (err < minerr) {
//                 minerr = err;
//                 minmrp = psi;
//                 minR = R_est;
//                 mint = t_est;
//             }
//         }
//     }

//     // Check if a valid solution was found
//     if (minerr == std::numeric_limits<double>::infinity()) {
//         R = Eigen::Matrix3d::Zero();
//         t = Eigen::Vector3d::Zero();
//         minrepr = 0.0;
//         // Optionally log a warning
//         std::cerr << "optDLS_mrp_seqrot: No solutions found, singular rotation?" << std::endl;
//         return false;
//     }

//     // Set the results
//     R = minR;
//     t = mint;
//     minrepr = minerr;

//     return true;
// }


}  // namespace rm_auto_aim