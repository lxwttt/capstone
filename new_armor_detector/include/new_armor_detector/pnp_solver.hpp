#ifndef NEW_ARMOR_DETECTOR__PNP_SOLVER_HPP_
#define NEW_ARMOR_DETECTOR__PNP_SOLVER_HPP_

// OpenCV
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>

// Ceres
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// STD
#include <array>
#include <vector>

#include "new_armor_detector/stamp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace rm_auto_aim
{

constexpr double ANGLE_UP_15   = M_PI / 12;
constexpr double ANGLE_DOWN_15 = -M_PI / 12;
constexpr double ANGLE_UP_75   = 5 * M_PI / 12;

constexpr double ANGLE_BOUNDARY_UP   = M_PI / 4;
constexpr double ANGLE_BOUNDARY_DOWN = 0;
enum ArmorElevation
{
    ARMOR_ELEVATION_UP_15,
    ARMOR_ELEVATION_UP_75,
    ARMOR_ELEVATION_DOWN_15,
    ARMOR_ELEVATION_NONE
};

// 0 for small armor, 1 for large armor
static bool cur_armor_type = 0;

class PnPSolver
{
   public:
    PnPSolver(const std::array<double, 9> &camera_matrix, const std::vector<double> &distortion_coefficients);

    // Get 3d position
    bool solvePnP(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solvePnP_BA(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solveYawPnP(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solvePnP_Iterative(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solvePnPSQ(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solvePnPRansac_IPPE(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool sovlePnPRansac_Iterative(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);
    bool solvePnPRansac_SQ(const Armor &armor, cv::Mat &rvec, cv::Mat &tvec);

    // TEST
    void setWorldPoints(const std::vector<cv::Point3f> &object_points);
    void setImagePoints(const std::vector<cv::Point2f> &image_points);

    void setIncline(double incline);
    ArmorElevation setElevation(double pitch);
    // ArmorElevation setElevation(rm::ArmorID armor_id);
    std::vector<Eigen::Vector4d> getMapping(double append_yaw) const;
    std::vector<Eigen::Vector2d> getProject(const std::vector<Eigen::Vector4d> &P_world) const;
    double getCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const;
    double getPixelCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const;
    double getAngleCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const;
    double getRatioCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const;
    double getJiaoCost(const std::vector<Eigen::Vector2d> &P_project, double append_yaw) const;

    double getCost(double append_yaw) const;
    double getPixelCost(double append_yaw) const;
    double getAngleCost(double append_yaw) const;
    double getRatioCost(double append_yaw) const;
    double getJiaoCost(double append_yaw) const;

    double getYawByPixelCost(double left, double right, double epsilon) const;
    double getYawByAngleCost(double left, double right, double epsilon) const;
    double getYawByRatioCost(double left, double right, double epsilon) const;
    double getYawByJiaoCost(double left, double right, double epsilon) const;
    double getYawByMix(double pixel_yaw, double angle_yaw) const;

    // Calculate the distance between armor center and image center
    float calculateDistanceToCenter(const cv::Point2f &image_point);
    void setPose(const Eigen::Vector4d pose_);
    void setRP(const double &roll, const double &pitch);
    void setArmorType(bool armor_type_);
    void setT_Odom_to_Camera(const Eigen::Matrix4d &T_odom_to_camera_);
    void passDebugImg(cv::Mat debug_img_);

    void displayYawPnP();

    void fixArmorYaw(Armor &armor, const cv::Mat &rvec, const cv::Mat &tvec);
    double calculateLoss(const cv::Mat &rvec, const cv::Mat &tvec, const cv::Mat &tvec_world);

    double sys_yaw              = 0;
    double armor_pitch          = 0;
    double armor_roll           = 0;
    double lightbar_avg_incline = 0;

   private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // TEST
    Eigen::Matrix3d camera_matrix_eigen;
    ArmorElevation elevation;
    Eigen::Vector4d pose;
    Eigen::Matrix4d T_odom_to_camera;
    cv::Mat debug_img;
    cv::Mat debug_img_angle;

    // ceres coefficients
    // rotation vector and translation vector
    double vecs[6];
    // The camera is parameterized using 4 parameters: 2 for focal length and 2 for center.
    double camera_coeffs[4], observations[8];

    // Regularization coefficients
    static constexpr double WEIGHT = 0.1;
    static constexpr double CENTER = -15 * CV_PI / 180;

    // Unit: mm
    static constexpr float SMALL_ARMOR_WIDTH  = 135;  // 135
    static constexpr float SMALL_ARMOR_HEIGHT = 55;   // 55
    static constexpr float LARGE_ARMOR_WIDTH  = 225;  // 225
    static constexpr float LARGE_ARMOR_HEIGHT = 55;   // 55
    static constexpr double SMALL_HALF_WEIGHT = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
    static constexpr double SMALL_HALF_HEIGHT = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
    static constexpr double LARGE_HALF_WEIGHT = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
    static constexpr double LARGE_HALF_HEIGHT = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

    // Four vertices of armor in 3d
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
    std::vector<cv::Point3f> test_points_;
    std::vector<Eigen::Vector4d> P_world_small;
    std::vector<Eigen::Vector4d> P_world_big;
    std::vector<Eigen::Vector4d> *P_world = &P_world_small;
    std::vector<Eigen::Vector2d> P_pixel;

    // Ceres solver
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

#define NUM_RESIDUALS 10  // 2 for length ratio, 4 for angle, 2 for ud length, 2 for center
#define NUM_PARAMS 6

    // Ceres solver using auto diff
    struct ReprojectionError_AutoDiff
    {
        ReprojectionError_AutoDiff(double *observed, double *camera_coeffs) : observed(observed), camera_coeffs(camera_coeffs) {}

        template <typename T>
        bool operator()(const T *const vecs, T *residuals) const
        {
            T p[4][3];  // 3d points in camera coordinate

            T points_SMALL[4][3] = {
                {T(-SMALL_HALF_WEIGHT), T(-SMALL_HALF_HEIGHT), T(0)},
                {T(-SMALL_HALF_WEIGHT), T(SMALL_HALF_HEIGHT), T(0)},
                {T(SMALL_HALF_WEIGHT), T(SMALL_HALF_HEIGHT), T(0)},
                {T(SMALL_HALF_WEIGHT), T(-SMALL_HALF_HEIGHT), T(0)},
            };

            T points_LARGE[4][3] = {
                {T(-LARGE_HALF_WEIGHT), T(-LARGE_HALF_HEIGHT), T(0)},
                {T(-LARGE_HALF_WEIGHT), T(LARGE_HALF_HEIGHT), T(0)},
                {T(LARGE_HALF_WEIGHT), T(LARGE_HALF_HEIGHT), T(0)},
                {T(LARGE_HALF_WEIGHT), T(-LARGE_HALF_HEIGHT), T(0)},
            };

            /* 1----2
             * |    |
             * 0----3
             */
            // vecs[0 - 3] roll pitch yaw

            // T rotation[3][3];  // yaw pitch roll
            // rotation[0][0] =

            if (cur_armor_type)
            {
                for (int i = 0; i < 4; i++)
                {
                    ceres::AngleAxisRotatePoint(vecs, points_LARGE[i], p[i]);
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    ceres::AngleAxisRotatePoint(vecs, points_SMALL[i], p[i]);
                }
            }

            const double &focal_x = camera_coeffs[0], &focal_y = camera_coeffs[1];
            const double &center_x = camera_coeffs[2], &center_y = camera_coeffs[3];

            // T predicted_x[4], predicted_y[4];
            T P_predicted[4][2];  // x, y in camera coordinate

            /* 1----2
             * |    |
             * 0----3
             */

            // camera[3,4,5] are the translation.
            for (int i = 0; i < 4; i++)
            {
                p[i][0] += vecs[3];
                p[i][1] += vecs[4];
                p[i][2] += vecs[5];

                P_predicted[i][0] = focal_x * p[i][0] / p[i][2] + center_x;
                P_predicted[i][1] = focal_y * p[i][1] / p[i][2] + center_y;
            }

            // Lengh ratio cost
            // left right lightbar length ratio loss
            double origin_left  = ((observed[0] - observed[2]) * (observed[0] - observed[2]) +
                                  (observed[1] - observed[3]) * (observed[1] - observed[3]));  // left lightbar length observed
            double origin_right = ((observed[4] - observed[6]) * (observed[4] - observed[6]) +
                                   (observed[5] - observed[7]) * (observed[5] - observed[7]));  // right lightbar length observed
            T predict_left      = ((P_predicted[0][0] - P_predicted[1][0]) * (P_predicted[0][0] - P_predicted[1][0]) +
                              (P_predicted[0][1] - P_predicted[1][1]) * (P_predicted[0][1] - P_predicted[1][1]));  // left lightbar length predicted
            T predict_right     = ((P_predicted[2][0] - P_predicted[3][0]) * (P_predicted[2][0] - P_predicted[3][0]) +
                               (P_predicted[2][1] - P_predicted[3][1]) * (P_predicted[2][1] - P_predicted[3][1]));  // right lightbar length predicted
            T ratio_lr          = (predict_left / origin_left) * (origin_right / predict_right);  // ratio of left right lightbar length

            residuals[0] = (ratio_lr > T(1) ? ratio_lr : T(1) / ratio_lr) - T(1);  // loss of left right lightbar length ratio

            // up down lightbar length ratio loss
            double origin_down = ((observed[0] - observed[6]) * (observed[0] - observed[6]) +
                                  (observed[1] - observed[7]) * (observed[1] - observed[7]));  // up down lightbar length observed
            double origin_up   = ((observed[2] - observed[4]) * (observed[2] - observed[4]) +
                                (observed[3] - observed[5]) * (observed[3] - observed[5]));  // up up lightbar length observed
            T predict_down     = ((P_predicted[0][0] - P_predicted[3][0]) * (P_predicted[0][0] - P_predicted[3][0]) +
                              (P_predicted[0][1] - P_predicted[3][1]) * (P_predicted[0][1] - P_predicted[3][1]));  // down lightbar length predicted
            T predict_up       = ((P_predicted[1][0] - P_predicted[2][0]) * (P_predicted[1][0] - P_predicted[2][0]) +
                            (P_predicted[1][1] - P_predicted[2][1]) * (P_predicted[1][1] - P_predicted[2][1]));  // up lightbar length predicted
            T ratio_ud         = (predict_down / origin_down) * (origin_up / predict_up);

            residuals[1] = (ratio_ud > T(1) ? ratio_ud : T(1) / ratio_ud) - T(1);

            // Angle cost
            // double L_up_observed[2]    = {observed[2] - observed[4], observed[3] - observed[5]};  // up lightbar vector observed
            // double L_down_observed[2]  = {observed[0] - observed[6], observed[1] - observed[7]};  // down lightbar vector observed
            // double L_left_observed[2]  = {observed[0] - observed[2], observed[1] - observed[3]};  // left lightbar vector observed
            // double L_right_observed[2] = {observed[4] - observed[6], observed[5] - observed[7]};  // right lightbar vector observed

            // T L_up_predict[2]    = {P_predicted[1][0] - P_predicted[2][0], P_predicted[1][1] - P_predicted[2][1]};  // up lightbar vector predicted
            // T L_down_predict[2]  = {P_predicted[0][0] - P_predicted[3][0], P_predicted[0][1] - P_predicted[3][1]};  // down lightbar vector
            // predicted T L_left_predict[2]  = {P_predicted[0][0] - P_predicted[1][0], P_predicted[0][1] - P_predicted[1][1]};  // left lightbar
            // vector predicted T L_right_predict[2] = {P_predicted[2][0] - P_predicted[3][0], P_predicted[2][1] - P_predicted[3][1]};  // right
            // lightbar vector predicted

            // residuals[2] = ceres::atan2(L_up_predict[1], L_up_predict[0]) - ceres::atan2(L_up_observed[1], L_up_observed[0]);
            // residuals[3] = ceres::atan2(L_down_predict[1], L_down_predict[0]) - ceres::atan2(L_down_observed[1], L_down_observed[0]);
            // residuals[4] = ceres::atan2(L_left_predict[1], L_left_predict[0]) - ceres::atan2(L_left_observed[1], L_left_observed[0]);
            // residuals[5] = ceres::atan2(L_right_predict[1], L_right_predict[0]) - ceres::atan2(L_right_observed[1], L_right_observed[0]);

            // pixel cost

            T dist_UL_2 = (P_predicted[0][0] - observed[0]) * (P_predicted[0][0] - observed[0]) +
                          (P_predicted[0][1] - observed[1]) * (P_predicted[0][1] - observed[1]);

            T dist_UR_2 = (P_predicted[1][0] - observed[2]) * (P_predicted[1][0] - observed[2]) +
                          (P_predicted[1][1] - observed[3]) * (P_predicted[1][1] - observed[3]);

            T dist_DR_2 = (P_predicted[2][0] - observed[4]) * (P_predicted[2][0] - observed[4]) +
                          (P_predicted[2][1] - observed[5]) * (P_predicted[2][1] - observed[5]);

            T dist_DL_2 = (P_predicted[3][0] - observed[6]) * (P_predicted[3][0] - observed[6]) +
                          (P_predicted[3][1] - observed[7]) * (P_predicted[3][1] - observed[7]);

            residuals[2] = dist_UL_2;
            residuals[3] = dist_UR_2;
            residuals[4] = dist_DR_2;
            residuals[5] = dist_DL_2;

#define DIST_IGNORE_THRES 1e2

            if (residuals[2] < DIST_IGNORE_THRES)
            {
                residuals[2] = T(0);
            }
            if (residuals[3] < DIST_IGNORE_THRES)
            {
                residuals[3] = T(0);
            }
            if (residuals[4] < DIST_IGNORE_THRES)
            {
                residuals[4] = T(0);
            }
            if (residuals[5] < DIST_IGNORE_THRES)
            {
                residuals[5] = T(0);
            }

#define DIST_WEIGHT 1e3
            residuals[2] *= DIST_WEIGHT;
            residuals[3] *= DIST_WEIGHT;
            residuals[4] *= DIST_WEIGHT;
            residuals[5] *= DIST_WEIGHT;

            // ud length cost
            T down_length_ratio = origin_down / predict_down;
            T up_length_ratio   = origin_up / predict_up;

            residuals[6] = (down_length_ratio > T(1) ? down_length_ratio : T(1) / down_length_ratio) - T(1);
            residuals[7] = (up_length_ratio > T(1) ? up_length_ratio : T(1) / up_length_ratio) - T(1);

            // center cost
            residuals[8] = (observed[0] + observed[2] + observed[4] + observed[6]) / T(4) - center_x;
            residuals[9] = (observed[1] + observed[3] + observed[5] + observed[7]) / T(4) - center_y;

#define RATIO_WEIGHT 1000
#define ANGLE_WEIGHT 1000
#define LENGTH_WEIGHT 1000
#define CENTER_WEIGHT 100

            residuals[0] *= RATIO_WEIGHT;
            residuals[1] *= RATIO_WEIGHT;
            // residuals[2] *= ANGLE_WEIGHT;
            // residuals[3] *= ANGLE_WEIGHT;
            // residuals[4] *= ANGLE_WEIGHT;
            // residuals[5] *= ANGLE_WEIGHT;
            residuals[6] *= LENGTH_WEIGHT;
            residuals[7] *= LENGTH_WEIGHT;
            residuals[8] *= CENTER_WEIGHT;
            residuals[9] *= CENTER_WEIGHT;

            return true;
        }

        static ceres::CostFunction *Create(double *observed, double *camera_coeffs)
        {
            return (new ceres::AutoDiffCostFunction<ReprojectionError_AutoDiff, NUM_RESIDUALS, NUM_PARAMS>(
                new ReprojectionError_AutoDiff(observed, camera_coeffs)));
        }

        double *observed;
        double *camera_coeffs;
    };
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__PNP_SOLVER_HPP_