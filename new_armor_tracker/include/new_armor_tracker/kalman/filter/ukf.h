#ifndef NEW_ARMOR_TRACKER__KALMAN__FILTER__UKF_HPP_
#define NEW_ARMOR_TRACKER__KALMAN__FILTER__UKF_HPP_

#include <Eigen/Dense>
#include <cmath>

template <int dimX, int dimY> class UKF {
public:
  using MatXX = Eigen::Matrix<double, dimX, dimX>;
  using MatXY = Eigen::Matrix<double, dimX, dimY>;
  using MatYX = Eigen::Matrix<double, dimY, dimX>;
  using MatYY = Eigen::Matrix<double, dimY, dimY>;
  using VecX = Eigen::Matrix<double, dimX, 1>;
  using VecY = Eigen::Matrix<double, dimY, 1>;

  UKF()
      : estimate_X(VecX::Zero()), P(MatXX::Identity()), Q(MatXX::Identity()),
        R(MatYY::Identity()), lambda(3 - dimX),
        forgetting_factor(0.3) { // Default forgetting factor
    initializeWeights();
  }

  UKF(const MatXX &Q0, const MatYY &R0, double alpha = 1e-3, double beta = 2.0,
      double kappa = 0.0, double forgetting_factor = 0.3)
      : estimate_X(VecX::Zero()), P(MatXX::Identity()), Q(Q0), R(R0),
        alpha(alpha), beta(beta), kappa(kappa),
        lambda(alpha * alpha * (dimX + kappa) - dimX),
        forgetting_factor(forgetting_factor) {
    initializeWeights();
  }

  void restart() {
    estimate_X = VecX::Zero();
    P = MatXX::Identity();
  }

  // Adaptive noise adjustment method
  void adaptNoise(const VecY &innovation) {
    // Update R adaptively based on innovation
    R = (1.0 - forgetting_factor) * R +
        forgetting_factor * (innovation * innovation.transpose());

    // Update Q adaptively (basic implementation, can be application-specific)
    MatXX state_diff = MatXX::Zero();
    for (int i = 0; i < dimX; ++i) {
      state_diff(i, i) = std::pow(innovation.norm(), 2);
    }
    Q = (1.0 - forgetting_factor) * Q + forgetting_factor * state_diff;
  }

  template <class Func> VecX predict(Func &&func) {
    auto sigma_points = generateSigmaPoints();
    Eigen::Matrix<double, dimX, 2 * dimX + 1> predicted_sigma_points;

    // Transform sigma points through the process model
    for (int i = 0; i < 2 * dimX + 1; i++) {
      predicted_sigma_points.col(i) = func(sigma_points.col(i));
    }

    // Compute predicted mean
    predict_X = VecX::Zero();
    for (int i = 0; i < 2 * dimX + 1; i++) {
      predict_X += weights_mean[i] * predicted_sigma_points.col(i);
    }

    // Compute predicted covariance
    P = MatXX::Zero();
    for (int i = 0; i < 2 * dimX + 1; i++) {
      VecX diff = predicted_sigma_points.col(i) - predict_X;
      P += weights_cov[i] * (diff * diff.transpose());
    }
    P += Q;

    return predict_X;
  }

  template <class Func> VecX update(Func &&func, const VecY &Y) {
    auto sigma_points = generateSigmaPoints();
    Eigen::Matrix<double, dimY, 2 * dimX + 1> predicted_measurements;

    // Transform sigma points through the measurement model
    for (int i = 0; i < 2 * dimX + 1; i++) {
      predicted_measurements.col(i) = func(sigma_points.col(i));
    }

    // Compute predicted measurement mean
    predict_Y = VecY::Zero();
    for (int i = 0; i < 2 * dimX + 1; i++) {
      predict_Y += weights_mean[i] * predicted_measurements.col(i);
    }

    // Compute innovation covariance
    MatYY S = MatYY::Zero();
    for (int i = 0; i < 2 * dimX + 1; i++) {
      VecY diff = predicted_measurements.col(i) - predict_Y;
      S += weights_cov[i] * (diff * diff.transpose());
    }
    S += R;

    // Compute cross-covariance
    MatXY cross_cov = MatXY::Zero();
    for (int i = 0; i < 2 * dimX + 1; i++) {
      VecX state_diff = sigma_points.col(i) - predict_X;
      VecY measurement_diff = predicted_measurements.col(i) - predict_Y;
      cross_cov += weights_cov[i] * (state_diff * measurement_diff.transpose());
    }

    // Compute Kalman gain
    MatXY K = cross_cov * S.inverse();

    // Update state estimate and covariance
    VecY innovation = Y - predict_Y;
    estimate_X = predict_X + K * innovation;
    P = P - K * S * K.transpose();

    // Adaptively adjust noise covariance matrices
    adaptNoise(innovation);

    return estimate_X;
  }

private:
  void initializeWeights() {
    weights_mean[0] = lambda / (dimX + lambda);
    weights_cov[0] = weights_mean[0] + (1.0 - alpha * alpha + beta);

    for (int i = 1; i < 2 * dimX + 1; i++) {
      weights_mean[i] = 1.0 / (2.0 * (dimX + lambda));
      weights_cov[i] = weights_mean[i];
    }
  }

  Eigen::Matrix<double, dimX, 2 * dimX + 1> generateSigmaPoints() {
    Eigen::Matrix<double, dimX, 2 * dimX + 1> sigma_points;
    MatXX sqrt_P = P.llt().matrixL();

    sigma_points.col(0) = estimate_X;
    for (int i = 0; i < dimX; i++) {
      sigma_points.col(i + 1) =
          estimate_X + std::sqrt(dimX + lambda) * sqrt_P.col(i);
      sigma_points.col(i + 1 + dimX) =
          estimate_X - std::sqrt(dimX + lambda) * sqrt_P.col(i);
    }

    return sigma_points;
  }

public:
  VecX estimate_X;
  VecX predict_X;
  VecY predict_Y;
  MatXX P;
  MatXX Q;
  MatYY R;

  double alpha;             // Spread of sigma points
  double beta;              // Optimized for Gaussian distributions
  double kappa;             // Secondary scaling parameter
  double lambda;            // Scaling factor
  double forgetting_factor; // Forgetting factor for adaptive noise

  Eigen::Matrix<double, 2 * dimX + 1, 1> weights_mean;
  Eigen::Matrix<double, 2 * dimX + 1, 1> weights_cov;
};

#endif