#ifndef SCALE_OFFSET_ESTIMATOR_H
#define SCALE_OFFSET_ESTIMATOR_H

#include <RansacLib/ransac.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace acmpose {

using ScaleOffset = Eigen::Vector2d;

class ScaleOffsetEstimator {
public:
    ScaleOffsetEstimator(const Eigen::VectorXd& x, const Eigen::VectorXd& y, bool use_log_score = false) : x_(x), y_(y), use_log_score_(use_log_score) {}

    inline int min_sample_size() const { return 2; }

    inline int non_minimal_sample_size() const { return 6; }

    inline int num_data() const { return x_.rows(); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<ScaleOffset>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, ScaleOffset* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const ScaleOffset& solution, int i) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<int>& sample, ScaleOffset* solution) const;

private:
    Eigen::VectorXd x_;
    Eigen::VectorXd y_;
    bool use_log_score_;
};

class TwoViewScaleOffsetEstimator {
public:
    TwoViewScaleOffsetEstimator(const Eigen::MatrixXd &mkpts0, const Eigen::MatrixXd &mkpts1, const Eigen::VectorXd &depth0, const Eigen::VectorXd &depth1,
                                const Eigen::MatrixXd &triangulated_p3ds, const Eigen::Matrix4d &T_0to1) : 
                                mkpts0_(mkpts0), mkpts1_(mkpts1), depth0_(depth0), depth1_(depth1), triangulated_p3ds_(triangulated_p3ds), T_0to1_(T_0to1) {
        assert(mkpts0_.cols() == 2);
        assert(mkpts1_.cols() == 2);
        assert(mkpts0_.rows() == mkpts1_.rows());
        assert(depth0_.rows() == mkpts0_.rows());
        assert(depth1_.rows() == mkpts1_.rows());
        assert(triangulated_p3ds_.cols() == mkpts0_.rows());
    }

    inline int min_sample_size() const { return 2; }

    inline int non_minimal_sample_size() const { return 13; }

    inline int num_data() const { return mkpts0_.rows(); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Vector4d>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, Eigen::Vector4d* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const Eigen::Vector4d& solution, int i) const;

    // Linear least squares solver.
    void LeastSquares(const std::vector<int>& sample, Eigen::Vector4d* solution) const;

private:
    Eigen::MatrixXd mkpts0_;
    Eigen::MatrixXd mkpts1_;
    Eigen::VectorXd depth0_;
    Eigen::VectorXd depth1_;
    Eigen::MatrixXd triangulated_p3ds_;
    Eigen::Matrix4d T_0to1_;
};

std::pair<ScaleOffset, ransac_lib::RansacStatistics> 
EstimateScaleOffset(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const ransac_lib::LORansacOptions& options, bool use_log_score = false);

std::pair<Eigen::Vector4d, ransac_lib::RansacStatistics>
EstimateTwoViewScaleOffset(const Eigen::MatrixXd& mkpts0, const Eigen::MatrixXd& mkpts1, const Eigen::VectorXd& depth0, const Eigen::VectorXd& depth1,
                           const Eigen::MatrixXd& triangulated_p3ds, const Eigen::Matrix4d& T_0to1, const ransac_lib::LORansacOptions& options);

Eigen::Vector4d 
TwoViewScaleOffsetLSQ(const Eigen::MatrixXd& mkpts0, const Eigen::MatrixXd& mkpts1, const Eigen::VectorXd& depth0, const Eigen::VectorXd& depth1, const Eigen::Matrix4d& T_0to1);

} // namespace acmpose

#endif // SCALE_OFFSET_ESTIMATOR_H

