#include "scale_offset_estimator.h"
#include <assert.h>

namespace acmpose {

std::pair<ScaleOffset, ransac_lib::RansacStatistics> 
EstimateScaleOffset(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const ransac_lib::LORansacOptions& options, bool use_log_score) {
    ransac_lib::LORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = rand_dev();

    ScaleOffsetEstimator solver(x, y, use_log_score);

    ScaleOffset best_solution;
    ransac_lib::RansacStatistics ransac_stats;

    ransac_lib::LocallyOptimizedMSAC<ScaleOffset, std::vector<ScaleOffset>, ScaleOffsetEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

std::pair<Eigen::Vector4d, ransac_lib::RansacStatistics>
EstimateTwoViewScaleOffset(const Eigen::MatrixXd& mkpts0, const Eigen::MatrixXd& mkpts1, const Eigen::VectorXd& depth0, const Eigen::VectorXd& depth1,
                           const Eigen::MatrixXd& triangulated_p3ds, const Eigen::Matrix4d& T_0to1, const ransac_lib::LORansacOptions& options) {
    ransac_lib::LORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = rand_dev();

    TwoViewScaleOffsetEstimator solver(mkpts0, mkpts1, depth0, depth1, triangulated_p3ds, T_0to1);

    Eigen::Vector4d best_solution;
    ransac_lib::RansacStatistics ransac_stats;

    ransac_lib::LocallyOptimizedMSAC<Eigen::Vector4d, std::vector<Eigen::Vector4d>, TwoViewScaleOffsetEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

int ScaleOffsetEstimator::MinimalSolver(const std::vector<int>& sample,
                                        std::vector<ScaleOffset>* solutions) const {
    size_t sample_sz = min_sample_size();
    assert(sample_sz == 2);

    solutions->clear();

    double x0 = x_(sample[0]), x1 = x_(sample[1]);
    double y0 = y_(sample[0]), y1 = y_(sample[1]);
    if (x0 == x1) return 0;
    double a = (y1 - y0) / (x1 - x0);
    double b = y0 - (a * x0);

    solutions->emplace_back(ScaleOffset(a, b));
    return 1;
}

int ScaleOffsetEstimator::NonMinimalSolver(const std::vector<int>& sample, ScaleOffset* solution) const {
    if (sample.size() < non_minimal_sample_size()) return 0;
    
    LeastSquares(sample, solution);
    return 1;
}

// Evaluates the pose on the i-th data point.
double ScaleOffsetEstimator::EvaluateModelOnPoint(const ScaleOffset& solution, int i) const {
    Eigen::Vector2d x(x_(i), 1);
    double yy = solution.dot(x);
    if (yy <= 0 || solution(0) <= 0.1) return 10;

    if (use_log_score_) 
        return std::pow(std::log(1 + yy) - std::log(1 + y_(i)), 2);
    else 
        return std::pow(yy - y_(i), 2);
}

void ScaleOffsetEstimator::LeastSquares(const std::vector<int>& sample, ScaleOffset* solution) const {
    Eigen::MatrixXd X(sample.size(), 2);
    Eigen::VectorXd Y(sample.size());

    for (int i = 0; i < sample.size(); ++i) {
        X.row(i) << x_(sample[i]), 1;
        Y(i) = y_(sample[i]);
    }

    *solution = X.colPivHouseholderQr().solve(Y);
}

int TwoViewScaleOffsetEstimator::MinimalSolver(const std::vector<int>& sample,
                                               std::vector<Eigen::Vector4d>* solutions) const {
    size_t sample_sz = min_sample_size();
    assert(sample_sz == 2);

    solutions->clear();

    Eigen::Vector3d p3d0 = triangulated_p3ds_.row(sample[0]);
    Eigen::Vector3d p3d1 = triangulated_p3ds_.row(sample[1]);
    double p3d0_depth0 = p3d0(2);
    double p3d1_depth0 = p3d1(2);
    double p3d0_depth1 = (T_0to1_ * p3d0.homogeneous())(2);
    double p3d1_depth1 = (T_0to1_ * p3d1.homogeneous())(2);

    if (p3d0_depth0 <= 0 || p3d1_depth0 <= 0 || p3d0_depth1 <= 0 || p3d1_depth1 <= 0) return 0;

    double d00 = depth0_(sample[0]), d01 = depth0_(sample[1]);
    double d10 = depth1_(sample[0]), d11 = depth1_(sample[1]);
    if (d00 == d01 || d10 == d11) return 0;

    double a0 = (p3d0_depth0 - p3d1_depth0) / (d00 - d01);
    double b0 = p3d0_depth0 - (a0 * d00);
    double a1 = (p3d0_depth1 - p3d1_depth1) / (d10 - d11);
    double b1 = p3d0_depth1 - (a1 * d10);

    solutions->emplace_back(Eigen::Vector4d(a0, b0, a1, b1));
    return 1;
}

int TwoViewScaleOffsetEstimator::NonMinimalSolver(const std::vector<int>& sample, Eigen::Vector4d* solution) const {
    if (sample.size() < non_minimal_sample_size()) return 0;
    
    LeastSquares(sample, solution);
    return 1;
}

// Evaluates the pose on the i-th data point.
double TwoViewScaleOffsetEstimator::EvaluateModelOnPoint(const Eigen::Vector4d& solution, int i) const {
    double d0 = depth0_(i) * solution(0) + solution(1);
    double d1 = depth1_(i) * solution(2) + solution(3);
    if (d0 <= 0 || d1 <= 0 || solution(0) <= 0.1 || solution(2) <= 0.1) return 10;

    Eigen::Vector3d p0 = Eigen::Vector3d(mkpts0_.row(i).x(), mkpts0_.row(i).y(), 1) * d0;
    Eigen::Vector3d p1 = Eigen::Vector3d(mkpts1_.row(i).x(), mkpts1_.row(i).y(), 1) * d1;
    Eigen::Vector4d p0_1 = (T_0to1_ * p0.homogeneous());
    return (p1 - p0_1.head(3)).squaredNorm();
}

void TwoViewScaleOffsetEstimator::LeastSquares(const std::vector<int>& sample, Eigen::Vector4d* solution) const {
    Eigen::MatrixXd A(sample.size() * 3, 4);
    Eigen::VectorXd B(sample.size() * 3);
    Eigen::Matrix3d R = T_0to1_.block<3, 3>(0, 0);
    Eigen::Vector3d t = T_0to1_.block<3, 1>(0, 3);
    for (int i = 0; i < sample.size(); i++) {
        Eigen::Vector3d x0 = mkpts0_.row(sample[i]);
        Eigen::Vector3d x1 = mkpts1_.row(sample[i]);
        A.block<3, 4>(i * 3, 0) << (R * x0) * depth0_(sample[i]), R * x0, -x1 * depth1_(sample[i]), -x1;
        B.segment<3>(i * 3) = -t;
    }

    // Solve Ax = B using least squares
    *solution = A.colPivHouseholderQr().solve(B);
}

Eigen::Vector4d TwoViewScaleOffsetLSQ(const Eigen::MatrixXd& mkpts0, const Eigen::MatrixXd& mkpts1, const Eigen::VectorXd& depth0, const Eigen::VectorXd& depth1, const Eigen::Matrix4d& T_0to1) {
    assert(mkpts0.cols() == 2);
    assert(mkpts1.cols() == 2);
    assert(mkpts0.rows() == mkpts1.rows());
    assert(depth0.rows() == mkpts0.rows());
    assert(depth1.rows() == mkpts1.rows());

    Eigen::MatrixXd A(mkpts0.rows() * 3, 4);
    Eigen::VectorXd B(mkpts0.rows() * 3);
    Eigen::Matrix3d R = T_0to1.block<3, 3>(0, 0);
    Eigen::Vector3d t = T_0to1.block<3, 1>(0, 3);
    for (int i = 0; i < mkpts0.rows(); i++) {
        Eigen::Vector3d x0 = mkpts0.row(i);
        Eigen::Vector3d x1 = mkpts1.row(i);
        A.block<3, 4>(i * 3, 0) << (R * x0) * depth0(i), R * x0, -x1 * depth1(i), -x1;
        B.segment<3>(i * 3) = -t;
    }

    // Solve Ax = B using least squares
    return A.colPivHouseholderQr().solve(B);
}

} // namespace acmpose
