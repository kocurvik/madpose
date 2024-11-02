#include <PoseLib/poselib.h>
#include "point_cloud_pose_estimator.h"
#include "pose_scale_shift_estimator.h"
#include "uncertainty_ransac.h"
#include "solver.h"

namespace acmpose {

std::random_device rand_dev;
std::mt19937 rng(rand_dev());
std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

std::pair<PoseAndScale, ransac_lib::RansacStatistics> 
EstimatePointCloudPose(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                       const std::vector<double> &depth0, const std::vector<double> &depth1,
                       const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                       const ransac_lib::LORansacOptions& options,
                       const std::vector<double> &uncert_weight) {
    ransac_lib::LORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = rand_dev();

    PointCloudPoseEstimator solver(x0, x1, depth0, depth1, K0, K1, uncert_weight);

    PoseAndScale best_solution;
    ransac_lib::RansacStatistics ransac_stats;

    UncertaintyLOMSAC<PoseAndScale, std::vector<PoseAndScale>, PointCloudPoseEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

std::pair<PoseScaleOffset, ransac_lib::RansacStatistics>
EstimatePointCloudPoseWithOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                                 const std::vector<double> &depth0, const std::vector<double> &depth1,
                                 const Eigen::Vector2d &min_depth, 
                                 const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                                 const ransac_lib::LORansacOptions& options,
                                 const std::vector<double> &uncert_weight) {
    ransac_lib::LORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = rand_dev();

    PointCloudPoseEstimatorWithOffset solver(x0, x1, depth0, depth1, min_depth, 
        K0, K1, uncert_weight);

    PoseScaleOffset best_solution;
    ransac_lib::RansacStatistics ransac_stats;

    UncertaintyLOMSAC<PoseScaleOffset, std::vector<PoseScaleOffset>, PointCloudPoseEstimatorWithOffset> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

int estimate_scale_and_pose_p3p(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y, const Eigen::Matrix3d Ky, std::vector<PoseAndScale> *output) {
    std::vector<Eigen::Vector3d> y_homo(y.cols());
    std::vector<Eigen::Vector3d> X_vec(X.cols());
    for (size_t i = 0; i < y.cols(); i++) {
        y_homo[i] = Ky.inverse() * Eigen::Vector3d(y(0, i), y(1, i), 1.0);
        X_vec[i] = X.col(i);
    }

    std::vector<poselib::CameraPose> output_cam_poses;
    int ret = poselib::p3p(y_homo, X_vec, &output_cam_poses);

    output->clear();
    for (size_t i = 0; i < output_cam_poses.size(); i++) {
        Eigen::Matrix3d R = output_cam_poses[i].R();
        Eigen::Vector3d t = output_cam_poses[i].t;
        double scale = 1.0;
        output->push_back(PoseAndScale(R, t, scale));
    }
    return ret;
}

PoseAndScale estimate_scale_and_pose(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::VectorXd &W) {
    // X: 3 x N
    // Y: 3 x N
    // W: N x 1

    // 1. Compute the weighted centroid of X and Y
    Eigen::Vector3d centroid_X = X * W / W.sum();
    Eigen::Vector3d centroid_Y = Y * W / W.sum();

    Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
    Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

    Eigen::Matrix3d S = Y_centered * W.asDiagonal() * X_centered.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        U.col(2) *= -1;
    }
    Eigen::Matrix3d R = U * V.transpose();

    Eigen::MatrixXd X_rotated = R * X_centered;
    double scale = Y_centered.cwiseProduct(X_rotated).sum() / X_rotated.cwiseProduct(X_rotated).sum();

    Eigen::Vector3d t = centroid_Y - scale * R * centroid_X;
    return PoseAndScale(R, t, scale);
}

int PointCloudPoseEstimator::MinimalSolver(const std::vector<int>& sample,
                                           std::vector<PoseAndScale>* solutions) const {
    size_t sample_sz = min_sample_size();
    assert(sample_sz == sample.size());

    Eigen::MatrixXd X0 = X0_(Eigen::all, sample);
    Eigen::MatrixXd X1 = X1_(Eigen::all, sample);

    solutions->clear();
    solutions->push_back(estimate_scale_and_pose(X0, X1, Eigen::VectorXd::Ones(sample_sz)));
    return 1;

    // int ret = estimate_scale_and_pose_p3p(X0, x1_(Eigen::all, sample), K1_, solutions);
    // Eigen::Vector3d centroid_X = X0.rowwise().mean();
    // Eigen::Vector3d centroid_Y = X1.rowwise().mean();

    // Eigen::MatrixXd X_centered = X0.colwise() - centroid_X;
    // Eigen::MatrixXd Y_centered = X1.colwise() - centroid_Y;
    // for (auto &solution : *solutions) {
    //     Eigen::Matrix3d R = solution.R();
    //     Eigen::MatrixXd X_rotated = R * X_centered;
    //     double scale = Y_centered.cwiseProduct(X_rotated).sum() / X_rotated.cwiseProduct(X_rotated).sum();
    //     solution.scale = scale;
    // }
    // return ret;
}

int PointCloudPoseEstimatorWithOffset::MinimalSolver(const std::vector<int>& sample,
                                                     std::vector<PoseScaleOffset>* solutions) const {
    size_t sample_sz = min_sample_size();
    assert(sample_sz == sample.size());

    Eigen::Matrix3d x0_homo = X0_(Eigen::all, sample);
    Eigen::Matrix3d x1_homo = X1_(Eigen::all, sample);

    return estimate_scale_shift_pose(x0_homo, x1_homo, d0_(sample), d1_(sample), solutions, false);
}

int PointCloudPoseEstimator::NonMinimalSolver(const std::vector<int>& sample, PoseAndScale* solution) const {
    size_t sample_sz = sample.size();

    Eigen::MatrixXd X0 = X0_(Eigen::all, sample);
    Eigen::MatrixXd X1 = X1_(Eigen::all, sample);

    *solution = estimate_scale_and_pose(X0, X1, Eigen::VectorXd::Ones(sample_sz));
    return 1;
}

int PointCloudPoseEstimatorWithOffset::NonMinimalSolver(const std::vector<int>& sample, PoseScaleOffset *solution) const {
    OptimizerConfig config;
    config.use_geometric = true;
    config.use_reprojection = false;
    Eigen::Matrix3d ident = Eigen::Matrix3d::Identity();
    PoseScaleOffsetOptimizer optim(X0_, X1_, d0_, d1_, sample, {}, min_depth_(0), min_depth_(1), 
        *solution, ident, ident, uncert_weight_, config);
    optim.SetUp();
    if (!optim.Solve()) return 0;
    *solution = optim.GetSolution();

    return 1;
    // std::vector<bool> v(sample.size());
    // std::fill(v.begin(), v.begin() + 3, true);

    // double min_sq_error = std::numeric_limits<double>::max();
    // do {
    //     std::vector<int> min_sample;
    //     for (int i = 0; i < sample.size(); i++) {
    //         if (v[i]) min_sample.push_back(sample[i]);
    //     }

    //     Eigen::Matrix3d x0 = X0_(Eigen::all, min_sample);
    //     Eigen::Matrix3d x1 = X1_(Eigen::all, min_sample);

    //     std::vector<Eigen::Vector4d> ss = solve_scale_and_shift(x0, x1, d0_(min_sample), d1_(min_sample));

    //     for (auto &s : ss) {
    //         double offset0 = s[1], a1 = s[2], offset1 = s[3] / s[2];
    //         if (offset0 <= -min_depth_(0) || offset1 <= -min_depth_(1)) continue;

    //         Eigen::MatrixXd x_homo = X0_(Eigen::all, sample);
    //         Eigen::MatrixXd y_homo = X1_(Eigen::all, sample);
    //         Eigen::VectorXd dx = d0_(sample).array() + offset0;
    //         Eigen::VectorXd dy = (d1_(sample).array() + offset1) * a1;
            
    //         Eigen::MatrixXd X = x_homo.array().rowwise() * dx.transpose().array();
    //         Eigen::MatrixXd Y = y_homo.array().rowwise() * dy.transpose().array();

    //         Eigen::Vector3d centroid_X = X.rowwise().mean();
    //         Eigen::Vector3d centroid_Y = Y.rowwise().mean();

    //         Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
    //         Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

    //         Eigen::Matrix3d S = Y_centered * X_centered.transpose();

    //         Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //         Eigen::Matrix3d U = svd.matrixU();
    //         Eigen::Matrix3d V = svd.matrixV();

    //         if (U.determinant() * V.determinant() < 0) {
    //             U.col(2) *= -1;
    //         }
    //         Eigen::Matrix3d R = U * V.transpose();
            
    //         Eigen::MatrixXd X_rotated = R * X_centered;
    //         double scale = X_rotated.cwiseProduct(X_rotated).sum() / Y_centered.cwiseProduct(X_rotated).sum();
    //         Eigen::Vector3d t = scale * centroid_Y - R * centroid_X;

    //         Eigen::MatrixXd diff = R * X + t - scale * Y;
    //         double err = diff.colwise().squaredNorm().mean();
    //         if (err < min_sq_error) {
    //             min_sq_error = err;
    //             solution->R() = R;
    //             solution->t() = t;
    //             solution->scale *= scale;
    //         }
    //     }
    // } while (std::prev_permutation(v.begin(), v.end()));

    // return min_sq_error < std::numeric_limits<double>::max();
}

// Evaluates the pose on the i-th data point.
double PointCloudPoseEstimator::EvaluateModelOnPoint(const PoseAndScale& solution, int i) const {
    const Eigen::Vector3d &X0 = X0_.col(i);
    const Eigen::Vector3d &X1 = X1_.col(i);
    const Eigen::Vector2d &x0 = x0_.col(i);
    const Eigen::Vector2d &x1 = x1_.col(i);

    Eigen::Vector3d X1_hat = solution.scale * solution.R() * X0 + solution.t();
    Eigen::Vector3d x1_hat = K1_ * X1_hat;
    double squared_error = (X1 - X1_hat).squaredNorm();
    if (X1_hat(2) <= 0) {
        squared_error = 1e10;
    }

    return squared_error;
}

double PointCloudPoseEstimatorWithOffset::EvaluateModelOnPoint(const PoseScaleOffset& solution, int i) const {
    Eigen::Vector3d X0 = X0_.col(i);
    Eigen::Vector3d X1 = X1_.col(i);

    X0 *= d0_(i) + solution.offset0;
    X1 *= (d1_(i) + solution.offset1) * solution.scale;

    Eigen::Vector3d X1_hat = solution.R() * X0 + solution.t();
    double squared_error = (X1 - X1_hat).squaredNorm();
    if (X1_hat(2) <= 0) {
        squared_error = 1e10;
    }

    return squared_error;
}

void PointCloudPoseEstimator::LeastSquares(const std::vector<int>& sample, PoseAndScale* solution) const {
    size_t sample_sz = sample.size();

    Eigen::MatrixXd X0 = X0_(Eigen::all, sample);
    Eigen::MatrixXd X1 = X1_(Eigen::all, sample);

    *solution = estimate_scale_and_pose(X0, X1, uncert_weight_(sample));
}

void PointCloudPoseEstimatorWithOffset::LeastSquares(const std::vector<int>& sample, PoseScaleOffset* solution) const {
    // size_t sample_sz = sample.size();

    // Eigen::MatrixXd X0 = X0_(Eigen::all, sample);
    // Eigen::MatrixXd X1 = X1_(Eigen::all, sample);

    // Eigen::VectorXd d0 = d0_(sample).array() * solution->scale + solution->offset0;
    // Eigen::VectorXd d1 = d1_(sample).array() + solution->offset1;

    // X0 = X0.array().rowwise() * d0.transpose().array();
    // X1 = X1.array().rowwise() * d1.transpose().array();

    // PoseAndScale ps = estimate_scale_and_pose(X0, X1, uncert_weight_(sample));
    // solution->R() = ps.R();
    // solution->t() = ps.t();
    // solution->scale *= ps.scale;
    // solution->offset0 *= ps.scale;

    OptimizerConfig config;
    config.use_geometric = true;
    config.use_reprojection = false;
    config.geom_loss_function.reset(new ceres::CauchyLoss(0.1));
    Eigen::Matrix3d ident = Eigen::Matrix3d::Identity();
    PoseScaleOffsetOptimizer optim(X0_, X1_, d0_, d1_, sample, {}, min_depth_(0), min_depth_(1), 
        *solution, ident, ident, uncert_weight_, config);
    optim.SetUp();
    if (!optim.Solve()) return;
    *solution = optim.GetSolution();
}

void PointCloudPoseScaleOffsetOptimizer::AddResiduals() {
    ceres::LossFunction* loss_func = new ceres::TrivialLoss();

    int num_data = x_homo_.cols();
    for (int i = 0; i < num_data; i++) {
        const Eigen::Vector3d x_homo = x_homo_.col(i);
        const Eigen::Vector3d y_homo = y_homo_.col(i);
        const double &depth_x = depth_x_(i);
        const double &depth_y = depth_y_(i);

        ceres::CostFunction* cost_function = 
            P3DDistFunctor::Create(x_homo, y_homo, depth_x, depth_y);
        ceres::ResidualBlockId block_id = 
            problem_->AddResidualBlock(cost_function, loss_func, &a_, &b_, &c_, qvec_.data(), tvec_.data());
    }
}

void PointCloudPoseScaleOffsetOptimizer::SetUp() {
        // setup problem
    problem_.reset(new ceres::Problem(config_.problem_options));
    
    AddResiduals();

    if (problem_->HasParameterBlock(&a_)) {
        problem_->SetParameterLowerBound(&a_, 0, 1e-8);
    }

    if (problem_->HasParameterBlock(qvec_.data())) {
        if (config_.constant_pose) {
            problem_->SetParameterBlockConstant(qvec_.data());
            problem_->SetParameterBlockConstant(tvec_.data());
        }
        else {
        #ifdef CERES_PARAMETERIZATION_ENABLED
            ceres::LocalParameterization* quaternion_parameterization = 
                new ceres::QuaternionParameterization;
            problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
            ceres::LocalParameterization* homogeneous_vec_parameterization = 
                new ceres::HomogeneousVectorParameterization(3);
            problem_->SetParameterization(tvec_.data(), homogeneous_vec_parameterization);
        #else
            ceres::Manifold* quaternion_manifold = 
                new ceres::QuaternionManifold;
            problem_->SetManifold(qvec_.data(), quaternion_manifold);
            ceres::Manifold* sphere_manifold = 
                new ceres::SphereManifold<3>;
            problem_->SetManifold(tvec_.data(), sphere_manifold);
        #endif
        }
    }

    if (config_.constant_scale) {
        if (problem_->HasParameterBlock(&a_))
            problem_->SetParameterBlockConstant(&a_);
    }
    if (config_.constant_offset) {
        if (problem_->HasParameterBlock(&b_)) {
            problem_->SetParameterBlockConstant(&b_);
            problem_->SetParameterBlockConstant(&c_);
        }
    }
}

bool PointCloudPoseScaleOffsetOptimizer::Solve() {
    if (problem_->NumResiduals() == 0)
        return false;
    ceres::Solver::Options solver_options = config_.solver_options;
    
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem_.get(), &summary_);
    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    return true;
}

}  // namespace acmpose