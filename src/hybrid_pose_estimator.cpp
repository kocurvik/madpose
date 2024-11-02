#include <colmap/geometry/triangulation.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/solvers/relpose_5pt.h>
#include "solver.h"
#include "hybrid_pose_estimator.h"

namespace acmpose {

std::pair<PoseScaleOffset, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                              const std::vector<double> &depth0, const std::vector<double> &depth1, 
                              const Eigen::Vector2d &min_depth, 
                              const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                              const ExtendedHybridLORansacOptions& options,
                              const std::vector<double> &uncert_weights) {
    ExtendedHybridLORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = 0;

    ransac_options.data_type_weights_[1] *= 2 * ransac_options.squared_inlier_thresholds_[0] / ransac_options.squared_inlier_thresholds_[1];
    double sampson_squared_weight = ransac_options.data_type_weights_[1];

    // ransac_options.data_type_weights_[1] *= 0.5;
    // HybridPoseEstimator solver(x0, x1, depth0, depth1, min_depth, K0, K1, sampson_squared_weight, ransac_options.squared_inlier_thresholds_, uncert_weights);
    
    ransac_options.data_type_weights_.push_back(ransac_options.data_type_weights_[1]);
    // ransac_options.data_type_weights_.push_back(ransac_options.squared_inlier_thresholds_[0] / 1e-2);
    ransac_options.squared_inlier_thresholds_.push_back(ransac_options.squared_inlier_thresholds_[1]);
    // ransac_options.squared_inlier_thresholds_.push_back(1e-2);
    ransac_options.data_type_weights_[1] = ransac_options.data_type_weights_[0];
    ransac_options.squared_inlier_thresholds_[1] = ransac_options.squared_inlier_thresholds_[0];
    HybridPoseEstimator3 solver(x0, x1, depth0, depth1, min_depth, K0, K1, sampson_squared_weight, ransac_options.squared_inlier_thresholds_, uncert_weights);

    PoseScaleOffset best_solution;
    ransac_lib::HybridRansacStatistics ransac_stats;

    // HybridUncertaintyLOMSAC<PoseScaleOffset, std::vector<PoseScaleOffset>, HybridPoseEstimator> lomsac;
    HybridUncertaintyLOMSAC<PoseScaleOffset, std::vector<PoseScaleOffset>, HybridPoseEstimator3> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

int HybridPoseEstimator::MinimalSolver(const std::vector<std::vector<int>>& sample,
                                       const int solver_idx, std::vector<PoseScaleOffset>* models) const {
    models->clear();
    if (solver_idx == 0) {
        Eigen::Matrix3d x0 = K0_inv_ * x0_(Eigen::all, sample[0]);
        Eigen::Matrix3d x1 = K1_inv_ * x1_(Eigen::all, sample[0]);

        std::vector<PoseScaleOffset> sols;
        int num_sols = estimate_scale_shift_pose(x0, x1, d0_(sample[0]), d1_(sample[0]), &sols, false);
        for (int i = 0; i < num_sols; i++) {
            if (sols[i].offset0 > -min_depth_(0) && sols[i].offset1 > -min_depth_(1) * sols[i].scale) {
                PoseScaleOffset sol = sols[i];
                sol.offset1 /= sol.scale;
                models->push_back(sol);
            }
        }
    }
    else if (solver_idx == 1) {
        Eigen::MatrixXd x0 = K0_inv_ * x0_(Eigen::all, sample[1]);
        Eigen::MatrixXd x1 = K1_inv_ * x1_(Eigen::all, sample[1]);

        std::vector<Eigen::Vector3d> x0_vec(sample[1].size()), x1_vec(sample[1].size());
        std::vector<Eigen::Vector2d> x0_2dvec(sample[1].size()), x1_2dvec(sample[1].size());
        for (int i = 0; i < sample[1].size(); i++) {
            x0_vec[i] = x0.col(i).normalized();
            x1_vec[i] = x1.col(i).normalized();
            x0_2dvec[i] = x0.col(i).head<2>();
            x1_2dvec[i] = x1.col(i).head<2>();
        }
        std::vector<poselib::CameraPose> poses;
        poselib::relpose_5pt(x0_vec, x1_vec, &poses);

        for (auto &pose : poses) {
            Eigen::Matrix3x4d proj_matrix0 = Eigen::Matrix3x4d::Identity();
            Eigen::Matrix3x4d proj_matrix1 = pose.Rt();
            
            std::vector<Eigen::Vector3d> p3d_vec = colmap::TriangulatePoints(proj_matrix0, proj_matrix1, x0_2dvec, x1_2dvec);
            Eigen::MatrixXd p3d(3, p3d_vec.size());
            for (int i = 0; i < p3d_vec.size(); i++) {
                p3d.col(i) = p3d_vec[i];
            }  

            // Estimate scale and shift by least-squares
            Eigen::MatrixXd A(sample[1].size(), 2);
            A.col(0) = d0_(sample[1]);
            A.col(1) = Eigen::VectorXd::Ones(sample[1].size());
            Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
            double s0 = x(0);
            double offset0 = x(1) / s0;
            if (offset0 < -min_depth_(0)) continue;

            p3d = p3d / s0;
            pose.t = pose.t / s0;
            p3d = (pose.R() * p3d).colwise() + pose.t;
            A.col(0) = d1_(sample[1]);
            A.col(1) = Eigen::VectorXd::Ones(sample[1].size());
            x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
            double scale = x(0);
            double offset1 = x(1) / scale;
            if (offset1 < -min_depth_(1)) continue;

            PoseScaleOffset sol(pose.R(), pose.t, scale, offset0, offset1);

            models->push_back(sol);
        }
    }
    return models->size();
}

int HybridPoseEstimator::NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* solution) const {
    OptimizerConfig config;
    config.use_geometric = false;
    config.use_sampson = true;
    config.use_reprojection = true;
    config.solver_options.max_num_iterations = 25;
    config.weight_sampson = sampson_squared_weight_;
    if (sample[0].size() < 3 && sample[1].size() < 5) {
        return 0;
    }
    PoseScaleOffsetOptimizer optim(x0_, x1_, d0_, d1_, sample[0], sample[1], min_depth_(0), min_depth_(1), 
        *solution, K0_, K1_, uncert_weight_, config);
    optim.SetUp();
    if (!optim.Solve()) return 0;
    *solution = optim.GetSolution();
    return 1;
}

double HybridPoseEstimator::EvaluateModelOnPoint(const PoseScaleOffset& model, int t, int i, double squared_thres) const {
    if (t == 0) {
        Eigen::Vector3d p3d0 = (K0_inv_ * x0_.col(i)) * (d0_(i) + model.offset0);
        Eigen::Vector3d p3d0_1 = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K1_ * p3d0_1;
        if (p2d_project(2) < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double r0 = (p2d - x1_.col(i).head<2>()).squaredNorm();
        
        Eigen::Vector3d p3d1 = (K1_inv_ * x1_.col(i)) * (d1_(i) + model.offset1) * model.scale;
        Eigen::Vector3d p3d1_0 = (model.R().transpose() * p3d1 - model.R().transpose() * model.t());
        p2d_project = K0_ * p3d1_0;
        if (p2d_project(2) < 1e-2) 
            return std::numeric_limits<double>::max();
        p2d = p2d_project.head<2>() / p2d_project(2);
        double r1 = (p2d - x0_.col(i).head<2>()).squaredNorm();

        return std::min(r0, r1);
    }
    else if (t == 1) {
        poselib::CameraPose pose(model.R(), model.t());
        Eigen::Vector3d x0_calib = K0_inv_ * x0_.col(i);
        Eigen::Vector3d x1_calib = K1_inv_ * x1_.col(i);
        bool cheirality = poselib::check_cheirality(pose, x0_calib.normalized(), x1_calib.normalized(), 1e-2);
        if (!cheirality) {
            return std::numeric_limits<double>::max();
        }

        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());
        // F = K1_inv_.transpose() * E * K0_inv_;

        double sampson_error = compute_sampson_error(x0_calib.head<2>(), x1_calib.head<2>(), E);
        double loss_scale = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        return sampson_error / std::pow(loss_scale, 2);
    }
}

// Linear least squares solver. 
void HybridPoseEstimator::LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model, bool final) const {
    if (sample[0].size() < 3 && sample[1].size() < 5) {
        return;
    }

    OptimizerConfig config;
    config.use_geometric = false;
    config.weight_geometric = 1.0;

    config.use_reprojection = true;
    config.use_sampson = true;
    config.solver_options.max_num_iterations = 25;
    config.weight_sampson = sampson_squared_weight_;
    config.reproj_loss_function.reset(new ceres::CauchyLoss(1.0));
    config.sampson_loss_function.reset(new ceres::CauchyLoss(1.0));
    config.geom_loss_function.reset(new ceres::CauchyLoss(0.1));
    PoseScaleOffsetOptimizer optim(x0_, x1_, d0_, d1_, sample[0], sample[1], min_depth_(0), min_depth_(1), 
        *model, K0_, K1_, uncert_weight_, config);
    optim.SetUp();
    if (!optim.Solve()) return;
    *model = optim.GetSolution();
}

int HybridPoseEstimator3::NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* solution) const {
    double sampson_loss_scale = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
    
    OptimizerConfig config;
    config.use_geometric = false;
    config.solver_options.max_num_iterations = 25;

    if (solver_idx == 1) {
        config.use_sampson = true;
        config.use_reprojection = false;
        PoseScaleOffsetOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
            *solution, K0_, K1_, uncert_weight_, config);
        optim.SetUp();
        if (!optim.Solve()) return 0;
        *solution = optim.GetSolution();

        config.use_sampson = false;
        config.use_reprojection = true;
        config.constant_pose = true;
        PoseScaleOffsetOptimizer3 optim_scale(x0_, x1_, d0_, d1_, sample[2], sample[2], sample[2], min_depth_, 
            *solution, K0_, K1_, uncert_weight_, config);
        optim_scale.SetUp();
        if (!optim_scale.Solve()) return 0;
        *solution = optim_scale.GetSolution();
    }
    else {
        config.use_sampson = true;
        config.use_reprojection = true;
        config.weight_sampson = sampson_squared_weight_;
        PoseScaleOffsetOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
            *solution, K0_, K1_, uncert_weight_, config);
        optim.SetUp();
        if (!optim.Solve()) return 0;
        *solution = optim.GetSolution();
    }
    return 1;
}

double HybridPoseEstimator3::EvaluateModelOnPoint(const PoseScaleOffset& model, int t, int i, double squared_thres, bool gradcut) const {
    if (t == 0) {
        Eigen::Vector3d p3d0 = (K0_inv_ * x0_.col(i)) * (d0_(i) + model.offset0);
        Eigen::Vector3d q = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K1_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x1 = x1_.col(i).head<2>();
        double reproj_error = (p2d - x1).squaredNorm();

        return reproj_error;
    }
    else if (t == 1) {   
        Eigen::Vector3d p3d1 = (K1_inv_ * x1_.col(i)) * (d1_(i) + model.offset1) * model.scale;
        Eigen::Vector3d q = model.R().transpose() * p3d1 - model.R().transpose() * model.t();
        Eigen::Vector3d p2d_project = K0_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x0 = x0_.col(i).head<2>();
        double reproj_error = (p2d - x0).squaredNorm();

        return reproj_error;
    }
    else if (t == 2) {
        poselib::CameraPose pose(model.R(), model.t());
        Eigen::Vector3d x0_calib = K0_inv_ * x0_.col(i);
        Eigen::Vector3d x1_calib = K1_inv_ * x1_.col(i);
        bool cheirality = poselib::check_cheirality(pose, x0_calib.normalized(), x1_calib.normalized(), 1e-2);
        if (!cheirality) {
            return std::numeric_limits<double>::max();
        }

        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());

        double sampson_error = compute_sampson_error(x0_calib.head<2>(), x1_calib.head<2>(), E);
        double loss_scale = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        return sampson_error / std::pow(loss_scale, 2);
    }
}

// Linear least squares solver. 
void HybridPoseEstimator3::LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model, bool final) const {
    OptimizerConfig config;
    config.use_geometric = false;

    config.reproj_loss_function.reset(new ceres::CauchyLoss(1.0));
    config.sampson_loss_function.reset(new ceres::CauchyLoss(1.0));

    if (sample[0].size() < 3 || sample[1].size() < 3 || sample[2].size() < 5) {
        return;
    }

    config.use_sampson = true;
    config.use_reprojection = true;
    config.weight_sampson = sampson_squared_weight_;
    PoseScaleOffsetOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
        *model, K0_, K1_, uncert_weight_, config);
    optim.SetUp();
    if (!optim.Solve()) return;
    *model = optim.GetSolution();
}

} // namespace acmpose