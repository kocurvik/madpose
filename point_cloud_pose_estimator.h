#ifndef POINT_CLOUD_POSE_ESTIMATOR_H
#define POINT_CLOUD_POSE_ESTIMATOR_H

#include <RansacLib/ransac.h>
#include "optimizer.h"

namespace py = pybind11;
namespace acmpose {

class PointCloudPoseEstimator {
public:
    PointCloudPoseEstimator(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                            const std::vector<double> &depth0, const std::vector<double> &depth1,
                            const Eigen::Matrix3d &K0, const Eigen::MatrixXd &K1,
                            const std::vector<double> &uncert_weight = {}) : K0_(K0), K1_(K1) {
                                assert(x0.size() == x1.size());
                                assert(depth0.size() == x0.size());
                                assert(depth1.size() == x1.size());

                                if (uncert_weight.size() == 0) {
                                    uncert_weight_ = Eigen::VectorXd::Ones(x0.size());
                                } else {
                                    uncert_weight_ = Eigen::Map<const Eigen::VectorXd>(uncert_weight.data(), uncert_weight.size());
                                }

                                Eigen::Matrix3d K0_inv_ = K0.inverse();
                                Eigen::Matrix3d K1_inv_ = K1.inverse();

                                X0_ = Eigen::MatrixXd(3, x0.size());
                                X1_ = Eigen::MatrixXd(3, x1.size());
                                x0_ = Eigen::MatrixXd(2, x0.size());
                                x1_ = Eigen::MatrixXd(2, x1.size());

                                for (size_t i = 0; i < x0.size(); i++) {
                                    Eigen::Vector3d x0_h = K0_inv_ * Eigen::Vector3d(x0[i](0), x0[i](1), 1.0);
                                    Eigen::Vector3d x1_h = K1_inv_ * Eigen::Vector3d(x1[i](0), x1[i](1), 1.0);

                                    X0_.col(i) = depth0[i] * x0_h;
                                    X1_.col(i) = depth1[i] * x1_h;
                                    x0_.col(i) = x0[i];
                                    x1_.col(i) = x1[i];
                                }
                            }

    inline int min_sample_size() const { return 3; }

    inline int non_minimal_sample_size() const { return 7 * min_sample_size() + 1; }

    inline int num_data() const { return X0_.cols(); }

    double GetWeight(int i) const { return uncert_weight_(i); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<PoseAndScale>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, PoseAndScale* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseAndScale& solution, int i) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<int>& sample, PoseAndScale* solution) const;

private:
    Eigen::Matrix3d K0_, K1_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::MatrixXd X0_, X1_;
    Eigen::VectorXd uncert_weight_;
};

class PointCloudPoseEstimatorWithOffset {
public:
    PointCloudPoseEstimatorWithOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                                      const std::vector<double> &depth0, const std::vector<double> &depth1,
                                      const Eigen::Vector2d &min_depth, 
                                      const Eigen::Matrix3d &K0, const Eigen::MatrixXd &K1,
                                      const std::vector<double> &uncert_weight = {}) : 
                                      K0_(K0), K1_(K1), min_depth_(min_depth) 
                                      {
                                assert(x0.size() == x1.size());
                                assert(depth0.size() == x0.size());
                                assert(depth1.size() == x1.size());

                                if (uncert_weight.size() == 0) {
                                    uncert_weight_ = Eigen::VectorXd::Ones(x0.size());
                                } else {
                                    uncert_weight_ = Eigen::Map<const Eigen::VectorXd>(uncert_weight.data(), uncert_weight.size());
                                }

                                Eigen::Matrix3d K0_inv_ = K0.inverse();
                                Eigen::Matrix3d K1_inv_ = K1.inverse();

                                d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
                                d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

                                X0_ = Eigen::MatrixXd(3, x0.size());
                                X1_ = Eigen::MatrixXd(3, x1.size());

                                for (size_t i = 0; i < x0.size(); i++) {
                                    Eigen::Vector3d x0_h = K0_inv_ * Eigen::Vector3d(x0[i](0), x0[i](1), 1.0);
                                    Eigen::Vector3d x1_h = K1_inv_ * Eigen::Vector3d(x1[i](0), x1[i](1), 1.0);
                                    X0_.col(i) = x0_h;
                                    X1_.col(i) = x1_h;
                                }
                            }

    inline int min_sample_size() const { return 3; }

    inline int non_minimal_sample_size() const { return 5 * min_sample_size() + 1; }

    inline int num_data() const { return X0_.cols(); }

    double GetWeight(int i) const { return uncert_weight_(i); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<PoseScaleOffset>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, PoseScaleOffset* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset& solution, int i) const;

    // Linear least squares solver.
    void LeastSquares(const std::vector<int>& sample, PoseScaleOffset* solution) const;

private:
    Eigen::Matrix3d K0_, K1_;
    Eigen::MatrixXd X0_, X1_;
    Eigen::VectorXd d0_, d1_;
    Eigen::VectorXd uncert_weight_;
    Eigen::Vector2d min_depth_;
};

class PointCloudPoseScaleOffsetOptimizer {
protected:
    Eigen::MatrixXd x_homo_, y_homo_;
    Eigen::VectorXd depth_x_, depth_y_;
    double a_;
    double b_, c_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;

    OptimizerConfig config_;

    void AddResiduals();

public:
    PointCloudPoseScaleOffsetOptimizer(
        const Eigen::MatrixXd &x_homo, const Eigen::MatrixXd &y_homo,
        const Eigen::VectorXd &depth_x, const Eigen::VectorXd &depth_y, 
        const Eigen::Matrix3d &R, const Eigen::Vector3d &t, 
        const double &scale, const double &offset0, const double &offset1,
        const OptimizerConfig &config = OptimizerConfig()) :
        x_homo_(x_homo), y_homo_(y_homo), depth_x_(depth_x), depth_y_(depth_y), 
        a_(scale), b_(offset0), c_(offset1), config_(config) {
        assert(x_homo.cols() == y_homo.cols());
        assert(x_homo.cols() == depth_x.size());
        assert(y_homo.cols() == depth_y.size());

        this->qvec_ = RotationMatrixToQuaternion<double>(R);
        this->tvec_ = t;
    }
    
    void SetUp();
    bool Solve();

    double GetScale() const { return a_; }
    std::pair<double, double> GetOffsets() const { return std::make_pair(b_, c_); }
    Eigen::Matrix4d GetTransform() const { return ComposeTransformationMatrix(qvec_, tvec_); }
    Eigen::Matrix3d GetRotation() const { return QuaternionToRotationMatrix(qvec_); }
    Eigen::Vector4d GetRotationQuaternion() { return qvec_; }
    Eigen::Vector3d GetTranslation() const { return tvec_; }

    double GetInitialCost() const { return std::sqrt(summary_.initial_cost / summary_.num_residuals_reduced); }
    double GetFinalCost() const { return std::sqrt(summary_.final_cost / summary_.num_residuals_reduced); }
    bool IsSolutionUsable() const { return summary_.IsSolutionUsable(); }

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
};

struct P3DDistFunctor {
public:
    P3DDistFunctor(const Eigen::Vector3d &x2d_normalized, const Eigen::Vector3d &y2d_normalized,
                   const double& x_depth, const double& y_depth) : 
        x2d_normalized_(x2d_normalized), y2d_normalized_(y2d_normalized), x_depth_(x_depth), y_depth_(y_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x2d_normalized, const Eigen::Vector3d &y2d_normalized,
                                       const double& x_depth, const double& y_depth) {
        return (new ceres::AutoDiffCostFunction<P3DDistFunctor, 3, 1, 1, 1, 4, 3>(
            new P3DDistFunctor(x2d_normalized, y2d_normalized, x_depth, y_depth)));
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c,
                    const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Vector<T, 3> x3d = (x_depth_ * a[0] + b[0]) * x2d_normalized_;
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_y = R * x3d + t;
        Eigen::Vector<T, 3> y3d = (y_depth_ + c[0]) * y2d_normalized_;

        residuals[0] = x3d_y[0] - y3d[0];
        residuals[1] = x3d_y[1] - y3d[1];
        residuals[2] = x3d_y[2] - y3d[2];
        return true;
    }
private:
    const Eigen::Vector3d x2d_normalized_, y2d_normalized_;
    const double x_depth_, y_depth_;
};

std::pair<PoseAndScale, ransac_lib::RansacStatistics> 
EstimatePointCloudPose(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                       const std::vector<double> &depth0, const std::vector<double> &depth1,
                       const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                       const ransac_lib::LORansacOptions& options,
                       const std::vector<double> &uncert_weight = {});

std::pair<PoseScaleOffset, ransac_lib::RansacStatistics>
EstimatePointCloudPoseWithOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                                 const std::vector<double> &depth0, const std::vector<double> &depth1,
                                 const Eigen::Vector2d &min_depth, 
                                 const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                                 const ransac_lib::LORansacOptions& options,
                                 const std::vector<double> &uncert_weight = {});

} // namespace acmpose

#endif

