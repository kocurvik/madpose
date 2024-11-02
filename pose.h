#ifndef POSE_H
#define POSE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace acmpose {

struct PoseAndScale {
    Eigen::Matrix<double, 3, 4> pose;
    double scale;

    PoseAndScale() : pose(Eigen::Matrix<double, 3, 4>::Zero()), scale(1.0) {}
    PoseAndScale(const Eigen::Matrix<double, 3, 4> &pose, double scale) : pose(pose), scale(scale) {}
    PoseAndScale(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double scale) : scale(scale) {
        pose.block<3, 3>(0, 0) = R;
        pose.block<3, 1>(0, 3) = t;
    }

    Eigen::Matrix3d R() const { return pose.block<3, 3>(0, 0); }
    Eigen::Vector3d t() const { return pose.block<3, 1>(0, 3); }
};

struct PoseScaleOffset : public PoseAndScale{
    double offset0, offset1;

    PoseScaleOffset() : PoseAndScale(), offset0(0), offset1(0) {}
    PoseScaleOffset(const Eigen::Matrix<double, 3, 4> &pose, double scale, double b, double c) : 
        PoseAndScale(pose, scale), offset0(b), offset1(c) {}
    PoseScaleOffset(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double scale, double b, double c) : 
        PoseAndScale(R, t, scale), offset0(b), offset1(c) {}

    Eigen::Matrix3d R() const { return pose.block<3, 3>(0, 0); }
    Eigen::Vector3d t() const { return pose.block<3, 1>(0, 3); }
};

struct PoseScaleOffsetSharedFocal : public PoseScaleOffset {
    double focal;

    PoseScaleOffsetSharedFocal() : PoseScaleOffset(), focal(1.0) {}
    PoseScaleOffsetSharedFocal(const Eigen::Matrix<double, 3, 4> &pose, double scale, double b, double c, double f) : 
        PoseScaleOffset(pose, scale, b, c), focal(f) {}
    PoseScaleOffsetSharedFocal(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double scale, double b, double c, double f) : 
        PoseScaleOffset(R, t, scale, b, c), focal(f) {}
};

struct PoseScaleOffsetTwoFocal : public PoseScaleOffset {
    double focal0, focal1;

    PoseScaleOffsetTwoFocal() : PoseScaleOffset(), focal0(1.0), focal1(1.0) {}
    PoseScaleOffsetTwoFocal(const Eigen::Matrix<double, 3, 4> &pose, double scale, double b, double c, double f0, double f1) : 
        PoseScaleOffset(pose, scale, b, c), focal0(f0), focal1(f1) {}
    PoseScaleOffsetTwoFocal(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double scale, double b, double c, double f0, double f1) : 
        PoseScaleOffset(R, t, scale, b, c), focal0(f0), focal1(f1) {}
};

} // namespace acmpose

#endif // POSE_H