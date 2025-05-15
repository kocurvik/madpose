#include "solver.h"

namespace madpose {

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

std::vector<Eigen::Vector4d> solve_scale_and_shift(const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
                                                   const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y) {
    // X: 3 x 3, column vectors are homogeneous of 2D points
    // Y: 3 x 3, column vectors are homogeneous of 2D points
    Eigen::Matrix3d x1 = x_homo.transpose();
    Eigen::Matrix3d x2 = y_homo.transpose();
    const Eigen::Vector3d &d1 = depth_x;
    const Eigen::Vector3d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(18);

    coeffs[0] = 2 * x2.row(0).dot(x2.row(1)) - x2.row(0).dot(x2.row(0)) - x2.row(1).dot(x2.row(1));
    coeffs[1] = x1.row(0).dot(x1.row(0)) + x1.row(1).dot(x1.row(1)) - 2 * x1.row(0).dot(x1.row(1));
    coeffs[2] = 2 * (d2[0] + d2[1]) * x2.row(0).dot(x2.row(1)) - 2 * d2[0] * x2.row(0).dot(x2.row(0)) -
                2 * d2[1] * x2.row(1).dot(x2.row(1));
    coeffs[3] = 2 * d2[0] * d2[1] * x2.row(0).dot(x2.row(1)) - d2[0] * d2[0] * x2.row(0).dot(x2.row(0)) -
                d2[1] * d2[1] * x2.row(1).dot(x2.row(1));
    coeffs[4] = 2 * d1[0] * x1.row(0).dot(x1.row(0)) + 2 * d1[1] * x1.row(1).dot(x1.row(1)) -
                2 * (d1[0] + d1[1]) * x1.row(0).dot(x1.row(1));
    coeffs[5] = d1[0] * d1[0] * x1.row(0).dot(x1.row(0)) + d1[1] * d1[1] * x1.row(1).dot(x1.row(1)) -
                2 * d1[0] * d1[1] * x1.row(0).dot(x1.row(1));
    coeffs[6] = 2 * x2.row(0).dot(x2.row(2)) - x2.row(0).dot(x2.row(0)) - x2.row(2).dot(x2.row(2));
    coeffs[7] = x1.row(0).dot(x1.row(0)) + x1.row(2).dot(x1.row(2)) - 2 * x1.row(0).dot(x1.row(2));
    coeffs[8] = 2 * (d2[0] + d2[2]) * x2.row(0).dot(x2.row(2)) - 2 * d2[0] * x2.row(0).dot(x2.row(0)) -
                2 * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[9] = 2 * d2[0] * d2[2] * x2.row(0).dot(x2.row(2)) - d2[0] * d2[0] * x2.row(0).dot(x2.row(0)) -
                d2[2] * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[10] = 2 * d1[0] * x1.row(0).dot(x1.row(0)) + 2 * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * (d1[0] + d1[2]) * x1.row(0).dot(x1.row(2));
    coeffs[11] = d1[0] * d1[0] * x1.row(0).dot(x1.row(0)) + d1[2] * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * d1[0] * d1[2] * x1.row(0).dot(x1.row(2));
    coeffs[12] = 2 * x2.row(1).dot(x2.row(2)) - x2.row(1).dot(x2.row(1)) - x2.row(2).dot(x2.row(2));
    coeffs[13] = x1.row(1).dot(x1.row(1)) + x1.row(2).dot(x1.row(2)) - 2 * x1.row(1).dot(x1.row(2));
    coeffs[14] = 2 * (d2[1] + d2[2]) * x2.row(1).dot(x2.row(2)) - 2 * d2[1] * x2.row(1).dot(x2.row(1)) -
                 2 * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[15] = 2 * d2[1] * d2[2] * x2.row(1).dot(x2.row(2)) - d2[1] * d2[1] * x2.row(1).dot(x2.row(1)) -
                 d2[2] * d2[2] * x2.row(2).dot(x2.row(2));
    coeffs[16] = 2 * d1[1] * x1.row(1).dot(x1.row(1)) + 2 * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * (d1[1] + d1[2]) * x1.row(1).dot(x1.row(2));
    coeffs[17] = d1[1] * d1[1] * x1.row(1).dot(x1.row(1)) + d1[2] * d1[2] * x1.row(2).dot(x1.row(2)) -
                 2 * d1[1] * d1[2] * x1.row(1).dot(x1.row(2));

    const std::vector<int> coeff_ind0 = {0, 6,  12, 1,  7,  13, 2,  8, 0,  6,  12, 14, 6, 0, 12, 1,  7,  13, 3,
                                         9, 2,  8,  14, 15, 4,  10, 7, 1,  16, 13, 8,  2, 6, 12, 0,  14, 9,  3,
                                         8, 14, 2,  15, 3,  9,  15, 4, 10, 16, 7,  13, 1, 5, 11, 10, 4,  17, 16};
    const std::vector<int> coeff_ind1 = {11, 17, 5, 9, 15, 3, 5, 11, 17, 10, 16, 4, 11, 5, 17};
    const std::vector<int> ind0 = {0,   1,   9,   12,  13,  21,  24,  25,  26,  28,  29,  33,  39,  42,  47,
                                   50,  52,  53,  60,  61,  62,  64,  65,  69,  72,  73,  75,  78,  81,  83,
                                   87,  90,  91,  92,  94,  95,  99,  102, 103, 104, 106, 107, 110, 112, 113,
                                   122, 124, 125, 127, 128, 130, 132, 133, 135, 138, 141, 143};
    const std::vector<int> ind1 = {7, 8, 10, 19, 20, 22, 26, 28, 29, 31, 32, 34, 39, 42, 47};
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(12, 12);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(12, 4);

    for (int k = 0; k < ind0.size(); k++) {
        int i = ind0[k] % 12;
        int j = ind0[k] / 12;
        C0(i, j) = coeffs[coeff_ind0[k]];
    }

    for (int k = 0; k < ind1.size(); k++) {
        int i = ind1[k] % 12;
        int j = ind1[k] / 12;
        C1(i, j) = coeffs[coeff_ind1[k]];
    }

    Eigen::MatrixXd C2 = C0.colPivHouseholderQr().solve(C1);
    Eigen::Matrix4d AM;
    AM << Eigen::RowVector4d(0, 0, 1, 0), -C2.row(9), -C2.row(10), -C2.row(11);

    Eigen::EigenSolver<Eigen::Matrix4d> es(AM);
    Eigen::Vector4cd D = es.eigenvalues();
    Eigen::Matrix4cd V = es.eigenvectors();

    Eigen::MatrixXcd sols = Eigen::MatrixXcd(4, 3);
    sols.col(0) = V.row(1).array() / V.row(0).array();
    sols.col(1) = D;
    sols.col(2) = V.row(3).array() / V.row(0).array();

    std::vector<Eigen::Vector4d> solutions;
    for (int i = 0; i < 4; i++) {
        if (D[i].imag() != 0)
            continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector4d sol;
        sol << 1.0, b1, a2, b2 * a2;
        solutions.push_back(sol);
    }

    return solutions;
}

std::vector<Eigen::Vector<double, 5>> solve_scale_and_shift_shared_focal(const Eigen::Matrix3x4d &x_homo,
                                                                         const Eigen::Matrix3x4d &y_homo,
                                                                         const Eigen::Vector4d &depth_x,
                                                                         const Eigen::Vector4d &depth_y) {
    Eigen::Matrix<double, 4, 3> x1 = x_homo.transpose();
    Eigen::Matrix<double, 4, 3> x2 = y_homo.transpose();

    double f1_0 = x1.block<4, 2>(0, 0).cwiseAbs().mean();
    double f2_0 = x2.block<4, 2>(0, 0).cwiseAbs().mean();
    double f0 = 0.5 * (f1_0 + f2_0);
    x1.block<4, 2>(0, 0) /= f0;
    x2.block<4, 2>(0, 0) /= f0;

    const Eigen::Vector4d &d1 = depth_x;
    const Eigen::Vector4d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(32);

    coeffs[0] = 2 * x2(0, 0) * x2(1, 0) + 2 * x2(0, 1) * x2(1, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1);
    coeffs[1] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(1, 1) - 2 * x1(0, 0) * x1(1, 0) + x1(0, 1) * x1(0, 1) +
                x1(1, 0) * x1(1, 0) + x1(1, 1) * x1(1, 1);
    coeffs[2] = 2 * d2(0) * x2(0, 0) * x2(1, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) -
                2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(1) * x2(0, 0) * x2(1, 0) +
                2 * d2(0) * x2(0, 1) * x2(1, 1) + 2 * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[3] = 2 * d2(0) * d2(1) * x2(0, 0) * x2(1, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                d2(1) * d2(1) * x2(1, 0) * x2(1, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[4] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(1) * x1(1, 0) * x1(1, 0) +
                2 * d1(1) * x1(1, 1) * x1(1, 1) - 2 * d1(0) * x1(0, 0) * x1(1, 0) - 2 * d1(1) * x1(0, 0) * x1(1, 0) -
                2 * d1(0) * x1(0, 1) * x1(1, 1) - 2 * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[5] = 2 * d2(0) * d2(1) - d2(0) * d2(0) - d2(1) * d2(1);
    coeffs[6] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) -
                2 * d1(0) * d1(1) * x1(0, 0) * x1(1, 0) - 2 * d1(0) * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[7] = d1(0) * d1(0) - 2 * d1(0) * d1(1) + d1(1) * d1(1);
    coeffs[8] = 2 * x2(0, 0) * x2(2, 0) + 2 * x2(0, 1) * x2(2, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[9] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(2, 1) - 2 * x1(0, 0) * x1(2, 0) + x1(0, 1) * x1(0, 1) +
                x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[10] = 2 * d2(0) * x2(0, 0) * x2(2, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(2, 1) +
                 2 * d2(2) * x2(0, 0) * x2(2, 0) + 2 * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[11] = 2 * d2(0) * d2(2) * x2(0, 0) * x2(2, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[12] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(0) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * x1(0, 1) * x1(2, 1) -
                 2 * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[13] = 2 * d2(0) * d2(2) - d2(0) * d2(0) - d2(2) * d2(2);
    coeffs[14] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(0) * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[15] = d1(0) * d1(0) - 2 * d1(0) * d1(2) + d1(2) * d1(2);
    coeffs[16] = 2 * x2(1, 0) * x2(2, 0) + 2 * x2(1, 1) * x2(2, 1) - x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1) -
                 x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[17] = x1(1, 0) * x1(1, 0) - 2 * x1(1, 1) * x1(2, 1) - 2 * x1(1, 0) * x1(2, 0) + x1(1, 1) * x1(1, 1) +
                 x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[18] = 2 * d2(1) * x2(1, 0) * x2(2, 0) - 2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(2) * x2(1, 0) * x2(2, 0) +
                 2 * d2(1) * x2(1, 1) * x2(2, 1) + 2 * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[19] = 2 * d2(1) * d2(2) * x2(1, 0) * x2(2, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(1) * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[20] = 2 * d1(1) * x1(1, 0) * x1(1, 0) + 2 * d1(1) * x1(1, 1) * x1(1, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(1) * x1(1, 0) * x1(2, 0) - 2 * d1(2) * x1(1, 0) * x1(2, 0) -
                 2 * d1(1) * x1(1, 1) * x1(2, 1) - 2 * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[21] = 2 * d2(1) * d2(2) - d2(1) * d2(1) - d2(2) * d2(2);
    coeffs[22] = d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(1) * d1(2) * x1(1, 0) * x1(2, 0) - 2 * d1(1) * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[23] = d1(1) * d1(1) - 2 * d1(1) * d1(2) + d1(2) * d1(2);
    coeffs[24] = 2 * x2(0, 0) * x2(3, 0) + 2 * x2(0, 1) * x2(3, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                 x2(3, 0) * x2(3, 0) - x2(3, 1) * x2(3, 1);
    coeffs[25] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(3, 1) - 2 * x1(0, 0) * x1(3, 0) + x1(0, 1) * x1(0, 1) +
                 x1(3, 0) * x1(3, 0) + x1(3, 1) * x1(3, 1);
    coeffs[26] = 2 * d2(0) * x2(0, 0) * x2(3, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(3) * x2(3, 0) * x2(3, 0) -
                 2 * d2(3) * x2(3, 1) * x2(3, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(3, 1) +
                 2 * d2(3) * x2(0, 0) * x2(3, 0) + 2 * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[27] = 2 * d2(0) * d2(3) * x2(0, 0) * x2(3, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(3) * d2(3) * x2(3, 0) * x2(3, 0) - d2(3) * d2(3) * x2(3, 1) * x2(3, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[28] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(3) * x1(3, 0) * x1(3, 0) +
                 2 * d1(3) * x1(3, 1) * x1(3, 1) - 2 * d1(0) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * x1(0, 1) * x1(3, 1) -
                 2 * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[29] = 2 * d2(0) * d2(3) - d2(0) * d2(0) - d2(3) * d2(3);
    coeffs[30] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(3) * d1(3) * x1(3, 0) * x1(3, 0) + d1(3) * d1(3) * x1(3, 1) * x1(3, 1) -
                 2 * d1(0) * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[31] = d1(0) * d1(0) - 2 * d1(0) * d1(3) + d1(3) * d1(3);

    const std::vector<int> coeff_ind0 = {
        0,  8,  16, 24, 1,  9,  17, 25, 2,  10, 0,  8,  16, 18, 24, 26, 0,  8,  16, 24, 0,  8,  16, 24, 0,  8,  16, 24,
        1,  9,  17, 25, 1,  9,  17, 25, 3,  11, 2,  10, 18, 19, 26, 27, 4,  12, 2,  1,  10, 9,  20, 18, 17, 28, 26, 25,
        1,  9,  17, 25, 2,  10, 8,  0,  16, 18, 24, 26, 2,  10, 0,  16, 18, 8,  24, 26, 8,  0,  16, 24, 5,  13, 21, 29,
        3,  11, 19, 27, 4,  3,  12, 11, 20, 9,  28, 1,  19, 17, 27, 25, 4,  12, 1,  17, 20, 9,  25, 28, 3,  11, 10, 2,
        18, 19, 26, 27, 6,  14, 4,  3,  12, 11, 2,  22, 18, 19, 20, 10, 26, 30, 28, 27, 4,  12, 9,  20, 1,  17, 25, 28,
        10, 2,  18, 0,  16, 24, 26, 8,  5,  13, 21, 29, 11, 3,  19, 27, 6,  14, 22, 12, 3,  30, 4,  19, 20, 11, 27, 28,
        6,  14, 4,  20, 22, 12, 1,  17, 25, 28, 30, 9,  6,  14, 11, 3,  19, 22, 2,  18, 26, 27, 30, 10, 6,  14, 12, 22,
        4,  20, 28, 30, 14, 6,  22, 3,  19, 27, 30, 11, 14, 6,  22, 30, 5,  13, 21, 29, 6,  22, 14, 4,  20, 28, 30, 12,
        7,  15, 23, 31, 5,  13, 21, 29, 7,  15, 23, 31, 7,  15, 5,  13, 23, 21, 31, 29};
    const std::vector<int> coeff_ind1 = {7,  23, 31, 15, 6,  22, 30, 14, 7,  23, 15, 31, 7,  15, 23,
                                         5,  31, 21, 13, 29, 15, 7,  23, 31, 7,  15, 13, 5,  21, 23,
                                         29, 31, 15, 7,  23, 5,  21, 29, 31, 13, 13, 5,  21, 29};
    const std::vector<int> ind0 = {
        0,    1,    14,   30,   36,   37,   50,   66,   72,   73,   74,   78,   80,   86,   87,   102,  111,  115,
        126,  139,  148,  153,  167,  177,  185,  190,  199,  215,  218,  222,  224,  231,  255,  259,  270,  283,
        288,  289,  290,  294,  296,  302,  303,  318,  324,  325,  327,  328,  331,  333,  338,  342,  347,  354,
        355,  357,  365,  370,  379,  395,  400,  405,  407,  412,  418,  419,  428,  429,  437,  442,  444,  449,
        451,  456,  461,  467,  481,  488,  489,  496,  504,  505,  518,  534,  542,  546,  548,  555,  578,  579,
        582,  583,  584,  587,  591,  592,  594,  598,  607,  608,  615,  619,  624,  629,  630,  636,  641,  643,
        652,  657,  659,  664,  670,  671,  680,  681,  684,  685,  688,  689,  693,  694,  696,  698,  701,  703,
        707,  708,  713,  714,  717,  719,  725,  730,  733,  739,  740,  741,  748,  755,  769,  776,  777,  781,
        782,  783,  784,  790,  796,  801,  815,  825,  839,  844,  850,  860,  866,  870,  872,  875,  876,  879,
        880,  881,  886,  888,  893,  896,  903,  907,  912,  917,  918,  924,  925,  926,  927,  929,  931,  934,
        940,  945,  949,  956,  957,  959,  961,  962,  963,  964,  969,  970,  977,  982,  985,  991,  992,  993,
        1000, 1007, 1019, 1024, 1030, 1033, 1034, 1035, 1040, 1042, 1057, 1064, 1065, 1072, 1082, 1086, 1088, 1095,
        1128, 1133, 1140, 1141, 1142, 1143, 1145, 1150, 1155, 1159, 1170, 1183, 1191, 1195, 1206, 1219, 1229, 1234,
        1243, 1259, 1260, 1261, 1265, 1270, 1274, 1279, 1290, 1295};
    const std::vector<int> ind1 = {25,  26,  27,  34,  61,  62,  63,  70,  84,  89,  96,  101, 110, 114, 116,
                                   120, 123, 125, 132, 137, 157, 164, 165, 172, 184, 189, 193, 200, 201, 203,
                                   208, 213, 227, 232, 238, 241, 242, 243, 248, 250, 263, 268, 274, 284};
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(36, 36);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(36, 8);

    for (int k = 0; k < ind0.size(); k++) {
        int i = ind0[k] % 36;
        int j = ind0[k] / 36;
        C0(i, j) = coeffs[coeff_ind0[k]];
    }

    for (int k = 0; k < ind1.size(); k++) {
        int i = ind1[k] % 36;
        int j = ind1[k] / 36;
        C1(i, j) = coeffs[coeff_ind1[k]];
    }

    Eigen::MatrixXd C2 = C0.fullPivHouseholderQr().solve(C1);

    Eigen::Matrix<double, 8, 8> AM;
    AM << Eigen::RowVector<double, 8>(0, 0, 1, 0, 0, 0, 0, 0), -C2.row(31), -C2.row(32), -C2.row(33), -C2.row(34),
        -C2.row(35), Eigen::RowVector<double, 8>(0, 0, 0, 1, 0, 0, 0, 0), -C2.row(30);

    Eigen::EigenSolver<Eigen::Matrix<double, 8, 8>> es(AM);
    Eigen::Vector<std::complex<double>, 8> D = es.eigenvalues();
    Eigen::Matrix<std::complex<double>, 8, 8> V = es.eigenvectors();

    Eigen::MatrixXcd sols = Eigen::MatrixXcd(8, 4);
    sols.col(0) = V.row(6).array() / V.row(0).array();
    sols.col(1) = D;
    sols.col(2) = V.row(4).array() / V.row(0).array();
    sols.col(3) = V.row(1).array() / V.row(0).array();

    std::vector<Eigen::Vector<double, 5>> solutions;
    for (int i = 0; i < sols.rows(); i++) {
        if (D[i].imag() != 0)
            continue;
        if (sols(i, 3).real() < 0)
            continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector<double, 5> sol;
        sol << 1.0, b1, a2, b2 * a2, f0 / std::sqrt(sols(i, 3).real());
        solutions.push_back(sol);
    }

    return solutions;
}

std::vector<Eigen::Vector<double, 6>> solve_scale_and_shift_two_focal(const Eigen::Matrix3x4d &x_homo,
                                                                      const Eigen::Matrix3x4d &y_homo,
                                                                      const Eigen::Vector4d &depth_x,
                                                                      const Eigen::Vector4d &depth_y) {
    Eigen::Matrix<double, 4, 3> x1 = x_homo.transpose();
    Eigen::Matrix<double, 4, 3> x2 = y_homo.transpose();

    double f1_0 = x1.block<4, 2>(0, 0).cwiseAbs().mean();
    double f2_0 = x2.block<4, 2>(0, 0).cwiseAbs().mean();
    x1.block<4, 2>(0, 0) /= f1_0;
    x2.block<4, 2>(0, 0) /= f2_0;

    const Eigen::Vector4d &d1 = depth_x;
    const Eigen::Vector4d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(40);

    coeffs[0] = 2 * x2(0, 0) * x2(1, 0) + 2 * x2(0, 1) * x2(1, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1);
    coeffs[1] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(1, 1) - 2 * x1(0, 0) * x1(1, 0) + x1(0, 1) * x1(0, 1) +
                x1(1, 0) * x1(1, 0) + x1(1, 1) * x1(1, 1);
    coeffs[2] = 2 * d2(0) * x2(0, 0) * x2(1, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) -
                2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(1) * x2(0, 0) * x2(1, 0) +
                2 * d2(0) * x2(0, 1) * x2(1, 1) + 2 * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[3] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(1) * x1(1, 0) * x1(1, 0) +
                2 * d1(1) * x1(1, 1) * x1(1, 1) - 2 * d1(0) * x1(0, 0) * x1(1, 0) - 2 * d1(1) * x1(0, 0) * x1(1, 0) -
                2 * d1(0) * x1(0, 1) * x1(1, 1) - 2 * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[4] = 2 * d2(0) * d2(1) * x2(0, 0) * x2(1, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                d2(1) * d2(1) * x2(1, 0) * x2(1, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(1) * x2(0, 1) * x2(1, 1);
    coeffs[5] = 2 * d2(0) * d2(1) - d2(0) * d2(0) - d2(1) * d2(1);
    coeffs[6] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) -
                2 * d1(0) * d1(1) * x1(0, 0) * x1(1, 0) - 2 * d1(0) * d1(1) * x1(0, 1) * x1(1, 1);
    coeffs[7] = d1(0) * d1(0) - 2 * d1(0) * d1(1) + d1(1) * d1(1);
    coeffs[8] = 2 * x2(0, 0) * x2(2, 0) + 2 * x2(0, 1) * x2(2, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[9] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(2, 1) - 2 * x1(0, 0) * x1(2, 0) + x1(0, 1) * x1(0, 1) +
                x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[10] = 2 * d2(0) * x2(0, 0) * x2(2, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(2, 1) +
                 2 * d2(2) * x2(0, 0) * x2(2, 0) + 2 * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[11] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(0) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * x1(0, 1) * x1(2, 1) -
                 2 * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[12] = 2 * d2(0) * d2(2) * x2(0, 0) * x2(2, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(2) * x2(0, 1) * x2(2, 1);
    coeffs[13] = 2 * d2(0) * d2(2) - d2(0) * d2(0) - d2(2) * d2(2);
    coeffs[14] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(0) * d1(2) * x1(0, 0) * x1(2, 0) - 2 * d1(0) * d1(2) * x1(0, 1) * x1(2, 1);
    coeffs[15] = d1(0) * d1(0) - 2 * d1(0) * d1(2) + d1(2) * d1(2);
    coeffs[16] = 2 * x2(1, 0) * x2(2, 0) + 2 * x2(1, 1) * x2(2, 1) - x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1) -
                 x2(2, 0) * x2(2, 0) - x2(2, 1) * x2(2, 1);
    coeffs[17] = x1(1, 0) * x1(1, 0) - 2 * x1(1, 1) * x1(2, 1) - 2 * x1(1, 0) * x1(2, 0) + x1(1, 1) * x1(1, 1) +
                 x1(2, 0) * x1(2, 0) + x1(2, 1) * x1(2, 1);
    coeffs[18] = 2 * d2(1) * x2(1, 0) * x2(2, 0) - 2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(2) * x2(2, 0) * x2(2, 0) -
                 2 * d2(2) * x2(2, 1) * x2(2, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(2) * x2(1, 0) * x2(2, 0) +
                 2 * d2(1) * x2(1, 1) * x2(2, 1) + 2 * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[19] = 2 * d1(1) * x1(1, 0) * x1(1, 0) + 2 * d1(1) * x1(1, 1) * x1(1, 1) + 2 * d1(2) * x1(2, 0) * x1(2, 0) +
                 2 * d1(2) * x1(2, 1) * x1(2, 1) - 2 * d1(1) * x1(1, 0) * x1(2, 0) - 2 * d1(2) * x1(1, 0) * x1(2, 0) -
                 2 * d1(1) * x1(1, 1) * x1(2, 1) - 2 * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[20] = 2 * d2(1) * d2(2) * x2(1, 0) * x2(2, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                 d2(2) * d2(2) * x2(2, 0) * x2(2, 0) - d2(2) * d2(2) * x2(2, 1) * x2(2, 1) -
                 d2(1) * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * d2(2) * x2(1, 1) * x2(2, 1);
    coeffs[21] = 2 * d2(1) * d2(2) - d2(1) * d2(1) - d2(2) * d2(2);
    coeffs[22] = d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) +
                 d1(2) * d1(2) * x1(2, 0) * x1(2, 0) + d1(2) * d1(2) * x1(2, 1) * x1(2, 1) -
                 2 * d1(1) * d1(2) * x1(1, 0) * x1(2, 0) - 2 * d1(1) * d1(2) * x1(1, 1) * x1(2, 1);
    coeffs[23] = d1(1) * d1(1) - 2 * d1(1) * d1(2) + d1(2) * d1(2);
    coeffs[24] = 2 * x2(0, 0) * x2(3, 0) + 2 * x2(0, 1) * x2(3, 1) - x2(0, 0) * x2(0, 0) - x2(0, 1) * x2(0, 1) -
                 x2(3, 0) * x2(3, 0) - x2(3, 1) * x2(3, 1);
    coeffs[25] = x1(0, 0) * x1(0, 0) - 2 * x1(0, 1) * x1(3, 1) - 2 * x1(0, 0) * x1(3, 0) + x1(0, 1) * x1(0, 1) +
                 x1(3, 0) * x1(3, 0) + x1(3, 1) * x1(3, 1);
    coeffs[26] = 2 * d2(0) * x2(0, 0) * x2(3, 0) - 2 * d2(0) * x2(0, 1) * x2(0, 1) - 2 * d2(3) * x2(3, 0) * x2(3, 0) -
                 2 * d2(3) * x2(3, 1) * x2(3, 1) - 2 * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * x2(0, 1) * x2(3, 1) +
                 2 * d2(3) * x2(0, 0) * x2(3, 0) + 2 * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[27] = 2 * d1(0) * x1(0, 0) * x1(0, 0) + 2 * d1(0) * x1(0, 1) * x1(0, 1) + 2 * d1(3) * x1(3, 0) * x1(3, 0) +
                 2 * d1(3) * x1(3, 1) * x1(3, 1) - 2 * d1(0) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * x1(0, 1) * x1(3, 1) -
                 2 * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[28] = 2 * d2(0) * d2(3) * x2(0, 0) * x2(3, 0) - d2(0) * d2(0) * x2(0, 1) * x2(0, 1) -
                 d2(3) * d2(3) * x2(3, 0) * x2(3, 0) - d2(3) * d2(3) * x2(3, 1) * x2(3, 1) -
                 d2(0) * d2(0) * x2(0, 0) * x2(0, 0) + 2 * d2(0) * d2(3) * x2(0, 1) * x2(3, 1);
    coeffs[29] = 2 * d2(0) * d2(3) - d2(0) * d2(0) - d2(3) * d2(3);
    coeffs[30] = d1(0) * d1(0) * x1(0, 0) * x1(0, 0) + d1(0) * d1(0) * x1(0, 1) * x1(0, 1) +
                 d1(3) * d1(3) * x1(3, 0) * x1(3, 0) + d1(3) * d1(3) * x1(3, 1) * x1(3, 1) -
                 2 * d1(0) * d1(3) * x1(0, 0) * x1(3, 0) - 2 * d1(0) * d1(3) * x1(0, 1) * x1(3, 1);
    coeffs[31] = d1(0) * d1(0) - 2 * d1(0) * d1(3) + d1(3) * d1(3);
    coeffs[32] = 2 * x2(1, 0) * x2(3, 0) + 2 * x2(1, 1) * x2(3, 1) - x2(1, 0) * x2(1, 0) - x2(1, 1) * x2(1, 1) -
                 x2(3, 0) * x2(3, 0) - x2(3, 1) * x2(3, 1);
    coeffs[33] = x1(1, 0) * x1(1, 0) - 2 * x1(1, 1) * x1(3, 1) - 2 * x1(1, 0) * x1(3, 0) + x1(1, 1) * x1(1, 1) +
                 x1(3, 0) * x1(3, 0) + x1(3, 1) * x1(3, 1);
    coeffs[34] = 2 * d2(1) * x2(1, 0) * x2(3, 0) - 2 * d2(1) * x2(1, 1) * x2(1, 1) - 2 * d2(3) * x2(3, 0) * x2(3, 0) -
                 2 * d2(3) * x2(3, 1) * x2(3, 1) - 2 * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * x2(1, 1) * x2(3, 1) +
                 2 * d2(3) * x2(1, 0) * x2(3, 0) + 2 * d2(3) * x2(1, 1) * x2(3, 1);
    coeffs[35] = 2 * d1(1) * x1(1, 0) * x1(1, 0) + 2 * d1(1) * x1(1, 1) * x1(1, 1) + 2 * d1(3) * x1(3, 0) * x1(3, 0) +
                 2 * d1(3) * x1(3, 1) * x1(3, 1) - 2 * d1(1) * x1(1, 0) * x1(3, 0) - 2 * d1(1) * x1(1, 1) * x1(3, 1) -
                 2 * d1(3) * x1(1, 0) * x1(3, 0) - 2 * d1(3) * x1(1, 1) * x1(3, 1);
    coeffs[36] = 2 * d2(1) * d2(3) * x2(1, 0) * x2(3, 0) - d2(1) * d2(1) * x2(1, 1) * x2(1, 1) -
                 d2(3) * d2(3) * x2(3, 0) * x2(3, 0) - d2(3) * d2(3) * x2(3, 1) * x2(3, 1) -
                 d2(1) * d2(1) * x2(1, 0) * x2(1, 0) + 2 * d2(1) * d2(3) * x2(1, 1) * x2(3, 1);
    coeffs[37] = 2 * d2(1) * d2(3) - d2(1) * d2(1) - d2(3) * d2(3);
    coeffs[38] = d1(1) * d1(1) * x1(1, 0) * x1(1, 0) + d1(1) * d1(1) * x1(1, 1) * x1(1, 1) +
                 d1(3) * d1(3) * x1(3, 0) * x1(3, 0) + d1(3) * d1(3) * x1(3, 1) * x1(3, 1) -
                 2 * d1(1) * d1(3) * x1(1, 0) * x1(3, 0) - 2 * d1(1) * d1(3) * x1(1, 1) * x1(3, 1);
    coeffs[39] = d1(1) * d1(1) - 2 * d1(1) * d1(3) + d1(3) * d1(3);

    const std::vector<int> coeff_ind0 = {
        0,  8,  16, 24, 32, 0,  8,  16, 24, 32, 1,  9,  17, 25, 33, 2,  10, 0,  8,  16, 18, 24, 26, 32, 34, 0,  8,  16,
        24, 32, 1,  9,  17, 25, 33, 2,  10, 8,  0,  16, 18, 24, 26, 32, 34, 8,  0,  16, 24, 32, 1,  9,  17, 25, 33, 3,
        11, 1,  9,  17, 19, 25, 27, 33, 35, 4,  12, 2,  10, 18, 20, 26, 28, 34, 36, 2,  10, 8,  16, 18, 0,  26, 24, 32,
        34, 9,  1,  17, 25, 33, 3,  11, 9,  1,  19, 17, 27, 25, 33, 35, 5,  4,  13, 12, 10, 2,  18, 21, 20, 26, 29, 28,
        34, 36, 37, 10, 2,  8,  0,  18, 24, 26, 32, 34, 16, 3,  11, 19, 9,  17, 1,  27, 25, 33, 35, 6,  14, 3,  11, 19,
        22, 27, 30, 35, 38, 4,  12, 20, 28, 36, 4,  12, 10, 18, 20, 2,  28, 26, 34, 36, 5,  13, 21, 29, 37, 11, 3,  19,
        9,  1,  27, 25, 33, 17, 35, 6,  14, 11, 3,  22, 19, 30, 27, 35, 38, 5,  12, 13, 21, 4,  20, 28, 29, 36, 37, 5,
        12, 13, 4,  10, 21, 2,  20, 26, 29, 28, 34, 36, 37, 18, 7,  15, 23, 31, 39, 6,  14, 22, 11, 19, 3,  30, 27, 35,
        38, 6,  14, 22, 30, 38, 12, 20, 4,  28, 36, 13, 5,  21, 29, 37, 13, 5,  21, 29, 37, 14, 6,  22, 30, 38, 13, 12,
        21, 5,  4,  28, 29, 37, 36, 20, 7,  15, 23, 31, 39, 14, 22, 6,  30, 38, 13, 5,  29, 37, 21, 15, 7,  23, 31, 39,
        7,  15, 23, 31, 39, 14, 6,  22, 11, 3,  30, 27, 35, 19, 38, 7,  15, 23, 31, 39};
    const std::vector<int> coeff_ind1 = {15, 7, 31, 39, 23, 15, 7, 23, 31, 39, 14, 6, 30, 38, 22, 15, 23, 7, 31, 39};
    const std::vector<int> ind0 = {
        0,    2,    18,   28,   39,   41,   45,   60,   69,   78,   80,   82,   98,   108,  119,  120,  122,  123,
        128,  130,  138,  145,  148,  157,  159,  164,  169,  177,  186,  194,  201,  205,  220,  229,  238,  241,
        245,  246,  252,  254,  260,  263,  269,  276,  278,  287,  293,  302,  310,  313,  323,  328,  330,  345,
        357,  360,  362,  364,  369,  377,  378,  386,  388,  394,  399,  400,  402,  403,  408,  410,  418,  425,
        428,  437,  439,  444,  449,  451,  456,  457,  459,  466,  467,  471,  474,  486,  492,  494,  503,  516,
        521,  525,  527,  533,  540,  542,  549,  550,  553,  558,  560,  561,  562,  565,  566,  572,  574,  578,
        580,  583,  588,  589,  596,  598,  599,  607,  613,  615,  621,  622,  624,  630,  632,  633,  635,  643,
        648,  650,  651,  656,  659,  665,  667,  671,  677,  680,  682,  684,  689,  697,  698,  706,  708,  714,
        719,  723,  728,  730,  745,  757,  764,  769,  771,  776,  777,  779,  786,  787,  791,  794,  801,  805,
        820,  829,  838,  846,  852,  854,  855,  861,  863,  864,  872,  875,  876,  881,  885,  887,  893,  900,
        902,  909,  910,  913,  918,  923,  926,  928,  930,  932,  934,  943,  945,  956,  957,  964,  967,  969,
        973,  975,  977,  981,  982,  984,  986,  990,  992,  993,  994,  995,  1000, 1002, 1018, 1028, 1039, 1043,
        1048, 1050, 1051, 1056, 1059, 1065, 1067, 1071, 1077, 1084, 1089, 1097, 1106, 1114, 1131, 1136, 1139, 1147,
        1151, 1166, 1172, 1174, 1183, 1196, 1207, 1213, 1222, 1230, 1233, 1247, 1253, 1262, 1270, 1273, 1291, 1295,
        1296, 1299, 1301, 1304, 1307, 1311, 1312, 1315, 1324, 1329, 1337, 1346, 1354, 1371, 1376, 1379, 1387, 1391,
        1415, 1421, 1424, 1432, 1435, 1446, 1452, 1454, 1463, 1476, 1481, 1485, 1500, 1509, 1518, 1526, 1532, 1534,
        1535, 1541, 1543, 1544, 1552, 1555, 1556, 1563, 1568, 1570, 1585, 1597};
    const std::vector<int> ind1 = {15, 21,  24,  32,  35,  47,  53,  62,  70,  73,
                                   95, 101, 104, 112, 115, 131, 136, 139, 147, 151};
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(40, 40);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(40, 4);

    for (int k = 0; k < ind0.size(); k++) {
        int i = ind0[k] % 40;
        int j = ind0[k] / 40;
        C0(i, j) = coeffs[coeff_ind0[k]];
    }

    for (int k = 0; k < ind1.size(); k++) {
        int i = ind1[k] % 40;
        int j = ind1[k] / 40;
        C1(i, j) = coeffs[coeff_ind1[k]];
    }

    Eigen::MatrixXd C2 = C0.fullPivHouseholderQr().solve(C1);

    Eigen::Matrix4d AM;
    AM << -C2.row(36), -C2.row(37), -C2.row(38), -C2.row(39);

    Eigen::EigenSolver<Eigen::Matrix4d> es(AM);
    Eigen::Vector4cd D = es.eigenvalues();
    Eigen::Matrix4cd V = es.eigenvectors();

    Eigen::MatrixXcd sols = Eigen::MatrixXcd(4, 5);
    sols.col(0) = (-C2.row(35) * V).array() / V.row(0).array();
    sols.col(1) = D;
    sols.col(2) = V.row(1).array() / V.row(0).array();
    sols.col(3) = V.row(2).array() / V.row(0).array();
    sols.col(4) = V.row(3).array() / V.row(0).array();

    std::vector<Eigen::Vector<double, 6>> solutions;
    for (int i = 0; i < sols.rows(); i++) {
        if (D[i].imag() != 0)
            continue;
        if (sols(i, 3).real() < 0 || sols(i, 4).real() < 0)
            continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector<double, 6> sol;
        sol << 1.0, b1, a2, b2 * a2, f1_0 / std::sqrt(sols(i, 3).real()), f2_0 / std::sqrt(sols(i, 4).real());
        solutions.push_back(sol);
    }

    return solutions;
}

int solve_scale_shift_pose(const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo, const Eigen::Vector3d &depth_x,
                           const Eigen::Vector3d &depth_y, std::vector<PoseScaleOffset> *output, bool scale_on_x) {
    // X: 3 x 3, column vectors are homogeneous 2D points
    // Y: 3 x 3, column vectors are homogeneous 2D points
    std::vector<Eigen::Vector4d> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol : solutions) {
        Eigen::Vector3d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        } else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0)
            continue;

        Eigen::Matrix3d X = x_homo.array().rowwise() * d1.transpose().array();
        Eigen::Matrix3d Y = y_homo.array().rowwise() * d2.transpose().array();

        Eigen::Vector3d centroid_X = X.rowwise().mean();
        Eigen::Vector3d centroid_Y = Y.rowwise().mean();

        Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
        Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

        Eigen::Matrix3d S = Y_centered * X_centered.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        if (U.determinant() * V.determinant() < 0) {
            U.col(2) *= -1;
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::Vector3d t = centroid_Y - R * centroid_X;

        double b2 = sol(1), a1 = sol(2), b1 = sol(3);
        if (!scale_on_x)
            std::swap(b1, b2);
        output->push_back(PoseScaleOffset(R, t, a1, b1, b2));
        sol_count++;
    }
    return sol_count;
}

Eigen::MatrixXd solver_p3p_mono_3d(const Eigen::VectorXd &data) {
    // Action =  y
    // Quotient ring basis (V) = 1,x,y,z,
    // Available monomials (RR*V) = x*y,y^2,y*z,1,x,y,z,

    const double *d = data.data();
    Eigen::VectorXd coeffs(18);
    coeffs[0] = std::pow(d[6],2) - 2*d[6]*d[7] + std::pow(d[7],2) + std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2);
    coeffs[1] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[3],2) + 2*d[3]*d[4] - std::pow(d[4],2);
    coeffs[2] = 2*std::pow(d[6],2)*d[15] - 2*d[6]*d[7]*d[15] + 2*std::pow(d[9],2)*d[15] - 2*d[9]*d[10]*d[15] - 2*d[6]*d[7]*d[16] + 2*std::pow(d[7],2)*d[16] - 2*d[9]*d[10]*d[16] + 2*std::pow(d[10],2)*d[16];
    coeffs[3] = std::pow(d[6],2)*std::pow(d[15],2) + std::pow(d[9],2)*std::pow(d[15],2) - 2*d[6]*d[7]*d[15]*d[16] - 2*d[9]*d[10]*d[15]*d[16] + std::pow(d[7],2)*std::pow(d[16],2) + std::pow(d[10],2)*std::pow(d[16],2) + std::pow(d[15],2) - 2*d[15]*d[16] + std::pow(d[16],2);
    coeffs[4] = -2*std::pow(d[0],2)*d[12] + 2*d[0]*d[1]*d[12] - 2*std::pow(d[3],2)*d[12] + 2*d[3]*d[4]*d[12] + 2*d[0]*d[1]*d[13] - 2*std::pow(d[1],2)*d[13] + 2*d[3]*d[4]*d[13] - 2*std::pow(d[4],2)*d[13];
    coeffs[5] = -std::pow(d[0],2)*std::pow(d[12],2) - std::pow(d[3],2)*std::pow(d[12],2) + 2*d[0]*d[1]*d[12]*d[13] + 2*d[3]*d[4]*d[12]*d[13] - std::pow(d[1],2)*std::pow(d[13],2) - std::pow(d[4],2)*std::pow(d[13],2) - std::pow(d[12],2) + 2*d[12]*d[13] - std::pow(d[13],2);
    coeffs[6] = std::pow(d[6],2) - 2*d[6]*d[8] + std::pow(d[8],2) + std::pow(d[9],2) - 2*d[9]*d[11] + std::pow(d[11],2);
    coeffs[7] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[3],2) + 2*d[3]*d[5] - std::pow(d[5],2);
    coeffs[8] = 2*std::pow(d[6],2)*d[15] - 2*d[6]*d[8]*d[15] + 2*std::pow(d[9],2)*d[15] - 2*d[9]*d[11]*d[15] - 2*d[6]*d[8]*d[17] + 2*std::pow(d[8],2)*d[17] - 2*d[9]*d[11]*d[17] + 2*std::pow(d[11],2)*d[17];
    coeffs[9] = std::pow(d[6],2)*std::pow(d[15],2) + std::pow(d[9],2)*std::pow(d[15],2) - 2*d[6]*d[8]*d[15]*d[17] - 2*d[9]*d[11]*d[15]*d[17] + std::pow(d[8],2)*std::pow(d[17],2) + std::pow(d[11],2)*std::pow(d[17],2) + std::pow(d[15],2) - 2*d[15]*d[17] + std::pow(d[17],2);
    coeffs[10] = -2*std::pow(d[0],2)*d[12] + 2*d[0]*d[2]*d[12] - 2*std::pow(d[3],2)*d[12] + 2*d[3]*d[5]*d[12] + 2*d[0]*d[2]*d[14] - 2*std::pow(d[2],2)*d[14] + 2*d[3]*d[5]*d[14] - 2*std::pow(d[5],2)*d[14];
    coeffs[11] = -std::pow(d[0],2)*std::pow(d[12],2) - std::pow(d[3],2)*std::pow(d[12],2) + 2*d[0]*d[2]*d[12]*d[14] + 2*d[3]*d[5]*d[12]*d[14] - std::pow(d[2],2)*std::pow(d[14],2) - std::pow(d[5],2)*std::pow(d[14],2) - std::pow(d[12],2) + 2*d[12]*d[14] - std::pow(d[14],2);
    coeffs[12] = std::pow(d[7],2) - 2*d[7]*d[8] + std::pow(d[8],2) + std::pow(d[10],2) - 2*d[10]*d[11] + std::pow(d[11],2);
    coeffs[13] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[14] = 2*std::pow(d[7],2)*d[16] - 2*d[7]*d[8]*d[16] + 2*std::pow(d[10],2)*d[16] - 2*d[10]*d[11]*d[16] - 2*d[7]*d[8]*d[17] + 2*std::pow(d[8],2)*d[17] - 2*d[10]*d[11]*d[17] + 2*std::pow(d[11],2)*d[17];
    coeffs[15] = std::pow(d[7],2)*std::pow(d[16],2) + std::pow(d[10],2)*std::pow(d[16],2) - 2*d[7]*d[8]*d[16]*d[17] - 2*d[10]*d[11]*d[16]*d[17] + std::pow(d[8],2)*std::pow(d[17],2) + std::pow(d[11],2)*std::pow(d[17],2) + std::pow(d[16],2) - 2*d[16]*d[17] + std::pow(d[17],2);
    coeffs[16] = -2*std::pow(d[1],2)*d[13] + 2*d[1]*d[2]*d[13] - 2*std::pow(d[4],2)*d[13] + 2*d[4]*d[5]*d[13] + 2*d[1]*d[2]*d[14] - 2*std::pow(d[2],2)*d[14] + 2*d[4]*d[5]*d[14] - 2*std::pow(d[5],2)*d[14];
    coeffs[17] = -std::pow(d[1],2)*std::pow(d[13],2) - std::pow(d[4],2)*std::pow(d[13],2) + 2*d[1]*d[2]*d[13]*d[14] + 2*d[4]*d[5]*d[13]*d[14] - std::pow(d[2],2)*std::pow(d[14],2) - std::pow(d[5],2)*std::pow(d[14],2) - std::pow(d[13],2) + 2*d[13]*d[14] - std::pow(d[14],2);

    Eigen::MatrixXd C0(3,3);
    C0 << coeffs[0], coeffs[2], coeffs[3],
        coeffs[6], coeffs[8], coeffs[9],
        coeffs[12], coeffs[14], coeffs[15];

    Eigen::MatrixXd C1(3,3);
    C1 << coeffs[1], coeffs[4], coeffs[5],
        coeffs[7], coeffs[10], coeffs[11],
        coeffs[13], coeffs[16], coeffs[17];

    Eigen::MatrixXd C2 = -C0.partialPivLu().solve(C1);

    double k0 = C2(0,0);
    double k1 = C2(0,1);
    double k2 = C2(0,2);
    double k3 = C2(1,0);
    double k4 = C2(1,1);
    double k5 = C2(1,2);
    double k6 = C2(2,0);
    double k7 = C2(2,1);
    double k8 = C2(2,2);

    double c4 = 1.0 / (k3*k3 - k0*k6);
    double c3 = c4 * (2*k3*k4 - k1*k6 - k0*k7);
    double c2 = c4 * (k4*k4 - k0*k8 - k1*k7 - k2*k6 + 2*k3*k5);
    double c1 = c4 * (2*k4*k5 - k2*k7 - k1*k8);
    double c0 = c4 * (k5*k5 - k2*k8);
    //    double roots[4];
    //    int n_roots = univariate::solve_quartic_real(c3, c2, c1, c0, roots);

    Eigen::Matrix4d CC;

    CC << 0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        -c0, -c1, -c2, -c3;

    Eigen::EigenSolver<Eigen::Matrix4d> es(CC, false);
    Eigen::Matrix<std::complex<double>, 4, 1> D = es.eigenvalues();
    int n_roots = 0;
    double roots[4];
    for (int i = 0; i < 4; ++i) {
        if (std::abs(D(i).imag()) > 1e-8)
            continue;
        roots[n_roots++] = D(i).real();
    }
    int m = 0;
    Eigen::MatrixXd sols(3, n_roots);
    for (int ii = 0; ii < n_roots; ii++) {
        double ss = k6*roots[ii]*roots[ii] + k7*roots[ii] + k8;
        if (ss < 0.01)
            continue;
        sols(1,ii) = roots[ii];
        sols(0,ii) = k6*roots[ii]*roots[ii] + k7*roots[ii] + k8;
        sols(2,ii) = (k3*roots[ii]*roots[ii] + k4*roots[ii] + k5)/sols(0,ii);
        ++m;
    }
    sols.conservativeResize(3, m);
    return sols;
}

int solve_scale_shift_pose_ours(const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
                                const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y,
                                std::vector<PoseScaleOffset> *output, bool scale_on_x) {
    output->clear();
    std::vector<Eigen::Vector3d> x1h(3);
    std::vector<Eigen::Vector3d> x2h(3);
    for (int i = 0; i < 3; ++i) {
        x1h[i] = x_homo.col(i);
        x2h[i] = y_homo.col(i);
    }

    double depth1[3];
    double depth2[3];
    for (int i = 0; i < 3; ++i) {
        depth1[i] = depth_x(i);
        depth2[i] = depth_y(i);
    }

    Eigen::VectorXd datain(18);
    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[0][1], x1h[1][1], x1h[2][1], x2h[0][0], x2h[1][0], x2h[2][0],
        x2h[0][1], x2h[1][1], x2h[2][1], depth1[0], depth1[1], depth1[2], depth2[0], depth2[1], depth2[2];

    Eigen::MatrixXd sols = solver_p3p_mono_3d(datain);

    output->reserve(sols.cols());

    int num_sols = 0;
    for (int k = 0; k < sols.cols(); ++k) {

        double s = std::sqrt(sols(0, k));
        double u = sols(1, k);
        double v = sols(2, k);

        if (depth1[0] + u <= 0 || depth1[1] + u <= 0 || depth1[2] + u <= 0)
            continue;

        if (depth2[0] + v <= 0 || depth2[1] + v <= 0 || depth2[2] + v <= 0)
            continue;

        Eigen::Vector3d v1 = s * (depth2[0] + v) * x2h[0] - s * (depth2[1] + v) * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * x2h[0] - s * (depth2[2] + v) * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * x1h[0] - (depth1[1] + u) * x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * x1h[0] - (depth1[2] + u) * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d t = s * (depth2[0] + v) * x2h[0] - (depth1[0] + u) * rot * x1h[0];
        output->emplace_back(PoseScaleOffset(rot, t, s, u, s*v));
        num_sols++;
    }
    return num_sols;
}

int solve_scale_shift_pose_shared_focal(const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
                                        const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y,
                                        std::vector<PoseScaleOffsetSharedFocal> *output, bool scale_on_x) {
    std::vector<Eigen::Vector<double, 5>> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift_shared_focal(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift_shared_focal(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol : solutions) {
        Eigen::Vector4d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        } else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0)
            continue;

        double focal = sol(4);
        Eigen::Matrix3x4d xu = x_homo;
        Eigen::Matrix3x4d yu = y_homo;
        xu.block<2, 4>(0, 0) /= focal;
        yu.block<2, 4>(0, 0) /= focal;

        Eigen::Matrix3x4d X = xu.array().rowwise() * d1.transpose().array();
        Eigen::Matrix3x4d Y = yu.array().rowwise() * d2.transpose().array();

        Eigen::Vector3d centroid_X = X.rowwise().mean();
        Eigen::Vector3d centroid_Y = Y.rowwise().mean();

        Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
        Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

        Eigen::Matrix3d S = Y_centered * X_centered.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        if (U.determinant() * V.determinant() < 0) {
            U.col(2) *= -1;
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::Vector3d t = centroid_Y - R * centroid_X;

        double b2 = sol(1), a1 = sol(2), b1 = sol(3), f = sol(4);
        if (!scale_on_x)
            std::swap(b1, b2);
        output->push_back(PoseScaleOffsetSharedFocal(R, t, a1, b1, b2, f));
        sol_count++;
    }
    return sol_count;
}

Eigen::MatrixXd solver_p3p_s00f(Eigen::VectorXd d)
{
    Eigen::VectorXd coeffs(18);
    coeffs[0] = d[0];
    coeffs[1] = d[2];
    coeffs[2] = d[1];
    coeffs[3] = d[3];
    coeffs[4] = d[4];
    coeffs[5] = d[5];
    coeffs[6] = d[8];
    coeffs[7] = d[6];
    coeffs[8] = d[9];
    coeffs[9] = d[10];
    coeffs[10] = d[7];
    coeffs[11] = d[11];
    coeffs[12] = d[12];
    coeffs[13] = d[13];
    coeffs[14] = d[15];
    coeffs[15] = d[16];
    coeffs[16] = d[14];
    coeffs[17] = d[17];

    static const int coeffs_ind[] = {0,4,12,0,5,12,13,2,4,0,7,13,2,5,12,16,7,2,13,10,0,16,1,6,14,1,8,14,15,1,9,15,3,6,3,8,14,17,9,3,15,11,1,17,11,17,3};

    static const int C_ind[] = {0,5,8,10,14,16,17,18,20,30,32,34,37,38,40,44,47,48,49,50,51,52,54,59,62,64,68,70,71,75,77,79,81,83,91,92,94,98,101,102,103,104,105,106,110,112,114};

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(9,13);
    for (int i = 0; i < 47; i++) {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Eigen::MatrixXd C0 = C.leftCols(9);
    Eigen::MatrixXd C1 = C.rightCols(4);
    Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
    Eigen::MatrixXd RR(7, 4);
    RR << -C12.bottomRows(3), Eigen::MatrixXd::Identity(4, 4);

    static const int AM_ind[] = {0,1,2,5};
    Eigen::MatrixXd AM(4, 4);
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(3).replicate(4, 1)).eval();

    Eigen::MatrixXd sols(3, 4); // s f d
    int m = 0;
    Eigen::MatrixXcd V0(1, 4);
    // sols.row(0) = D.transpose(); // s
    V0 = V.row(0) / (V.row(1)); //d3

    for (int k = 0; k < 4; ++k) {

        if (abs(D(k).imag()) > 0.01 || D(k).real() < 0.1 || D(k).real() > 10.0 || abs(V0(0, k).imag()) > 0.01 || V0(0, k).real() < 0.0)
            continue;

        double s2 = D(k).real(); // s^2
        double f2 = -(d[2] * s2 + d[3]) / (d[0] * s2 + d[1]); // f^2

        if (f2 < 0.0)
            continue;


        sols(0, m) = std::sqrt(s2);  // s
        sols(1, m) = std::sqrt(f2); // f
        sols(2, m) = V0(0, k).real();   // d3
        ++m;
    }

    sols.conservativeResize(3, m);

    return sols;
}

int solve_scale_shift_pose_shared_focal_ours(const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
                                             const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y,
                                             std::vector<PoseScaleOffsetSharedFocal> *output, bool scale_on_x) {
    int sol_count = 0;
    output->clear();
    output->reserve(4);
    std::vector<Eigen::Vector3d> x1h(3);
    std::vector<Eigen::Vector3d> x2h(3);
    for (int i = 0; i < 3; ++i) {
        x1h[i] = x_homo.col(i);
        x2h[i] = y_homo.col(i);
    }

    double depth1[3];
    double depth2[3];
    for (int i = 0; i < 3; ++i) {
        depth1[i] = depth_x[i];
        depth2[i] = depth_y[i];
    }

    Eigen::Matrix3d X1;
    X1.col(0) = depth1[0] * x1h[0];
    X1.col(1) = depth1[1] * x1h[1];
    X1.col(2) = depth1[2] * x1h[2];

    Eigen::Matrix3d X2;
    X2.col(0) = depth2[0] * x2h[0];
    X2.col(1) = depth2[1] * x2h[1];
    X2.col(2) = x2h[2];

    double a[17];

    a[0] = X1(0, 0); a[1] = X1(0, 1); a[2] = X1(0, 2);
    a[3] = X1(1, 0); a[4] = X1(1, 1); a[5] = X1(1, 2);
    a[6] = X1(2, 0); a[7] = X1(2, 1); a[8] = X1(2, 2);

    a[9]  = X2(0, 0); a[10] = X2(0, 1); a[11] = X2(0, 2);
    a[12] = X2(1, 0); a[13] = X2(1, 1); a[14] = X2(1, 2);
    a[15] = X2(2, 0); a[16] = X2(2, 1);

    double b[12];
    b[0] = a[0] - a[1]; b[1] = a[3] - a[4]; b[2] = a[6] - a[7];
    b[3] = a[0] - a[2]; b[4] = a[3] - a[5]; b[5] = a[6] - a[8];
    b[6] = a[1] - a[2]; b[7] = a[4] - a[5]; b[8] = a[7] - a[8];
    b[9]  = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];

    double c[18];
    c[0] = -std::pow(b[11], 2);
    c[1] = std::pow(b[2], 2);
    c[2] = -std::pow(b[9], 2) - std::pow(b[10], 2);
    c[3] = std::pow(b[0], 2) + std::pow(b[1], 2);

    c[4] = -1.0;
    c[5] = 2 * a[15];
    c[6] = -std::pow(a[15], 2);
    c[7] = std::pow(b[5], 2);
    c[8] = -std::pow(a[11], 2) - std::pow(a[14], 2);
    c[9] = 2 * a[9] * a[11] + 2 * a[12] * a[14];
    c[10] = -std::pow(a[9], 2) - std::pow(a[12], 2);
    c[11] = std::pow(b[3], 2) + std::pow(b[4], 2);

    c[12] = 2 * a[16] - 2 * a[15];
    c[13] = std::pow(a[15], 2) - std::pow(a[16], 2);
    c[14] = std::pow(b[8], 2) - std::pow(b[5], 2);
    c[15] = 2 * a[10] * a[11] - 2 * a[9] * a[11] - 2 * a[12] * a[14] + 2 * a[13] * a[14];
    c[16] = std::pow(a[9], 2) - std::pow(a[10], 2) + std::pow(a[12], 2) - std::pow(a[13], 2);
    c[17] = -std::pow(b[3], 2) - std::pow(b[4], 2) + std::pow(b[6], 2) + std::pow(b[7], 2);

    double d[21];

    d[6] = 1 / (a[6] - a[7]);
    d[0] = (-c[3] * c[8]) * d[6];
    d[1] = (-c[3] * c[9]) * d[6];
    d[2] = (c[2] * c[11] - c[3] * c[10]) * d[6];
    d[3] = (-c[3] * c[4] - c[1] * c[8]) * d[6];
    d[4] = (-c[3] * c[5] - c[1] * c[9]) * d[6];
    d[5] = (c[2] * c[7] - c[3] * c[6] + c[0] * c[11] - c[1] * c[10]) * d[6];
    d[7] = (a[6] * a[16] - 2 * a[6] * a[15] + a[7] * a[15] + a[8] * a[15] - a[8] * a[16]) * d[6];

    d[8] = 1 / (2 * (a[6] - a[7]) * (a[15] - a[16]));
    d[9] = (-c[3] * c[15]) * d[8];
    d[10] = (c[2] * c[17] - c[3] * c[16]) * d[8];
    d[11] = (-c[3] * c[12] - c[1] * c[15]) * d[8];
    d[12] = (c[2] * c[14] - c[3] * c[13] + c[0] * c[17] - c[1] * c[16]) * d[8];

    d[13] = 1 / (a[6] + a[7] - 2 * a[8]);
    d[14] = (a[8] * a[15] - a[7] * a[15] - a[6] * a[16] + a[8] * a[16]) * d[13];
    d[15] = (c[8] * c[17]) * d[13];
    d[16] = (c[9] * c[17] - c[11] * c[15]) * d[13];
    d[17] = (c[10] * c[17] - c[11] * c[16]) * d[13];
    d[18] = (c[4] * c[17] + c[8] * c[14]) * d[13];
    d[19] = (c[5] * c[17] - c[7] * c[15] + c[9] * c[14] - c[11] * c[12]) * d[13];
    d[20] = (c[6] * c[17] - c[7] * c[16] + c[10] * c[14] - c[11] * c[13]) * d[13];

    Eigen::MatrixXd C0(3, 3);
    C0 << d[2], d[5], d[7],
        d[10], d[12], 1.0,
        d[17], d[20], d[14];

    Eigen::MatrixXd C1(3, 4);
    C1 << d[0]-d[9],  d[3]-d[11], d[1]-d[10],  d[4]-d[12],
        0,     0, d[9],     d[11],
        d[15]-d[9], d[18]-d[11], d[16]-d[10], d[19]-d[12];

    Eigen::MatrixXd C2 = -C0.partialPivLu().solve(C1);

    Eigen::MatrixXd AM(4, 4);
    AM << 0, 0, 1.0, 0,
        0, 0, 0, 1.0,
        C2(0,0), C2(0,1), C2(0,2), C2(0,3),
        C2(1,0), C2(1,1), C2(1,2), C2(1,3);

    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(AM, false);
    Eigen::ArrayXcd D = es.eigenvalues();

    for (int k = 0; k < 4; ++k) {

        if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0)
            continue;

        double d3 = 1.0 / D(k).real();

        Eigen::MatrixXd A0(2, 2);
        A0 << (d[3]-d[11])*d3*d3 + (d[4]-d[12])*d3 + d[5], d[7],
            d[12] + d[11]*d3, 1.0;

        Eigen::VectorXd A1(2);
        A1 << (d[0]-d[9])*d3*d3 + (d[1]-d[10])*d3 + d[2], d[10] + d[9]*d3;
        Eigen::VectorXd A2 = -A0.partialPivLu().solve(A1);

        if (A2(0) < 0.0)
            continue;

        double s2 = -(c[1]*A2(0) + c[3])/(c[0]*A2(0) + c[2]);
        if (s2 < 0.001)
            continue;

        double s = std::sqrt(s2);
        double f = std::sqrt(A2(0));

        Eigen::Matrix3d Kinv;
        Kinv << 1.0 / f, 0, 0, 0, 1.0 / f, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0]) * Kinv * x2h[0] - s * (depth2[1]) * Kinv * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0]) * Kinv * x2h[0] - s * (d3) * Kinv * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * Kinv * x1h[0] - (depth1[1]) * Kinv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * Kinv * x1h[0] - (depth1[2]) * Kinv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * Kinv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * Kinv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        output->emplace_back(PoseScaleOffsetSharedFocal(rot, trans, s, 0.0, 0.0, f));
        sol_count++;
    }
    return sol_count;
}

int solve_scale_shift_pose_two_focal(const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
                                     const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y,
                                     std::vector<PoseScaleOffsetTwoFocal> *output, bool scale_on_x) {
    std::vector<Eigen::Vector<double, 6>> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift_two_focal(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift_two_focal(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol : solutions) {
        Eigen::Vector4d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        } else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0)
            continue;

        double focal1 = sol(4), focal2 = sol(5);
        Eigen::Matrix3x4d xu = x_homo;
        Eigen::Matrix3x4d yu = y_homo;
        xu.block<2, 4>(0, 0) /= focal1;
        yu.block<2, 4>(0, 0) /= focal2;

        Eigen::Matrix3x4d X = xu.array().rowwise() * d1.transpose().array();
        Eigen::Matrix3x4d Y = yu.array().rowwise() * d2.transpose().array();

        Eigen::Vector3d centroid_X = X.rowwise().mean();
        Eigen::Vector3d centroid_Y = Y.rowwise().mean();

        Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
        Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

        Eigen::Matrix3d S = Y_centered * X_centered.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        if (U.determinant() * V.determinant() < 0) {
            U.col(2) *= -1;
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::Vector3d t = centroid_Y - R * centroid_X;

        double b2 = sol(1), a1 = sol(2), b1 = sol(3), f1 = sol(4), f2 = sol(5);
        if (!scale_on_x)
            std::swap(b1, b2);
        output->push_back(PoseScaleOffsetTwoFocal(R, t, a1, b1, b2, f1, f2));
        sol_count++;
    }
    return sol_count;
}

int solve_scale_shift_pose_two_focal_ours(const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
                                          const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y,
                                          std::vector<PoseScaleOffsetTwoFocal> *output, bool scale_on_x) {
    output->clear();
    output->reserve(1);
    std::vector<Eigen::Vector3d> x1h(3);
    std::vector<Eigen::Vector3d> x2h(3);
    for (int i = 0; i < 3; ++i) {
        x1h[i] = x_homo.col(i);
        x2h[i] = y_homo.col(i);
    }

    double depth1[3];
    double depth2[3];
    for (int i = 0; i < 3; ++i) {
        depth1[i] = depth_x[i];
        depth2[i] = depth_y[i];
    }
    double a[18];
    a[0] = x1h[0][0]*depth1[0];
    a[1] = x1h[1][0]*depth1[1];
    a[2] = x1h[2][0]*depth1[2];
    a[3] = x1h[0][1]*depth1[0];
    a[4] = x1h[1][1]*depth1[1];
    a[5] = x1h[2][1]*depth1[2];
    a[6] = depth1[0];
    a[7] = depth1[1];
    a[8] = depth1[2];

    a[9] = x2h[0][0]*depth2[0];
    a[10] = x2h[1][0]*depth2[1];
    a[11] = x2h[2][0]*depth2[2];
    a[12] = x2h[0][1]*depth2[0];
    a[13] = x2h[1][1]*depth2[1];
    a[14] = x2h[2][1]*depth2[2];
    a[15] = depth2[0];
    a[16] = depth2[1];
    a[17] = depth2[2];

    double b[18];
    b[0] = a[0] - a[1];
    b[1] = a[3] - a[4];
    b[2] = a[6] - a[7];
    b[3] = a[0] - a[2];
    b[4] = a[3] - a[5];
    b[5] = a[6] - a[8];
    b[6] = a[1] - a[2];
    b[7] = a[4] - a[5];
    b[8] = a[7] - a[8];
    b[9] = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];
    b[12] = a[9] - a[11];
    b[13] = a[12] - a[14];
    b[14] = a[15] - a[17];
    b[15] = a[10] - a[11];
    b[16] = a[13] - a[14];
    b[17] = a[16] - a[17];

    Eigen::Matrix3d A;
    A << std::pow(b[0], 2) + std::pow(b[1], 2), -std::pow(b[9], 2) - std::pow(b[10], 2), -std::pow(b[11], 2),
        std::pow(b[3], 2) + std::pow(b[4], 2), -std::pow(b[12], 2) - std::pow(b[13], 2), -std::pow(b[14], 2),
        std::pow(b[6], 2) + std::pow(b[7], 2), -std::pow(b[15], 2) - std::pow(b[16], 2), -std::pow(b[17], 2);
    Eigen::Vector3d B;
    B << b[2] * b[2], b[5] * b[5], b[8] * b[8];
    Eigen::Vector3d sol = -A.partialPivLu().solve(B);

    if (sol(0) > 0 && sol(1) > 0 && sol(2) > 0)
    {
        double f = std::sqrt(sol(0));
        double s = std::sqrt(sol(2));
        double w = std::sqrt(sol(1) / sol(2));

        Eigen::Matrix3d K1inv;
        K1inv << f, 0, 0,
            0, f, 0,
            0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << w, 0, 0,
            0,  w, 0,
            0, 0, 1;

        Eigen::Vector3d v1 = s * ((depth2[0]) * K2inv*x2h[0] - (depth2[1]) * K2inv*x2h[1]);
        Eigen::Vector3d v2 = s * ((depth2[0]) * K2inv*x2h[0] - (depth2[2]) * K2inv*x2h[2]);
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * K1inv*x1h[0] - (depth1[1]) * K1inv*x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * K1inv*x1h[0] - (depth1[2]) * K1inv*x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * K1inv*x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * K2inv*x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        double focal1 = 1.0 / f;
        double focal2 = 1.0 / w;

        output->emplace_back(PoseScaleOffsetTwoFocal(rot, trans, s, 0.0, 0.0, focal1, focal2));

        return 1;
    }
    return 0;
}


int solve_scale_shift_pose_two_focal_4p4d(const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
                                          const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y,
                                          std::vector<PoseScaleOffsetTwoFocal> *output, bool scale_on_x) {
    output->clear();
    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = x_homo.col(i);
        x2h[i] = y_homo.col(i);
    }

    std::vector<Eigen::Vector2d> sigma(4);
    for (int i = 0; i < 4; ++i) {
        sigma[i](0) = depth_x[i];
        sigma[i](1) = depth_y[i];
    }

    Eigen::MatrixXd coefficients(12, 12);
    int i;

    // Form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    size_t row = 0;
    for (i = 0; i < 4; i++)
    {
        double u11 = x1h[i](0), v11 = x1h[i](1), u12 = x2h[i](0), v12 = x2h[i](1);
        double q1 = sigma[i](0), q2 = sigma[i](1);
        double q = q2 / q1;

        coefficients(row, 0) = -u11;
        coefficients(row, 1) = -v11;
        coefficients(row, 2) = -1;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = 0;
        coefficients(row, 10) = q;
        coefficients(row, 11) = -q * v12;
        ++row;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = -u11;
        coefficients(row, 4) = -v11;
        coefficients(row, 5) = -1;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = -q;
        coefficients(row, 10) = 0;
        coefficients(row, 11) = q * u12;
        ++row;

        if (i == 3)
            break;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = -u11;
        coefficients(row, 7) = -v11;
        coefficients(row, 8) = -1;
        coefficients(row, 9) = q * v12;
        coefficients(row, 10) = -q * u12;
        coefficients(row, 11) = 0;
        ++row;
    }

    Eigen::Matrix<double, 12, 1> f1 = coefficients.block<11, 11>(0, 0).partialPivLu().solve(-coefficients.block<11, 1>(0, 11)).homogeneous();

    Eigen::Matrix3d F;
    F << f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8];

    //    std::cout << "F: " << std::endl << F << std::endl;
    //    std::cout << "Ep: " << x2h[0].transpose() * F * x1h[0] << std::endl;
    //    std::cout << "Det: " << F.determinant() << std::endl;

    double f0, f1;
    std::tie(f0, f1) = bougnoux_focals(F);
    f0 = std::sqrt(std::abs(f0));
    f1 = std::sqrt(std::abs(f1));

    if (std::isnan(f0))
        return 0;
    if (std::isnan(f1))
        return 0;

    Eigen::Matrix3d K0, K1;
    K0 << f0, 0.0, 0.0, 0.0, f0, 0.0, 0.0, 0.0, 1.0;
    K1 << f1, 0.0, 0.0, 0.0, f1, 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix3d E = K1.transpose() * F * K0;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    cv::Mat cv_E, cv_R, cv_tr;
    cv::eigen2cv(E, cv_E);

    cv::Mat cv_x0_2dvec(4, 2, CV_64F);
    cv::Mat cv_x1_2dvec(4, 2, CV_64F);
    for (int i = 0; i < 4; i++) {
        cv_x0_2dvec.at<double>(i, 0) = x1h[i](0);
        cv_x0_2dvec.at<double>(i, 1) = x1h[i](1);
        cv_x1_2dvec.at<double>(i, 0) = x2h[i](0);
        cv_x1_2dvec.at<double>(i, 1) = x2h[i](1);
    }
    cv::recoverPose(cv_E, cv_x0_2dvec, cv_x1_2dvec, cv::Mat_<float>::eye(3, 3), cv_R, cv_tr, 1e9);

    cv::cv2eigen(cv_R, R);
    cv::cv2eigen(cv_tr, t);
    output->emplace_back(PoseScaleOffsetTwoFocal(R, t, 1.0, 0.0, 0.0, f0, f1));
    return output->size();
}

std::vector<PoseScaleOffset> solve_scale_shift_pose_wrapper(const Eigen::Matrix3d &x_homo,
                                                            const Eigen::Matrix3d &y_homo,
                                                            const Eigen::Vector3d &depth_x,
                                                            const Eigen::Vector3d &depth_y) {
    std::vector<PoseScaleOffset> output;
    int sol_num = solve_scale_shift_pose(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

std::vector<PoseScaleOffsetSharedFocal> solve_scale_shift_pose_shared_focal_wrapper(const Eigen::Matrix3x4d &x_homo,
                                                                                    const Eigen::Matrix3x4d &y_homo,
                                                                                    const Eigen::Vector4d &depth_x,
                                                                                    const Eigen::Vector4d &depth_y) {
    std::vector<PoseScaleOffsetSharedFocal> output;
    int sol_num = solve_scale_shift_pose_shared_focal(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

std::vector<PoseScaleOffsetTwoFocal> solve_scale_shift_pose_two_focal_wrapper(const Eigen::Matrix3x4d &x_homo,
                                                                              const Eigen::Matrix3x4d &y_homo,
                                                                              const Eigen::Vector4d &depth_x,
                                                                              const Eigen::Vector4d &depth_y) {
    std::vector<PoseScaleOffsetTwoFocal> output;
    int sol_num = solve_scale_shift_pose_two_focal(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

}; // namespace madpose