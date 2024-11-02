#include "solver.h"

namespace acmpose {

std::vector<Eigen::Vector4d> solve_scale_and_shift(
    const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo, 
    const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y) {
    // X: 3 x 3, column vectors are homogeneous of 2D points
    // Y: 3 x 3, column vectors are homogeneous of 2D points
    Eigen::Matrix3d x1 = x_homo.transpose();
    Eigen::Matrix3d x2 = y_homo.transpose();
    const Eigen::Vector3d &d1 = depth_x;
    const Eigen::Vector3d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(18);

    coeffs[0] = 2*x2.row(0).dot(x2.row(1)) - x2.row(0).dot(x2.row(0)) - x2.row(1).dot(x2.row(1));
    coeffs[1] = x1.row(0).dot(x1.row(0)) + x1.row(1).dot(x1.row(1)) - 2*x1.row(0).dot(x1.row(1));
    coeffs[2] = 2*(d2[0]+d2[1])*x2.row(0).dot(x2.row(1)) - 2*d2[0]*x2.row(0).dot(x2.row(0)) - 2*d2[1]*x2.row(1).dot(x2.row(1));
    coeffs[3] = 2*d2[0]*d2[1]*x2.row(0).dot(x2.row(1)) - d2[0]*d2[0]*x2.row(0).dot(x2.row(0)) - d2[1]*d2[1]*x2.row(1).dot(x2.row(1));
    coeffs[4] = 2*d1[0]*x1.row(0).dot(x1.row(0)) + 2*d1[1]*x1.row(1).dot(x1.row(1)) - 2*(d1[0]+d1[1])*x1.row(0).dot(x1.row(1));
    coeffs[5] = d1[0]*d1[0]*x1.row(0).dot(x1.row(0)) + d1[1]*d1[1]*x1.row(1).dot(x1.row(1)) - 2*d1[0]*d1[1]*x1.row(0).dot(x1.row(1));
    coeffs[6] = 2*x2.row(0).dot(x2.row(2)) - x2.row(0).dot(x2.row(0)) - x2.row(2).dot(x2.row(2));
    coeffs[7] = x1.row(0).dot(x1.row(0)) + x1.row(2).dot(x1.row(2)) - 2*x1.row(0).dot(x1.row(2));
    coeffs[8] = 2*(d2[0]+d2[2])*x2.row(0).dot(x2.row(2)) - 2*d2[0]*x2.row(0).dot(x2.row(0)) - 2*d2[2]*x2.row(2).dot(x2.row(2));
    coeffs[9] = 2*d2[0]*d2[2]*x2.row(0).dot(x2.row(2)) - d2[0]*d2[0]*x2.row(0).dot(x2.row(0)) - d2[2]*d2[2]*x2.row(2).dot(x2.row(2));
    coeffs[10] = 2*d1[0]*x1.row(0).dot(x1.row(0)) + 2*d1[2]*x1.row(2).dot(x1.row(2)) - 2*(d1[0]+d1[2])*x1.row(0).dot(x1.row(2));
    coeffs[11] = d1[0]*d1[0]*x1.row(0).dot(x1.row(0)) + d1[2]*d1[2]*x1.row(2).dot(x1.row(2)) - 2*d1[0]*d1[2]*x1.row(0).dot(x1.row(2));
    coeffs[12] = 2*x2.row(1).dot(x2.row(2)) - x2.row(1).dot(x2.row(1)) - x2.row(2).dot(x2.row(2));
    coeffs[13] = x1.row(1).dot(x1.row(1)) + x1.row(2).dot(x1.row(2)) - 2*x1.row(1).dot(x1.row(2));
    coeffs[14] = 2*(d2[1]+d2[2])*x2.row(1).dot(x2.row(2)) - 2*d2[1]*x2.row(1).dot(x2.row(1)) - 2*d2[2]*x2.row(2).dot(x2.row(2));
    coeffs[15] = 2*d2[1]*d2[2]*x2.row(1).dot(x2.row(2)) - d2[1]*d2[1]*x2.row(1).dot(x2.row(1)) - d2[2]*d2[2]*x2.row(2).dot(x2.row(2));
    coeffs[16] = 2*d1[1]*x1.row(1).dot(x1.row(1)) + 2*d1[2]*x1.row(2).dot(x1.row(2)) - 2*(d1[1]+d1[2])*x1.row(1).dot(x1.row(2));
    coeffs[17] = d1[1]*d1[1]*x1.row(1).dot(x1.row(1)) + d1[2]*d1[2]*x1.row(2).dot(x1.row(2)) - 2*d1[1]*d1[2]*x1.row(1).dot(x1.row(2));

    const std::vector<int> coeff_ind0 = 
        {0,6,12,1,7,13,2,8,0,6,12,14,6,0,12,1,7,13,3,9,2,8,14,15,4,10,7,1,16,13,8,2,6,12,0,14,9,3,8,14,2,15,3,9,15,4,10,16,7,13,1,5,11,10,4,17,16};
    const std::vector<int> coeff_ind1 = {11,17,5,9,15,3,5,11,17,10,16,4,11,5,17};
    const std::vector<int> ind0 = 
        {0,1,9,12,13,21,24,25,26,28,29,33,39,42,47,50,52,53,60,61,62,64,65,69,72,73,75,78,81,83,87,90,91,92,94,95,99,102,103,104,106,107,110,112,113,122,124,125,127,128,130,132,133,135,138,141,143};
    const std::vector<int> ind1 = {7,8,10,19,20,22,26,28,29,31,32,34,39,42,47};
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
        if (D[i].imag() != 0) continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector4d sol;
        sol << 1.0, b1, a2, b2 * a2;
        solutions.push_back(sol);
    }

    return solutions;
}

std::vector<Eigen::Vector<double, 5>> solve_scale_and_shift_shared_focal(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo, 
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y) {
    
    Eigen::Matrix<double, 4, 3> x1 = x_homo.transpose(); 
    Eigen::Matrix<double, 4, 3> x2 = y_homo.transpose();

    double f1_0 = x1.block<4,2>(0,0).cwiseAbs().mean();
    double f2_0 = x2.block<4,2>(0,0).cwiseAbs().mean();
    double f0 = 0.5 * (f1_0 + f2_0);
    x1.block<4,2>(0,0) /= f0;
    x2.block<4,2>(0,0) /= f0;
    
    const Eigen::Vector4d &d1 = depth_x;
    const Eigen::Vector4d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(32);

    coeffs[0] = 2*x2(0,0)*x2(1,0) + 2*x2(0,1)*x2(1,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(1,0)*x2(1,0) - x2(1,1)*x2(1,1);
    coeffs[1] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(1,1) - 2*x1(0,0)*x1(1,0) + x1(0,1)*x1(0,1) + x1(1,0)*x1(1,0) + x1(1,1)*x1(1,1);
    coeffs[2] = 2*d2(0)*x2(0,0)*x2(1,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(1)*x2(1,0)*x2(1,0) - 2*d2(1)*x2(1,1)*x2(1,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(1)*x2(0,0)*x2(1,0) + 2*d2(0)*x2(0,1)*x2(1,1) + 2*d2(1)*x2(0,1)*x2(1,1);
    coeffs[3] = 2*d2(0)*d2(1)*x2(0,0)*x2(1,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(1)*d2(1)*x2(1,0)*x2(1,0) - d2(1)*d2(1)*x2(1,1)*x2(1,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(1)*x2(0,1)*x2(1,1);
    coeffs[4] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(1)*x1(1,0)*x1(1,0) + 2*d1(1)*x1(1,1)*x1(1,1) - 2*d1(0)*x1(0,0)*x1(1,0) - 2*d1(1)*x1(0,0)*x1(1,0) - 2*d1(0)*x1(0,1)*x1(1,1) - 2*d1(1)*x1(0,1)*x1(1,1);
    coeffs[5] = 2*d2(0)*d2(1) - d2(0)*d2(0) - d2(1)*d2(1);
    coeffs[6] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(1)*d1(1)*x1(1,0)*x1(1,0) + d1(1)*d1(1)*x1(1,1)*x1(1,1) - 2*d1(0)*d1(1)*x1(0,0)*x1(1,0) - 2*d1(0)*d1(1)*x1(0,1)*x1(1,1);
    coeffs[7] = d1(0)*d1(0) - 2*d1(0)*d1(1) + d1(1)*d1(1);
    coeffs[8] = 2*x2(0,0)*x2(2,0) + 2*x2(0,1)*x2(2,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(2,0)*x2(2,0) - x2(2,1)*x2(2,1);
    coeffs[9] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(2,1) - 2*x1(0,0)*x1(2,0) + x1(0,1)*x1(0,1) + x1(2,0)*x1(2,0) + x1(2,1)*x1(2,1);
    coeffs[10] = 2*d2(0)*x2(0,0)*x2(2,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(2)*x2(2,0)*x2(2,0) - 2*d2(2)*x2(2,1)*x2(2,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*x2(0,1)*x2(2,1) + 2*d2(2)*x2(0,0)*x2(2,0) + 2*d2(2)*x2(0,1)*x2(2,1);
    coeffs[11] = 2*d2(0)*d2(2)*x2(0,0)*x2(2,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(2)*d2(2)*x2(2,0)*x2(2,0) - d2(2)*d2(2)*x2(2,1)*x2(2,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(2)*x2(0,1)*x2(2,1);
    coeffs[12] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(2)*x1(2,0)*x1(2,0) + 2*d1(2)*x1(2,1)*x1(2,1) - 2*d1(0)*x1(0,0)*x1(2,0) - 2*d1(0)*x1(0,1)*x1(2,1) - 2*d1(2)*x1(0,0)*x1(2,0) - 2*d1(2)*x1(0,1)*x1(2,1);
    coeffs[13] = 2*d2(0)*d2(2) - d2(0)*d2(0) - d2(2)*d2(2);
    coeffs[14] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(2)*d1(2)*x1(2,0)*x1(2,0) + d1(2)*d1(2)*x1(2,1)*x1(2,1) - 2*d1(0)*d1(2)*x1(0,0)*x1(2,0) - 2*d1(0)*d1(2)*x1(0,1)*x1(2,1);
    coeffs[15] = d1(0)*d1(0) - 2*d1(0)*d1(2) + d1(2)*d1(2);
    coeffs[16] = 2*x2(1,0)*x2(2,0) + 2*x2(1,1)*x2(2,1) - x2(1,0)*x2(1,0) - x2(1,1)*x2(1,1) - x2(2,0)*x2(2,0) - x2(2,1)*x2(2,1);
    coeffs[17] = x1(1,0)*x1(1,0) - 2*x1(1,1)*x1(2,1) - 2*x1(1,0)*x1(2,0) + x1(1,1)*x1(1,1) + x1(2,0)*x1(2,0) + x1(2,1)*x1(2,1);
    coeffs[18] = 2*d2(1)*x2(1,0)*x2(2,0) - 2*d2(1)*x2(1,1)*x2(1,1) - 2*d2(2)*x2(2,0)*x2(2,0) - 2*d2(2)*x2(2,1)*x2(2,1) - 2*d2(1)*x2(1,0)*x2(1,0) + 2*d2(2)*x2(1,0)*x2(2,0) + 2*d2(1)*x2(1,1)*x2(2,1) + 2*d2(2)*x2(1,1)*x2(2,1);
    coeffs[19] = 2*d2(1)*d2(2)*x2(1,0)*x2(2,0) - d2(1)*d2(1)*x2(1,1)*x2(1,1) - d2(2)*d2(2)*x2(2,0)*x2(2,0) - d2(2)*d2(2)*x2(2,1)*x2(2,1) - d2(1)*d2(1)*x2(1,0)*x2(1,0) + 2*d2(1)*d2(2)*x2(1,1)*x2(2,1);
    coeffs[20] = 2*d1(1)*x1(1,0)*x1(1,0) + 2*d1(1)*x1(1,1)*x1(1,1) + 2*d1(2)*x1(2,0)*x1(2,0) + 2*d1(2)*x1(2,1)*x1(2,1) - 2*d1(1)*x1(1,0)*x1(2,0) - 2*d1(2)*x1(1,0)*x1(2,0) - 2*d1(1)*x1(1,1)*x1(2,1) - 2*d1(2)*x1(1,1)*x1(2,1);
    coeffs[21] = 2*d2(1)*d2(2) - d2(1)*d2(1) - d2(2)*d2(2);
    coeffs[22] = d1(1)*d1(1)*x1(1,0)*x1(1,0) + d1(1)*d1(1)*x1(1,1)*x1(1,1) + d1(2)*d1(2)*x1(2,0)*x1(2,0) + d1(2)*d1(2)*x1(2,1)*x1(2,1) - 2*d1(1)*d1(2)*x1(1,0)*x1(2,0) - 2*d1(1)*d1(2)*x1(1,1)*x1(2,1);
    coeffs[23] = d1(1)*d1(1) - 2*d1(1)*d1(2) + d1(2)*d1(2);
    coeffs[24] = 2*x2(0,0)*x2(3,0) + 2*x2(0,1)*x2(3,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(3,0)*x2(3,0) - x2(3,1)*x2(3,1);
    coeffs[25] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(3,1) - 2*x1(0,0)*x1(3,0) + x1(0,1)*x1(0,1) + x1(3,0)*x1(3,0) + x1(3,1)*x1(3,1);
    coeffs[26] = 2*d2(0)*x2(0,0)*x2(3,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(3)*x2(3,0)*x2(3,0) - 2*d2(3)*x2(3,1)*x2(3,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*x2(0,1)*x2(3,1) + 2*d2(3)*x2(0,0)*x2(3,0) + 2*d2(3)*x2(0,1)*x2(3,1);
    coeffs[27] = 2*d2(0)*d2(3)*x2(0,0)*x2(3,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(3)*d2(3)*x2(3,0)*x2(3,0) - d2(3)*d2(3)*x2(3,1)*x2(3,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(3)*x2(0,1)*x2(3,1);
    coeffs[28] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(3)*x1(3,0)*x1(3,0) + 2*d1(3)*x1(3,1)*x1(3,1) - 2*d1(0)*x1(0,0)*x1(3,0) - 2*d1(0)*x1(0,1)*x1(3,1) - 2*d1(3)*x1(0,0)*x1(3,0) - 2*d1(3)*x1(0,1)*x1(3,1);
    coeffs[29] = 2*d2(0)*d2(3) - d2(0)*d2(0) - d2(3)*d2(3);
    coeffs[30] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(3)*d1(3)*x1(3,0)*x1(3,0) + d1(3)*d1(3)*x1(3,1)*x1(3,1) - 2*d1(0)*d1(3)*x1(0,0)*x1(3,0) - 2*d1(0)*d1(3)*x1(0,1)*x1(3,1);
    coeffs[31] = d1(0)*d1(0) - 2*d1(0)*d1(3) + d1(3)*d1(3);

    const std::vector<int> coeff_ind0 = 
        {0,8,16,24,1,9,17,25,2,10,0,8,16,18,24,26,0,8,16,24,0,8,16,24,0,8,16,24,1,9,17,25,1,9,17,25,3,11,2,10,18,19,26,27,4,12,2,1,10,9,20,18,17,28,26,25,1,9,17,25,2,10,8,0,16,18,24,26,2,10,0,16,18,8,24,26,8,0,16,
         24,5,13,21,29,3,11,19,27,4,3,12,11,20,9,28,1,19,17,27,25,4,12,1,17,20,9,25,28,3,11,10,2,18,19,26,27,6,14,4,3,12,11,2,22,18,19,20,10,26,30,28,27,4,12,9,20,1,17,25,28,10,2,18,0,16,24,26,8,5,13,21,29,11,3,
         19,27,6,14,22,12,3,30,4,19,20,11,27,28,6,14,4,20,22,12,1,17,25,28,30,9,6,14,11,3,19,22,2,18,26,27,30,10,6,14,12,22,4,20,28,30,14,6,22,3,19,27,30,11,14,6,22,30,5,13,21,29,6,22,14,4,20,28,30,12,7,15,23,31,
         5,13,21,29,7,15,23,31,7,15,5,13,23,21,31,29};
    const std::vector<int> coeff_ind1 = {7,23,31,15,6,22,30,14,7,23,15,31,7,15,23,5,31,21,13,29,15,7,23,31,7,15,13,5,21,23,29,31,15,7,23,5,21,29,31,13,13,5,21,29};
    const std::vector<int> ind0 = 
        {0,1,14,30,36,37,50,66,72,73,74,78,80,86,87,102,111,115,126,139,148,153,167,177,185,190,199,215,218,222,224,231,255,259,270,283,288,289,290,294,296,302,303,318,324,325,327,328,331,333,338,342,347,354,355,
         357,365,370,379,395,400,405,407,412,418,419,428,429,437,442,444,449,451,456,461,467,481,488,489,496,504,505,518,534,542,546,548,555,578,579,582,583,584,587,591,592,594,598,607,608,615,619,624,629,630,636,
         641,643,652,657,659,664,670,671,680,681,684,685,688,689,693,694,696,698,701,703,707,708,713,714,717,719,725,730,733,739,740,741,748,755,769,776,777,781,782,783,784,790,796,801,815,825,839,844,850,860,866,
         870,872,875,876,879,880,881,886,888,893,896,903,907,912,917,918,924,925,926,927,929,931,934,940,945,949,956,957,959,961,962,963,964,969,970,977,982,985,991,992,993,1000,1007,1019,1024,1030,1033,1034,1035,
         1040,1042,1057,1064,1065,1072,1082,1086,1088,1095,1128,1133,1140,1141,1142,1143,1145,1150,1155,1159,1170,1183,1191,1195,1206,1219,1229,1234,1243,1259,1260,1261,1265,1270,1274,1279,1290,1295};
    const std::vector<int> ind1 = {25,26,27,34,61,62,63,70,84,89,96,101,110,114,116,120,123,125,132,137,157,164,165,172,184,189,193,200,201,203,208,213,227,232,238,241,242,243,248,250,263,268,274,284};
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
    AM << Eigen::RowVector<double, 8>(0, 0, 1, 0, 0, 0, 0, 0), 
          -C2.row(31), -C2.row(32), -C2.row(33), -C2.row(34), -C2.row(35), 
          Eigen::RowVector<double, 8>(0, 0, 0, 1, 0, 0, 0, 0), -C2.row(30);

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
        if (D[i].imag() != 0) continue;
        if (sols(i, 3).real() < 0) continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector<double, 5> sol;
        sol << 1.0, b1, a2, b2 * a2, f0 / std::sqrt(sols(i, 3).real());
        solutions.push_back(sol);
    }

    return solutions;
}

std::vector<Eigen::Vector<double, 6>> solve_scale_and_shift_two_focal(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo, 
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y) {
    
    Eigen::Matrix<double, 4, 3> x1 = x_homo.transpose(); 
    Eigen::Matrix<double, 4, 3> x2 = y_homo.transpose();

    double f1_0 = x1.block<4,2>(0,0).cwiseAbs().mean();
    double f2_0 = x2.block<4,2>(0,0).cwiseAbs().mean();
    x1.block<4,2>(0,0) /= f1_0;
    x2.block<4,2>(0,0) /= f2_0;
    
    const Eigen::Vector4d &d1 = depth_x;
    const Eigen::Vector4d &d2 = depth_y;
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(40);

    coeffs[0] = 2*x2(0,0)*x2(1,0) + 2*x2(0,1)*x2(1,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(1,0)*x2(1,0) - x2(1,1)*x2(1,1);
    coeffs[1] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(1,1) - 2*x1(0,0)*x1(1,0) + x1(0,1)*x1(0,1) + x1(1,0)*x1(1,0) + x1(1,1)*x1(1,1);
    coeffs[2] = 2*d2(0)*x2(0,0)*x2(1,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(1)*x2(1,0)*x2(1,0) - 2*d2(1)*x2(1,1)*x2(1,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(1)*x2(0,0)*x2(1,0) + 2*d2(0)*x2(0,1)*x2(1,1) + 2*d2(1)*x2(0,1)*x2(1,1);
    coeffs[3] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(1)*x1(1,0)*x1(1,0) + 2*d1(1)*x1(1,1)*x1(1,1) - 2*d1(0)*x1(0,0)*x1(1,0) - 2*d1(1)*x1(0,0)*x1(1,0) - 2*d1(0)*x1(0,1)*x1(1,1) - 2*d1(1)*x1(0,1)*x1(1,1);
    coeffs[4] = 2*d2(0)*d2(1)*x2(0,0)*x2(1,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(1)*d2(1)*x2(1,0)*x2(1,0) - d2(1)*d2(1)*x2(1,1)*x2(1,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(1)*x2(0,1)*x2(1,1);
    coeffs[5] = 2*d2(0)*d2(1) - d2(0)*d2(0) - d2(1)*d2(1);
    coeffs[6] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(1)*d1(1)*x1(1,0)*x1(1,0) + d1(1)*d1(1)*x1(1,1)*x1(1,1) - 2*d1(0)*d1(1)*x1(0,0)*x1(1,0) - 2*d1(0)*d1(1)*x1(0,1)*x1(1,1);
    coeffs[7] = d1(0)*d1(0) - 2*d1(0)*d1(1) + d1(1)*d1(1);
    coeffs[8] = 2*x2(0,0)*x2(2,0) + 2*x2(0,1)*x2(2,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(2,0)*x2(2,0) - x2(2,1)*x2(2,1);
    coeffs[9] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(2,1) - 2*x1(0,0)*x1(2,0) + x1(0,1)*x1(0,1) + x1(2,0)*x1(2,0) + x1(2,1)*x1(2,1);
    coeffs[10] = 2*d2(0)*x2(0,0)*x2(2,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(2)*x2(2,0)*x2(2,0) - 2*d2(2)*x2(2,1)*x2(2,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*x2(0,1)*x2(2,1) + 2*d2(2)*x2(0,0)*x2(2,0) + 2*d2(2)*x2(0,1)*x2(2,1);
    coeffs[11] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(2)*x1(2,0)*x1(2,0) + 2*d1(2)*x1(2,1)*x1(2,1) - 2*d1(0)*x1(0,0)*x1(2,0) - 2*d1(0)*x1(0,1)*x1(2,1) - 2*d1(2)*x1(0,0)*x1(2,0) - 2*d1(2)*x1(0,1)*x1(2,1);
    coeffs[12] = 2*d2(0)*d2(2)*x2(0,0)*x2(2,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(2)*d2(2)*x2(2,0)*x2(2,0) - d2(2)*d2(2)*x2(2,1)*x2(2,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(2)*x2(0,1)*x2(2,1);
    coeffs[13] = 2*d2(0)*d2(2) - d2(0)*d2(0) - d2(2)*d2(2);
    coeffs[14] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(2)*d1(2)*x1(2,0)*x1(2,0) + d1(2)*d1(2)*x1(2,1)*x1(2,1) - 2*d1(0)*d1(2)*x1(0,0)*x1(2,0) - 2*d1(0)*d1(2)*x1(0,1)*x1(2,1);
    coeffs[15] = d1(0)*d1(0) - 2*d1(0)*d1(2) + d1(2)*d1(2);
    coeffs[16] = 2*x2(1,0)*x2(2,0) + 2*x2(1,1)*x2(2,1) - x2(1,0)*x2(1,0) - x2(1,1)*x2(1,1) - x2(2,0)*x2(2,0) - x2(2,1)*x2(2,1);
    coeffs[17] = x1(1,0)*x1(1,0) - 2*x1(1,1)*x1(2,1) - 2*x1(1,0)*x1(2,0) + x1(1,1)*x1(1,1) + x1(2,0)*x1(2,0) + x1(2,1)*x1(2,1);
    coeffs[18] = 2*d2(1)*x2(1,0)*x2(2,0) - 2*d2(1)*x2(1,1)*x2(1,1) - 2*d2(2)*x2(2,0)*x2(2,0) - 2*d2(2)*x2(2,1)*x2(2,1) - 2*d2(1)*x2(1,0)*x2(1,0) + 2*d2(2)*x2(1,0)*x2(2,0) + 2*d2(1)*x2(1,1)*x2(2,1) + 2*d2(2)*x2(1,1)*x2(2,1);
    coeffs[19] = 2*d1(1)*x1(1,0)*x1(1,0) + 2*d1(1)*x1(1,1)*x1(1,1) + 2*d1(2)*x1(2,0)*x1(2,0) + 2*d1(2)*x1(2,1)*x1(2,1) - 2*d1(1)*x1(1,0)*x1(2,0) - 2*d1(2)*x1(1,0)*x1(2,0) - 2*d1(1)*x1(1,1)*x1(2,1) - 2*d1(2)*x1(1,1)*x1(2,1);
    coeffs[20] = 2*d2(1)*d2(2)*x2(1,0)*x2(2,0) - d2(1)*d2(1)*x2(1,1)*x2(1,1) - d2(2)*d2(2)*x2(2,0)*x2(2,0) - d2(2)*d2(2)*x2(2,1)*x2(2,1) - d2(1)*d2(1)*x2(1,0)*x2(1,0) + 2*d2(1)*d2(2)*x2(1,1)*x2(2,1);
    coeffs[21] = 2*d2(1)*d2(2) - d2(1)*d2(1) - d2(2)*d2(2);
    coeffs[22] = d1(1)*d1(1)*x1(1,0)*x1(1,0) + d1(1)*d1(1)*x1(1,1)*x1(1,1) + d1(2)*d1(2)*x1(2,0)*x1(2,0) + d1(2)*d1(2)*x1(2,1)*x1(2,1) - 2*d1(1)*d1(2)*x1(1,0)*x1(2,0) - 2*d1(1)*d1(2)*x1(1,1)*x1(2,1);
    coeffs[23] = d1(1)*d1(1) - 2*d1(1)*d1(2) + d1(2)*d1(2);
    coeffs[24] = 2*x2(0,0)*x2(3,0) + 2*x2(0,1)*x2(3,1) - x2(0,0)*x2(0,0) - x2(0,1)*x2(0,1) - x2(3,0)*x2(3,0) - x2(3,1)*x2(3,1);
    coeffs[25] = x1(0,0)*x1(0,0) - 2*x1(0,1)*x1(3,1) - 2*x1(0,0)*x1(3,0) + x1(0,1)*x1(0,1) + x1(3,0)*x1(3,0) + x1(3,1)*x1(3,1);
    coeffs[26] = 2*d2(0)*x2(0,0)*x2(3,0) - 2*d2(0)*x2(0,1)*x2(0,1) - 2*d2(3)*x2(3,0)*x2(3,0) - 2*d2(3)*x2(3,1)*x2(3,1) - 2*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*x2(0,1)*x2(3,1) + 2*d2(3)*x2(0,0)*x2(3,0) + 2*d2(3)*x2(0,1)*x2(3,1);
    coeffs[27] = 2*d1(0)*x1(0,0)*x1(0,0) + 2*d1(0)*x1(0,1)*x1(0,1) + 2*d1(3)*x1(3,0)*x1(3,0) + 2*d1(3)*x1(3,1)*x1(3,1) - 2*d1(0)*x1(0,0)*x1(3,0) - 2*d1(0)*x1(0,1)*x1(3,1) - 2*d1(3)*x1(0,0)*x1(3,0) - 2*d1(3)*x1(0,1)*x1(3,1);
    coeffs[28] = 2*d2(0)*d2(3)*x2(0,0)*x2(3,0) - d2(0)*d2(0)*x2(0,1)*x2(0,1) - d2(3)*d2(3)*x2(3,0)*x2(3,0) - d2(3)*d2(3)*x2(3,1)*x2(3,1) - d2(0)*d2(0)*x2(0,0)*x2(0,0) + 2*d2(0)*d2(3)*x2(0,1)*x2(3,1);
    coeffs[29] = 2*d2(0)*d2(3) - d2(0)*d2(0) - d2(3)*d2(3);
    coeffs[30] = d1(0)*d1(0)*x1(0,0)*x1(0,0) + d1(0)*d1(0)*x1(0,1)*x1(0,1) + d1(3)*d1(3)*x1(3,0)*x1(3,0) + d1(3)*d1(3)*x1(3,1)*x1(3,1) - 2*d1(0)*d1(3)*x1(0,0)*x1(3,0) - 2*d1(0)*d1(3)*x1(0,1)*x1(3,1);
    coeffs[31] = d1(0)*d1(0) - 2*d1(0)*d1(3) + d1(3)*d1(3);
    coeffs[32] = 2*x2(1,0)*x2(3,0) + 2*x2(1,1)*x2(3,1) - x2(1,0)*x2(1,0) - x2(1,1)*x2(1,1) - x2(3,0)*x2(3,0) - x2(3,1)*x2(3,1);
    coeffs[33] = x1(1,0)*x1(1,0) - 2*x1(1,1)*x1(3,1) - 2*x1(1,0)*x1(3,0) + x1(1,1)*x1(1,1) + x1(3,0)*x1(3,0) + x1(3,1)*x1(3,1);
    coeffs[34] = 2*d2(1)*x2(1,0)*x2(3,0) - 2*d2(1)*x2(1,1)*x2(1,1) - 2*d2(3)*x2(3,0)*x2(3,0) - 2*d2(3)*x2(3,1)*x2(3,1) - 2*d2(1)*x2(1,0)*x2(1,0) + 2*d2(1)*x2(1,1)*x2(3,1) + 2*d2(3)*x2(1,0)*x2(3,0) + 2*d2(3)*x2(1,1)*x2(3,1);
    coeffs[35] = 2*d1(1)*x1(1,0)*x1(1,0) + 2*d1(1)*x1(1,1)*x1(1,1) + 2*d1(3)*x1(3,0)*x1(3,0) + 2*d1(3)*x1(3,1)*x1(3,1) - 2*d1(1)*x1(1,0)*x1(3,0) - 2*d1(1)*x1(1,1)*x1(3,1) - 2*d1(3)*x1(1,0)*x1(3,0) - 2*d1(3)*x1(1,1)*x1(3,1);
    coeffs[36] = 2*d2(1)*d2(3)*x2(1,0)*x2(3,0) - d2(1)*d2(1)*x2(1,1)*x2(1,1) - d2(3)*d2(3)*x2(3,0)*x2(3,0) - d2(3)*d2(3)*x2(3,1)*x2(3,1) - d2(1)*d2(1)*x2(1,0)*x2(1,0) + 2*d2(1)*d2(3)*x2(1,1)*x2(3,1);
    coeffs[37] = 2*d2(1)*d2(3) - d2(1)*d2(1) - d2(3)*d2(3);
    coeffs[38] = d1(1)*d1(1)*x1(1,0)*x1(1,0) + d1(1)*d1(1)*x1(1,1)*x1(1,1) + d1(3)*d1(3)*x1(3,0)*x1(3,0) + d1(3)*d1(3)*x1(3,1)*x1(3,1) - 2*d1(1)*d1(3)*x1(1,0)*x1(3,0) - 2*d1(1)*d1(3)*x1(1,1)*x1(3,1);
    coeffs[39] = d1(1)*d1(1) - 2*d1(1)*d1(3) + d1(3)*d1(3);

    const std::vector<int> coeff_ind0 = 
        {0,8,16,24,32,0,8,16,24,32,1,9,17,25,33,2,10,0,8,16,18,24,26,32,34,0,8,16,24,32,1,9,17,25,33,2,10,8,0,16,18,24,26,32,34,8,0,16,24,32,1,9,17,25,33,3,11,1,9,17,19,25,27,33,35,4,12,2,10,18,20,26,28,34,36,2,10,8,16,18,0,
         26,24,32,34,9,1,17,25,33,3,11,9,1,19,17,27,25,33,35,5,4,13,12,10,2,18,21,20,26,29,28,34,36,37,10,2,8,0,18,24,26,32,34,16,3,11,19,9,17,1,27,25,33,35,6,14,3,11,19,22,27,30,35,38,4,12,20,28,36,4,12,10,18,20,2,28,26,34,
         36,5,13,21,29,37,11,3,19,9,1,27,25,33,17,35,6,14,11,3,22,19,30,27,35,38,5,12,13,21,4,20,28,29,36,37,5,12,13,4,10,21,2,20,26,29,28,34,36,37,18,7,15,23,31,39,6,14,22,11,19,3,30,27,35,38,6,14,22,30,38,12,20,4,28,36,13,
         5,21,29,37,13,5,21,29,37,14,6,22,30,38,13,12,21,5,4,28,29,37,36,20,7,15,23,31,39,14,22,6,30,38,13,5,29,37,21,15,7,23,31,39,7,15,23,31,39,14,6,22,11,3,30,27,35,19,38,7,15,23,31,39};
    const std::vector<int> coeff_ind1 = {15,7,31,39,23,15,7,23,31,39,14,6,30,38,22,15,23,7,31,39};
    const std::vector<int> ind0 = 
        {0,2,18,28,39,41,45,60,69,78,80,82,98,108,119,120,122,123,128,130,138,145,148,157,159,164,169,177,186,194,201,205,220,229,238,241,245,246,252,254,260,263,269,276,278,287,293,302,310,313,323,328,330,345,357,360,362,364,
         369,377,378,386,388,394,399,400,402,403,408,410,418,425,428,437,439,444,449,451,456,457,459,466,467,471,474,486,492,494,503,516,521,525,527,533,540,542,549,550,553,558,560,561,562,565,566,572,574,578,580,583,588,589,
         596,598,599,607,613,615,621,622,624,630,632,633,635,643,648,650,651,656,659,665,667,671,677,680,682,684,689,697,698,706,708,714,719,723,728,730,745,757,764,769,771,776,777,779,786,787,791,794,801,805,820,829,838,846,
         852,854,855,861,863,864,872,875,876,881,885,887,893,900,902,909,910,913,918,923,926,928,930,932,934,943,945,956,957,964,967,969,973,975,977,981,982,984,986,990,992,993,994,995,1000,1002,1018,1028,1039,1043,1048,1050,
         1051,1056,1059,1065,1067,1071,1077,1084,1089,1097,1106,1114,1131,1136,1139,1147,1151,1166,1172,1174,1183,1196,1207,1213,1222,1230,1233,1247,1253,1262,1270,1273,1291,1295,1296,1299,1301,1304,1307,1311,1312,1315,1324,
         1329,1337,1346,1354,1371,1376,1379,1387,1391,1415,1421,1424,1432,1435,1446,1452,1454,1463,1476,1481,1485,1500,1509,1518,1526,1532,1534,1535,1541,1543,1544,1552,1555,1556,1563,1568,1570,1585,1597};
    const std::vector<int> ind1 = {15,21,24,32,35,47,53,62,70,73,95,101,104,112,115,131,136,139,147,151};
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
        if (D[i].imag() != 0) continue;
        if (sols(i, 3).real() < 0 || sols(i, 4).real() < 0) continue;
        double a2 = std::sqrt(sols(i, 0).real());
        double b1 = sols(i, 1).real(), b2 = sols(i, 2).real();
        Eigen::Vector<double, 6> sol;
        sol << 1.0, b1, a2, b2 * a2, f1_0 / std::sqrt(sols(i, 3).real()), f2_0 / std::sqrt(sols(i, 4).real());
        solutions.push_back(sol);
    }

    return solutions;
}

int estimate_scale_shift_pose(
    const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
    const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y, std::vector<PoseScaleOffset> *output, bool scale_on_x) {
    // X: 3 x 3, column vectors are homogeneous 2D points
    // Y: 3 x 3, column vectors are homogeneous 2D points
    std::vector<Eigen::Vector4d> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol: solutions) {
        Eigen::Vector3d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        }
        else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0) continue;

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
        if (!scale_on_x) std::swap(b1, b2);
        output->push_back(PoseScaleOffset(R, t, a1, b1, b2));
        sol_count++;
    }   
    return sol_count;
}

int estimate_scale_shift_pose_shared_focal(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y, 
    std::vector<PoseScaleOffsetSharedFocal> *output, bool scale_on_x) {
    std::vector<Eigen::Vector<double, 5>> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift_shared_focal(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift_shared_focal(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol: solutions) {
        Eigen::Vector4d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        }
        else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0) continue;

        double focal = sol(4);
        Eigen::Matrix3x4d xu = x_homo;
        Eigen::Matrix3x4d yu = y_homo;
        xu.block<2,4>(0,0) /= focal;
        yu.block<2,4>(0,0) /= focal;

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
        if (!scale_on_x) std::swap(b1, b2);
        output->push_back(PoseScaleOffsetSharedFocal(R, t, a1, b1, b2, f));
        sol_count++;
    }   
    return sol_count;
}

int estimate_scale_shift_pose_two_focal(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y, 
    std::vector<PoseScaleOffsetTwoFocal> *output, bool scale_on_x) {
    std::vector<Eigen::Vector<double, 6>> solutions;
    if (scale_on_x)
        solutions = solve_scale_and_shift_two_focal(y_homo, x_homo, depth_y, depth_x);
    else
        solutions = solve_scale_and_shift_two_focal(x_homo, y_homo, depth_x, depth_y);
    output->clear();

    int sol_count = 0;
    for (auto &sol: solutions) {
        Eigen::Vector4d d1, d2;
        if (scale_on_x) {
            d1 = depth_x.array() * sol(2) + sol(3);
            d2 = depth_y.array() + sol(1);
        }
        else {
            d1 = depth_x.array() + sol(1);
            d2 = depth_y.array() * sol(2) + sol(3);
        }
        if (d1.minCoeff() <= 0 || d2.minCoeff() <= 0) continue;

        double focal1 = sol(4), focal2 = sol(5);
        Eigen::Matrix3x4d xu = x_homo;
        Eigen::Matrix3x4d yu = y_homo;
        xu.block<2,4>(0,0) /= focal1;
        yu.block<2,4>(0,0) /= focal2;

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
        if (!scale_on_x) std::swap(b1, b2);
        output->push_back(PoseScaleOffsetTwoFocal(R, t, a1, b1, b2, f1, f2));
        sol_count++;
    }   
    return sol_count;
}

std::vector<PoseScaleOffset> estimate_scale_shift_pose_wrapper(
    const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
    const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y) {
    std::vector<PoseScaleOffset> output;
    int sol_num = estimate_scale_shift_pose(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

std::vector<PoseScaleOffsetSharedFocal> estimate_scale_shift_pose_shared_focal_wrapper(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y) {
    std::vector<PoseScaleOffsetSharedFocal> output;
    int sol_num = estimate_scale_shift_pose_shared_focal(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

std::vector<PoseScaleOffsetTwoFocal> estimate_scale_shift_pose_two_focal_wrapper(
    const Eigen::Matrix3x4d &x_homo, const Eigen::Matrix3x4d &y_homo,
    const Eigen::Vector4d &depth_x, const Eigen::Vector4d &depth_y) {
    std::vector<PoseScaleOffsetTwoFocal> output;
    int sol_num = estimate_scale_shift_pose_two_focal(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

int estimate_scale_and_pose_with_offset_3pts(
    const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
    const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y,
    std::vector<PoseScaleOffset> *output) {
    // X: 3 x 3, column vectors are normalized 2D points
    // Y: 3 x 3, column vectors are normalized 2D points
    Eigen::Matrix<double, 3, 10> coeffs;

    Eigen::Matrix3d X = x_homo.array().rowwise() * depth_x.transpose().array();
    Eigen::Matrix3d Y = y_homo.array().rowwise() * depth_y.transpose().array();

    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        const Eigen::Vector3d &x1 = x_homo.col(i);
        const Eigen::Vector3d &y1 = y_homo.col(i);
        const Eigen::Vector3d &x2 = x_homo.col(j);
        const Eigen::Vector3d &y2 = y_homo.col(j);
        const double &x1d = depth_x(i), &x2d = depth_x(j);
        const double &y1d = depth_y(i), &y2d = depth_y(j);

        const double &x1u = x1(0), &x1v = x1(1), &x1w = x1(2);
        const double &x2u = x2(0), &x2v = x2(1), &x2w = x2(2);
        const double &y1u = y1(0), &y1v = y1(1), &y1w = y1(2);
        const double &y2u = y2(0), &y2v = y2(1), &y2w = y2(2);

        Eigen::RowVector<double, 10> coeff;
        // double coeff_a_2 =  x1u * x1u * x1d * x1d + x2u * x2u * x2d * x2d - 2 * x1u * x2u * x1d * x2d +
        //                     x1v * x1v * x1d * x1d + x2v * x2v * x2d * x2d - 2 * x1v * x2v * x1d * x2d +
        //                     x1w * x1w * x1d * x1d + x2w * x2w * x2d * x2d - 2 * x1w * x2w * x1d * x2d;
        // double coeff_ab  = (x1u * x1u * x1d + x2u * x2u * x2d - x1u * x2u * (x1d + x2d) + 
        //                     x1v * x1v * x1d + x2v * x2v * x2d - x1v * x2v * (x1d + x2d) +
        //                     x1w * x1w * x1d + x2w * x2w * x2d - x1w * x2w * (x1d + x2d)) * 2;
        // double coeff_b_2 =  x1u * x1u + x2u * x2u - 2 * x1u * x2u +
        //                     x1v * x1v + x2v * x2v - 2 * x1v * x2v +
        //                     x1w * x1w + x2w * x2w - 2 * x1w * x2w;
        // double coeff_c_2 = (y1u * y1u + y2u * y2u - 2 * y1u * y2u +
        //                     y1v * y1v + y2v * y2v - 2 * y1v * y2v +
        //                     y1w * y1w + y2w * y2w - 2 * y1w * y2w) * -1; 
        // double coeff_c   = (y1u * y1u * y1d + y2u * y2u * y2d - y1u * y2u * (y1d + y2d) +
        //                     y1v * y1v * y1d + y2v * y2v * y2d - y1v * y2v * (y1d + y2d) +
        //                     y1w * y1w * y1d + y2w * y2w * y2d - y1w * y2w * (y1d + y2d)) * -2;
        // double coeff_1   = (y1u * y1u * y1d * y1d + y2u * y2u * y2d * y2d - 2 * y1u * y2u * y1d * y2d +
        //                     y1v * y1v * y1d * y1d + y2v * y2v * y2d * y2d - 2 * y1v * y2v * y1d * y2d +
        //                     y1w * y1w * y1d * y1d + y2w * y2w * y2d * y2d - 2 * y1w * y2w * y1d * y2d) * -1;

        double coeff_a_2 =  (x1u * x1d - x2u * x2d) * (x1u * x1d - x2u * x2d) +
                            (x1v * x1d - x2v * x2d) * (x1v * x1d - x2v * x2d) +
                            (x1w * x1d - x2w * x2d) * (x1w * x1d - x2w * x2d);
        double coeff_ab  = ((x1u * x1d - x2u * x2d) * (x1u - x2u) +
                            (x1v * x1d - x2v * x2d) * (x1v - x2v) +
                            (x1w * x1d - x2w * x2d) * (x1w - x2w)) * 2;
        double coeff_b_2 =  (x1u - x2u) * (x1u - x2u) +
                            (x1v - x2v) * (x1v - x2v) +
                            (x1w - x2w) * (x1w - x2w);
        double coeff_c_2 = ((y1u - y2u) * (y1u - y2u) +
                            (y1v - y2v) * (y1v - y2v) +
                            (y1w - y2w) * (y1w - y2w)) * -1;
        double coeff_c   = ((y1u * y1d - y2u * y2d) * (y1u - y2u) +
                            (y1v * y1d - y2v * y2d) * (y1v - y2v) +
                            (y1w * y1d - y2w * y2d) * (y1w - y2w)) * -2;
        double coeff_1   = ((y1u * y1d - y2u * y2d) * (y1u * y1d - y2u * y2d) +
                            (y1v * y1d - y2v * y2d) * (y1v * y1d - y2v * y2d) +
                            (y1w * y1d - y2w * y2d) * (y1w * y1d - y2w * y2d)) * -1;
        coeff << coeff_a_2, coeff_ab, 0, coeff_b_2, 0, coeff_c_2, 0, 0, coeff_c, coeff_1;
        coeffs.row(i) = coeff;
    }

    Eigen::Matrix<double, 3, 8> solutions;
    int num_sols = poselib::re3q3::re3q3(coeffs, &solutions);

    Eigen::Matrix<double, 3, 3> Ax, Ay, Az;
    Ax << coeffs.col(3), coeffs.col(5), coeffs.col(4); // y^2, z^2, yz
    Ay << coeffs.col(0), coeffs.col(5), coeffs.col(2); // x^2, z^2, xz
    Az << coeffs.col(3), coeffs.col(0), coeffs.col(1); // y^2, x^2, yx

    // We check det(A) as a cheaper proxy for condition number
    int elim_var = 0;
    double detx = std::abs(Ax.determinant());
    double dety = std::abs(Ay.determinant());
    double detz = std::abs(Az.determinant());

    // std::cout << "detx: " << detx << std::endl;
    // std::cout << "dety: " << dety << std::endl;
    // std::cout << "detz: " << detz << std::endl;

    // std::cout << Az.inverse() << std::endl;

    // double a = 1.8526, b = 4.7592, c = 2.0301;
    // Eigen::Vector<double, 10> mon;
    // mon << a*a, a*b, a*c, b*b, b*c, c*c, a, b, c, 1.0;
    // std::cout << coeffs * mon << std::endl;

    // std::cout << solutions << std::endl;

    // std::cout << "num_sols: " << num_sols << std::endl; 
    // for (int i = 0; i < num_sols; i++) {
    //     double a = solutions(0, i), b = solutions(1, i), c = solutions(2, i);
    //     Eigen::Vector<double, 10> mons;
    //     mons << a * a, a * b, a * c, b * b, b * c, c * c, a, b, c, 1;
    //     Eigen::Vector<double, 3> res = coeffs * mons;
    //     std::cout << a << " " << b << " " << c << std::endl; 
    // }

    output->clear();
    const double &x0d = depth_x(0), &x1d = depth_x(1), &x2d = depth_x(2);
    const double &y0d = depth_y(0), &y1d = depth_y(1), &y2d = depth_y(2);
    for (int i = 0; i < num_sols; i++) {
        double a = solutions(0, i), b = solutions(1, i), c = solutions(2, i);
        if (a <= 0) continue;
        if (x0d * a + b <= 0 || x1d * a + b <= 0 || x2d * a + b <= 0) continue;
        if (y0d + c <= 0 || y1d + c <= 0 || y2d + c <= 0) continue;

        Eigen::Vector3d X = X.array().colwise() * (depth_x.array() * a + b);
        Eigen::Vector3d Y = Y.array().colwise() * (depth_y.array() + c);

        Eigen::Vector3d centroid_X = X.rowwise().mean();
        Eigen::Vector3d centroid_Y = Y.rowwise().mean();

        Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
        Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

        Eigen::Matrix3d W = Y_centered * X_centered.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        if (U.determinant() * V.determinant() < 0) {
            U.col(2) *= -1;
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::MatrixXd X_rotated = R * X_centered;
        Eigen::Vector3d t = centroid_Y - R * centroid_X;

        output->push_back(PoseScaleOffset(R, t, a, b, c));
    }
    return output->size();
}

std::vector<PoseScaleOffset> estimate_scale_and_pose_with_offset_3pts_wrap(
    const Eigen::Matrix3d &x_homo, const Eigen::Matrix3d &y_homo,
    const Eigen::Vector3d &depth_x, const Eigen::Vector3d &depth_y) {
    // X: 3 x 3, column vectors are homogeneous 2D points
    // Y: 3 x 3, column vectors are homogeneous 2D points
    std::vector<PoseScaleOffset> output;
    estimate_scale_and_pose_with_offset_3pts(x_homo, y_homo, depth_x, depth_y, &output);
    return output;
}

}; // namespace acmpose