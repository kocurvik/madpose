# MADPose

<a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>

This repo contains the official implementation of the solvers and estimators proposed in the paper "Relative Pose Estimation through Affine Corrections of Monocular Depth Priors". The solvers and estimators are implemented using C++, and we provide easy-to-use Python bindings. 

Note: "**MAD**" is an acronym for "**M**onocular **A**ffine **D**epth".

## Installation
### Install from PyPI
We are working on setting up wheel for easy installation using PyPI. Currently please use the following method to install from source.

### Install from source
#### Install dependencies
```bash
sudo apt-get install libeigen3-dev libceres-dev libopencv-dev
```
_Note: The two-focal estimator currently relies on `cv::recoverPose` from OpenCV, we plan to remove dependency on OpenCV in future updates._

#### Clone the repo
```bash
git clone --recursive https://github.com/MarkYu98/madpose
```

#### Build and install the Python bindings
```bash
pip install .
```
If you would like to see the building process (e.g. CMake logs) you can add `-v` option to the above command.

#### Check the installation
```bash
python -c "import madpose"
```
You should not see any errors if MADPose is successfully installed.

## Usage
### Estimators
We provide Python bindings of our 3 hybrid estimators for image pairs with calibrated cameras, shared-focal cameras, and cameras with unknown focal lengths (two-focal). 

The estimators take `HybridLORansacOptions` and `EstimatorConfig` for related settings, some useful settings are:
```python
import madpose

options = madpose.HybridLORansacOptions()
options.min_num_iterations = 1000
options.max_num_iterations = 10000
options.success_probability = 0.9999
options.random_seed = 0 # for reproducibility
options.final_least_squares = True
options.threshold_multiplier = 5.0
options.num_lo_steps = 4
# squared px thresholds for reprojection error and epipolar error
options.squared_inlier_thresholds = [reproj_pix_thres ** 2, epipolar_pix_thres ** 2]
# weight when scoring for the two types of errors
options.data_type_weights = [1.0, epipolar_weight]

est_config = madpose.EstimatorConfig()
# if enabled, the input min_depth values are guaranteed to be positive with the estimated depth offsets (shifts), default: True
est_config.min_depth_constraint = True
# if disabled, will model the depth with only scale (only applicable to the calibrated camera case)
est_config.use_shift = True
```

We provide a example image pairs and code snippets in [examples/](examples/) to test the hybrid estimators. More demos and evaluations will be added in the future.

#### Calibrated estimator
```python
pose, stats = madpose.HybridEstimatePoseScaleOffset(
                  mkpts0, mkpts1, 
                  depth0, depth1,
                  [depth_map0.min(), depth_map1.min()], 
                  K0, K1, options, est_config
              )
# rotation and translation of the estimated pose
R_est, t_est = pose.R(), pose.t()
# scale and offsets of the affine corrected depth maps
s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
```
The parameters are: keypoint matches(`mkpts0`, `mkpts1`), their corresponding depth prior values(`depth0`, `depth1`), min depth values for both views (used when `est_config.min_depth_constraint` is `True`), camera intrinsics(`K0`, `K1`),`options`, and `est_config`.

See [examples/calibrated.py](examples/calibrated.py) for a complete code example, evaluation, and comparison with point-based estimation using PoseLib.

#### Shared-focal estimator
```python
pose, stats = monodepth.HybridEstimatePoseScaleOffsetSharedFocal(
                  mkpts0, mkpts1, 
                  depth0, depth1,
                  [depth_map0.min(), depth_map1.min()], 
                  pp0, pp1, options, est_config
              )
# rotation and translation of the estimated pose
R_est, t_est = pose.R(), pose.t()
# scale and offsets of the affine corrected depth maps
s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
# estimated shared focal length
f_est = pose.focal
```
Different from the calibrated estimator, now instead of intrinsics(`K0`, `K1`), the shared-focal estimator now takes as input the principal points(`pp0`, `pp1`).

See [examples/shared_focal.py](examples/shared_focal.py) for complete example.

#### Two-focal estimator
```python
pose, stats = monodepth.HybridEstimatePoseScaleOffsetTwoFocal(
                  mkpts0, mkpts1, depth0, depth1,
                  [depth_map0.min(), depth_map1.min()], 
                  pp0, pp1, options, est_config
              )
```
The parameters are same with the shared-focal estimator, but now the estimator will estimate two independent focal lengths.

See [examples/two_focal.py](examples/two_focal.py) for complete example.

#### Point-based baseline
You can compare with point-based estimators from [PoseLib](https://github.com/PoseLib/PoseLib). You need to install the [Poselib's Python bindings](https://github.com/PoseLib/PoseLib?tab=readme-ov-file#python-bindings). 

The corresponding point-based estimation are included in each of the three example scripts above.

### Solvers

## TODO List

- [ ] Remove dependency on OpenCV.
- [ ] Setup wheel for PyPI
- [ ] Add experiment scirpts on datasets

## Acknowledgement
Our codebase is inspired by and built upon many research work and opensource projects, we thank the authors and contributors for their work.
- [PoseLib](https://github.com/PoseLib/PoseLib)
- [COLMAP](https://github.com/colmap/colmap)
- [LIMAP](https://github.com/cvg/limap)
- [RansacLib](https://github.com/tsattler/RansacLib)
- [pybind11](https://github.com/pybind/pybind11)

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@misc{}
```
