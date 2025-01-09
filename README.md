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

We provide a few sample image pairs in `samples/` to test the hybrid estimators. More demos and evaluations will be added in the future.

#### Calibrated estimator
```python
```

#### Shared-focal estimator
```python
```

#### Two-focal estimator
```python
```

#### Point-based baseline
You can compare with point-based estimators from [PoseLib](https://github.com/PoseLib/PoseLib). You need to install the Poselib's Python bindings. 

**Calibrated cameras**:
```python
ransac_opt_dict = {'max_epipolar_error': args.epi_pix_thres, 'min_iterations': 1000, 'max_iterations': 10000}
cam0_dict = {'model': 'PINHOLE', 'width': image0.shape[1], 'height': image0.shape[0], 'params': [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]}
cam1_dict = {'model': 'PINHOLE', 'width': image1.shape[1], 'height': image1.shape[0], 'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]}

pose, stats = poselib.estimate_relative_pose(mkpts0, mkpts1, cam0_dict, cam1_dict, ransac_opt_dict)
R_ponly, t_ponly = est_pose.R, est_pose.t
```

**Shared-focal cameras**:
```python
pp = (np.array(image0.shape[:2][::-1]) - 1) / 2
ransac_opt_dict = {'max_epipolar_error': args.epi_pix_thres, 'min_iterations': 1000, 'max_iterations': 10000}

img_pair, stats = poselib.estimate_shared_focal_relative_pose(mkpts0, mkpts1, pp, ransac_opt_dict)
shared_focal_pose = img_pair.pose
R_ponly, t_ponly = shared_focal_pose.R, shared_focal_pose.t
f_ponly = img_pair.camera1.focal()
```

**Two-focal cameras**

### Solvers

## TODO List

- [ ] Add example image pairs with depth priors and image correspondences.
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
