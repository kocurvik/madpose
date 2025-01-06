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
_Note: The two-focal estimators currently rely on `cv::recoverPose` from OpenCV, we plan to remove dependency on OpenCV in future updates._

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
If you find our work useful in your research, please consider consider citing our paper:
```
@misc{}
```
