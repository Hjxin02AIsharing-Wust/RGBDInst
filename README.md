# RGBDInst
## Table of Contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Quantitative Result](#quantitative-result)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Background
This is our work in 2023 on the application of instance segmentation network with fused depth information to dense stacked target segmentation.We have improved the [queryinst](https://arxiv.org/pdf/2105.01928.pdf) based model for dense stacked target scenario characteristics.This is our network structure with quantitative results.Compared with other models, our segmentation achieves the best results in dense stacked target scenario.
![image]
![image]

## Install

Our code is built in the **mmdetection** framework, which you can refer to [here](https://mmdetection.readthedocs.io/en/latest/get_started.html) to build your runtime environment.

Please note that you may have trouble installing the installation package `mmcv`, here we provide a way to download the mmcv installation package. You can run the following command:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Where cu_version and torch_version are the cuda and torch versions. If your cuda version is 11.0 and torch is version 1.7.0. The installation command is `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html`.

## Usage

### - Data



### - Model




