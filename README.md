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
Where cu_version and torch_version are the cuda and torch versions. If your cuda version is 11.0 and torch is version 1.7.0. The installation command is
`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html`.

## Usage

### - Data

In order to efficiently create densely stacked target datasets with large scale and diversity, we designed a dense stacked target data generation method based on diffusion model and 3D simulation.You can see this research in another one of our highly innovative open source projects [DiffSimuDateGene](https://github.com/Hjxin02AIsharing-Wust/DiffSimuDateGene.git).

We generated a multimodal dataset containing 20,000 RGB-D  data with 1,209,951 individually segmented cartons in various  environments. We are about to publicly release this dataset.

Here we provide a small dataset containing 4000 RGB-D training data and 400 test data, you can download it [here](https://drive.google.com/drive/folders/1ggZXYTYaE5fEqmtBNdbWQEwS4TeO_n0K).


### - Model
Here we provide a [model weight](https://drive.google.com/drive/folders/1Bqk9WpueXeedxWn0Y2CKoARU_znpsiUy), you can load our weights for training or train your own model from 0 using the small dataset we provide.


### - Train


### -Test



