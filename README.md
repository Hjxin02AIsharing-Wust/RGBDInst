# RGBDInst
## Table of Contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Result](#result)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Background
This is our work in 2023 on the application of instance segmentation network with fused depth information to dense stacked target segmentation.We have improved the [queryinst](https://arxiv.org/pdf/2105.01928.pdf) based model for dense stacked target scenario characteristics.This is our network structure with quantitative results.Compared with other models, our segmentation achieves the best results in dense stacked target scenario.

The architecture of our proposed RGBDInst for RGB-D based instance segmentation:
![image](https://github.com/Hjxin02AIsharing-Wust/RGBDInst/blob/main/texture-image/RGBDInst4.png)


Overview of RGB-D Fusion Modul:
![image](https://github.com/Hjxin02AIsharing-Wust/RGBDInst/blob/main/texture-image/RGBDInst3.jpg)

## Install

Our code is built in the **mmdetection** framework, which you can refer to [here](https://mmdetection.readthedocs.io/en/latest/get_started.html) to build your runtime environment.

Please note that you may have trouble installing the installation package `mmcv`, here we provide a way to download the mmcv installation package. You can run the following command:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Where cu_version and torch_version are the cuda and torch versions. If your cuda version is 11.0 and torch is version 1.7.0. The installation command is
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

## Usage

### - Data

In order to efficiently create densely stacked target datasets with large scale and diversity, we designed a dense stacked target data generation method based on diffusion model and 3D simulation.You can see this research in another one of our highly innovative open source projects [DiffSimuDateGene](https://github.com/Hjxin02AIsharing-Wust/DiffSimuDateGene.git).

We generated a multimodal dataset containing 20,000 RGB-D  data with 1,209,951 individually segmented cartons in various  environments. We are about to publicly release this dataset.

Here we provide a small dataset containing 4000 RGB-D training data and 400 test data, you can download it [here](https://drive.google.com/drive/folders/1ggZXYTYaE5fEqmtBNdbWQEwS4TeO_n0K).

After downloading the dataset you need to change the dataset path in this configuration folder：`./configs/rgbd-inst/rgbdinst_r50_fpn_mstrain_480-800_3x_coco.py`. Our dataset class name is **surface**.


### - Model
Here we provide a [model weight](https://drive.google.com/drive/folders/1Bqk9WpueXeedxWn0Y2CKoARU_znpsiUy), you can load our weights for training or testing  or train your own model from 0 using the small dataset we provide.


### - Train
- **Single GPU**
```shell
python tools/train.py configs/rgbd-inst/rgbdinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py
```
- **Multi GPUS**
```shell
./tools/dist_train.sh configs/rgbd-inst/rgbdinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py 8
```

### -Test
- **Single GPU**
```shell
python tools/test.py configs/rgbd-inst/rgbdinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py PATH/TO/CKPT.pth --eval bbox segm
```


- **Multi GPUS**

```shell
./tools/dist_test.sh configs/rgbd-inst/rgbdinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py PATH/TO/CKPT.pth 8 --eval bbox segm
```
## Result

### Quantitative results

![image](https://github.com/Hjxin02AIsharing-Wust/RGBDInst/blob/main/texture-image/RGBDInst1.png)

### Qualitative results

<p align="center">
  <img src="https://github.com/Hjxin02AIsharing-Wust/RGBDInst/blob/main/texture-image/RGBDInst2.png" alt="example input output gif" width="500" />
</p>



## Contributing
This project exists thanks to these people who contribute

```shell
Jiaxin Hu 
```
We are doing further research, welcoming to communicate. If your suggestions and work advance this project, we will regard you as one of the contributors.

## Acknowledgements
Our research is based on [Queryinst](https://arxiv.org/pdf/2105.01928.pdf) and [mmdetection](https://github.com/open-mmlab/mmdetection), and we pay a high respect to them.


