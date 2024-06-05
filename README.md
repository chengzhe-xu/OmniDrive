# Environment Setup

**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**2. Download infos files**
Download the [Info Files](https://drive.google.com/file/d/1YHJA29BCWgUxxRh7UImoaoTAIi8hBOEO/view?usp=sharing).

Unzip the data_nusc.tar.gz and organize the files following the Folder structure.

## Pretrained Weights
```shell
cd /path/to/StreamPETR
mkdir ckpts
```
Please download the pretrained [2D llm weights](https://drive.google.com/drive/folders/1yqNyAp3Pp9CdENpah6AiMt3IfOXbqeUO?usp=sharing) to ./ckpts. 



**3. Install Packages**
```shell
cd /path/to/StreamPETR
conda create -n demo python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extraindex-url https://download.pytorch.org/whl/cu117
pip install flash-attn==0.2.8
pip install transformers==4.31.0 
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
pip uninstall setuptools
pip install setuptools==59.5.0
```


After preparation, you will be able to see the following directory structure:  

**4. Folder structure**
```
OmniDrive
├── projects/
├── iter_10548.pth
├── mmdetection3d/
├── tools/
├── configs/
├── ckpts/
│   ├── final/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── conv/
│   │   ├── desc/
│   │   ├── keywords/
│   │   ├── vqa/
│   │   ├── nuscenes2d_ego_temporal_infos_train.pkl
│   │   ├── nuscenes2d_ego_temporal_infos_val.pkl
│   │   ├── data_dict_sample.pkl
│   │   ├── data_dict_subset_B.json
│   │   ├── data_dict_subset_B_val.pkl
│   │   ├── lane_obj_train.pkl
```

**5. Inference**
```shell
tools/dist_test.sh .projects/configs/StreamPETR/mask_eva_lane_det_vlm_all.py ./iter_10548.pth 1 --eval bbox
```