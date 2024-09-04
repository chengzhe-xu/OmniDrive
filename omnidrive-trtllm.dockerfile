FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN chmod 1777 /tmp
RUN pip install opencv-python==4.8.0.74 debugpy timm==0.6.13 termcolor yacs pyyaml scipy
RUN pip install transformers onnx onnxsim onnxruntime pycocotools einops tqdm torchprofile
RUN apt update
RUN apt install libgl1-mesa-glx libsm6 libxext6  -y
RUN pip install numba
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.2 mmdet==2.28.2 mmsegmentation==0.30.0
RUN pip install llvmlite==0.41.0
RUN pip install numba==0.58.0 scikit-image==0.18.3 "matplotlib<3.6.0"
RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.1.0 && cd mmdetection3d/ && python setup.py install
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 pip install flash-attn==2.3.4
RUN pip install openai==1.10.0 
RUN pip install accelerate==0.29.0 
RUN pip install plyfile openlanev2 peft fvcore sentencepiece
RUN pip install nvidia-cutlass
RUN pip install nuscenes-devkit --no-deps
RUN pip install "opencv-python-headless<4.3"
RUN (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash) && apt install git-lfs
RUN pip install s2-wrapper 
