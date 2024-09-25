```
cd ./deploy/
docker build -t omnidrive-deploy:v0 --file ./omnidrive-deploy.dockerfile .
docker run -it --name omnidrive-deploy --gpus all --shm-size=8g -v /data1:/data1 omnidrive-deploy:v0 /bin/bash

install tensorrt wheel
git clone trtllm
git submodule update --init --recursive
git lfs install && git lfs pull
python3 ./scripts/build_wheel.py --cuda_architectures=86 --trt_root=/data1/chengzhex/envs/TensorRT-10.4.0.11/cu118/x86_64/TensorRT-10.4.0.11/ -D ENABLE_MULTI_DEVICE=0 --job_count=4 --clean
pip3 install pynvml==11.5.3
pip3 install ./build/tensorrt_llm_*.whl --force-reinstall --no-deps
pip3 install transformers==4.31.0
```