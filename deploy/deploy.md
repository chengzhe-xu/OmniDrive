# OmniDrive Deployment with TensorRT and TensorTR-LLM
This document demonstrates the deployment of the [OmniDrive](https://arxiv.org/abs/2405.01533) utilizing TensorRT and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). Specifically, we will use [EVA](https://arxiv.org/abs/2303.11331)-base as the vision backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head. We will provide a step-by-step overview of the deployment process, including the overall strategy, environment setup, engine build, engine benchmarking and result analysis.

## Table of Contents

1. [Deployment strategy](#strategy)
2. [Environment setup](#env)
3. [Vision engine build](#vision)
4. [LLM engine build](#llm)
5. [Benchmark](#bench)
6. [Results analysis](#result)
    - [Accuracy performance](#acc)
    - [Inference latencies](#latency)
7. [Future works](#future)
8. [References](#ref)

## Deployment strategy <a name="strategy"></a>
The OmniDrive employs [EVA](https://arxiv.org/abs/2303.11331) as the vision backbone, [StreamPETR](https://arxiv.org/abs/2303.11926)s for both the BBOX detection head and the map head, and a LLM model as the planning head. For deployment, we will utilize ```EVA-base``` as the backbone and [TinyLlama](https://arxiv.org/abs/2401.02385) as the LLM head.

To enhence inference efficiency, engines are built seperately for the vision component (EVA backbone and StreamPETR necks) and the LLM component (TinyLlama). Below are the pipelines for deploying the two components:
 - The vision component: 
    1) export ONNX models;
    2) build engines with TensorRT;
 - The LLM component:
    1) convert the checkpoints to Hugging Face safetensor with TensorRT-LLM;
    2) build engines with ```trtllm-build```.

<img src="../assets/deployment_strategy.png" width="1024">

We will use TensorRT 10.4, and TensorRT-LLM 0.13 to deploy the OmniDrive on A100 GPU, X86_64 Linux platforms. (see this [config](../projects/configs/OmniDrive/eva_base_tinyllama.py) for model details.) Please notice that a patch must be applied to the TensorRT-LLM, please refer to [Environment setup](#env) section for details.

## Environment setup <a name="env"></a>
Please ensure that the folder structure complies with the requirements outlined in this [README](../README.md). We will need the following:
- The ```mmdetection3d``` folder;
- The ```data``` folder;
- The pretrained LLM weights;
- The model checkpoints.

We recommend starting with the [Dockerfile](./omnidrive-deploy.dockerfile):
```
cd ./deploy/

docker build -t omnidrive-deploy:v0 --file ./omnidrive-deploy.dockerfile .

docker run -it --name omnidrive-deploy --gpus all --shm-size=8g -v <workspace>:<workspace> omnidrive-deploy:v0 /bin/bash
```
We need to install the TensorRT Python wheels, which can be found in the TensorRT directory. Please select the wheels that correspond to the Python version in the environment.

```
cd <TensorRT_PATH>/python/
pip3 install ./tensorrt-*-cp3*-none-linux_x86_64.whl
```
we also need to build the TensorRT-LLM wheels within the Docker environment. Please note that a [patch]() must be applied to the official TensorRT-LLM repo to ensure compatibility between TensorRT-LLM and the environment needed for the vision component.
```
git clone ssh://git@gitlab-master.nvidia.com:12051/yuchaoj/tensorrt-llm.git
git checkout commit-xxxxx
git apply patch

cd ./tensorrt-llm/
git submodule update --init --recursive
git lfs install && git lfs pull

LD_LIBRARY_PATH=<TensorRT_PATH>/lib/:$LD_LIBRARY_PATH python3 ./scripts/build_wheel.py --cuda_architectures=<CUDA_ARCH> --trt_root=<TensorRT_PATH> -D ENABLE_MULTI_DEVICE=0 --job_count=4 --clean
```
Nest, we need to install the TensorRT-LLM and any other required Python packages.
```
pip3 install pynvml==11.5.3
pip3 install ./build/tensorrt_llm-*.whl --force-reinstall --no-deps
pip3 install transformers==4.31.0
```

## Vision engine build <a name="vision"></a>
The vision component of the OmniDrive includes the vision backbone, positional embedding, BBOX detection head and map head. We will export a unified ONNX model for the vision component and subsequently build engines based on the ONNX models.

To export the [OmniDrive](../projects/configs/OmniDrive/eva_base_tinyllama.py) ONNX model:
```
PYTHONPATH="./":$PYTHONPATH python3 ./deploy/export_vision.py ./projects/configs/OmniDrive/eva_base_tinyllama.py <checkpoint_path>
```
The ONNX model for the vision component should be generated in the ```./onnxs/``` folder.

Please note that to deploy with FP16 precision, we should convert only the backbone as FP16 and keep the remaining oprations in FP32. This approach is necessary due to the higher numerical sensitivity of coordinate-related computations in anchor-free networks, using FP16 for these operations can lead to significant performance degradation. To set the precision for specific parts of the vision network, we need to mark the operations and generate a seperate ONNX model for FP16 engine building, which has the filename ending with ```_fp16.onnx```.

We can then build the TensorRT engines for the vision component using ```trtexec```. Be sure to use the ONNX file ending with ```_fp16.onnx``` when building FP16 engines:
```
TRT_HOME=<TensorRT_PATH>

# FP32 enigne
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec --onnx=./onnxs/eva_base_tinyllama.onnx --skipInference --saveEngine=./engines/eva_base_tinyllama.engine --useCudaGraph

# FP16 enigne
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec --onnx=./onnxs/eva_base_tinyllama_fp16.onnx --skipInference --saveEngine=./engines/eva_base_tinyllama_fp16.engine --fp16 --precisionConstraints=obey --layerPrecisions=*_FORCEFP32:fp32 --useCudaGraph
```

## LLM engine build <a name="llm"></a>
The LLM head is fine-tuned based on a pretrained Hugging Face model, therefore, the engine requires the weights from both the pretrained model and OmniDrive's checkpoint. The first step is to save the actual weights for LLM component.
```
PYTHONPATH="./":$PYTHONPATH  python3 ./deploy/save_llm_checkpoint.py --config ./projects/configs/OmniDrive/eva_base_tinyllama.py --checkpoint <checkpoint_path> --llm_checkpoint <pretrained_LLM_path> --save_checkpoint_pth <LLM_checkpoint_path>
```
Next, we will need to convert the checkpoint to Hugging Face safetensors.
```
# FP16 engine
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH python3 ./deploy/convert_llm_checkpoint.py --model_dir <LLM_checkpoint_path> --output_dir <LLM_safetensor_path>/x86_1gpu_fp16/ --dtype float16

# FP16 activation, INT4 weight
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH python3 ./deploy/convert_llm_checkpoint.py --model_dir <LLM_checkpoint_path> --output_dir <LLM_safetensor_path>/x86_1gpu_afp16wint4/ --dtype float16 --use_weight_only --weight_only_precision int4
```
Please note that if you encountered ```llm_cfg``` related errors in the previous step (e.g., ```AttributeError: 'LlavaConfig' object has no attribute 'llm_cfg'```), you will need to comment out the following code starting from ```L111``` in ```/opt/conda/lib/python3.8/site-packages/tensorrt_llm/models/llama/config.py```, and then re-run the script.
```
# if hf_config.model_type == "llava_llama":
#     hf_config.llm_cfg["architecture"] = hf_config.llm_cfg["architectures"][0]
#     hf_config.llm_cfg["dtype"] = hf_config.llm_cfg["torch_dtype"]
#     hf_config = PretrainedConfig.from_dict(hf_config.llm_cfg)
```
After this step, the safetensors should be generated.

Since OmniDrive utilizes a modified version of TinyLlama, we need to change the ```architecture``` field in the safetensor's ```config.json``` from ```"architecture": "LlavaLlamaForCausalLM",``` to ```"architecture": "LlamaForCausalLM"```. Following this modification, we can run ```trtllm-build``` to build the TensorRT-LLM engine.
```
# FP16 engine and FP16 activation, INT4 weight
LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH trtllm-build --checkpoint_dir <LLM_safetensor_path>/x86_1gpu_<fp16_or_afp16wint4>/ --output_dir <LLM_engine_path>/x86_1gpu_<fp16_or_afp16wint4>/ --max_prompt_embedding_table_size 1024 --max_batch_size 1 --max_multimodal_len 2048 --gemm_plugin float16
```
At this point, we should have both the vision engine and the LLM engine ready.

## Benchmark <a name="bench"></a>
We have provided a shell script to run the full benchmark for the OmniDrive engine.
```
bash ./deploy/dist_test.sh <TensorRT_PATH> <config_path> <tokenizer_path> <vision_engine_path> <LLM_engine_path>/x86_1gpu_<fp16_or_afp16wint4>/ <QA_save_path>
```
Similar to the PyTorch benchmark, the script will display performance evaluation for BBOX detection, and will save all generated planning trajectories into a result folder. Please evaluate the planning results using [```eval_planning.py```](../evaluation/eval_planning.py).
```
python3 ./evaluation/eval_planning.py --base_path ./data/nuscenes/ --pred_path <QA_save_path>
```

## Results analysis <a name="result"></a>
### Accuracy performance <a name="acc"></a>
Here are the performance comparisons between the PyTorch model and engines.
Vision    | LLM  | BBOX mAP |  Planning L2 1s | Planning L2 2s | Planning L2 3s 
--------------------- | ---- | -------- | --- | --------------------------------| ------- 
PyTorch | PyTorch  | 0.354 |  0.151 | 0.314 | 0.585
FP32 engine | FP16 engine  | 0.354 | 0.150 | 0.312 | 0.581
FP32 engine | FP16 activation INT4 weight  |0.354|0.157|0.323|0.604
FP16 engine | FP16 engine  | 0.306 |0.166|0.337|0.615
FP16 engine | FP16 activation INT4 weight  | 0.306|0.171|0.349|0.634

### Inference latencies <a name="latency"></a>
Here is the runtime latency analysis for the engines. The data was collected on an A100.
metrics | PyTorch | FP32 Vision engine | fp16 Vision engine
------|---- | ---- | ---
engine latency | xxx| xxx | xxx  

metrics | PyTorch | FP16 LLM engine | FP16 activation INT4 weight
--- | --- | --- | ---
Time To First Token (TTFT) |xxx| xxx | xxx
Time Per Output Token (TPOT) |xxx| xxx | xxx
Time Per Frame | xxx|xxx | xxx


## Future works <a name="future"></a>
- [ ] Better quantization (accuracy and latency)
- [ ] DriveOS deployment
- [ ] Pipelined engine excution

## References <a name="ref"></a>
1. [EVA paper](https://arxiv.org/abs/2303.11331)
2. [StreamPETR paper](https://arxiv.org/abs/2303.11926)
3. [TinyLlama paper](https://arxiv.org/abs/2401.02385)
4. [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM)
