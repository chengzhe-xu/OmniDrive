ONNX_NAME=eva_base_tinyllama

TRT_HOME=/home/nvidia/workspace/chengzhex/243_data1/envs/TensorRT-10.4.0.11/cu114/d6l-aarch64/TensorRT-10.4.0.11

PRECISION=--fp16

# EXTRA_FLAG="--precisionConstraints=obey --layerPrecisions=*Softmax:fp32,/img_backbone/patch_embed/proj/Conv:fp32"
# EXTRA_FLAG="--precisionConstraints=obey --layerPrecisions=*ReduceMean*:fp32,*Sqrt*:fp32,*Div*:fp32,*Pow*:fp32"
EXTRA_FLAG=

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec --onnx=./onnxs/${ONNX_NAME}.onnx ${EXTRA_FLAG} --skipInference ${PRECISION} --saveEngine=./engines/${ONNX_NAME}_${PRECISION}.engine --verbose --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=./engines/${ONNX_NAME}_${PRECISION}_layer_build.json > ./engines/${ONNX_NAME}_${PRECISION}.build-log

wait

# sudo /opt/nvidia/nsight_systems/nsys profile -t cuda,nvtx -o ${ONNX_NAME}_${PRECISION} --gpu-metrics-device=0 --cuda-graph-trace=node --env-var='LD_LIBRARY_PATH=/home/nvidia/workspace/chengzhex/243_data1/envs/TensorRT-10.3.0.17/cu114/aarch64/TensorRT-10.3.0.17/lib/:$LD_LIBRARY_PATH' ${TRT_HOME}/bin/trtexec ${PRECISION} ${EXTRA_FLAG} --loadEngine=./${ONNX_NAME}_${PRECISION}.engine --loadInputs="x":./x.bin --iterations=100 --verbose --profilingVerbosity=detailed --exportOutput=./${ONNX_NAME}_${PRECISION}_output_pre.json

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec ${PRECISION} ${EXTRA_FLAG} --loadEngine=./engines/${ONNX_NAME}_${PRECISION}.engine --loadInputs="img":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/img.bin,"pos_embed_input":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/pos_embed_input.bin,"command":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/command.bin,"can_bus":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/can_bus.bin,"is_first_frame":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/is_first_frame.bin,"ego_pose":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/ego_pose.bin,"timestamp":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/timestamp.bin,"ego_pose_inv":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/ego_pose_inv.bin,"memory_embedding_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/memory_embedding_bbox_in.bin,"memory_reference_point_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/memory_reference_point_bbox_in.bin,"memory_timestamp_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/memory_timestamp_bbox_in.bin,"memory_egopose_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/memory_egopose_bbox_in.bin,"memory_canbus_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/memory_canbus_bbox_in.bin,"sample_time_bbox_in":./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/sample_time_bbox_in.bin --iterations=100 --verbose --profilingVerbosity=detailed --dumpProfile --exportProfile=./engines/${ONNX_NAME}_${PRECISION}_layer_infer.json --exportOutput=./engines/${ONNX_NAME}_${PRECISION}_output.json > ./engines/${ONNX_NAME}_${PRECISION}.infer-log

wait
