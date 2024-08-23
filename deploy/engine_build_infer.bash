ONNX_NAME=eva_base_tinyllama

TRT_HOME=/home/nvidia/workspace/chengzhex/243_data1/envs/TensorRT-10.4.0.11/cu114/d6l-aarch64/TensorRT-10.4.0.11

DATA_PATH=./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/

PRECISION=--fp16

# EXTRA_FLAG="--precisionConstraints=obey --layerPrecisions=*Softmax:fp32,/img_backbone/patch_embed/proj/Conv:fp32"
EXTRA_FLAG=

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec --onnx=./onnxs/${ONNX_NAME}.onnx ${EXTRA_FLAG} --skipInference ${PRECISION} --saveEngine=./engines/${ONNX_NAME}_${PRECISION}.engine --verbose --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=./engines/${ONNX_NAME}_${PRECISION}_layer_build.json > ./engines/${ONNX_NAME}_${PRECISION}.build-log

wait

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH ${TRT_HOME}/bin/trtexec ${PRECISION} ${EXTRA_FLAG} --loadEngine=./engines/${ONNX_NAME}_${PRECISION}.engine \
 --loadInputs="img":${DATA_PATH}/img.bin,"pos_embed_input":${DATA_PATH}/pos_embed_input.bin,"command":${DATA_PATH}/command.bin,"can_bus":${DATA_PATH}/can_bus.bin,"is_first_frame":${DATA_PATH}/is_first_frame.bin,"ego_pose":${DATA_PATH}/ego_pose.bin,"timestamp":${DATA_PATH}/timestamp.bin,"ego_pose_inv":${DATA_PATH}/ego_pose_inv.bin,"memory_embedding_bbox_in":${DATA_PATH}/memory_embedding_bbox_in.bin,"memory_reference_point_bbox_in":${DATA_PATH}/memory_reference_point_bbox_in.bin,"memory_timestamp_bbox_in":${DATA_PATH}/memory_timestamp_bbox_in.bin,"memory_egopose_bbox_in":${DATA_PATH}/memory_egopose_bbox_in.bin,"memory_canbus_bbox_in":${DATA_PATH}/memory_canbus_bbox_in.bin,"sample_time_bbox_in":${DATA_PATH}/sample_time_bbox_in.bin,"memory_timestamp_map_in":${DATA_PATH}/memory_timestamp_map_in.bin,"sample_time_map_in":${DATA_PATH}/sample_time_map_in.bin,"memory_egopose_map_in":${DATA_PATH}/memory_egopose_map_in.bin,"memory_embedding_map_in":${DATA_PATH}/memory_embedding_map_in.bin,"memory_reference_point_map_in":${DATA_PATH}/memory_reference_point_map_in.bin \
 --iterations=100 --verbose --profilingVerbosity=detailed --dumpProfile --exportProfile=./engines/${ONNX_NAME}_${PRECISION}_layer_infer.json --exportOutput=./engines/${ONNX_NAME}_${PRECISION}_output.json > ./engines/${ONNX_NAME}_${PRECISION}.infer-log

wait

sudo /opt/nvidia/nsight_systems/nsys profile -t cuda,nvtx --output=./engines/${ONNX_NAME}_${PRECISION}.nsys-rep --gpu-metrics-device=0 --cuda-graph-trace=node --env-var="LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH" ${TRT_HOME}/bin/trtexec ${PRECISION} ${EXTRA_FLAG} --loadEngine=./engines/${ONNX_NAME}_${PRECISION}.engine \
 --loadInputs="img":${DATA_PATH}/img.bin,"pos_embed_input":${DATA_PATH}/pos_embed_input.bin,"command":${DATA_PATH}/command.bin,"can_bus":${DATA_PATH}/can_bus.bin,"is_first_frame":${DATA_PATH}/is_first_frame.bin,"ego_pose":${DATA_PATH}/ego_pose.bin,"timestamp":${DATA_PATH}/timestamp.bin,"ego_pose_inv":${DATA_PATH}/ego_pose_inv.bin,"memory_embedding_bbox_in":${DATA_PATH}/memory_embedding_bbox_in.bin,"memory_reference_point_bbox_in":${DATA_PATH}/memory_reference_point_bbox_in.bin,"memory_timestamp_bbox_in":${DATA_PATH}/memory_timestamp_bbox_in.bin,"memory_egopose_bbox_in":${DATA_PATH}/memory_egopose_bbox_in.bin,"memory_canbus_bbox_in":${DATA_PATH}/memory_canbus_bbox_in.bin,"sample_time_bbox_in":${DATA_PATH}/sample_time_bbox_in.bin,"memory_timestamp_map_in":${DATA_PATH}/memory_timestamp_map_in.bin,"sample_time_map_in":${DATA_PATH}/sample_time_map_in.bin,"memory_egopose_map_in":${DATA_PATH}/memory_egopose_map_in.bin,"memory_embedding_map_in":${DATA_PATH}/memory_embedding_map_in.bin,"memory_reference_point_map_in":${DATA_PATH}/memory_reference_point_map_in.bin \
 --iterations=100 --verbose --profilingVerbosity=detailed --dumpProfile --exportProfile=./engines/${ONNX_NAME}_${PRECISION}_layer_infer.json --exportOutput=./engines/${ONNX_NAME}_${PRECISION}_output.json

wait
