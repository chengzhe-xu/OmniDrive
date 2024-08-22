import json
import numpy as np
import onnx
import onnxruntime as ort
import os
import onnx_graphsurgeon as gs

if __name__=='__main__':
    data_pth = "/data1/chengzhex/omnidrive-trt/omnidrive-deploy/omnidrive/onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a"
    onnx_pth = "/data1/chengzhex/omnidrive-trt/omnidrive-deploy/omnidrive/onnxs/eva_base_tinyllama.onnx"
    trt_output_json_pth = "./engines/fp16_forcefp32/eva_base_tinyllama_--fp16_output.json"
    input_precision = np.float32

    graph = gs.import_onnx(onnx.load(onnx_pth))
    input_data = {}
    for _in in graph.inputs:
        input_data[_in.name] = np.fromfile(
            os.path.join(data_pth, _in.name+".bin"),
            dtype=np.float32
        ).astype(input_precision).reshape(_in.shape)
        
    output_names = [_out.name for _out in graph.outputs]
    output_lst = ort.InferenceSession(onnx_pth).run(None, input_data)

    assert len(output_names) == len(output_lst), "[ERROR] ONNX: Incorrect number of outputs."
    ref_output_data = {}
    for i in range(len(output_names)):
        ref_output_data[output_names[i]] = output_lst[i]

    trt_output_json = json.load(open(trt_output_json_pth))
    trt_output_data = {}
    for _trt_out in trt_output_json:
        trt_output_data[_trt_out["name"]] = np.array(
            _trt_out["values"]
        ).reshape(
            [int(eval(_s)) for _s in _trt_out["dimensions"].split("x")]
        ).astype(input_precision)

    for output_name in trt_output_data.keys():
        assert ref_output_data[output_name].shape == trt_output_data[output_name].shape, f"[ERROR] {output_name}: Shape miss-match."
        model_output = trt_output_data[output_name].reshape(-1)
        onnx_output = ref_output_data[output_name].reshape(-1)
        diff = np.abs(model_output - onnx_output)
        original_candidate = np.min(np.abs(onnx_output[np.argwhere(diff == np.max(diff)).flatten().tolist()]))
        if original_candidate > 0:
            print(f"{output_name}:\tmax diff: {np.max(diff)},\twhich is {100.0*np.max(diff)/original_candidate}%\ton {original_candidate}")
        else:
            print(f"{output_name}:\tmax diff: {np.max(diff)},\ton {original_candidate}")
