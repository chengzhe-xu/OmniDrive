import json
import numpy as np
import onnx
import onnxruntime as ort
import os
import onnx_graphsurgeon as gs

if __name__=='__main__':
    data_pth = "/data1/chengzhex/omnidrive-trt/omnidrive-deploy/omnidrive/onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a"
    onnx_pth = "/data1/chengzhex/omnidrive-trt/omnidrive-deploy/omnidrive/onnxs/eva_base_tinyllama.onnx"
    input_precision = np.float32

    graph = gs.import_onnx(onnx.load(onnx_pth))
    ref_output_data = {}
    for _out in graph.outputs:
        ref_output_data[_out.name] = np.fromfile(
            os.path.join(data_pth, _out.name+".bin"),
            dtype=np.float32
        ).astype(input_precision).reshape(_out.shape)

    # trt_output_json = json.load(open("./engines/eva_base_tinyllama__output.json"))
    trt_output_json = json.load(open("./engines/eva_base_tinyllama_--fp16_output.json"))
    output_data = {}
    for _trt_out in trt_output_json:
        output_data[_trt_out["name"]] = np.array(
            _trt_out["values"]
        ).reshape(
            [int(eval(_s)) for _s in _trt_out["dimensions"].split("x")]
        ).astype(input_precision)

    for output_name in output_data.keys():
        print(output_name)
        assert ref_output_data[output_name].shape == output_data[output_name].shape, "Shape miss-match."
        model_output = output_data[output_name].reshape(-1)
        pytorch_output = ref_output_data[output_name].reshape(-1)
        diff = np.abs(model_output - pytorch_output)
        print(f"max diff: {np.max(diff)}, which is {100.0*np.max(diff)/np.abs(pytorch_output[np.argmax(diff)])}% on {np.abs(pytorch_output[np.argmax(diff)])}")