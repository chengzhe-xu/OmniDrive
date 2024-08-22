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
    # load_string = ""
    input_data = {}
    for _in in graph.inputs:
        input_data[_in.name] = np.fromfile(
            os.path.join(data_pth, _in.name+".bin"),
            dtype=np.float32
        ).astype(input_precision).reshape(_in.shape)
    #     load_string += f""""{_in.name}":{os.path.join(data_pth, _in.name+'.bin').replace("/data1/chengzhex/omnidrive-trt/omnidrive-deploy/omnidrive", ".")},"""
    # print(load_string[:-1])
    if "topk_indexes_bbox" in input_data.keys():
        input_data["topk_indexes_bbox"] = input_data["topk_indexes_bbox"].astype(np.int64)
    if "topk_indexes_map" in input_data.keys():
        input_data["topk_indexes_map"] = input_data["topk_indexes_map"].astype(np.int64)
    
    ref_output_data = {}
    for _out in graph.outputs:
        ref_output_data[_out.name] = np.fromfile(
            os.path.join(data_pth, _out.name+".bin"),
            dtype=np.float32
        ).astype(input_precision).reshape(_out.shape)
        
    output_names = [_out.name for _out in graph.outputs]
    output_lst = ort.InferenceSession(onnx_pth).run(None, input_data)

    assert len(output_names) == len(output_lst), "[ERROR] Incorrect number of outputs."
    output_data = {}
    for i in range(len(output_names)):
        output_data[output_names[i]] = output_lst[i]

    for output_name in output_data.keys():
        assert ref_output_data[output_name].shape == output_data[output_name].shape, "[ERROR] Shape miss-match."
        model_output = output_data[output_name].reshape(-1)
        pytorch_output = ref_output_data[output_name].reshape(-1)
        diff = np.abs(model_output - pytorch_output)
        original_candidate = np.min(np.abs(pytorch_output[np.argwhere(diff == np.max(diff)).flatten().tolist()]))
        if original_candidate > 0:
            print(f"{output_name}:\tmax diff: {np.max(diff)},\twhich is {100.0*np.max(diff)/original_candidate}%\ton {original_candidate}")
        else:
            print(f"{output_name}:\tmax diff: {np.max(diff)},\ton {original_candidate}")
