import numpy as np
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import onnxsim

if __name__=='__main__':
    onnx_pth = "./onnxs/eva_base_tinyllama.onnx"
    onnx_model = onnx.load(onnx_pth)
    graph = gs.import_onnx(onnx_model)

    for idx in range(len(graph.nodes)):
        if "img_backbone" in graph.nodes[idx].name or len(graph.nodes[idx].name) == 0:
            # backbone, fp16
            pass
        else:
            graph.nodes[idx].name = graph.nodes[idx].name + "_FORCEFP32"
    
    graph.toposort().cleanup()
    onnx_model = gs.export_onnx(graph)
    onnx_model, _ = onnxsim.simplify(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model) 
    onnx.save(onnx_model, onnx_pth.replace(".onnx", "_fp16.onnx"))
