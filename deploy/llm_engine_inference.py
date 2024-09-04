import os
import time
import argparse
import numpy as np
import torch
import torch
import tensorrt as trt
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pth', type=str, default=None)
    parser.add_argument('--llm_engine_pth', type=str, default=None)
    parser.add_argument('--tokenizer_pth', type=str, default=None)
    args = parser.parse_args()
    return args


class Runner:
    def __init__(self, llm_engine_pth, tokenizer_pth) -> None:
        self.rank = 0
        device_id = 0
        self.IMAGE_TOKEN_INDEX = -200
        self.llm_engine_pth = llm_engine_pth
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, model_max_length=2048, padding_side="right", use_fast=False,)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model_type = "llava_llama"
        self.init_llm()
    
    def init_llm(self):
        self.model = ModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=False, stream=self.stream)
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def image_to_ptuning(self, input_ids, vision_embeded):
        updated_input_ids = []
        current_vocab_size = self.tokenizer.vocab_size
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                updated_input_ids.append(cur_input_ids)
                continue
            im_token_ids = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist()
            im_token_ids = [-1] + im_token_ids + [cur_input_ids.shape[0]]
            im_idx = 0
            for i in range(len(im_token_ids) - 1):
                updated_input_ids.append(cur_input_ids[im_token_ids[i]+1:im_token_ids[i+1]])
                if im_idx < vision_embeded.shape[0]:
                    im = vision_embeded[im_idx]
                    im_size = im.shape[0]
                    im_indices = torch.from_numpy(np.arange(current_vocab_size, current_vocab_size + im_size)).cuda()
                    updated_input_ids.append(im_indices)
                    im_idx += 1
        return torch.cat(updated_input_ids).unsqueeze(0), vision_embeded.reshape(1, -1, vision_embeded.shape[2])

    def generate(self, input_ids, vision_embeded):
        input_ids, prompt_table = self.image_to_ptuning(input_ids, vision_embeded)
        input_ids = input_ids.contiguous().to(dtype=torch.int32)
        prompt_table = prompt_table.cuda().contiguous().to(dtype=torch.float16)
        t_start = time.time()
        output_ids = self.model.generate(
            input_ids, 
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=320,
            use_cache=False)
        print(f"Generation time: {time.time() - t_start}s.")
        output_ids = torch.masked_select(output_ids, output_ids.lt(self.tokenizer.vocab_size)).reshape([1, -1])
        return output_ids

if __name__=='__main__':
    args = parse_arguments()

    input_ids = np.fromfile(os.path.join(args.data_pth, "input_ids_0.bin"), dtype=np.float32).reshape([65]).astype(np.int64)
    input_ids = torch.LongTensor(torch.from_numpy(input_ids)).cuda().unsqueeze(0)
    vlm_memory_bbox = np.fromfile(os.path.join(args.data_pth, "vlm_memory_bbox.bin"), dtype=np.float32).reshape([1,257,2048])
    vlm_memory_bbox = torch.from_numpy(vlm_memory_bbox).cuda()
    vlm_memory_map = np.fromfile(os.path.join(args.data_pth, "vlm_memory_map.bin"), dtype=np.float32).reshape([1,256,2048])
    vlm_memory_map = torch.from_numpy(vlm_memory_map).cuda()

    vision_embeded = torch.cat([vlm_memory_bbox, vlm_memory_map], dim=1)
    runner = Runner(llm_engine_pth=args.llm_engine_pth, tokenizer_pth=args.tokenizer_pth)

    output_ids = runner.generate(input_ids, vision_embeded)
    print(runner.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
