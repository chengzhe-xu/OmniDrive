# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.apis import set_random_seed
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import NMSFreeCoder
from mmdet.datasets import replace_ImageToTensor
import time
import tensorrt as trt
import pycuda.driver as cuda
import os.path as osp


class InferTrt(object):
    def __init__(self, logger, torch_ref_model=None):        
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        self.cuda_ctx.push()

        self.builder = trt.Builder(logger)
        self.logger = logger
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.opt = self.builder.create_optimization_profile()

        self.config = self.builder.create_builder_config()
        self.config.add_optimization_profile(self.opt)
        # self.config.max_workspace_size = 2 << 34
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # self.config.set_flag(trt.BuilderFlag.FP16)  # control this
        self.stream = cuda.Stream()
        self.cuda_ctx.pop()
        self.curr_scene_token = None
        self.start_timestamp = None
        self.bindings = {}
        self.position_range = torch.nn.Parameter(torch.tensor(
            [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], device='cuda:0'), requires_grad=False)
        self.coords_d = torch.nn.Parameter(torch.tensor(
            [ 1.0000,  1.0289,  1.0868,  1.1737,  1.2894,  1.4341,  1.6078,  1.8104,
            2.0419,  2.3024,  2.5918,  2.9102,  3.2575,  3.6337,  4.0389,  4.4731,
            4.9362,  5.4282,  5.9491,  6.4990,  7.0779,  7.6857,  8.3224,  8.9881,
            9.6827, 10.4062, 11.1587, 11.9402, 12.7506, 13.5899, 14.4582, 15.3554,
            16.2815, 17.2366, 18.2207, 19.2337, 20.2756, 21.3464, 22.4463, 23.5750,
            24.7327, 25.9193, 27.1349, 28.3794, 29.6529, 30.9553, 32.2866, 33.6469,
            35.0362, 36.4543, 37.9014, 39.3775, 40.8825, 42.4164, 43.9793, 45.5712,
            47.1919, 48.8416, 50.5203, 52.2279, 53.9644, 55.7299, 57.5243, 59.3477], device='cuda:0'), requires_grad=False)
        self.bbox_coder = NMSFreeCoder(
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            voxel_size=[0.2, 0.2, 8],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=300,
            score_threshold=None,
            num_classes=10
        )
        self.torch_ref_model = torch_ref_model

    
    def from_onnx(self, onnx_mod):
        parser = trt.OnnxParser(self.network, self.logger)
        result = parser.parse(onnx_mod.SerializeToString())
        if not result:
            print("failed parsing onnx")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
        self.buf = self.builder.build_serialized_network(self.network, self.config)
        self._build_engine()
        
    def _build_engine(self):
        self.runtime = trt.Runtime(self.logger)        
        self.engine = self.runtime.deserialize_cuda_engine(self.buf)
        self.context = self.engine.create_execution_context()
        # self.context.profiler = CustomProfiler()
        self.names = []
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            self.names.append(self.engine.get_tensor_name(i))

    def write(self, path):
        with open(path, "wb") as fp:
            fp.write(self.buf)

    def read(self, path):
        with open(path, "rb") as fp:
            self.buf = fp.read()
        self._build_engine()
    
    def prepare_location(self, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = 1, 6
        location = self.locations(16, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def locations(self, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = 40, 40
        device = "cuda:0"
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.reshape(h, w, 2)
        return locations
    
    def position_embeding(self, memory_centers, img_metas, intrinsics, lidar2img):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = intrinsics.size(0)

        intrinsic = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        return pos_embed

    def get_bbox(self, all_cls_scores, all_bbox_preds, img_metas):
        bbox_results = []
        preds_dicts = self.bbox_coder.decode({
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict': None,
        })
        num_samples = len(preds_dicts)

        bbox_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            bbox_list.append([bboxes, scores, labels])

        for bboxes, scores, labels in bbox_list:
            bbox_results.append(bbox3d2result(bboxes, scores, labels))
        return bbox_results
    
    def get_lane(self, all_lane_cls_one2one, all_lane_preds_one2one, img_metas):
        cls_scores = all_lane_cls_one2one[-1]
        bbox_preds = all_lane_preds_one2one[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            assert len(cls_score) == len(bbox_pred)
            cls_score = cls_score.sigmoid()
            det_bboxes = bbox_pred
            for p in range(11):
                det_bboxes[..., 3 * p].clamp_(min=-51.2000, max=51.2000)
                det_bboxes[..., 3 * p + 1].clamp_(min=-51.2000, max=51.2000)
                
            # det_bboxes = self.control_points_to_lane_points(det_bboxes)
            det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)
            result_list.append([det_bboxes.cpu().numpy(), cls_score.cpu().numpy()])
        return result_list
    
    def eval(self):
        if self.torch_ref_model is not None:            
            self.torch_ref_model.eval()
        if len(self.bindings) == 0:
            create_bindings_tensor = True
        else:
            create_bindings_tensor = False
        n_io = self.engine.num_io_tensors
        metas_in = []
        metas_out = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = str(self.engine.get_tensor_shape(tname))
            tdtype = str(self.engine.get_tensor_dtype(tname))
            tmode = str(self.engine.get_tensor_mode(tname))
            m = f"{i}\t{tname}\t{tshape}\t{tdtype}"
            if "INPUT" in tmode:
                metas_in.append(m)
            elif "OUTPUT" in tmode:
                metas_out.append(m)
            else:
                assert False, f"Unrecognized tensor mode: {tname}: {tmode}."
            if create_bindings_tensor:
                self.bindings[tname] = torch.zeros(list(self.engine.get_tensor_shape(tname)), 
                                        dtype=torch.float32, 
                                        device="cuda:0").contiguous()
        print("##### Input Bindings: ")
        print("\n".join(metas_in))
        print("##### Output Bindings: ")
        print("\n".join(metas_out))
        return

    def __call__(self, img_metas, input_ids, img, lidar2img, intrinsics, extrinsics, timestamp, img_timestamp, 
                ego_pose, ego_pose_inv, command, can_bus,
                return_loss=False, rescale=True):
        data_dict = {
            "img_metas": img_metas, "input_ids": input_ids, "img": img, "lidar2img": lidar2img, 
            "intrinsics": intrinsics, "extrinsics": extrinsics, "timestamp": timestamp, "img_timestamp": img_timestamp, 
            "ego_pose": ego_pose, "ego_pose_inv": ego_pose_inv, "command": command, "can_bus": can_bus,
        }
        if self.torch_ref_model is not None:            
            ref_result_list = self.torch_ref_model(return_loss=False, rescale=True, **data_dict)
        else:
            ref_result_list = None
        result_list = self.forward(img_metas=img_metas, 
                            input_ids=input_ids, 
                            img=img, 
                            lidar2img=lidar2img, 
                            intrinsics=intrinsics, 
                            timestamp=timestamp, 
                            ego_pose=ego_pose, 
                            ego_pose_inv=ego_pose_inv, 
                            command=command, 
                            can_bus=can_bus)
        return result_list

    def forward(self, img_metas, input_ids, img, lidar2img, intrinsics, timestamp, 
                ego_pose, ego_pose_inv, command, can_bus):
        if len(self.bindings) == 0:
            print("Need to call eval() before forward!.")
            exit(-1)
        # re-format the data
        img_metas = img_metas[0].data[0]
        input_ids = input_ids[0].data[0]
        img = img[0].data[0].to(device="cuda:0").contiguous()
        lidar2img = lidar2img[0].data[0][0].unsqueeze(0).to(device="cuda:0")
        intrinsics = intrinsics[0].data[0][0].unsqueeze(0).to(device="cuda:0")
        timestamp = timestamp[0].data[0][0].unsqueeze(0)
        ego_pose = ego_pose[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        ego_pose_inv = ego_pose_inv[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        command = command[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        can_bus = can_bus[0].data[0][0].unsqueeze(0).to(device="cuda:0").contiguous()
        # convert timestamp from fp64 to fp32
        if self.curr_scene_token is None or img_metas[0]["scene_token"] != self.curr_scene_token:
            self.start_timestamp = timestamp[0].item()
            self.curr_scene_token = img_metas[0]["scene_token"]
            is_first_frame = torch.ones([1]).to(device="cuda:0").contiguous()
        else:
            is_first_frame = torch.zeros([1]).to(device="cuda:0").contiguous()
        timestamp -= self.start_timestamp
        timestamp = timestamp.type(torch.float32).to(device="cuda:0").contiguous()
        location = self.prepare_location(img_metas)
        pos_embed_input = self.position_embeding(location, img_metas, intrinsics, lidar2img)

        # copy the input values to the binding buffer
        self.bindings["img"].copy_(img)
        self.bindings["pos_embed_input"].copy_(pos_embed_input.type(torch.float32).to(device="cuda:0").contiguous())
        self.bindings["command"].copy_(command)
        self.bindings["can_bus"].copy_(can_bus)
        self.bindings["is_first_frame"].copy_(is_first_frame)
        self.bindings["ego_pose"].copy_(ego_pose)
        self.bindings["ego_pose_inv"].copy_(ego_pose_inv)
        self.bindings["timestamp"].copy_(timestamp)
        self.bindings["memory_embedding_bbox_in"].copy_(self.bindings["memory_embedding_bbox_out"])
        self.bindings["memory_reference_point_bbox_in"].copy_(self.bindings["memory_reference_point_bbox_out"])
        self.bindings["memory_timestamp_bbox_in"].copy_(self.bindings["memory_timestamp_bbox_out"])
        self.bindings["memory_egopose_bbox_in"].copy_(self.bindings["memory_egopose_bbox_out"])
        self.bindings["memory_canbus_bbox_in"].copy_(self.bindings["memory_canbus_bbox_out"])
        self.bindings["sample_time_bbox_in"].copy_(self.bindings["sample_time_bbox_out"])
        self.bindings["memory_timestamp_map_in"].copy_(self.bindings["memory_timestamp_map_out"])
        self.bindings["sample_time_map_in"].copy_(self.bindings["sample_time_map_out"])
        self.bindings["memory_egopose_map_in"].copy_(self.bindings["memory_egopose_map_out"])
        self.bindings["memory_embedding_map_in"].copy_(self.bindings["memory_embedding_map_out"])
        self.bindings["memory_reference_point_map_in"].copy_(self.bindings["memory_reference_point_map_out"])
        # inference
        self.cuda_ctx.push()
        for i in range(len(self.names)):
            self.context.set_tensor_address(self.names[i], self.bindings[str(self.names[i])].data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        self.cuda_ctx.pop()

        bbox_results = self.get_bbox(self.bindings["all_cls_scores"].clone(), 
                                     self.bindings["all_bbox_preds"].clone(), 
                                     img_metas)
        
        lane_results = self.get_lane(self.bindings["all_lane_cls_one2one"].clone(), 
                                     self.bindings["all_lane_preds_one2one"].clone(), 
                                     img_metas)
        
        # for i, input_ids in enumerate(input_ids[0]):
        #     input_ids = input_ids.unsqueeze(0)
        #     output_ids = self.lm_head.generate(
        #         inputs=input_ids,
        #         images=vision_embeded,
        #         do_sample=True,
        #         temperature=0.1,
        #         top_p=0.75,
        #         num_beams=1,
        #         max_new_tokens=320,
        #         use_cache=True
        #     )

        result_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(result_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        result_list[0]['text_out'] = [dict(
            Q=img_metas[0]['vlm_labels'].data[0],
            A="",
        )]
        result_list[0]['lane_results'] = lane_results
        return result_list

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) an engine')
    parser.add_argument('--config',help='test config file path')
    parser.add_argument('--engine_pth', help='engine file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
        # wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # # old versions did not save class info in checkpoints, this walkaround is
    # # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # # palette for visualization in segmentation tasks
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # elif hasattr(dataset, 'PALETTE'):
    #     # segmentation dataset has `PALETTE` attribute
    #     model.PALETTE = dataset.PALETTE
    
    # build the engine
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine = InferTrt(logger)
    engine.read(args.engine_pth)

    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False)
        # engine.torch_ref_model = model
        outputs = custom_multi_gpu_test(engine, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    cuda.init()
    torch.cuda.init()
    main()
