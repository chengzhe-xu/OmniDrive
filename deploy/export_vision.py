# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import numpy as np
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import onnx
import onnxsim
import io
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet.models.utils.transformer import inverse_sigmoid
import onnxruntime as ort
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points_lane, transform_reference_points, memory_refresh, SELayer_Linear

def parse_args():
    parser = argparse.ArgumentParser(
        description='OmniDrive OOX export(Vision part).')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--show', action='store_true', help='show results')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

class OmniDriveVisionTrtProxy(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod
    
    def mapPETR_forward(self, img_feats, pos_embed, timestamp, ego_pose, ego_pose_inv, is_first_frame,
                        memory_timestamp, sample_time, memory_egopose, memory_embedding, memory_reference_point):
        head = self.mod.map_head

        B = img_feats.size(0)
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1.-is_first_frame)
        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1.-is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1.-is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1.-is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1.-is_first_frame)
        sample_time = timestamp.new_zeros(B)

        x = img_feats
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)

        memory = head.input_projection(memory)
        lane_embedding = head.instance_embedding_lane.weight.unsqueeze(-2) + head.points_embedding_lane.weight.unsqueeze(0) 
        reference_points_lane = head.reference_points_lane(lane_embedding).sigmoid().flatten(-2).unsqueeze(0).repeat(B, 1, 1)
        query_pos = head.query_pos(nerf_positional_encoding(reference_points_lane))
        tgt = head.instance_embedding_lane.weight.unsqueeze(0).repeat(B, 1, 1)
        query_embedding = head.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        self_attn_mask = (
            torch.zeros([head.num_lane+head.num_extra, head.num_lane+head.num_extra]).bool().to(x.device)
        )
        self_attn_mask[head.num_lanes_one2one+head.num_extra:, 0: head.num_lanes_one2one+head.num_extra] = True
        self_attn_mask[0: head.num_lanes_one2one+head.num_extra, head.num_lanes_one2one+head.num_extra:] = True
        temporal_attn_mask = (
            torch.zeros([head.num_lane+head.num_extra, head.num_lane+head.num_extra+head.memory_len]).bool().to(x.device)
        )
        temporal_attn_mask[:self_attn_mask.size(0), :self_attn_mask.size(1)] = self_attn_mask
        if head.with_mask:
            temporal_attn_mask[head.num_extra:, :head.num_extra] = True

        B_query_pos = query_pos.size(0)
        temp_reference_point = (memory_reference_point - head.pc_range[:3]) / (head.pc_range[3:6] - head.pc_range[0:3])
        temp_pos = head.query_pos(nerf_positional_encoding(temp_reference_point.flatten(-2))) 
        temp_memory = memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B_query_pos, query_pos.size(1), 1, 1)
        
        if head.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(tgt[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(tgt[...,:1])))
        temp_pos += head.time_embedding(pos2posemb1d(memory_timestamp).float())

        tgt = torch.cat([query_embedding, tgt], dim=1)
        query_pos = torch.cat([torch.zeros_like(query_embedding), query_pos], dim=1)
        
        outs_dec = head.transformer(tgt, memory, query_pos, pos_embed, temporal_attn_mask, temp_memory, temp_pos)

        vlm_memory = outs_dec[-1, :, :head.num_extra, :]
        outs_dec = outs_dec[:, :, head.num_extra:, :]
        
        outs_dec = torch.nan_to_num(outs_dec)
        
        lane_queries = outs_dec
        outputs_lane_preds = []
        outputs_lane_clses = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points_lane.clone())
            reference = reference.view(B, head.num_lane, head.n_control*3)
            tmp = head.reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = head.cls_branches[lvl](lane_queries[lvl])

            tmp = tmp.reshape(B, head.num_lane, head.n_control*3)
            tmp += reference
            tmp = tmp.sigmoid()

            outputs_coord = tmp
            outputs_coord = outputs_coord.reshape(B, head.num_lane, head.n_control, 3)
            outputs_lane_preds.append(outputs_coord)
            outputs_lane_clses.append(outputs_lanecls)

        all_lane_preds = torch.stack(outputs_lane_preds)
        all_lane_clses = torch.stack(outputs_lane_clses)

        all_lane_preds[..., 0:3] = (all_lane_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3])
        all_lane_preds = all_lane_preds.flatten(-2)

        all_lane_cls_one2one = all_lane_clses[:, :, 0: head.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0: head.num_lanes_one2one, :]
        all_lane_cls_one2many = all_lane_clses[:, :, head.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, head.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0: head.num_lanes_one2one, :]
        outs_dec_one2many = outs_dec[:, :, head.num_lanes_one2one:, :]

        rec_reference_points = all_lane_preds_one2one[-1].reshape(outs_dec_one2one.shape[1], -1, head.n_control, 3)
        out_memory = outs_dec_one2one[-1]
        rec_score = all_lane_cls_one2one[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        # topk_indexes = topk_indexes_input
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        memory_embedding = torch.cat([rec_memory, memory_embedding], dim=1)
        memory_timestamp = torch.cat([rec_timestamp, memory_timestamp], dim=1)
        memory_egopose= torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_reference_point = transform_reference_points_lane(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose

        if head.output_dims is not None:
            vlm_memory = head.output_projection(vlm_memory)
        return all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many, all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, vlm_memory, \
            memory_embedding, memory_timestamp, memory_egopose, memory_reference_point, sample_time

    def bboxStreamPETR_forward(self, img_feats, pos_embed, command, can_bus, is_first_frame,
                               memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_canbus,
                               sample_time, ego_pose, timestamp, ego_pose_inv):
        head = self.mod.pts_bbox_head

        B = img_feats.size(0)
        memory_timestamp += timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time += timestamp
        sample_time_x = (torch.abs(sample_time) < 2.0).to(img_feats.dtype)
        sample_time_x *= (1.-is_first_frame)
        memory_egopose = ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose_inv, reverse=False)
        memory_timestamp = memory_refresh(memory_timestamp[:, :head.memory_len], sample_time_x)
        memory_timestamp *= (1.-is_first_frame)
        memory_reference_point = memory_refresh(memory_reference_point[:, :head.memory_len], sample_time_x)
        memory_reference_point *= (1.-is_first_frame)
        memory_embedding = memory_refresh(memory_embedding[:, :head.memory_len], sample_time_x)
        memory_embedding *= (1.-is_first_frame)
        memory_egopose = memory_refresh(memory_egopose[:, :head.memory_len], sample_time_x)
        memory_egopose *= (1.-is_first_frame)
        memory_canbus = memory_refresh(memory_canbus[:, :head.can_bus_len], sample_time_x)
        memory_canbus *= (1.-is_first_frame)
        sample_time = timestamp.new_zeros(B)
        sample_time *= (1.-is_first_frame)

        pseudo_reference_points = head.pseudo_reference_points.weight * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3]
        memory_reference_point[:, :head.num_propagated]  = memory_reference_point[:, :head.num_propagated] + (1-sample_time_x).view(B, 1, 1) * pseudo_reference_points
        memory_egopose[:, :head.num_propagated]  = memory_egopose[:, :head.num_propagated] + (1-sample_time_x).view(B, 1, 1, 1) * torch.eye(4, device=sample_time_x.device)

        x = img_feats
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = head.input_projection(memory)

        reference_points = head.reference_points.weight
        reference_points = torch.cat([torch.zeros_like(reference_points[:head.num_extra]), reference_points], dim=0)

        attn_mask = None
        if head.with_mask:
            tgt_size = head.num_query + head.memory_len + head.num_extra
            query_size = head.num_query + head.num_propagated + head.num_extra
            attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            attn_mask[head.num_extra:, :head.num_extra] = True
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)
    
        query_pos = head.query_pos(nerf_positional_encoding(reference_points.repeat(1, 1, head.n_control)))
        tgt = torch.zeros_like(query_pos)

        query_pos_B = query_pos.size(0)

        temp_reference_point = (memory_reference_point - head.pc_range[:3]) / (head.pc_range[3:6] - head.pc_range[0:3])
        temp_pos = head.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, head.n_control))) 
        temp_memory = memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(query_pos_B, query_pos.size(1), 1, 1)
        
        if head.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([memory_timestamp, memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += head.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += head.time_embedding(pos2posemb1d(memory_timestamp).float())

        if head.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :head.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :head.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :head.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(query_pos_B, query_pos.shape[1]+head.num_propagated, 1, 1)
            temp_memory = temp_memory[:, head.num_propagated:]
            temp_pos = temp_pos[:, head.num_propagated:]

        tgt[:, :head.num_extra, :] = head.query_embedding.weight.unsqueeze(0)
        query_pos[:, :head.num_extra, :] = query_pos[:, :head.num_extra, :] * 0

        outs_dec = head.transformer(tgt, memory, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        reference_points = reference_points[:, head.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)
        vlm_memory = outs_dec[-1, :, :head.num_extra, :]
        outs_dec = outs_dec[:, :, head.num_extra:, :]

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = head.cls_branches[lvl](outs_dec[lvl])
            tmp = head.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3])

        rec_can_bus = torch.cat([command.unsqueeze(-1), can_bus], dim=-1)
        memory_ego_pose = memory_egopose.reshape(B, -1, head.topk_proposals, 4, 4).flatten(-2)
        vlm_memory = head.output_projection(vlm_memory)
        can_bus_input = torch.cat([rec_can_bus, memory_canbus.flatten(-2), memory_ego_pose.mean(-2).flatten(-2)], dim=-1)
        can_bus_embed = head.can_bus_embed(can_bus_input)
        vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2)

        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        # topk_indexes = topk_indexes_input
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        memory_embedding = torch.cat([rec_memory, memory_embedding], dim=1)
        memory_timestamp = torch.cat([rec_timestamp, memory_timestamp], dim=1)
        memory_egopose= torch.cat([rec_ego_pose, memory_egopose], dim=1)
        memory_reference_point = torch.cat([rec_reference_points, memory_reference_point], dim=1)
        memory_canbus = torch.cat([rec_can_bus.unsqueeze(-2), memory_canbus], dim=1)
        memory_reference_point = transform_reference_points(memory_reference_point, ego_pose, reverse=False)
        memory_timestamp -= timestamp.unsqueeze(-1).unsqueeze(-1)
        sample_time -= timestamp
        memory_egopose = ego_pose.unsqueeze(1) @ memory_egopose
        return all_cls_scores, all_bbox_preds, vlm_memory, \
            memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_canbus, sample_time

    def forward(self, img, pos_embed_input, command, can_bus, is_first_frame, ego_pose, timestamp, ego_pose_inv,
                memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox,
                memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox,
                memory_timestamp_map, sample_time_map, memory_egopose_map, 
                memory_embedding_map, memory_reference_point_map):
        
        img_feats = self.mod.extract_img_feat(img)
        pos_embed = self.mod.position_encoder(pos_embed_input)

        all_cls_scores, all_bbox_preds, vlm_memory_bbox, \
        memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox, memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox = \
                self.bboxStreamPETR_forward(img_feats, pos_embed, 
                                    command, can_bus, is_first_frame,
                                    memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox,
                                    memory_egopose_bbox, memory_canbus_bbox,
                                    sample_time_bbox, ego_pose, timestamp, ego_pose_inv)
        all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many, all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, vlm_memory_map, \
        memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map = \
                self.mapPETR_forward(img_feats, pos_embed, 
                                     timestamp, ego_pose, ego_pose_inv, is_first_frame,
                                     memory_timestamp_map, sample_time_map, memory_egopose_map, 
                                     memory_embedding_map, memory_reference_point_map)
        
        return all_cls_scores, all_bbox_preds, vlm_memory_bbox, \
            all_lane_cls_one2one, all_lane_preds_one2one, all_lane_cls_one2many, all_lane_preds_one2many, outs_dec_one2one, outs_dec_one2many, vlm_memory_map, \
            memory_embedding_bbox, memory_reference_point_bbox, memory_timestamp_bbox, memory_egopose_bbox, memory_canbus_bbox, sample_time_bbox, \
            memory_embedding_map, memory_timestamp_map, memory_egopose_map, memory_reference_point_map, sample_time_map

def main():
    args = parse_args()
    onnx_device = 'cpu'
    input_precision = np.float32
    cfg = Config.fromfile(args.config)
    output_onnx_pth = "./onnxs/"+args.config.split("/")[-1].split(".")[0]+".onnx"
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

    cfg.model.pretrained = None
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
    assert distributed == False, "onnx export only support non distributed launch."

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=onnx_device)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    model = model.float().cpu()
    # model = model.half()
    model.eval()
    model.training = False

    dumped_input_pth = "./onnxs_data/4f9ad42bb4a24970b770ba0a87baf47a/"
    img = np.fromfile(os.path.join(dumped_input_pth, "img.bin"), dtype=np.float32).astype(np.float32).reshape([1,6,3,640,640])
    pos_embed_input = np.fromfile(os.path.join(dumped_input_pth, "pos_embed_input.bin"), dtype=np.float32).astype(np.float32).reshape([1,9600,192])
    command = np.fromfile(os.path.join(dumped_input_pth, "command.bin"), dtype=np.float32).astype(np.float32).reshape([1])
    can_bus = np.fromfile(os.path.join(dumped_input_pth, "can_bus.bin"), dtype=np.float32).astype(np.float32).reshape([1,13])
    is_first_frame = np.fromfile(os.path.join(dumped_input_pth, "is_first_frame.bin"), dtype=np.float32).astype(np.float32).reshape([1])
    ego_pose = np.fromfile(os.path.join(dumped_input_pth, "ego_pose.bin"), dtype=np.float32).astype(np.float32).reshape([1,4,4])
    timestamp = np.fromfile(os.path.join(dumped_input_pth, "timestamp.bin"), dtype=np.float32).astype(np.float32).reshape([1])
    ego_pose_inv = np.fromfile(os.path.join(dumped_input_pth, "ego_pose_inv.bin"), dtype=np.float32).astype(np.float32).reshape([1,4,4])
    memory_embedding_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "memory_embedding_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,256])
    memory_reference_point_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "memory_reference_point_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,3])
    memory_timestamp_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "memory_timestamp_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,1])
    memory_egopose_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "memory_egopose_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,4,4])
    memory_canbus_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "memory_canbus_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,3,14])
    sample_time_bbox_in = np.fromfile(os.path.join(dumped_input_pth, "sample_time_bbox_in.bin"), dtype=np.float32).astype(np.float32).reshape([1])
    memory_timestamp_map_in = np.fromfile(os.path.join(dumped_input_pth, "memory_timestamp_map_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,1])
    sample_time_map_in = np.fromfile(os.path.join(dumped_input_pth, "sample_time_map_in.bin"), dtype=np.float32).astype(np.float32).reshape([1])
    memory_egopose_map_in = np.fromfile(os.path.join(dumped_input_pth, "memory_egopose_map_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,4,4])
    memory_embedding_map_in = np.fromfile(os.path.join(dumped_input_pth, "memory_embedding_map_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,256])
    memory_reference_point_map_in = np.fromfile(os.path.join(dumped_input_pth, "memory_reference_point_map_in.bin"), dtype=np.float32).astype(np.float32).reshape([1,900,11,3])

    # topk_indexes_bbox = np.fromfile(os.path.join(dumped_input_pth, "topk_indexes_bbox.bin"), dtype=np.float32).astype(np.int64).reshape([1,300,1])
    # topk_indexes_map = np.fromfile(os.path.join(dumped_input_pth, "topk_indexes_map.bin"), dtype=np.float32).astype(np.int64).reshape([1,300,1])

    proxy = OmniDriveVisionTrtProxy(model.module)
    args = [
        torch.from_numpy(img.astype(input_precision)).to(onnx_device),
        torch.from_numpy(pos_embed_input.astype(input_precision)).to(onnx_device),
        torch.from_numpy(command.astype(input_precision)).to(onnx_device),
        torch.from_numpy(can_bus.astype(input_precision)).to(onnx_device),
        torch.from_numpy(is_first_frame.astype(input_precision)).to(onnx_device),
        torch.from_numpy(ego_pose.astype(input_precision)).to(onnx_device),
        torch.from_numpy(timestamp.astype(input_precision)).to(onnx_device),
        torch.from_numpy(ego_pose_inv.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_embedding_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_reference_point_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_timestamp_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_egopose_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_canbus_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(sample_time_bbox_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_timestamp_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(sample_time_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_egopose_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_embedding_map_in.astype(input_precision)).to(onnx_device),
        torch.from_numpy(memory_reference_point_map_in.astype(input_precision)).to(onnx_device),

        # torch.from_numpy(topk_indexes_bbox.astype(np.int64)).to(onnx_device),
        # torch.from_numpy(topk_indexes_map.astype(np.int64)).to(onnx_device),
    ]
    input_names = [
        "img", 
        "pos_embed_input", 
        "command", 
        "can_bus", 
        "is_first_frame", 
        "ego_pose", 
        "timestamp", 
        "ego_pose_inv", 
        "memory_embedding_bbox_in", 
        "memory_reference_point_bbox_in", 
        "memory_timestamp_bbox_in", 
        "memory_egopose_bbox_in", 
        "memory_canbus_bbox_in", 
        "sample_time_bbox_in",
        "memory_timestamp_map_in",
        "sample_time_map_in",
        "memory_egopose_map_in",
        "memory_embedding_map_in",
        "memory_reference_point_map_in",

        # "topk_indexes_bbox", 
        # "topk_indexes_map"
    ]

    output_names = [
        "all_cls_scores", 
        "all_bbox_preds", 
        "vlm_memory_bbox", 
        "all_lane_cls_one2one",
        "all_lane_preds_one2one",
        "all_lane_cls_one2many",
        'all_lane_preds_one2many',
        "outs_dec_one2one",
        "outs_dec_one2many",
        "vlm_memory_map",
        "memory_embedding_bbox_out", 
        "memory_reference_point_bbox_out", 
        "memory_timestamp_bbox_out", 
        "memory_egopose_bbox_out", 
        "memory_canbus_bbox_out", 
        "sample_time_bbox_out", 
        "memory_embedding_map_out",
        "memory_timestamp_map_out",
        "memory_egopose_map_out",
        "memory_reference_point_map_out",
        "sample_time_map_out",

        # "rec_score_bbox", 
        # "rec_score_map"
    ]

    torch.onnx.export(
        proxy, tuple(args),
        output_onnx_pth,
        input_names=input_names, output_names=output_names,
        do_constant_folding=True,
        opset_version=17,
        verbose=True)
    
    onnx_mod = onnx.load(output_onnx_pth)
    onnx_mod, _ = onnxsim.simplify(onnx_mod)
    onnx_mod = onnx.shape_inference.infer_shapes(onnx_mod) 
    onnx.save(onnx_mod, output_onnx_pth)
    print(output_onnx_pth)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
