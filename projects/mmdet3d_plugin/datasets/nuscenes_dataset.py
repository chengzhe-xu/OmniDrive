# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import tempfile
import numpy as np
import json
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
from os import path as osp
import mmcv
import random
import math
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from .utils.data_utils import preprocess
from .utils.constants import DEFAULT_IMAGE_TOKEN
import copy
import pickle
import os
from openlanev2.centerline.evaluation import evaluate as openlanev2_evaluate
@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 tokenizer,
                 max_pair=10,
                 seq_mode=False, 
                 seq_split_num=1, 
                 drivelm_path='./data/nuscenes/restructured_drivelm_data.json',
                 nuscqa_path='./data/nuscenes/restructured_nuscQA_train_data.json',
                 lane_path='./data/nuscenes/data_dict_sample.pkl',
                 eval_mode=['text', 'det'],
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.seq_mode = seq_mode
        self.eval_mode = eval_mode
        self.lane_info = pickle.load(open(lane_path, 'rb'))
        self.vqa_data = dict()
        self.lane_anno_file = './data/nuscenes/data_dict_subset_B_val.pkl'
        
        if drivelm_path is not None:
            drive_lm = self.preprocess_drivelm(drivelm_path)
            merge_dicts(self.vqa_data, drive_lm)
        
        if nuscqa_path is not None:
            nuscqa = self.preprocess_nuscqa(nuscqa_path)
            merge_dicts(self.vqa_data, nuscqa)
            
        self.max_pair = max_pair
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                           model_max_length=1024,
                                           padding_side="right",
                                           use_fast=False,
                                           )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        if seq_mode:
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def preprocess_drivelm(self, vqa_path):
        vqa_data = dict()
        with open(vqa_path,'r',encoding='utf8')as fp:
            ori_data = json.load(fp)
        for sample_token in ori_data.keys():
            sources = []
            for _, data in ori_data[sample_token]['frame_data']['QA'].items():
                for qa in data:
                    sources.append(
                        [{"from": 'human',
                                    "value": qa['Q']},
                                    {"from": 'gpt',
                                    "value": qa['A']}]
                    )
            vqa_data[sample_token] = sources
        return vqa_data
  
    def preprocess_nuscqa(self, vqa_path):
        vqa_data = dict()
        with open(vqa_path,'r',encoding='utf8')as fp:
            ori_data = json.load(fp)
        for sample_token in ori_data.keys():
            sources = []
            for qa in ori_data[sample_token]:
                sources.append(
                    [{"from": 'human',
                                "value": qa['Q']},
                                {"from": 'gpt',
                                "value": qa['A']}]
                )
            vqa_data[sample_token] = sources
        return vqa_data  
    
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                # assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        ego_pose = e2g_matrix

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        vqa_anno = []
        if info['token'] in self.vqa_data.keys():
            vqa_anno = copy.deepcopy(self.vqa_data[info['token']])
        else:
            vqa_anno = [[{"from": 'human',
                                "value": ''},
                                {"from": 'gpt',
                                "value": ''}]]
        if not self.test_mode:         
            random.shuffle(vqa_anno)
            vqa_anno = vqa_anno[:self.max_pair]
            vqa_anno = [item for pair in vqa_anno for item in pair]
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + vqa_anno[0]['value']  
            vqa_converted = preprocess([vqa_anno], self.tokenizer, True)
            input_ids = vqa_converted['input_ids'][0]
            vlm_labels = vqa_converted['labels'][0]
        else:
            vlm_labels = [anno[0]['value'] for anno in vqa_anno]     
            for anno in vqa_anno:
                anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + anno[0]['value']  
                anno[1]['value'] = ''
            vqa_converted = preprocess(vqa_anno, self.tokenizer, True, False)
            input_ids = vqa_converted['input_ids']
            
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            input_ids=input_ids,
            vlm_labels=vlm_labels,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                cam2lidar_t = cam_info['sensor2ego_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            if 'lane_info' not in info.keys():
                lane_pts = []
            else:
                centerline = self.lane_info[info['lane_info']]['annotation']['lane_centerline']
                lane_pts = [lane['points'] for lane in centerline] 
                   
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore']),
                    lane_pts=lane_pts,

            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        results_dict = dict()
        if 'lane' in self.eval_mode:
            predictions = {
            'method': 'dummy',
            'authors': ['dummy'],
            'e-mail': 'dummy',
            'institution / company': 'dummy',
            'country / region': 'China',
            'results': {},
        }
            for index, result in enumerate(results):
                prediction = {                
                    'lane_centerline': [],
                    'traffic_element': [],
                    'topology_lclc': np.zeros((300, 300)),
                    'topology_lcte': np.zeros((300, 0)),
                }
                lanes, confidences = result['lane_results'][0]
                for i, (lane, confidence) in enumerate(zip(lanes, confidences)):
                    prediction['lane_centerline'].append({
                        'id': i,
                        'points': lane.astype(np.float32),
                        'confidence': confidence.item(),
                    })
                sample_token = self.data_infos[index]['lane_info']
                predictions['results'][sample_token] = {
                    'predictions': prediction,
                }
            metric_results = {}
            for key, val in openlanev2_evaluate(ground_truth=self.lane_anno_file, predictions=predictions).items():
                for k, v in val.items():
                    metric_results[k if k != 'score' else key] = v
            print(metric_results)
                
        
        if 'det' in self.eval_mode:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            if isinstance(result_files, dict):
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files)

            if tmp_dir is not None:
                tmp_dir.cleanup()

            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)
            
        if 'text' in self.eval_mode:
            from datetime import datetime
            text_out = {}
            for sample_id, text in enumerate(mmcv.track_iter_progress(results)):
                sample_token = self.data_infos[sample_id]['token']
                text_out[sample_token] = text['text_out']
                
            now = datetime.now()
            # 格式化时间字符串
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # 创建文件名
            mmcv.mkdir_or_exist("./vlm_out")
            file_name = f"./vlm_out/text_out_{timestamp}.json"

            # 将数据写入 JSON 文件
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(text_out, f, ensure_ascii=False, indent=4)

        return results_dict

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        if 'det' in self.eval_mode:
            metric_prefix = f'{result_name}_NuScenes'
            for name in self.CLASSES:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix,
                                        self.ErrNameMapping[k])] = val

            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))
        
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

 
        result_files = dict()
        name = 'pts_bbox'
        print(f'\nFormating bboxes of {name}')
        results_ = [out[name] for out in results]
        tmp_file_ = osp.join(jsonfile_prefix, name)
        result_files.update(
            {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        # box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

def merge_dicts(base_dict, new_dict):
    for key, new_value in new_dict.items():
        if key in base_dict:
            # 确保基础字典中的值也是列表类型
            base_value = base_dict[key]
            if not isinstance(base_value, list):
                raise ValueError(f'在基础字典中，键 "{key}" 对应的值不是列表。')
            # 合并列表
            base_dict[key].extend(new_value)
        else:
            # 如果键不存在，直接添加
            base_dict[key] = new_value