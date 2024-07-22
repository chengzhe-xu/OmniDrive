point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesDatasetv2'
data_root = './data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.37, 0.45),
            final_dim=(320, 640),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=False),
        training=True),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(
        type='LoadAnnoatationVQA',
        base_vqa_path='./data/nuscenes/vqa/train/',
        base_desc_path='./data/nuscenes/desc/train/',
        base_conv_path='./data/nuscenes/conv/train/',
        tokenizer='ckpts/final/',
        max_length=2048,
        ignore_type=[],
        lane_objs_info='./data/nuscenes/lane_obj_train.pkl'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus',
            'prev_exists'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'lane_pts', 'input_ids', 'vlm_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d',
            'depths', 'prev_exists', 'lidar2img', 'intrinsics', 'extrinsics',
            'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv',
            'command', 'can_bus'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                   'gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.37, 0.45),
            final_dim=(320, 640),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=False),
        training=False),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadAnnoatationVQATest',
        base_vqa_path='./data/nuscenes/vqa/val/',
        base_conv_path='./data/nuscenes/conv/val/',
        base_counter_path='./eval_cf/',
        load_type=['conv'],
        tokenizer='ckpts/final/',
        max_length=2048),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command',
                    'can_bus'
                ],
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'input_ids', 'img', 'lidar2img', 'intrinsics',
                    'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose',
                    'ego_pose_inv', 'command', 'can_bus'
                ],
                meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'scene_token'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CustomNuScenesDatasetv2',
        data_root='./data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_bbox_depth=True),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=True),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(
                type='LoadAnnoatationVQA',
                base_vqa_path='./data/nuscenes/vqa/train/',
                base_desc_path='./data/nuscenes/desc/train/',
                base_conv_path='./data/nuscenes/conv/train/',
                tokenizer='ckpts/final/',
                max_length=2048,
                ignore_type=[],
                lane_objs_info='./data/nuscenes/lane_obj_train.pkl'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='PETRFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command',
                    'can_bus', 'prev_exists'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'lane_pts', 'input_ids', 'vlm_labels', 'gt_bboxes_3d',
                    'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
                    'centers2d', 'depths', 'prev_exists', 'lidar2img',
                    'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                    'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token',
                           'gt_bboxes_3d', 'gt_labels_3d'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        seq_split_num=1,
        seq_mode=True,
        use_valid_flag=True,
        filter_empty_gt=False),
    val=dict(
        type='CustomNuScenesDatasetv2',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=False),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadAnnoatationVQATest',
                base_vqa_path='./data/nuscenes/vqa/val/',
                base_conv_path='./data/nuscenes/conv/val/',
                base_counter_path='./eval_cf/',
                load_type=['conv'],
                tokenizer='ckpts/final/',
                max_length=2048),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv', 'command', 'can_bus'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'input_ids', 'img', 'lidar2img', 'intrinsics',
                            'extrinsics', 'timestamp', 'img_timestamp',
                            'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
                        ],
                        meta_keys=('sample_idx', 'vlm_labels', 'filename',
                                   'ori_shape', 'img_shape', 'pad_shape',
                                   'scale_factor', 'flip', 'box_mode_3d',
                                   'box_type_3d', 'img_norm_cfg',
                                   'scene_token'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        eval_mode=['lane', 'det']),
    test=dict(
        type='CustomNuScenesDatasetv2',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=False),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadAnnoatationVQATest',
                base_vqa_path='./data/nuscenes/vqa/val/',
                base_conv_path='./data/nuscenes/conv/val/',
                base_counter_path='./eval_cf/',
                load_type=['conv'],
                tokenizer='ckpts/final/',
                max_length=2048),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv', 'command', 'can_bus'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'input_ids', 'img', 'lidar2img', 'intrinsics',
                            'extrinsics', 'timestamp', 'img_timestamp',
                            'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
                        ],
                        meta_keys=('sample_idx', 'vlm_labels', 'filename',
                                   'ori_shape', 'img_shape', 'pad_shape',
                                   'scale_factor', 'flip', 'box_mode_3d',
                                   'box_type_3d', 'img_norm_cfg',
                                   'scene_token'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        eval_mode=['lane', 'det']),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        warmup_split_num=10,
        num_iters_to_seq=1758),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=42192,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='ResizeCropFlipRotImage',
            data_aug_conf=dict(
                resize_lim=(0.37, 0.45),
                final_dim=(320, 640),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=False),
            training=False),
        dict(
            type='ResizeMultiview3D',
            img_scale=(640, 640),
            keep_ratio=False,
            multiscale_mode='value'),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='LoadAnnoatationVQATest',
            base_vqa_path='./data/nuscenes/vqa/val/',
            base_conv_path='./data/nuscenes/conv/val/',
            base_counter_path='./eval_cf/',
            load_type=['conv'],
            tokenizer='ckpts/final/',
            max_length=2048),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PETRFormatBundle3D',
                    collect_keys=[
                        'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                        'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command',
                        'can_bus'
                    ],
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'input_ids', 'img', 'lidar2img', 'intrinsics',
                        'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose',
                        'ego_pose_inv', 'command', 'can_bus'
                    ],
                    meta_keys=('sample_idx', 'vlm_labels', 'filename',
                               'ori_shape', 'img_shape', 'pad_shape',
                               'scale_factor', 'flip', 'box_mode_3d',
                               'box_type_3d', 'img_norm_cfg', 'scene_token'))
            ])
    ])
checkpoint_config = dict(interval=879, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mask_eva_base_lane_det'
load_from = 'ckpts/eva02_B_coco_remapped.pth'
resume_from = './work_dirs/mask_eva_base_lane_det/latest.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_gpus = 8
batch_size = 2
num_iters_per_epoch = 1758
num_epochs = 24
llm_path = 'ckpts/final/'
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
]
model = dict(
    type='Petr3D',
    save_path='./results_planning_only/',
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    tokenizer='ckpts/final/',
    lm_head=None,
    img_backbone=dict(
        type='EVAViT',
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=2.6666666666666665,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        qkv_bias=True,
        drop_path_rate=0.1,
        flash_attn=True,
        with_cp=True,
        frozen=False),
    map_head=dict(
        type='PETRHeadM',
        num_classes=1,
        in_channels=768,
        out_dims=None,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        code_weights=[1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='LaneHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.02),
                iou_cost=dict(type='IoUCost', weight=0.0))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.02),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)),
    pts_bbox_head=dict(
        type='StreamPETRHead',
        num_classes=10,
        in_channels=768,
        out_dims=None,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,
        n_control=11,
        match_with_velo=False,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
ida_aug_conf = dict(
    resize_lim=(0.37, 0.45),
    final_dim=(320, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=False)
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
find_unused_parameters = False
runner = dict(type='IterBasedRunner', max_iters=42192)
gpu_ids = range(0, 8)
