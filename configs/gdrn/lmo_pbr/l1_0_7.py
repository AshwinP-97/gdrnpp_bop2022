OUTPUT_ROOT = 'output'
OUTPUT_DIR = 'output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo'
EXP_NAME = ''
DEBUG = False
SEED = -1
CUDNN_BENCHMARK = True
IM_BACKEND = 'cv2'
VIS_PERIOD = 0
INPUT = dict(
    FORMAT='BGR',
    MIN_SIZE_TRAIN=480,
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING='choice',
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=False,
    BP_DEPTH=False,
    AUG_DEPTH=False,
    NORM_DEPTH=False,
    DROP_DEPTH_RATIO=0.2,
    DROP_DEPTH_PROB=0.5,
    ADD_NOISE_DEPTH_LEVEL=0.01,
    ADD_NOISE_DEPTH_PROB=0.9,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE='code',
    COLOR_AUG_CODE=
    'Sequential([Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),Sometimes(0.4, GaussianBlur((0., 3.))),Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),Sometimes(0.5, Add((-25, 25), per_channel=0.3)),Sometimes(0.3, Invert(0.2, per_channel=True)),Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),Sometimes(0.5, Multiply((0.6, 1.4))),Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),], random_order=True)',
    COLOR_AUG_SYN_ONLY=False,
    RANDOM_FLIP='none',
    WITH_BG_DEPTH=False,
    BG_DEPTH_FACTOR=10000.0,
    BG_TYPE='VOC_table',
    BG_IMGS_ROOT='/gdrnpp_bop2022/datasets/VOCdevkit/VOC2012/',
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.5,
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    DZI_TYPE='uniform',
    DZI_PAD_SCALE=1.5,
    DZI_SCALE_RATIO=0.25,
    DZI_SHIFT_RATIO=0.25,
    SMOOTH_XYZ=False)
DATASETS = dict(
    TRAIN=('lmo_pbr_train', ),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    DATA_LEN_WITH_TRAIN2=True,
    PROPOSAL_FILES_TRAIN=(),
    PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
    TEST=('lmo_bop_test', ),
    PROPOSAL_FILES_TEST=(),
    PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    DET_FILES_TRAIN=(),
    DET_TOPK_PER_OBJ_TRAIN=1,
    DET_TOPK_PER_IM_TRAIN=30,
    DET_THR_TRAIN=0.0,
    DET_FILES_TEST=
    ('datasets/BOP_DATASETS/lmo/test/test_bboxes/yolox_x_640_lmo_pbr_lmo_bop_test.json',
     ),
    DET_TOPK_PER_OBJ=1,
    DET_TOPK_PER_IM=30,
    DET_THR=0.0,
    INIT_POSE_FILES_TEST=(),
    INIT_POSE_TOPK_PER_OBJ=1,
    INIT_POSE_TOPK_PER_IM=30,
    INIT_POSE_THR=0.0,
    SYM_OBJS=['bowl', 'cup', 'eggbox', 'glue'],
    EVAL_SCENE_IDS=None)
DATALOADER = dict(
    NUM_WORKERS=8,
    PERSISTENT_WORKERS=False,
    MAX_OBJS_TRAIN=120,
    ASPECT_RATIO_GROUPING=False,
    SAMPLER_TRAIN='TrainingSampler',
    REPEAT_THRESHOLD=0.0,
    FILTER_EMPTY_ANNOTATIONS=True,
    FILTER_EMPTY_DETS=True,
    FILTER_VISIB_THR=0.3,
    REMOVE_ANNO_KEYS=[])
SOLVER = dict(
    IMS_PER_BATCH=48,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=40,
    OPTIMIZER_CFG=dict(type='Ranger', lr=0.0008, weight_decay=0.01),
    GAMMA=0.1,
    BIAS_LR_FACTOR=1.0,
    LR_SCHEDULER_NAME='flat_and_anneal',
    WARMUP_METHOD='linear',
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    ANNEAL_METHOD='cosine',
    ANNEAL_POINT=0.72,
    POLY_POWER=0.9,
    REL_STEPS=(0.5, 0.75),
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
    CLIP_GRADIENTS=dict(
        ENABLED=False, CLIP_TYPE='value', CLIP_VALUE=1.0, NORM_TYPE=2.0),
    SET_NAN_GRAD_TO_ZERO=False,
    AMP=dict(ENABLED=False),
    WEIGHT_DECAY=0.01,
    OPTIMIZER_NAME='Ranger',
    BASE_LR=0.0008,
    MOMENTUM=0.9)
TRAIN = dict(PRINT_FREQ=100, VERBOSE=False, VIS=False, VIS_IMG=False)
VAL = dict(
    DATASET_NAME='lmo',
    SCRIPT_PATH='lib/pysixd/scripts/eval_pose_results_more.py',
    RESULTS_PATH='',
    TARGETS_FILENAME='test_targets_bop19.json',
    ERROR_TYPES='mspd,mssd,vsd,ad,reS,teS',
    RENDERER_TYPE='cpp',
    SPLIT='test',
    SPLIT_TYPE='',
    N_TOP=1,
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=True,
    SAVE_BOP_CSV_ONLY=False)
TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE='est',
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    AMP_TEST=False,
    COLOR_AUG=False,
    USE_PNP=False,
    SAVE_RESULTS_ONLY=False,
    PNP_TYPE='ransac_pnp',
    USE_DEPTH_REFINE=False,
    DEPTH_REFINE_ITER=2,
    DEPTH_REFINE_THRESHOLD=0.8,
    USE_COOR_Z_REFINE=False)
DIST_PARAMS = dict(backend='nccl')
MODEL = dict(
    DEVICE='cuda',
    WEIGHTS=
    '/gdrnpp_bop2022/output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.pth',
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_DETS_TEST=True,
    BBOX_CROP_REAL=False,
    BBOX_CROP_SYN=False,
    BBOX_TYPE='AMODAL_CLIP',
    EMA=dict(ENABLED=False, INIT_CFG=dict(decay=0.9999, updates=0)),
    POSE_NET=dict(
        NAME='GDRN_double_mask',
        XYZ_ONLINE=True,
        XYZ_BP=True,
        NUM_CLASSES=8,
        USE_MTL=False,
        INPUT_RES=256,
        OUTPUT_RES=64,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED='timm',
            INIT_CFG=dict(
                type='timm/convnext_base',
                in_chans=3,
                features_only=True,
                pretrained=True,
                out_indices=(3, ))),
        DEPTH_BACKBONE=dict(
            ENABLED=False,
            FREEZE=False,
            PRETRAINED='timm',
            INIT_CFG=dict(
                type='timm/resnet18',
                in_chans=1,
                features_only=True,
                pretrained=True,
                out_indices=(4, ))),
        FUSE_RGBD_TYPE='cat',
        NECK=dict(
            ENABLED=False,
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4)),
        GEO_HEAD=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type='TopDownDoubleMaskXyzRegionHead',
                in_dim=1024,
                up_types=('deconv', 'bilinear', 'bilinear'),
                deconv_kernel_size=3,
                num_conv_per_block=2,
                feat_dim=256,
                feat_kernel_size=3,
                norm='GN',
                num_gn_groups=32,
                act='GELU',
                out_kernel_size=1,
                out_layer_shared=True),
            XYZ_BIN=64,
            XYZ_CLASS_AWARE=True,
            MASK_CLASS_AWARE=True,
            REGION_CLASS_AWARE=True,
            MASK_THR_TEST=0.5,
            NUM_REGIONS=64),
        PNP_NET=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type='ConvPnPNet',
                norm='GN',
                act='gelu',
                num_gn_groups=32,
                drop_prob=0.0,
                denormalize_by_extent=True),
            WITH_2D_COORD=True,
            COORD_2D_TYPE='abs',
            REGION_ATTENTION=True,
            MASK_ATTENTION='none',
            ROT_TYPE='allo_rot6d',
            TRANS_TYPE='centroid_z',
            Z_TYPE='REL'),
        LOSS_CFG=dict(
            XYZ_LOSS_TYPE='L1',
            XYZ_LOSS_MASK_GT='visib',
            XYZ_LW=1.0,
            FULL_MASK_LOSS_TYPE='L1',
            FULL_MASK_LW=1.0,
            MASK_LOSS_TYPE='L1',
            MASK_LOSS_GT='trunc',
            MASK_LW=1.0,
            REGION_LOSS_TYPE='CE',
            REGION_LOSS_MASK_GT='visib',
            REGION_LW=1.0,
            NUM_PM_POINTS=3000,
            PM_LOSS_TYPE='L1',
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=True,
            PM_NORM_BY_EXTENT=False,
            PM_R_ONLY=True,
            PM_DISENTANGLE_T=False,
            PM_DISENTANGLE_Z=False,
            PM_T_USE_POINTS=True,
            PM_LW=1.0,
            ROT_LOSS_TYPE='angular',
            ROT_LW=0.0,
            CENTROID_LOSS_TYPE='L1',
            CENTROID_LW=1.0,
            Z_LOSS_TYPE='L1',
            Z_LW=1.0,
            TRANS_LOSS_TYPE='L1',
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            BIND_LOSS_TYPE='L1',
            BIND_LW=0.0)),
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False)
FAST = dict(
    FLAG=True,
    OUTPUT_DIR=
    '/gdrnpp_bop2022/output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/Fast',
    WEIGHTS=
    '/gdrnpp_bop2022/output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/Fast/prune_l1_0_7.pth',
    INSPECT=True,
    THRESHOLD_GEO=0,
    THRESHOLD_PNP=7,
    MODEL_CFG=dict(
        PRUNE=True,
        GEO_HEAD_feat=256,
        GEO_HEAD_num_groups=32,
        PNP_NET_Input=100,
        PNP_NET_num_groups=25,
        PNP_NET_fc_out=996,
        PNP_NET_fc2_out=228))
EXP_ID = 'lmo_fast_test'
RESUME = False
