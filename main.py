import argparse
import os, tqdm
import shutil
import time
from os.path import join

# 限制底层库的并行线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0) # 禁止 OpenCV 多线程
cv2.ocl.setUseOpenCL(False) # 禁止 OpenCL
import numpy as np

# 新增 nuScenes 相关库
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion 
import json

from datasets.coordinate_transformation import convert_x1y1x2y2c_to_tlwhc
from datasets.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from tracking.DeepFusionMOT import DeepFusionMOT
from utils.config import Config
from evaluation.KITTI.evaluation_HOTA.scripts.run_kitti import eval_kitti
from utils.combine_trk_cat import combine_category_result
from datasets.data_fusion import data_fusion
from utils.save_results import save_results

# 假设你把上一轮写的 Loader 保存为 utils/nuscenes_2d_loader.py
from utils.nuscenes_2d_loader import NuScenes2DLoader
# 保存nuscenes格式数据
from utils.save_nusc_results import format_nusc_outputs, save_nuscenes_json

# ==============================================================================
# 新增的两个核心格式转换与投影工具函数
# ==============================================================================
def project_nusc_to_kitti_format(nusc, current_token, raw_3d_boxes, category, img_shape=(1600, 900)):
    """
    将 nuScenes 的 Global 3D 检测结果，投影至前视相机(CAM_FRONT)并组装为原有代码所需的 numpy array。
    """
    dets_3d = []
    dets_2d_proj = []
    add_info = []

    # 1. 提取相机的内外参以及自车运动补偿数据
    sample = nusc.get('sample', current_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])

    # 2. 类别过滤 (CenterPoint 输出中类别字段通常是 detection_name)
    boxes_filtered = [
            b for b in raw_3d_boxes 
            if b.get('detection_name', b.get('tracking_name', '')).lower() == category.lower()
        ]
    for box_dict in boxes_filtered:
        w, l, h = box_dict['size']
        x, y, z = box_dict['translation']
        q = box_dict['rotation']
        score = box_dict.get('detection_score', box_dict.get('tracking_score', 1.0))
        
        quat = Quaternion(q)
        yaw = quat.yaw_pitch_roll[0] # 获取绕 Z 轴的偏航角

        # A. 组装 3D 观测数据 [h, w, l, x, y, z, theta] -> 注意：你的卡尔曼滤波将在全局坐标系下直接平稳运行！
        dets_3d.append([h, w, l, x, y, z, yaw])

        # B. 跨坐标系投影：计算 3D 框在 2D 图像上的投影 (用于 Ro_GDIoU 匹配)
        box = Box(box_dict['translation'], box_dict['size'], Quaternion(box_dict['rotation']))
        # 一步步逆推: Global -> 自车坐标系 (Ego)
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        # 自车坐标系 -> 相机传感器坐标系 (Cam)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # 判断物体是否在相机前方
        if box.center[2] > 0.1: 
            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            x1, x2 = corners[0, :].min(), corners[0, :].max()
            y1, y2 = corners[1, :].min(), corners[1, :].max()
            # 限制在图像边界内
            # 彻底在外面的情况：最大值小于0，或最小值大于分辨率
            if x2 < 0 or x1 > img_shape[0] or y2 < 0 or y1 > img_shape[1]:
                bbox_2d = [0, 0, 0, 0] # 彻底不可见，安全丢弃
            else:
                # 此时必然存在至少一部分在屏幕内，安全截断
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_shape[0], x2), min(img_shape[1], y2)
                bbox_2d = [x1, y1, x2, y2]
        else:
            bbox_2d = [0, 0, 0, 0] # 物体在车后方，不投影

        dets_2d_proj.append(bbox_2d)

        # C. 组装 additional_info: [alpha, type_id, x1, y1, x2, y2, score]
        type_id = 1 if category == 'car' else 2
        alpha = 0.0 # nuScenes 评测不强依赖 alpha，设为 0 防止报错
        add_info.append([alpha, type_id, bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3], score])

    if len(dets_3d) == 0:
        return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 4))
    
    return np.array(dets_3d), np.array(add_info), np.array(dets_2d_proj)


def format_nusc_2d_to_kitti_format(raw_2d_boxes_dict, frame_idx):
    """
    将 NuScenes2DLoader 吐出的字典，转换为你原系统兼容的 N x 7 numpy 数组
    """
    # 为了完美兼容你代码后续的单图片读取和匹配逻辑，我们主要提取前视相机的 2D 框作为 RV 的匹配依据
    front_boxes = raw_2d_boxes_dict.get('CAM_FRONT', [])
    dets_2d = []
    for det in front_boxes:
        b = det['bbox']
        score = det['score']
        # KITTI 格式: [frame_idx, x1, y1, x2, y2, score, loc_score(用 score 兜底)]
        dets_2d.append([frame_idx, b[0], b[1], b[2], b[3], score, score])
        
    if len(dets_2d) == 0:
        return np.empty((0, 7))
    return np.array(dets_2d)


# ==============================================================================
# 主函数部分
# ==============================================================================
def tracking(cfg):
    spilt = cfg.spilt
    seq_list = cfg.tracking_seqs
    total_time, total_frames = 0, 0
    
    nusc = None
    det_3d_results = {}
    loader_2d = None
    nusc_seq_list = [] # 新增：存放真实的 nuScenes 场景名
    
    nusc_submissions = {
            "meta": {
                "use_camera": True, "use_lidar": True, "use_radar": False, 
                "use_map": False, "use_external": False
            },
            "results": {}
        }
    
    if cfg.dataset == 'nuscenes':
        # 动态决定 version
        if cfg.spilt in ['train', 'val']:
            nusc_version = 'v1.0-trainval'
        elif cfg.spilt == 'test':
            nusc_version = 'v1.0-test'
        else:
            nusc_version = 'v1.0-trainval'
            
        nusc = NuScenes(version=nusc_version, dataroot=cfg.dataset_path, verbose=False) 
        print("NuScenes initialized.")
        
        print("Loading CenterPoint 3D JSON...")
        with open(cfg.nusc_3d_json_path, 'r') as f:
            det_3d_results = json.load(f)['results']
            
        print("Initializing NuScenes 2D Loader...")
        loader_2d = NuScenes2DLoader(cfg.dataset_path, cfg.nusc_2d_json_dir)

        # --- 【核心修复：自动获取合法场景名】 ---
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        # 根据 val/test 获取对应的全部合法场景名 (如 'scene-0014')
        nusc_seq_list = splits[cfg.spilt] 
        
        # # 如果是 mini 验证 (通过 YAML 中的 tracking_seqs 长度控制测试数量)
        # if len(seq_list) < 10: 
        #     nusc_seq_list = nusc_seq_list[:len(seq_list)]
        #     print(f"Mini test: automatically running on {nusc_seq_list}")

    for category in cfg.cat_list:
        junk_conf = cfg[category]["fga"]["junk_conf"]
        high_conf = cfg[category]["fga"]["high_conf"]
        loc_thresh = cfg[category]["fga"]["loc_thresh"] 
        
        # 兼容两种数据集的循环列表
        loop_seqs = nusc_seq_list if cfg.dataset == 'nuscenes' else seq_list

        # 修正原来的 enumerate(tqdm(len(xxx))) 逻辑
        for real_seq_id in tqdm.tqdm(loop_seqs):
            seq_id = real_seq_id
            tracker = DeepFusionMOT(cfg, category)
            
            if cfg.dataset == 'nuscenes':
                target_scene_name = real_seq_id  # 此时这里已经是真实的名称，比如 'scene-0014'
                seq_name = real_seq_id
                
                try:
                    scene = next(s for s in nusc.scene if s['name'] == target_scene_name)
                except StopIteration:
                    print(f"[Warning] Scene {target_scene_name} not found in nuScenes!")
                    continue
                
                # 获取该场景所有对齐好的 2D 框字典
                aligned_2d_boxes = loader_2d.load_aligned_2d_boxes(target_scene_name)
                current_token = scene['first_sample_token']
                frame_idx = 0 
                
                # --- 补全的 nuScenes 核心循环结构 ---
                while current_token != "":
                    start_time = time.time()
                    
                    # 获取当前帧的前视图片 (为了兼容你的 save_results 可视化)
                    sample = nusc.get('sample', current_token)
                    cam_token = sample['data']['CAM_FRONT']
                    cam_data = nusc.get('sample_data', cam_token)
                    img0_path = os.path.join(nusc.dataroot, cam_data['filename'])
                    img_0 = cv2.imread(img0_path)
                    
                    # 1. 获取检测框
                    raw_3d_boxes = det_3d_results.get(current_token, [])
                    raw_2d_boxes_dict = aligned_2d_boxes.get(current_token, {})
                    
                    # 📍 探针 1：看看 CenterPoint 当前帧有没有读出数据
                    # print(f"[{frame_idx}] 读入的原始 3D 框总数: {len(raw_3d_boxes)}")
                    
                    # 2. 转换和投影
                    dets_3d_camera, additional_info, dets_3dto2d_image = project_nusc_to_kitti_format(
                        nusc, current_token, raw_3d_boxes, category
                    )
                    dets_2d_input = format_nusc_2d_to_kitti_format(raw_2d_boxes_dict, frame_idx)
                    
                    # 📍 探针 2：看看过滤类别后，还剩几个 3D 框
                    # if frame_idx < 3: # 只打印前 3 帧避免刷屏
                    #     print(f"[{frame_idx}] 类别 {category} 过滤后的 3D 框数量: {len(dets_3d_camera)}")
                        
                    # === 3. 复用原有的融合与跟踪逻辑 ===
                    frame_mask = (dets_2d_input[:, 0] == frame_idx) & (dets_2d_input[:, 5] > junk_conf)
                    current_dets = dets_2d_input[frame_mask]
                    dets_high = np.empty((0, 7))
                    dets_low_valid = np.empty((0, 7))
                    
                    if len(current_dets) > 0:
                        scores_final = current_dets[:, 5]
                        mask_high = scores_final > high_conf
                        dets_high = current_dets[mask_high]
                        if cfg["use_fga"] == "True" and current_dets.shape[1] > 6:
                            scores_loc = current_dets[:, 6]
                            mask_low_score = (scores_final < high_conf) & (scores_final > junk_conf)
                            mask_loc_valid = scores_loc > loc_thresh
                            dets_low_valid = current_dets[mask_low_score & mask_loc_valid]
                            
                    if cfg["use_fga"] == "True":
                        if len(dets_high) > 0 and len(dets_low_valid) > 0:
                            dets_2d_combined = np.concatenate((dets_high, dets_low_valid), axis=0)
                        elif len(dets_high) > 0:
                            dets_2d_combined = dets_high
                        elif len(dets_low_valid) > 0:
                            dets_2d_combined = dets_low_valid
                        else:
                            dets_2d_combined = np.empty((0, 7))
                    else:
                        dets_2d_combined = dets_high
                        
                    dets_2d_input_fused = dets_2d_combined
                    
                    # Data Fusion
                    dets_3d_fusion, dets_3d_only, dets_2d_only_list = \
                        data_fusion(dets_3d_camera, dets_2d_input_fused, dets_3dto2d_image, additional_info)

                    dets_2d_high_tlwhc = []
                    dets_2d_low_tlwhc = []

                    if len(dets_2d_only_list) > 0:
                        dets_2d_only_array = np.array(dets_2d_only_list) 
                        scores = dets_2d_only_array[:, 5]
                        high_mask = scores >= high_conf
                        low_mask = ~high_mask 
                        raw_high = dets_2d_only_array[high_mask]
                        raw_low = dets_2d_only_array[low_mask]

                        if len(raw_high) > 0:
                            dets_2d_high_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_high])
                        if len(raw_low) > 0:
                            dets_2d_low_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_low])
                    
                    if len(dets_2d_high_tlwhc) == 0: dets_2d_high_tlwhc = np.empty((0, 5))
                    if len(dets_2d_low_tlwhc) == 0: dets_2d_low_tlwhc = np.empty((0, 5))

                    trackers = tracker.update(dets_3d_fusion,
                                              dets_2d_high_tlwhc, 
                                              dets_2d_low_tlwhc, 
                                              dets_3d_only,
                                              cfg,
                                              frame_idx,
                                              seq_id)
                    
                    # 📍 探针 3：看看跟踪器最终有没有输出
                    # if frame_idx < 3:
                    #     print(f"[{frame_idx}] Tracker 吐出的轨迹数: {len(trackers)}\n")
                        
                    cycle_time = time.time() - start_time
                    total_time += cycle_time
                    total_frames += 1
                    
                    # 将这一帧的结果转为字典并追加到总表
                    formatted_res = format_nusc_outputs(trackers, current_token, category)
                    # 确保该 token 在字典中被初始化
                    if current_token not in nusc_submissions["results"]:
                        nusc_submissions["results"][current_token] = []
                    
                    nusc_submissions["results"][current_token].extend(formatted_res)
                    # ==================

                    # 原有的可视化代码，你可以保留或者暂时注释掉
                    # save_results(trackers, cfg, seq_name, frame_idx, category, img_0)
                    
                    # 指针步进
                    current_token = sample['next']
                    frame_idx += 1
            else:
                # =========================================================
                # 保持你原有的 KITTI 处理逻辑完全不变
                # =========================================================
                seq_name = str(real_seq_id).zfill(4)
                dets_path_3d = os.path.join(cfg.dets_path_3d, cfg.detector_3d, spilt, category) + "/" + seq_name + '.txt'
                dets_path_2d = os.path.join(cfg.dets_path_2d, cfg.detector_2d, spilt, category) + "/" + seq_name + '.txt'
                
                image_02_path = os.path.join(cfg.dataset_path, spilt, 'image_02') + "/" + seq_name
                if os.path.exists(image_02_path):
                    filenames = os.listdir(image_02_path)
                    sorted_filenames = sorted(filenames)
                    image_filenames = [join(image_02_path, x) for x in sorted_filenames]
                else:
                    image_filenames = []
                
                dets_3d = np.loadtxt(dets_path_3d, delimiter=',') 
                dets_2d = np.loadtxt(dets_path_2d, delimiter=',')

                min_frame, max_frame = 0, len(image_filenames)
                for frame in tqdm.trange(max_frame):
                    img0_path = image_filenames[frame]
                    img_0 = cv2.imread(img0_path)
                    dets_3d_camera = dets_3d[dets_3d[:, 0] == frame, 7:14]  

                    ori_array = dets_3d[dets_3d[:, 0] == frame, -1].reshape((-1, 1)) 
                    other_array = dets_3d[dets_3d[:, 0] == frame, 1:7] 
                    additional_info = np.concatenate((ori_array, other_array), axis=1)
                    dets_3dto2d_image = dets_3d[dets_3d[:, 0] == frame, 2:6] 

                    frame_mask = (dets_2d[:, 0] == frame) & (dets_2d[:, 5] > junk_conf)
                    current_dets = dets_2d[frame_mask]
                    dets_high = np.empty((0, 7))
                    dets_low_valid = np.empty((0, 7))
                    if len(current_dets) > 0:
                        scores_final = current_dets[:, 5]
                        mask_high = scores_final > high_conf
                        dets_high = current_dets[mask_high]
                        if  cfg["use_fga"] == "True":
                            if current_dets.shape[1] > 6:
                                scores_loc = current_dets[:, 6]
                                mask_low_score = (scores_final < high_conf) & (scores_final > junk_conf)
                                mask_loc_valid = scores_loc > loc_thresh
                                dets_low_valid = current_dets[mask_low_score & mask_loc_valid]
                            
                    if cfg["use_fga"] == "True":
                        if len(dets_high) > 0 and len(dets_low_valid) > 0:
                            dets_2d_combined = np.concatenate((dets_high, dets_low_valid), axis=0)
                        elif len(dets_high) > 0:
                            dets_2d_combined = dets_high
                        elif len(dets_low_valid) > 0:
                            dets_2d_combined = dets_low_valid
                        else:
                            dets_2d_combined = np.empty((0, 7))
                    else:
                        dets_2d_combined = dets_high
                        
                    dets_2d_input = dets_2d_combined
                    
                    dets_3d_fusion, dets_3d_only, dets_2d_only_list = \
                        data_fusion(dets_3d_camera, dets_2d_input, dets_3dto2d_image, additional_info)

                    dets_2d_high_tlwhc = []
                    dets_2d_low_tlwhc = []

                    if len(dets_2d_only_list) > 0:
                        dets_2d_only_array = np.array(dets_2d_only_list)
                        scores = dets_2d_only_array[:, 5]
                        high_mask = scores >= high_conf
                        low_mask = ~high_mask 
                        raw_high = dets_2d_only_array[high_mask]
                        raw_low = dets_2d_only_array[low_mask]

                        if len(raw_high) > 0:
                            dets_2d_high_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_high])
                        if len(raw_low) > 0:
                            dets_2d_low_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_low])
                    
                    if len(dets_2d_high_tlwhc) == 0: dets_2d_high_tlwhc = np.empty((0, 5))
                    if len(dets_2d_low_tlwhc) == 0: dets_2d_low_tlwhc = np.empty((0, 5))

                    start_time = time.time()
                    trackers = tracker.update(dets_3d_fusion,
                                              dets_2d_high_tlwhc,
                                              dets_2d_low_tlwhc, 
                                              dets_3d_only,
                                              cfg,
                                              frame,
                                              seq_id)
                    
                    cycle_time = time.time() - start_time
                    total_time += cycle_time
                    total_frames += 1
                    save_results(trackers, cfg, seq_name, frame, category, img_0)

    print('--------------The total time is {}s --------------'.format(total_time))
    print('--------------FPS = {} --------------'.format(total_frames / total_time))
    if cfg.dataset == 'nuscenes':
        save_dir = os.path.join(cfg.save_path, 'nuscenes_results')
        save_nuscenes_json(nusc_submissions, save_dir)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFusionMOT')
    parser.add_argument('--cfg', type=str, default='./config/kitti.yaml', help='data')
    args = parser.parse_args()
    cfg, _ = Config(args.cfg)

    tracking(cfg)
    if cfg.dataset != 'nuscenes':
        combine_category_result(cfg)
        print("--------------Starting Evaluation-------------")
        results = eval_kitti(cfg)