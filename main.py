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

from utils.nuscenes_2d_loader import NuScenes2DLoader
from utils.save_nusc_results import format_nusc_outputs, save_nuscenes_json, save_nusc_results_vis

# ==============================================================================
# 升级 1：支持 6 相机动态投影，并携带全局索引 orig_idx 防止重复匹配
# ==============================================================================
def project_nusc_to_kitti_format(nusc, current_token, boxes_3d_filtered, indices_to_project, category, cam_name, img_shape=(1600, 900)):
    dets_3d = []
    dets_2d_proj = []
    add_info = []

    sample = nusc.get('sample', current_token)
    cam_token = sample['data'][cam_name] # 动态指定相机
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])

    for idx in indices_to_project:
        box_dict = boxes_3d_filtered[idx]
        w, l, h = box_dict['size']
        x, y, z = box_dict['translation']
        q = box_dict['rotation']
        score = box_dict.get('detection_score', box_dict.get('tracking_score', 1.0))
        
        velo = box_dict.get('velocity', [0.0, 0.0])
        vx, vy = velo[0], velo[1]
        
        quat = Quaternion(q)
        yaw = quat.yaw_pitch_roll[0] 

        box = Box(box_dict['translation'], box_dict['size'], Quaternion(box_dict['rotation']))
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        if box.center[2] > 0.1: 
            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            x1, x2 = corners[0, :].min(), corners[0, :].max()
            y1, y2 = corners[1, :].min(), corners[1, :].max()
            
            if x2 < 0 or x1 > img_shape[0] or y2 < 0 or y1 > img_shape[1]:
                bbox_2d = [0, 0, 0, 0] 
            else:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_shape[0], x2), min(img_shape[1], y2)
                bbox_2d = [x1, y1, x2, y2]
        else:
            bbox_2d = [0, 0, 0, 0] 

        if bbox_2d != [0, 0, 0, 0]:
            dets_3d.append([h, w, l, x, y, z, yaw, vx, vy])
            dets_2d_proj.append(bbox_2d)
            type_id = 1 if category.lower() == 'car' else 2
            
            # 妙笔：把它在原始列表中的 idx 塞到最后一位 (info[7])，方便融合后认领！
            add_info.append([0.0, type_id, bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3], score, idx])

    if len(dets_3d) == 0:
        return np.empty((0, 7)), np.empty((0, 8)), np.empty((0, 4))
    
    return np.array(dets_3d), np.array(add_info), np.array(dets_2d_proj)

# ==============================================================================
# 升级 2：动态指定提取某个相机的 2D 框
# ==============================================================================
def format_nusc_2d_to_kitti_format(raw_2d_boxes_dict, frame_idx, cam_name):
    cam_boxes = raw_2d_boxes_dict.get(cam_name, [])
    dets_2d = []
    for det in cam_boxes:
        b = det['bbox']
        score = det['score']
        dets_2d.append([frame_idx, b[0], b[1], b[2], b[3], score, score])
        
    if len(dets_2d) == 0:
        return np.empty((0, 7))
    return np.array(dets_2d)


def nms_2d(boxes, scores, iou_threshold=0.5):
    """
    标准的 2D NMS (非极大值抑制)
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def nms_3d_center_distance(boxes_3d, scores, dist_threshold=0.5):
    """
    针对 CenterPoint 的极速 3D NMS (基于中心点欧氏距离)
    CenterPoint本身有NMS，但这里作为二次保险，防止不同类别误检或脏数据
    """
    if len(boxes_3d) == 0:
        return []
    
    centers = np.array([box['translation'] for box in boxes_3d]) # (N, 3)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
            
        dist = np.linalg.norm(centers[order[1:]] - centers[i], axis=1)
        inds = np.where(dist > dist_threshold)[0]
        order = order[inds + 1]
        
    return keep

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
    nusc_seq_list = [] 
    
    nusc_submissions = {
            "meta": {
                "use_camera": True, "use_lidar": True, "use_radar": False, 
                "use_map": False, "use_external": False
            },
            "results": {}
        }
    
    if cfg.dataset == 'nuscenes':
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

        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        nusc_seq_list = splits[cfg.spilt] 

    for category in cfg.cat_list:
        junk_conf = cfg[category]["fga"]["junk_conf"]
        high_conf = cfg[category]["fga"]["high_conf"]
        loc_thresh = cfg[category]["fga"]["loc_thresh"] 
        
        loop_seqs = nusc_seq_list if cfg.dataset == 'nuscenes' else seq_list

        for real_seq_id in tqdm.tqdm(loop_seqs):
            seq_id = real_seq_id
            tracker = DeepFusionMOT(cfg, category)
            
            if cfg.dataset == 'nuscenes':
                target_scene_name = real_seq_id  
                seq_name = real_seq_id
                
                try:
                    scene = next(s for s in nusc.scene if s['name'] == target_scene_name)
                except StopIteration:
                    print(f"[Warning] Scene {target_scene_name} not found in nuScenes!")
                    continue
                
                aligned_2d_boxes = loader_2d.load_aligned_2d_boxes(target_scene_name)
                current_token = scene['first_sample_token']
                frame_idx = 0 
                
                # 定义 6 相机列表
                CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                while current_token != "":
                    start_time = time.time()
                    
                    # 仅保留读取前视图像，为了向后兼容你的可视化接口不报错（如果需要用的话）
                    sample = nusc.get('sample', current_token)
                    cam_token = sample['data']['CAM_FRONT']
                    cam_data = nusc.get('sample_data', cam_token)
                    img0_path = os.path.join(nusc.dataroot, cam_data['filename'])
                    img_0 = cv2.imread(img0_path)
                    
                    raw_3d_boxes = det_3d_results.get(current_token, [])
                    raw_2d_boxes_dict = aligned_2d_boxes.get(current_token, {})
                    
                    # 在一帧开始前，先全局过滤该类别的 3D 框
                    boxes_3d_filtered = [b for b in raw_3d_boxes if b.get('detection_name', b.get('tracking_name', '')).lower() == category.lower()]
                    # NMS
                    if len(boxes_3d_filtered) > 0:
                        scores_3d = np.array([b.get('detection_score', b.get('tracking_score', 1.0)) for b in boxes_3d_filtered])
                        # 距离小于 0.5 米的认为是重复框，进行抑制
                        keep_3d_indices = nms_3d_center_distance(boxes_3d_filtered, scores_3d, dist_threshold=0.5)
                        boxes_3d_filtered = [boxes_3d_filtered[i] for i in keep_3d_indices]

                    # 2. ================= 2D 单相机 NMS 预处理 =================
                    # 提前清洗字典里每个相机的 2D 框
                    for c_name in raw_2d_boxes_dict.keys():
                        cam_boxes = raw_2d_boxes_dict[c_name]
                        if len(cam_boxes) > 0:
                            bboxes = np.array([b['bbox'] for b in cam_boxes])
                            scores = np.array([b['score'] for b in cam_boxes])
                            # 2D 框 IoU > 0.4 且类别相同，抑制低分框
                            keep_2d = nms_2d(bboxes, scores, iou_threshold=0.4)
                            raw_2d_boxes_dict[c_name] = [cam_boxes[i] for i in keep_2d]
                            
                    # 【核心】6 相机全局去重遮罩与汇总收集器
                    matched_3d_mask = np.zeros(len(boxes_3d_filtered), dtype=bool)
                    global_dets_3d_fusion_cam = []
                    global_dets_3d_fusion_info = []
                    global_dets_2d_high_tlwhc = []
                    global_dets_2d_low_tlwhc = []
                    
                    # 🚀 开始遍历 6 个相机进行并发融合
                    for cam_name in CAMERAS:
                        # 仅取出当前尚未被匹配掉的 3D 框的索引
                        unmatched_indices = np.where(~matched_3d_mask)[0].tolist()
                        
                        # 如果没有剩余的 3D 框，或者该相机压根没有 2D 观测，直接跳过加速运算
                        if len(unmatched_indices) == 0 and len(raw_2d_boxes_dict.get(cam_name, [])) == 0:
                            continue
                            
                        dets_3d_camera, additional_info, dets_3dto2d_image = project_nusc_to_kitti_format(
                            nusc, current_token, boxes_3d_filtered, unmatched_indices, category, cam_name
                        )
                        dets_2d_input = format_nusc_2d_to_kitti_format(raw_2d_boxes_dict, frame_idx, cam_name)
                        
                        # --- FGA 拆分逻辑 (原封不动) ---
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
                        if cam_name == 'CAM_FRONT' and frame_idx < 20: 
                            vis_img = img_0.copy()
                            
                            # 1. 画 2D 检测框 (红色)
                            for det in dets_2d_combined:
                                x1, y1, x2, y2 = map(int, det[1:5])
                                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(vis_img, f"2D:{det[5]:.2f}", (x1, y1 - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                            # 2. 画 3D 投影转 2D 框 (绿色)
                            for proj_box in dets_3dto2d_image:
                                x1, y1, x2, y2 = map(int, proj_box)
                                if x1 == 0 and x2 == 0: 
                                    continue # 过滤掉不在视野内的框
                                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(vis_img, "3D-Proj", (x1, y2 + 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                            os.makedirs("./debug_dets_vis", exist_ok=True)
                            cv2.imwrite(f"./debug_dets_vis/frame_{frame_idx:04d}.jpg", vis_img)
                        # --- 融合逻辑 ---
                        if len(dets_3d_camera) > 0 or len(dets_2d_combined) > 0:
                            dets_3d_fusion, _, dets_2d_only_list = \
                                data_fusion(dets_3d_camera, dets_2d_combined, dets_3dto2d_image, additional_info)

                            # 收集匹配成功的 3D 轨迹
                            if len(dets_3d_fusion['dets_3d_fusion']) > 0:
                                for i, info in enumerate(dets_3d_fusion['dets_3d_fusion_info']):
                                    orig_idx = int(info[7]) # 拿到户口本索引
                                    matched_3d_mask[orig_idx] = True # 注销户口！防止在下个相机里重复匹配
                                    
                                    global_dets_3d_fusion_cam.append(dets_3d_fusion['dets_3d_fusion'][i])
                                    global_dets_3d_fusion_info.append(info[:7]) # 去掉索引标志，还原为7维
                            
                            # 收集 2D-only 轨迹
                            if len(dets_2d_only_list) > 0:
                                dets_2d_only_array = np.array(dets_2d_only_list) 
                                scores = dets_2d_only_array[:, 5]
                                high_mask = scores >= high_conf
                                raw_high = dets_2d_only_array[high_mask]
                                raw_low = dets_2d_only_array[~high_mask]

                                if len(raw_high) > 0:
                                    global_dets_2d_high_tlwhc.extend([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_high])
                                if len(raw_low) > 0:
                                    global_dets_2d_low_tlwhc.extend([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_low])
                    
                    # 🚀 6 相机轮询结束，打扫战场
                    global_dets_3d_only_cam = []
                    global_dets_3d_only_info = []
                    
                    # 遍历户口本，把没人认领的 3D 框全部分入 3D-Only 队伍
                    for idx, is_matched in enumerate(matched_3d_mask):
                        if not is_matched:
                            box_dict = boxes_3d_filtered[idx]
                            w, l, h = box_dict['size']
                            x, y, z = box_dict['translation']
                            q = box_dict['rotation']
                            score = box_dict.get('detection_score', box_dict.get('tracking_score', 1.0))
                            yaw = Quaternion(q).yaw_pitch_roll[0]
                            
                            velo = box_dict.get('velocity', [0.0, 0.0])
                            vx, vy = velo[0], velo[1]
                            
                            global_dets_3d_only_cam.append([h, w, l, x, y, z, yaw, vx, vy])
                            type_id = 1 if category.lower() == 'car' else 2
                            global_dets_3d_only_info.append([0.0, type_id, 0, 0, 0, 0, score])
                    
                    # 构建送入 Tracker 的终极数据结构
                    final_dets_3d_fusion = {
                        'dets_3d_fusion': np.array(global_dets_3d_fusion_cam) if len(global_dets_3d_fusion_cam) > 0 else np.empty((0, 9)),
                        'dets_3d_fusion_info': np.array(global_dets_3d_fusion_info) if len(global_dets_3d_fusion_info) > 0 else np.empty((0, 7))
                    }
                    final_dets_3d_only = {
                        'dets_3d_only': np.array(global_dets_3d_only_cam) if len(global_dets_3d_only_cam) > 0 else np.empty((0, 9)),
                        'dets_3d_only_info': np.array(global_dets_3d_only_info) if len(global_dets_3d_only_info) > 0 else np.empty((0, 7))
                    }
                    final_dets_2d_high = np.array(global_dets_2d_high_tlwhc) if len(global_dets_2d_high_tlwhc) > 0 else np.empty((0, 5))
                    final_dets_2d_low = np.array(global_dets_2d_low_tlwhc) if len(global_dets_2d_low_tlwhc) > 0 else np.empty((0, 5))

                    # 👑 Tracker 全局更新
                    trackers = tracker.update(final_dets_3d_fusion,
                                              final_dets_2d_high, 
                                              final_dets_2d_low, 
                                              final_dets_3d_only,
                                              cfg,
                                              frame_idx,
                                              seq_id)
                        
                    cycle_time = time.time() - start_time
                    total_time += cycle_time
                    total_frames += 1
                    
                    formatted_res = format_nusc_outputs(trackers, current_token, category)
                    if current_token not in nusc_submissions["results"]:
                        nusc_submissions["results"][current_token] = []
                    nusc_submissions["results"][current_token].extend(formatted_res)
                    
                    # ================= 📍 轨迹可视化探针 =================
                    # 我们只画 CAM_FRONT 的前 50 帧，防止生成太多图片占硬盘
                    if frame_idx < 50:
                        save_nusc_results_vis(trackers, nusc, current_token, 'CAM_FRONT', img_0, frame_idx)
                    # ==================================================
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