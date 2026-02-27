import json
import os
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

# ================= 全局配置区域 (请在此处修改路径) =================

# CenterPoint 结果 JSON 文件路径
JSON_PATH = "/datav/DeepFusionMOT/data/detections/3D/nuscenes/nusc_results_flip/infos_test_10sweeps_withvelo_painted.json"

# nuScenes 数据集根目录 (包含 maps, samples, sweeps, v1.0-test 等文件夹)
NUSC_ROOT = "/datav/DeepFusionMOT/data/nuscenes/test"

# nuScenes 版本 ('v1.0-trainval' 或 'v1.0-test')
VERSION = "v1.0-test"

# 输出 KITTI 格式 txt 的文件夹
OUTPUT_DIR = "/datav/DeepFusionMOT/data/detections/3D/nusc_centerpoint"

# 默认使用前视相机进行转换 (你的 Tracker 需要基于某个相机视角)
CAMERA_CHANNEL = 'CAM_FRONT' 

# 类别映射：将 nuScenes 类别映射为你代码中的整数 ID
# 对应 DeepFusionMOT 的 config 配置
CLASS_MAP = {
    'car': 2,
    'truck': 2,        # 很多跟踪器将卡车也视为车辆处理
    'bus': 2,
    'trailer': 2,
    'construction_vehicle': 2,
    'pedestrian': 1,   # 假设行人是 1
    'bicycle': 3,      # 假设骑行者是 3
    'motorcycle': 3
}
# =================================================================

def transform_box_to_kitti_camera(box, nusc, sample_token):
    """
    将 Global 坐标系下的 Box 转换到 KITTI Camera 坐标系
    """
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data'][CAMERA_CHANNEL]
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # 1. Global -> Ego
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)

    # 2. Ego -> Camera (nuScenes definition: x-right, y-down, z-forward)
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)

    # 3. Camera (nuScenes) -> Camera (KITTI)
    # nuScenes Camera 坐标系定义与 KITTI 相机坐标系定义基本一致 (x-right, y-down, z-forward)
    # 但需注意 Box 的 W/L/H 定义差异
    
    return box, cs_record['camera_intrinsic']

def project_3d_to_2d(box, intrinsic):
    """
    将 3D Box 投影到 2D 图像平面计算 BBox
    """
    # 过滤掉完全在相机背后的物体 (z < 0.1)
    if box.center[2] < 0.1:
        return None

    # 获取 3D 框的 8 个角点
    corners_3d = box.corners()
    
    # 投影到图像平面
    # view_points 返回 (3, N)，其中第三行是深度，前两行是像素坐标
    corners_img = view_points(corners_3d, np.array(intrinsic), normalize=True)[:2, :]

    # 计算 2D 框 (x1, y1, x2, y2)
    x1, x2 = np.min(corners_img[0, :]), np.max(corners_img[0, :])
    y1, y2 = np.min(corners_img[1, :]), np.max(corners_img[1, :])

    # 简单边界处理 (防止坐标为负)
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    return [x1, y1, x2, y2]

def calc_kitti_alpha(rot_y, x, z):
    """
    计算观测角 Alpha
    alpha = rot_y - theta
    theta = arctan2(x, z)
    """
    theta = np.arctan2(x, z)
    alpha = rot_y - theta
    
    # Normalize to [-pi, pi]
    while alpha > np.pi: alpha -= 2*np.pi
    while alpha < -np.pi: alpha += 2*np.pi
    return alpha

def main():
    # 打印配置信息
    print(f"Loading nuScenes {VERSION} from {NUSC_ROOT}...")
    try:
        nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=True)
    except Exception as e:
        print(f"Error loading nuScenes: {e}")
        print(f"请检查路径 {NUSC_ROOT} 下是否存在 {VERSION} 文件夹")
        return

    print(f"Loading Predictions from {JSON_PATH}...")
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    results = data['results']

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Converting results to KITTI format in {OUTPUT_DIR}...")
    
    # 遍历所有 Scene
    for scene in tqdm(nusc.scene):
        # 获取序列ID，如 scene-0001 -> 1
        # 注意：CenterPoint 结果可能不覆盖所有 scene，或者 scene name 格式不同，需根据实际情况调整
        try:
            seq_id = int(scene['name'].replace('scene-', '')) 
        except ValueError:
            # 如果 scene 名字不是 scene-XXXX 格式，尝试直接使用索引或其他逻辑
            continue

        seq_name = f"{seq_id:04d}"
        output_file = os.path.join(OUTPUT_DIR, f"{seq_name}.txt")
        file_lines = []
        
        # 遍历该 Scene 下的所有 Frame
        first_token = scene['first_sample_token']
        current_token = first_token
        frame_idx = 0
        
        while current_token:
            if current_token in results:
                dets = results[current_token]
                
                for det in dets:
                    det_name = det['detection_name']
                    
                    # 1. 过滤不在映射表中的类别
                    if det_name not in CLASS_MAP:
                        continue
                    
                    type_id = CLASS_MAP[det_name]
                    score = det['detection_score']
                    
                    # 过滤低分检测框 (可选，根据你的需求调整阈值，例如 0.1)
                    if score < 0.1: 
                        continue

                    # 2. 构建 nuScenes Box 对象 (Global Frame)
                    # CenterPoint 输出通常是 Global 坐标
                    box = Box(
                        center=det['translation'],
                        size=det['size'], # w, l, h
                        orientation=Quaternion(det['rotation']),
                        score=score,
                        name=det_name
                    )

                    # 3. 坐标转换: Global -> KITTI Camera
                    try:
                        box_cam, intrinsic = transform_box_to_kitti_camera(box, nusc, current_token)
                    except Exception as e:
                        # 某些帧可能缺少传感器数据
                        continue

                    # 4. 2D 投影 (计算 2D BBox)
                    bbox_2d = project_3d_to_2d(box_cam, intrinsic)
                    if bbox_2d is None:
                        continue # 在相机后面

                    # 5. 准备 KITTI 格式数据
                    # 格式: Frame, Type, x1, y1, x2, y2, Score, H, W, L, X, Y, Z, RotY, Alpha
                    
                    # 尺寸: nuScenes Box.wlh -> [w, l, h]
                    # KITTI 需要: h, w, l
                    w, l, h = box_cam.wlh
                    
                    # 位置: KITTI Camera Frame (x, y, z) 
                    # y is down (height), z is forward (depth)
                    x, y, z = box_cam.center
                    
                    # 旋转: RotY (Yaw around Y-axis in Camera Frame)
                    # 简单近似：计算 Box 前向向量在 x-z 平面的角度
                    v = np.dot(box_cam.rotation_matrix, np.array([1, 0, 0]))
                    rot_y = -np.arctan2(v[2], v[0]) 
                    
                    # Alpha (观测角)
                    alpha = calc_kitti_alpha(rot_y, x, z)

                    # 格式化行
                    # FrameID, TypeID, 2D_x1, 2D_y1, 2D_x2, 2D_y2, Score, H, W, L, X, Y, Z, RotY, Alpha
                    line = f"{frame_idx},{type_id},{bbox_2d[0]:.4f},{bbox_2d[1]:.4f},{bbox_2d[2]:.4f},{bbox_2d[3]:.4f}," \
                           f"{score:.4f},{h:.4f},{w:.4f},{l:.4f},{x:.4f},{y:.4f},{z:.4f},{rot_y:.4f},{alpha:.4f}\n"
                    
                    file_lines.append(line)

            # 移动到下一帧
            sample = nusc.get('sample', current_token)
            current_token = sample['next']
            frame_idx += 1
            
        # 写入文件
        if len(file_lines) > 0:
            with open(output_file, 'w') as f:
                f.writelines(file_lines)
        else:
            # 如果该序列没有任何检测结果，也可以选择创建一个空文件，或者跳过
            # 这里创建一个空文件以保持序列完整性
            with open(output_file, 'w') as f:
                pass

    print("Done!")

if __name__ == "__main__":
    main()