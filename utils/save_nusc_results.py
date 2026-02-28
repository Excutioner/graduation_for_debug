import os
import json
import numpy as np
from pyquaternion import Quaternion

def format_nusc_outputs(outputs, current_token, category):
    """
    将 DeepFusionMOT 输出的 numpy 数组转为 nuScenes 官方的字典格式
    outputs: [N, 15] 或 [N, 1, 15] 的 numpy array.
             含义: [track_id, h, w, l, x, y, z, yaw, alpha, type_id, x1, y1, x2, y2, score]
    """
    frame_results = []
    if len(outputs) == 0:
        return frame_results
        
    # 【核心修复】：不管传进来嵌套了多少层，强行统一重塑为 N x 15 的二维数组！
    outputs = np.array(outputs).reshape(-1, 15)
        
    for i in range(outputs.shape[0]):
        row = outputs[i]
        
        # 现在的 row 百分之百是一维数组了
        track_id = str(int(row[0])) 
        
        h, w, l = row[1], row[2], row[3]
        x, y, z = row[4], row[5], row[6]
        yaw = row[7]
        score = row[14]
        
        # 1. 恢复四元数: nuScenes 绕 Z 轴旋转
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        
        # 2. 组装字典
        det_dict = {
            "sample_token": current_token,
            "translation": [float(x), float(y), float(z)],
            "size": [float(w), float(l), float(h)],
            "rotation": list(q.elements), # [w, x, y, z] 格式
            "velocity": [0.0, 0.0],       # 传统系统暂不跟踪速度，直接补0
            "tracking_id": track_id,
            "tracking_name": category.lower(), # nuScenes 评测强制小写
            "tracking_score": float(score)
        }
        frame_results.append(det_dict)
        
    return frame_results

def save_nuscenes_json(nusc_submissions, save_dir, filename="nusc_tracking_val_results.json"):
    """
    在整个推理结束后，保存终极 JSON 文件
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(nusc_submissions, f, indent=4)
        
    print(f"\n[Success] NuScenes evaluation JSON saved to {save_path}")
    
import cv2
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

def save_nusc_results_vis(trackers, nusc, current_token, cam_name, img_0, frame_idx, save_dir='./debug_tracking_vis'):
    """
    可视化 nuScenes 跟踪结果 (将全局 3D 轨迹投影回特定相机的 2D 图像上)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 提取相机的内外参
    sample = nusc.get('sample', current_token)
    cam_token = sample['data'][cam_name]
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])

    vis_img = img_0.copy()

    # 格式化 trackers (防止维度嵌套)
    if len(trackers) == 0:
        cv2.imwrite(os.path.join(save_dir, f"{cam_name}_{frame_idx:04d}.jpg"), vis_img)
        return
        
    outputs = np.array(trackers).reshape(-1, 15)

    for i in range(outputs.shape[0]):
        row = outputs[i]
        track_id = int(row[0])
        h, w, l = row[1], row[2], row[3]
        x, y, z = row[4], row[5], row[6]
        yaw = row[7]

        # 2. 组装 Global Box
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        box = Box([x, y, z], [w, l, h], q)

        # 3. 坐标系转换: Global -> Ego -> Camera
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # 过滤掉不在相机前方的目标
        if box.center[2] < 0.1:
            continue

        # 4. 投影到 2D 图像平面
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
        corners_int = corners_2d.astype(int).T # (8, 2)

        # 检查是否有一部分在图像内
        x_min, x_max = corners_int[:, 0].min(), corners_int[:, 0].max()
        y_min, y_max = corners_int[:, 1].min(), corners_int[:, 1].max()
        if x_max < 0 or x_min > vis_img.shape[1] or y_max < 0 or y_min > vis_img.shape[0]:
            continue
            
        # 5. 在图像上绘制 3D 框的连线 (绿色)
        colors = (0, 255, 0)
        thickness = 2
        
        # nuScenes 8个角点的连接规则
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), # Front face
                 (4, 5), (5, 6), (6, 7), (7, 4), # Back face
                 (0, 4), (1, 5), (2, 6), (3, 7)] # Connectors
                 
        for edge in edges:
            pt1 = tuple(corners_int[edge[0]])
            pt2 = tuple(corners_int[edge[1]])
            cv2.line(vis_img, pt1, pt2, colors, thickness)
            
        # 在左上角画出 Track ID (黄色，极度醒目)
        cv2.putText(vis_img, f"ID: {track_id}", (x_min, max(0, y_min - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 保存图片
    save_path = os.path.join(save_dir, f"{cam_name}_{frame_idx:04d}.jpg")
    cv2.imwrite(save_path, vis_img)