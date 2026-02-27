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