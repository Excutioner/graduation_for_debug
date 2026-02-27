import os
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# ================= 配置路径 =================
NUSC_ROOT = "/datav/DeepFusionMOT/data/nuscenes/test"
VERSION = "v1.0-test"
# 输出路径：将生成 pose 和 calib 文件夹
OUTPUT_ROOT = "/datav/DeepFusionMOT/data/nuscenes_kitti_format" 
CAMERA_CHANNEL = 'CAM_FRONT'
# ===========================================

def get_calib_str(P2, Tr_velo_to_cam):
    """生成 KITTI 格式的 calib 字符串"""
    # KITTI calib 需要 P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
    # 这里核心只需要 P2 (相机内参) 和 Tr_velo_to_cam (外参)
    # R0_rect 通常设为单位矩阵，因为 nuScenes 已经做过矫正或不使用此逻辑
    
    def mat2str(mat):
        return " ".join([f"{x:.6e}" for x in mat.flatten()])

    P_str = mat2str(P2)
    R0 = np.eye(3) # 假设 R0_rect 是单位阵
    Tr_v2c = mat2str(Tr_velo_to_cam[:3, :4]) # KITTI 只需要 3x4
    
    lines = []
    lines.append(f"P0: {P_str}")
    lines.append(f"P1: {P_str}")
    lines.append(f"P2: {P_str}") # 重点
    lines.append(f"P3: {P_str}")
    lines.append(f"R0_rect: {mat2str(R0)}")
    lines.append(f"Tr_velo_to_cam: {Tr_v2c}") # 重点
    lines.append(f"Tr_imu_to_velo: {Tr_v2c}") # 占位
    return "\n".join(lines)

def main():
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=True)
    
    pose_dir = os.path.join(OUTPUT_ROOT, "pose")
    calib_dir = os.path.join(OUTPUT_ROOT, "calib")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    print(f"Generating Pose and Calib to {OUTPUT_ROOT}...")

    # 如果你的文件名是场景名（如 scene-0614），这里需要保持一致
    # 按照 scene name 排序遍历
    scenes = sorted(nusc.scene, key=lambda x: x['name'])

    for scene in tqdm(scenes):
        # 获取场景名称作为文件名，或者你可以根据你的需要修改为 0000 格式
        # 这里假设你使用 '0614' 这种命名
        seq_name = scene['name'].replace('scene-', '') 
        
        pose_lines = []
        
        # --- 1. 获取 Calibration (使用序列第一帧的标定) ---
        # 注意：DeepFusionMOT 的 calibration.py 只读取一次标定文件
        first_sample = nusc.get('sample', scene['first_sample_token'])
        sd_token = first_sample['data'][CAMERA_CHANNEL]
        sd_record = nusc.get('sample_data', sd_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        
        # 构建 P2 (Intrinsic) 3x4
        P2 = np.zeros((3, 4))
        P2[:3, :3] = np.array(cs_record['camera_intrinsic'])
        
        # 构建 Tr_velo_to_cam (Extrinsic: Lidar -> Cam)
        # nuScenes: Lidar -> Ego -> Cam
        # T_lidar2cam = inv(T_cam2ego) * T_lidar2ego
        # 但这里为了简化，通常我们只需要相机坐标系下的处理。
        # DeepFusionMOT 主要用 P2 将 3D 投到 2D。
        # 如果你的检测结果已经是相机坐标系（你的转换脚本已做），这里的 Tr 可以设为特定值或单位阵的变换。
        # *重要*: 如果你的 3D 检测框已经是相机坐标系，这里的 Tr 并不影响投影（投影只用 P2）。
        # 但为了兼容性，我们计算真实的 Lidar 到 Cam 变换。
        
        # 获取 LIDAR 标定
        lidar_token = first_sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        
        # T_lidar_to_ego
        T_l2e = np.eye(4)
        T_l2e[:3, :3] = Quaternion(lidar_cs['rotation']).rotation_matrix
        T_l2e[:3, 3] = np.array(lidar_cs['translation'])
        
        # T_cam_to_ego
        T_c2e = np.eye(4)
        T_c2e[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        T_c2e[:3, 3] = np.array(cs_record['translation'])
        
        # T_ego_to_cam = inv(T_c2e)
        T_e2c = np.linalg.inv(T_c2e)
        
        # T_lidar_to_cam = T_e2c * T_l2e
        Tr_velo_to_cam = np.dot(T_e2c, T_l2e)
        
        # 写入 calib 文件
        with open(os.path.join(calib_dir, f"{seq_name}.txt"), 'w') as f:
            f.write(get_calib_str(P2, Tr_velo_to_cam))

        # --- 2. 获取 Pose (每一帧的 Ego Pose) ---
        # 写入格式：每一行是一个 flattened 3x4 matrix (12 floats)
        # 代表当前帧 Ego 坐标系 到 Global 坐标系的变换，或者相对变换
        # DeepFusionMOT 读取 oxts 后会转为 transforms。
        # 我们直接保存 T_ego_to_global 或者 T_global_to_ego。
        # 这里的标准是：保存这一帧时刻，车体在世界坐标系下的位姿 (T_ego_to_global)
        
        current_token = scene['first_sample_token']
        while current_token:
            sample = nusc.get('sample', current_token)
            # 使用 LIDAR 的 pose 作为基准 (通常更准，且用于 3D 跟踪)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_sd = nusc.get('sample_data', lidar_token)
            ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            
            # 构建 4x4 矩阵
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
            pose_mat[:3, 3] = np.array(ego_pose['translation'])
            
            # 展平为 12 个数 (3x4)
            pose_line = " ".join([f"{x:.6e}" for x in pose_mat[:3, :].flatten()])
            pose_lines.append(pose_line + "\n")
            
            current_token = sample['next']
            
        with open(os.path.join(pose_dir, f"{seq_name}.txt"), 'w') as f:
            f.writelines(pose_lines)

    print("Done!")

if __name__ == "__main__":
    main()