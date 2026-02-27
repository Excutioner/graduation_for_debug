import copy

import numpy as np
import cv2

from utils.kitti_oxts import egomotion_compensation_ID, get_ego_traj
from utils.coordinate_transformation import TransformationKitti
'''
  3D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated with any detections for several frames, it 
  is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4

class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_3D:
    def __init__(self, pose, kf_3d, track_id_3d, n_init, max_age,additional_info, feature=None):
        self.pose = pose
        self.kf_3d = kf_3d
        self.track_id_3d = track_id_3d
        self.hits = 1 # 每次update都+1，代表当前轨迹匹配上的帧数
        self.age = 1 # 轨迹存在的总帧数，2D和3D不同，2D只要predict就+1，3D则update才+1
        self.state = TrackState.Tentative
        self.n_init = n_init
        self._max_age = max_age
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_3D
        self.additional_info = additional_info
        self.time_since_update = 0
        self.fusion_time_update = 0 # 代表未与fusion后的det(高质量det)匹配的帧数
        self.compensated_2d = False # 代表3D轨迹的2D信息是否被运动补偿，在卡尔曼update处置False，在2D运动补偿处置True

    def predict_3d(self, trk_3d, apn_cfg=None):
        # 判断一下，如果
        if (apn_cfg.get('use_apn_ctra', False)):
            self.pose = trk_3d.predict(apn_cfg=apn_cfg)
        else:
            self.pose = trk_3d.predict()

    def update_3d(self, detection_3d, cg_akf_cfg=None):
        """
        创新点: 置信度引导的自适应卡尔曼滤波 (CG-AKF)
        """
        # 1. 获取检测置信度 (根据之前的分析，score 在 index 6)
        # additional_info: [alpha, type, x1, y1, x2, y2, score]
        try:
            current_logit = detection_3d.additional_info[6]
        except IndexError:
            current_logit = 5.0 # Fallback
            
        # 2. 动态调整 R 矩阵
        if cg_akf_cfg and cg_akf_cfg.get('use_cg_akf', False):
            # 获取参数 mu 和 tau
            params = cg_akf_cfg.get('cg_akf_params', {})
            mu = params.get('mu', 1.0)   # 论文推荐截断阈值
            tau = params.get('tau', 1.0) # 论文推荐灵敏系数
            
            # --- 核心修改：直接代入公式 ---
            # 论文公式: R_new = R_base * (1 + exp((mu - logit) / tau))
            # 当 logit < mu (遮挡/低置信度) -> 指数项变大 -> R 变大 -> 信任预测
            # 当 logit > mu (正常) -> 指数项趋近 0 -> R 保持 R_base
            adaptive_factor = 1.0 + np.exp((mu - current_logit) / tau)
            
            # 备份原始 R
            original_R = self.kf_3d.kf.R.copy()
            
            # 应用自适应因子
            self.kf_3d.kf.R *= adaptive_factor
            
            # 执行更新
            self.kf_3d.update(detection_3d.bbox)
            
            # 恢复原始 R (保持滤波器参数纯净)
            self.kf_3d.kf.R = original_R
            
        else:
            # 原始逻辑
            self.kf_3d.update(detection_3d.bbox)

        # 3. 状态维护 (保持不变)
        self.additional_info = detection_3d.additional_info
        self.compensated_2d = False
        self.pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
            
        if self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate
    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def fusion_state(self):
        if  self.fusion_time_update >= 2:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_track_id_3d(self):
        track_id_3d= self.track_id_3d
        return track_id_3d
            
    def ego_motion_compensation_3d(self, frame, calib_file, oxts, model_type="CTRA", with_rect2imu="False"):
            """
            3D Ego Motion Compensation adapted for different motion models.
            
            Args:
                frame: Current frame index
                calib_file: Path to calibration file
                oxts: IMU data
                model_type (str): 'CV', 'CA', 'CTRA', 'CTRV'
            """
            # 1. 获取自车运动 (反向变换: T-1 -> T)
            ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(oxts, frame, 1, 1, only_fut=True, inverse=True)
            
            # -----------------------------------------------------
            # Part A: 位置补偿 (所有模型通用)
            # -----------------------------------------------------
            # 补偿 Tracker 内部存储的 pose
            xyz = np.array([self.pose[0], self.pose[1], self.pose[2]]).reshape((1, -1))
            compensated = egomotion_compensation_ID(xyz, calib_file, ego_rot_imu, ego_xyz_imu, left, right)
            
            self.pose[0], self.pose[1], self.pose[2] = compensated[0]
            
            # 补偿 Kalman Filter 状态向量中的位置部分 (通常在前3位)
            try:
                self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1))
            except:
                self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

            # -----------------------------------------------------
            # Part B: 计算 Rect 坐标系下的旋转矩阵 R_rect_delta
            # -----------------------------------------------------
            R_imu_delta = np.array(ego_rot_imu[0]) # 3x3, IMU 坐标系旋转
            
            transformer = TransformationKitti(calib_file)
            R_cam2lidar = transformer.Tr_cam_to_lidar[:3, :3]
            R_lidar2imu = transformer.Tr_lidar_to_imu[:3, :3]
            R_rect_inv  = np.linalg.inv(transformer.R0_rect[:3, :3])
            
            # R_rect -> R_ref -> R_lidar -> R_imu
            R_rect2imu = R_lidar2imu @ R_cam2lidar @ R_rect_inv
            R_rect2imu_inv = np.linalg.inv(R_rect2imu)
            
            # 得到 Rect 系下的旋转增量
            R_rect_delta = R_rect2imu_inv @ R_imu_delta @ R_rect2imu

            # -----------------------------------------------------
            # Part C: 角度状态更新 (Yaw Compensation)
            # -----------------------------------------------------
            # 所有模型都包含 ry (索引通常为 3)
            if with_rect2imu == "True":
                delta_yaw = np.arctan2(R_rect_delta[0, 2], R_rect_delta[0, 0])
            else:
                delta_yaw = np.arctan2(R_imu_delta[0, 2], R_imu_delta[0, 0]) 
            THETA_IDX = 3 
            
            self.kf_3d.kf.x[THETA_IDX] += delta_yaw
            
            # -----------------------------------------------------
            # Part D: 协方差矩阵 P 和 矢量状态 的旋转
            # -----------------------------------------------------
            # 动态获取当前状态向量维度 (CTRA=10, CTRV=9, CV=10, CA=13)
            state_dim = self.kf_3d.kf.x.shape[0]
            J_rot = np.eye(state_dim)
            
            # 1. 旋转位置协方差 [0:3, 0:3] (所有模型通用)
            if with_rect2imu == "True":
                J_rot[0:3, 0:3] = R_rect_delta
            else:
                J_rot[0:3, 0:3] = R_imu_delta

            # 2. 根据模型类型处理速度/加速度的旋转
            # CV/CA 模型使用矢量速度 (vx, vy, vz)，需要旋转
            # CTRA/CTRV 模型使用标量速度 (v)，不需要旋转 (旋转只改变 ry)
            
            if model_type in ['CV', 'CA']:
                # 假设 CV/CA 内部状态排序为: [x, y, z, ry, l, w, h, vx, vy, vz, (ax, ay, az)]
                # 对应的索引: vx,vy,vz -> 7, 8, 9
                
                # --- 旋转速度协方差 ---
                if with_rect2imu == "True":
                    J_rot[7:10, 7:10] = R_rect_delta
                else:
                    J_rot[7:10, 7:10] = R_imu_delta
                # --- 旋转速度状态向量本身 (这是标量模型不需要的) ---
                # 因为 vx, vy, vz 是在旧坐标系下的分量，必须转到新坐标系
                vel_vec = self.kf_3d.kf.x[7:10].reshape(3, 1) # [vx, vy, vz]^T
                if with_rect2imu == "True":
                    vel_vec_rotated = R_rect_delta @ vel_vec
                else:
                    vel_vec_rotated = R_imu_delta @ vel_vec
                try:
                    self.kf_3d.kf.x[7:10] = vel_vec_rotated.reshape((-1))
                except:
                    self.kf_3d.kf.x[7:10] = vel_vec_rotated.reshape((-1, 1))
                    
                # 如果是 CA 模型，且包含加速度矢量 (ax, ay, az)
                if model_type == 'CA' and state_dim >= 13:
                    # 假设加速度在 10, 11, 12
                    J_rot[10:13, 10:13] = R_rect_delta
                    
                    acc_vec = self.kf_3d.kf.x[10:13].reshape(3, 1)
                    if with_rect2imu == "True":
                        acc_vec_rotated = R_rect_delta @ acc_vec
                    else:
                        acc_vec_rotated = R_imu_delta @ acc_vec
                    try:
                        self.kf_3d.kf.x[10:13] = acc_vec_rotated.reshape((-1))
                    except:
                        self.kf_3d.kf.x[10:13] = acc_vec_rotated.reshape((-1, 1))

            elif model_type in ['CTRA', 'CTRV', 'BICYCLE']:
                # 标量速度模型：
                # v (速度大小) 不随坐标系旋转而改变。
                # a (加速度大小) 也不改变。
                # 旋转的影响完全由 ry (角度) 的更新来体现 (Part C 已完成)。
                # 所以 J_rot 的其他部分保持单位阵即可。
                pass

            # 3. 执行协方差更新 P = J * P * J.T
            self.kf_3d.kf.P = J_rot @ self.kf_3d.kf.P @ J_rot.T
    def ego_motion_compensation_2d(self, frame, cmc_transforms):
        # 【建议修改】对于3D轨迹，我们不需要用纯2D的方式来更新 additional_info
        # 因为 additional_info 里的 2D 框在这一帧已经是“过时”的了。
        # 真正的 2D 位置应该由 3D Pose 投影得到。
        # 所以这里可以直接 return，或者只做保留而不覆盖。
        
        # 如果你必须保留它（比如为了可视化或者其他没改到的逻辑），
        # 请记住：这个 additional_info[2:6] 是不准确的（没有自身速度）。
        
        # 最佳做法：直接注释掉下面具体的更新逻辑，或者让它不生效
        pass 
        
        # 原有逻辑（已废弃/不推荐）：
        # if frame <= 1:
        #     return

        # predicted_pts_before_cmc = self.additional_info[2:6]
        # predicted_pts_before_cmc = predicted_pts_before_cmc.reshape(-1, 1, 2)
        # transform = np.array(cmc_transforms[frame-1][1:]).reshape(2, 3)
        # predicted_pts_after_cmc = cv2.transform(predicted_pts_before_cmc, transform)  # shape: (N, 1, 2)
        # predicted_pts_after_cmc = predicted_pts_after_cmc.reshape(-1, 4)  # 转换回(N, 2)形状
        # # 更新3D轨迹中的2D信息
        # self.additional_info[2:6] = predicted_pts_after_cmc[0]
        