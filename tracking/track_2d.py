# Note：The code code is referenced from https://github.com/nwojke/deep_sort
import cv2
import numpy as np
from utils.kitti_oxts import egomotion_compensation_ID, get_ego_traj


'''
  2D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated
  with any detections for several frames, it is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_2D:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):

        self.mean = mean
        self.covariance = covariance
        self.track_id_2d = track_id  #
        self.hits = 1 # 每次update都+1，代表当前轨迹匹配上的帧数
        self.age = 1 # 轨迹存在的总帧数，2D和3D不同，2D只要predict就+1，3D则update才+1
        self.state = TrackState.Tentative
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_2D  # 2D tracking
        self.time_since_update = 0 # 每次predict都+1，update时清零，代表当前轨迹丢失的帧数
        self.n_init = n_init    # 连续n_init帧被检测到，状态就被设为confirmed
        self._max_age = 20  # 一个跟踪对象丢失多少帧后会被删去（删去之后将不再进行特征匹配）

    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`.
        Returns
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_x1y1x2y2(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def to_x1y1x2y2c(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y, confidence)`.
        """
        ret = self.to_x1y1x2y2()
        conf = self.mean[4] 
        return np.append(ret, conf)

    def increment(self):
        self.age += 1
        self.time_since_update += 1

    def predict_2d(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment()
        
    def ltbr_predict_2d(self, kf):
        x1y1x2y2_mean = np.concatenate([self.to_x1y1x2y2(), self.mean[4:]])
        self.mean, self.covariance = kf.predict(x1y1x2y2_mean, self.covariance)
        self.increment()
    def ltbrc_predict_2d(self, kf):
        x1y1x2y2_mean = np.concatenate([self.to_x1y1x2y2(), self.mean[4:]])
        self.mean, self.covariance = kf.predict(x1y1x2y2_mean, self.covariance)
        self.increment()
        
    def update_2d(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        # self.features.append(detection.feature)
        self.hits += 1
        # self.age += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state = TrackState.Confirmed
            
    def ltbr_update_2d(self, kf, detection):
        x1y1x2y2_mean = np.concatenate([self.to_x1y1x2y2(), self.mean[4:]])
        self.mean, self.covariance = kf.update(x1y1x2y2_mean, self.covariance, detection.to_x1y1x2y2())
        # self.features.append(detection.feature)
        self.hits += 1
        # self.age += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state = TrackState.Confirmed
    def ltbrc_update_2d(self, kf, detection):
        x1y1x2y2_mean = np.concatenate([self.to_x1y1x2y2(), self.mean[4:]])
        detection_measure = np.append(detection.to_x1y1x2y2(), detection.get_confidence())
        self.mean, self.covariance = kf.update(x1y1x2y2_mean, self.covariance, detection_measure)
        # self.features.append(detection.feature)
        self.hits += 1
        # self.age += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state = TrackState.Confirmed
    def mark_missed(self):
        if self.state == TrackState.Tentative or self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
    
    def ego_motion_compensation_2d(self, frame, cmc_transforms):
        if frame <= 1:
            return

        predicted_pts_before_cmc = self.to_x1y1x2y2()
        predicted_pts_before_cmc = predicted_pts_before_cmc.reshape(-1, 1, 2)
        transform = np.array(cmc_transforms[frame-1][1:]).reshape(2, 3)
        predicted_pts_after_cmc = cv2.transform(predicted_pts_before_cmc, transform)  # shape: (N, 1, 2)
        predicted_pts_after_cmc = predicted_pts_after_cmc.reshape(-1, 4)  # 转换回(N, 2)形状
        x1, y1, x2, y2 = predicted_pts_after_cmc[0]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height != 0 else 0
        # 更新跟踪器的状态
        self.mean[:4] = [center_x, center_y, aspect_ratio, height]
        
        # 更新协方差矩阵
        # 提取变换矩阵的线性部分（2x2旋转缩放矩阵）
        linear_transform = transform[:, :2]  # 提取2x2的线性变换部分
        
        # 构造4x4的雅可比矩阵，只对位置相关的状态进行变换
        # 假设状态向量为 [cx, cy, aspect_ratio, height, vx, vy, ...]
        J = np.eye(len(self.mean))  # 创建与mean维度相同的单位矩阵
        J[:2, :2] = linear_transform  # 将位置部分的变换应用到协方差矩阵
        
        # 使用标准的协方差更新公式: P = J * P * J^T
        old_covariance = self.covariance.copy()
        self.covariance = J @ old_covariance @ J.T
        
    def ego_motion_compensation_2d_imu(self, frame, calib_file, oxts):
        """
        使用IMU数据对2D跟踪器进行自车运动补偿
        
        Parameters:
        -----------
        frame : int
            当前帧号
        calib_file : str
            标定文件路径
        oxts : array
            IMU姿态数据
        """
        if frame <= 1:
            return
        
        # 获取自车轨迹信息
        # 使用1帧过去的IMU数据进行补偿
        ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(oxts, frame, 1, 1, only_fut=True, inverse=True)
        
        # 获取当前2D检测框的坐标 (x1, y1, x2, y2)
        current_bbox = self.to_x1y1x2y2()
        x1, y1, x2, y2 = current_bbox
        
        # 将2D框的四个角点转换为3D点进行IMU补偿
        # 这里假设所有点都在一个平面上，使用一个固定的Z值
        # 在实际应用中，可能需要根据目标距离调整Z值
        corners_2d = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x1, y2],  # 左下
            [x2, y2]   # 右下
        ])
        
        # 为角点添加Z坐标，假设在图像平面上
        z_coord = 0.0  # 可根据实际情况调整
        corners_3d = np.hstack([corners_2d, np.full((4, 1), z_coord)])
        
        # 应用IMU补偿
        compensated_corners = egomotion_compensation_ID(
            corners_3d, calib_file, ego_rot_imu, ego_xyz_imu, left, right
        )
        
        # 将补偿后的3D点投影回2D
        # 这里简化处理，直接使用x,y坐标
        compensated_2d = compensated_corners[:, :2]
        
        # 计算补偿后的边界框
        x1_comp = np.min(compensated_2d[:, 0])
        y1_comp = np.min(compensated_2d[:, 1])
        x2_comp = np.max(compensated_2d[:, 0])
        y2_comp = np.max(compensated_2d[:, 1])
        
        # 更新跟踪器的状态
        center_x = (x1_comp + x2_comp) / 2.0
        center_y = (y1_comp + y2_comp) / 2.0
        width = x2_comp - x1_comp
        height = y2_comp - y1_comp
        aspect_ratio = width / height if height != 0 else 0
        
        self.mean[:4] = [center_x, center_y, aspect_ratio, height]
        
        # 更新协方差矩阵
        # 构造雅可比矩阵
        # 由于是2D到2D的变换，我们只考虑x,y的变化
        J = np.eye(len(self.mean))
        
        # 简化处理：假设IMU补偿主要影响位置，使用平均变换矩阵
        # 在实际应用中，应该根据具体的投影和变换计算精确的雅可比矩阵
        avg_transform = np.array(ego_rot_imu[0])[:2, :2]  # 取旋转矩阵的2D部分
        J[:2, :2] = avg_transform
        
        # 更新协方差矩阵
        old_covariance = self.covariance.copy()
        self.covariance = J @ old_covariance @ J.T