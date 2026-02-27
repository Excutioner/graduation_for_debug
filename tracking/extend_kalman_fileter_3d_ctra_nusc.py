# Author: wangxy
# Modified for nuScenes global coordinate system (X-Y BEV plane) via Coordinate Deception

import numpy as np
from filterpy.kalman import KalmanFilter
from motion_module.motion_model import CTRA

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        self.dt = 0.1  
        self.model = CTRA(has_velo=False, dt=self.dt)
        self.kf = KalmanFilter(dim_x=self.model.SD, dim_z=self.model.MD)
        self.kf.P = self.model.getInitCovP(2)
        self.kf.Q = self.model.getProcessNoiseQ()
        self.kf.R = self.model.getMeaNoiseR()
        self.kf.x[:7] = bbox3D.reshape((7, 1))   
        self.original_Q = self.kf.Q.copy()

    def DFM_to_CTRA(self, DFM_bbox3D):
        """
        坐标轴欺骗：将 nuScenes 的 (X, Y, Z, yaw) 映射到 CTRA 期望的 KITTI (X, Z, Y, ry)
        骗过底层的 X-Z BEV 雅可比矩阵
        """
        if DFM_bbox3D.ndim == 1 or DFM_bbox3D.shape[1] != 1:
            DFM_bbox3D = DFM_bbox3D.reshape(-1, 1)
        bbox_full = np.zeros((10, 1))
        
        # --- 核心轴对换 ---
        bbox_full[0] = DFM_bbox3D[0]      # CTRA x (横向) = nusc x (横向)
        bbox_full[1] = DFM_bbox3D[2]      # CTRA y (高度) = nusc z (高度)
        bbox_full[2] = DFM_bbox3D[1]      # CTRA z (前方) = nusc y (前方)
        
        bbox_full[3] = DFM_bbox3D[5]      # w
        bbox_full[4] = DFM_bbox3D[4]      # l
        bbox_full[5] = DFM_bbox3D[6]      # h
        bbox_full[6] = DFM_bbox3D[7]      # v
        bbox_full[7] = DFM_bbox3D[8]      # a
        
        # 角度完美映射：CTRA ry = pi/2 - nusc yaw
        bbox_full[8] = np.pi / 2.0 - DFM_bbox3D[3]      
        # 角速度也要反向
        bbox_full[9] = -DFM_bbox3D[9]      
        return bbox_full
    
    def CTRA_to_DFM(self, CTRA_bbox3D):
        """
        逆向映射，恢复为 nuScenes 全局坐标系输出
        """
        return np.mat([
            [CTRA_bbox3D[0, 0]],                     # nusc x = CTRA x
            [CTRA_bbox3D[2, 0]],                     # nusc y = CTRA z
            [CTRA_bbox3D[1, 0]],                     # nusc z = CTRA y
            [np.pi / 2.0 - CTRA_bbox3D[8, 0]],       # nusc yaw = pi/2 - CTRA ry
            [CTRA_bbox3D[4, 0]],                     # l
            [CTRA_bbox3D[3, 0]],                     # w
            [CTRA_bbox3D[5, 0]],                     # h
            [CTRA_bbox3D[6, 0]],                     # v
            [CTRA_bbox3D[7, 0]],                     # a
            [-CTRA_bbox3D[9, 0]]                     # nusc omega = -CTRA omega
        ])

    def update(self, bbox3D):
        # 角度跳变处理（复用原逻辑，因为 nuScenes 的 yaw 同样在 [-pi, pi] 之间）
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_ry = bbox3D[3]
        if new_ry >= np.pi: new_ry -= np.pi * 2
        if new_ry < -np.pi: new_ry += np.pi * 2
        bbox3D[3] = new_ry

        predicted_ry = self.kf.x[3]
        if abs(new_ry - predicted_ry) > np.pi / 2.0 and abs(new_ry - predicted_ry) < np.pi * 3 / 2.0:
            bbox3D[3] += np.pi
            if bbox3D[3] > np.pi: bbox3D[3] -= np.pi * 2
            if bbox3D[3] < -np.pi: bbox3D[3] += np.pi * 2
        new_ry = bbox3D[3]
        
        if abs(new_ry - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_ry > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2
                    
        # --- [核心修改] 组装 CTRA 测量的矩阵时，同样应用轴欺骗 ---
        meas_info = np.mat([
            [bbox3D[0]],                        # CTRA x = nusc x
            [bbox3D[2]],                        # CTRA y = nusc z
            [bbox3D[1]],                        # CTRA z = nusc y
            [bbox3D[5]],                        # w
            [bbox3D[4]],                        # l
            [bbox3D[6]],                        # h
            [np.pi / 2.0 - bbox3D[3]]           # CTRA ry = pi/2 - nusc yaw
        ])
        
        bbox_full = self.DFM_to_CTRA(self.kf.x)
        state_info = self.model.StateToMeasure(np.mat(bbox_full))
        
        _res = meas_info - state_info
        _res = self.model.warpResYawToPi(_res)
        
        self.kf.H = self.model.getMeaStateH(np.mat(bbox_full))
        _S = self.kf.H * self.kf.P * self.kf.H.T + self.kf.R
        _KF_GAIN = self.kf.P * self.kf.H.T * _S.I
        _I_KH = np.mat(np.eye(self.model.SD)) - _KF_GAIN * self.kf.H
        
        bbox_full += _KF_GAIN * _res
        self.kf.P = _I_KH * self.kf.P * _I_KH.T + _KF_GAIN * self.kf.R * _KF_GAIN.T

        self.kf.x = self.CTRA_to_DFM(bbox_full)
        
        while self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        while self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    def predict(self, apn_cfg=None):
        if apn_cfg and apn_cfg.get('use_apn_ctra', False):
            try:
                accel = self.kf.x[8]
                omega = self.kf.x[9]
                
                OMEGA_THRESH = 0.02
                ACCEL_THRESH = 0.5
                
                params = apn_cfg.get('apn_params', {})
                k_omega = params.get('maneuver_factor_omega', 10.0) 
                k_accel = params.get('maneuver_factor_accel', 0.5)
                
                maneuver_factor = 1.0
                
                if np.abs(omega) > OMEGA_THRESH:
                    maneuver_factor += k_omega * (np.abs(omega) - OMEGA_THRESH)
                if np.abs(accel) > ACCEL_THRESH:
                    maneuver_factor += k_accel * (np.abs(accel) - ACCEL_THRESH)
                
                self.kf.Q = np.multiply(self.original_Q, maneuver_factor)
            except IndexError:
                pass

        state_mat = np.mat(self.DFM_to_CTRA(self.kf.x))
        predicted_state = self.model.stateTransition(state_mat)
        predicted_state = self.CTRA_to_DFM(predicted_state)
        
        self.kf.x = predicted_state.A.flatten()
        
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
            
        F = self.model.getTransitionF(state_mat)
        self.kf.P = F * self.kf.P * F.T + self.kf.Q
        
        if apn_cfg and apn_cfg.get('use_apn_ctra', False):
            self.kf.Q = self.original_Q

        return self.kf.x[:7].flatten()

    def get_state(self):
        return self.kf.x[:7].flatten()