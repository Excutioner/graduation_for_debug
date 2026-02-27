# Author: wangxy
# Modified to implement CTRV model based on the provided CTRA implementation

import numpy as np
from filterpy.kalman import KalmanFilter
from motion_module.motion_model import CTRV

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        """
        Initialises a tracker using initial bounding box.
        bbox3D format: [x, y, z, ry, l, w, h]
        
        DFM Internal State (self.kf.x) Structure for CTRV:
        [x, y, z, ry, l, w, h, v, omega]  <-- 9 dims (Removed 'a')
        """
        # 初始化时间步长
        self.dt = 0.1  # 可根据实际情况调整
        
        # 创建CTRV模型实例
        # 注意：需要确保 motion_module.motion_model 中有我们之前定义的 CTRV 类
        self.model = CTRV(has_velo=False, dt=self.dt)

        # 定义CTRV模型的状态转移矩阵和测量矩阵维度
        # dim_x = 9, dim_z = 7 (assuming no velocity obs)
        self.kf = KalmanFilter(dim_x=self.model.SD, dim_z=self.model.MD)

        # 初始化协方差矩阵 (2代表cls为car)
        self.kf.P = self.model.getInitCovP(2)
        
        # 过程噪声协方差
        self.kf.Q = self.model.getProcessNoiseQ()
        
        # 测量噪声协方差
        self.kf.R = self.model.getMeaNoiseR()
        
        # 初始化状态向量
        # DFM State: [x, y, z, ry, l, w, h, v, omega]
        # bbox3D:    [x, y, z, ry, l, w, h]
        # 直接填充前7位
        self.kf.x[:7] = bbox3D.reshape((7, 1)) 

    def DFM_to_CTRV(self, DFM_bbox3D):
        """
        [State Mapping]
        From DFM internal format: [x, y, z, ry, l, w, h, v, omega]
        To CTRV standard format:  [x, y, z, w, l, h, v, ry, omega]
        """
        # 统一输入为列向量
        if DFM_bbox3D.ndim == 1 or DFM_bbox3D.shape[1] != 1:
            DFM_bbox3D = DFM_bbox3D.reshape(-1, 1)
            
        bbox_full = np.zeros((9, 1)) # CTRV is 9 dims
        
        bbox_full[0:3] = DFM_bbox3D[0:3]  # x, y, z
        bbox_full[3] = DFM_bbox3D[5]      # w (来自原self.kf.x的w位置)
        bbox_full[4] = DFM_bbox3D[4]      # l (来自原self.kf.x的l位置)
        bbox_full[5] = DFM_bbox3D[6]      # h (来自原self.kf.x的h位置)
        bbox_full[6] = DFM_bbox3D[7]      # v (速度)
        bbox_full[7] = DFM_bbox3D[3]      # ry (朝向角) - 注意索引变化
        bbox_full[8] = DFM_bbox3D[8]      # omega (角速度) - DFM中排第9位(idx 8)
        
        return bbox_full
    
    def CTRV_to_DFM(self, CTRV_bbox3D):
        """
        [State Mapping]
        From CTRV standard format: [x, y, z, w, l, h, v, ry, omega]
        To DFM internal format:    [x, y, z, ry, l, w, h, v, omega]
        """
        # 使用 np.mat 保持与原代码风格一致
        return np.mat([
            [CTRV_bbox3D[0, 0]], # x
            [CTRV_bbox3D[1, 0]], # y
            [CTRV_bbox3D[2, 0]], # z
            [CTRV_bbox3D[7, 0]], # ry (From CTRV idx 7)
            [CTRV_bbox3D[4, 0]], # l
            [CTRV_bbox3D[3, 0]], # w (From CTRV idx 3)
            [CTRV_bbox3D[5, 0]], # h
            [CTRV_bbox3D[6, 0]], # v
            [CTRV_bbox3D[8, 0]]  # omega
        ])

    def update(self, bbox3D):
        """
        Updates the state vector with observed bbox.
        bbox3D format: [x, y, z, ry, l, w, h]
        """
        # --- 角度预处理逻辑 (与CTRA代码完全保持一致) ---
        
        # 1. 状态角度标准化
        if self.kf.x[3] >= np.pi: 
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: 
            self.kf.x[3] += np.pi * 2

        new_ry = bbox3D[3]
        
        # 2. 观测角度标准化
        if new_ry >= np.pi: 
            new_ry -= np.pi * 2
        if new_ry < -np.pi: 
            new_ry += np.pi * 2
        bbox3D[3] = new_ry

        predicted_ry = self.kf.x[3]
        
        # 3. 处理方向模糊 (180度跳变)
        # 如果预测和观测方向相反，修改观测值来迁就预测值(防止自旋)
        if abs(new_ry - predicted_ry) > np.pi / 2.0 and abs(new_ry - predicted_ry) < np.pi * 3 / 2.0:
            bbox3D[3] += np.pi
            if bbox3D[3] > np.pi: 
                bbox3D[3] -= np.pi * 2
            if bbox3D[3] < -np.pi: 
                bbox3D[3] += np.pi * 2
        
        new_ry = bbox3D[3] # 更新修正后的观测角
        
        # 4. 处理周期性 (Wrap-around)
        # 确保数值计算连续，例如 -3.14 和 +3.14
        if abs(new_ry - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_ry > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2
                    
        # --- 执行卡尔曼更新 ---
        
        # 获取测量信息 (注意顺序: x, y, z, w, l, h, ry)
        # 对应 CTRV 的 Measure Space
        meas_info = np.mat([[bbox3D[0]], [bbox3D[1]], [bbox3D[2]], [bbox3D[5]], [bbox3D[4]], [bbox3D[6]], [bbox3D[3]]])
        
        # 格式转换 DFM -> CTRV
        bbox_full = self.DFM_to_CTRV(self.kf.x)
        
        # 计算 Hx (Prediction in Measure Space)
        state_info = self.model.StateToMeasure(np.mat(bbox_full))
        
        # 计算残差
        _res = meas_info - state_info
        
        # 残差角度归一化 (关键步骤)
        _res = self.model.warpResYawToPi(_res)
        
        # 获取雅可比矩阵 H
        self.kf.H = self.model.getMeaStateH(np.mat(bbox_full))
        
        # 计算卡尔曼增益 K
        _S = self.kf.H * self.kf.P * self.kf.H.T + self.kf.R
        _KF_GAIN = self.kf.P * self.kf.H.T * _S.I
        
        # 更新协方差 P (使用 Joseph form 保证数值稳定性)
        _I_KH = np.mat(np.eye(self.model.SD)) - _KF_GAIN * self.kf.H
        self.kf.P = _I_KH * self.kf.P * _I_KH.T + _KF_GAIN * self.kf.R * _KF_GAIN.T
        
        # 更新状态 x
        # 注意：这里混合了 array 和 matrix 运算，沿用原代码风格
        bbox_full += _KF_GAIN * _res

        # 格式转换回 DFM 内部格式
        self.kf.x = self.CTRV_to_DFM(bbox_full)
        
        # 确保更新后的角度仍在 [-π, π] 范围内
        while self.kf.x[3] >= np.pi: 
            self.kf.x[3] -= np.pi * 2
        while self.kf.x[3] < -np.pi: 
            self.kf.x[3] += np.pi * 2

    def predict(self, apn_cfg=None): # [Modified] Add apn_cfg
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # [APN Logic for CTRV]
        if apn_cfg and apn_cfg.get('use_apn_ctra', False):
            try:
                # CTRV state in DFM: [x, y, z, ry, l, w, h, v, omega]
                # omega is at index 8
                omega = self.kf.x[8]
                
                params = apn_cfg.get('apn_params', {})
                k_omega = params.get('maneuver_factor_omega', 2.0)
                # CTRV assumes constant turn rate, no linear acceleration
                
                maneuver_factor = 1.0 + k_omega * np.abs(float(omega))
                self.kf.Q = self.original_Q * maneuver_factor
            except IndexError:
                pass

        # 格式转换
        state_mat = np.mat(self.DFM_to_CTRV(self.kf.x))
        predicted_state = self.model.stateTransition(state_mat)
        predicted_state = self.CTRV_to_DFM(predicted_state)
        
        self.kf.x = predicted_state.A.flatten()
        
        # ... (angle normalization same as before)
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
            
        F = self.model.getTransitionF(state_mat)
        self.kf.P = F * self.kf.P * F.T + self.kf.Q
        
        # [APN] Restore Q
        if apn_cfg and apn_cfg.get('use_apn_ctra', False):
            self.kf.Q = self.original_Q
        
        return self.kf.x[:7].flatten()
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].flatten()