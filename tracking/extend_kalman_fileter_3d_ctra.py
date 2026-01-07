# Author: wangxy
# The code refers to https://github.com/xinshuoweng/AB3DMOT
# Modified to implement CTRA model based on kalman_filter_copy.py

import numpy as np
from filterpy.kalman import KalmanFilter
from motion_module.motion_model import CTRA

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        """
        Initialises a tracker using initial bounding box.
        bbox3D format: [x, y, z, ry, l, w, h]
        self.kf.x: [x, y, z, ry, l, w, h, v, a, omega]
        """
        # 初始化时间步长
        self.dt = 0.1  # 可根据实际情况调整
        
        # 创建CTRA模型实例
        self.model = CTRA(has_velo=False, dt=self.dt)

        # 定义CTRA模型的状态转移矩阵和测量矩阵
        self.kf = KalmanFilter(dim_x=self.model.SD, dim_z=self.model.MD)

        # 初始化协方差矩阵(2代表cls为car)
        self.kf.P = self.model.getInitCovP(2)
        
        # 过程噪声协方差
        self.kf.Q = self.model.getProcessNoiseQ()
        
        # 测量噪声协方差
        self.kf.R = self.model.getMeaNoiseR()
        
        # MARK:  如果检测器的输出有速度，则此处需要修改
        self.kf.x[:7] = bbox3D.reshape((7, 1))   # [x,y,z,ry,l,w,h,v,a,omega]

    def DFM_to_CTRA(self,DFM_bbox3D):
        """
        格式从[x, y, z, ry, l, w, h, v, a, omega]转换为
        [x, y, z, w, l, h, v, a, ry, omega]
        """
        # 统一输入为列向量
        if DFM_bbox3D.ndim == 1 or DFM_bbox3D.shape[1] != 1:
            DFM_bbox3D = DFM_bbox3D.reshape(-1, 1)
        bbox_full = np.zeros((10, 1))
        bbox_full[0:3] = DFM_bbox3D[0:3]  # x, y, z
        bbox_full[3] = DFM_bbox3D[5]      # w (来自原self.kf.x的w位置)
        bbox_full[4] = DFM_bbox3D[4]      # l (来自原self.kf.x的l位置)
        bbox_full[5] = DFM_bbox3D[6]      # h (来自原self.kf.x的h位置)
        bbox_full[6] = DFM_bbox3D[7]      # v (速度)
        bbox_full[7] = DFM_bbox3D[8]      # a (加速度)
        bbox_full[8] = DFM_bbox3D[3]      # ry (朝向角)
        bbox_full[9] = DFM_bbox3D[9]      # omega (角速度)
        return bbox_full
    
    def CTRA_to_DFM(self,CTRA_bbox3D):
        """
        格式从[x, y, z, w, l, h, v, a, ry, omega]转换为
        [x, y, z, ry, l, w, h, v, a, omega]
        """
        
        return np.mat([[CTRA_bbox3D[0, 0]], [CTRA_bbox3D[1, 0]], [CTRA_bbox3D[2, 0]], \
                        [CTRA_bbox3D[8, 0]], 
                        [CTRA_bbox3D[4, 0]], [CTRA_bbox3D[3, 0]], [CTRA_bbox3D[5, 0]], 
                        [CTRA_bbox3D[6, 0]], [CTRA_bbox3D[7, 0]], [CTRA_bbox3D[9, 0]]])
    def update(self, bbox3D):
        """
        Updates the state vector with observed bbox.
        bbox3D format: [x, y, z, ry, l, w, h]
        """
        # 角度标准化处理
        if self.kf.x[3] >= np.pi: 
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: 
            self.kf.x[3] += np.pi * 2

        new_ry = bbox3D[3]
        if new_ry >= np.pi: 
            new_ry -= np.pi * 2
        if new_ry < -np.pi: 
            new_ry += np.pi * 2
        bbox3D[3] = new_ry

        predicted_ry = self.kf.x[3]
        
        # 处理角度跳变问题
        if abs(new_ry - predicted_ry) > np.pi / 2.0 and abs(new_ry - predicted_ry) < np.pi * 3 / 2.0:
            bbox3D[3] += np.pi
            if bbox3D[3] > np.pi: 
                bbox3D[3] -= np.pi * 2
            if bbox3D[3] < -np.pi: 
                bbox3D[3] += np.pi * 2
        new_ry = bbox3D[3]
        # 处理大于270度的角度差异
        if abs(new_ry - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_ry > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2
                    
        # 执行卡尔曼更新
        # 获取测量信息并投影状态到测量空间，此处还需要将bbox3D的顺序修改为[x, y, z, w, l, h, ry]
        meas_info = np.mat([[bbox3D[0]], [bbox3D[1]], [bbox3D[2]], [bbox3D[5]], [bbox3D[4]], [bbox3D[6]], [bbox3D[3]]])
        
        # 格式转换
        bbox_full = self.DFM_to_CTRA(self.kf.x)
        
        # 计算Hx
        state_info = self.model.StateToMeasure(np.mat(bbox_full))
        
        # 计算残差并处理角度
        _res = meas_info - state_info
        _res = self.model.warpResYawToPi(_res)
        
        # 获取雅可比矩阵H
        self.kf.H = self.model.getMeaStateH(np.mat(bbox_full))
        
        # 计算卡尔曼增益和更新状态及协方差
        _S = self.kf.H * self.kf.P * self.kf.H.T + self.kf.R
        _KF_GAIN = self.kf.P * self.kf.H.T * _S.I
        _I_KH = np.mat(np.eye(self.model.SD)) - _KF_GAIN * self.kf.H
        
        bbox_full += _KF_GAIN * _res
        self.kf.P = _I_KH * self.kf.P * _I_KH.T + _KF_GAIN * self.kf.R * _KF_GAIN.T

        # 将bbox_full从[x, y, z, w, l, h, v, a, ry, omega]转换为
        # [x,y,z,ry,l,w,h,v,a,omega]
        self.kf.x = self.CTRA_to_DFM(bbox_full)
        
        # 确保更新后的角度仍在[-π, π]范围内
        while self.kf.x[3] >= np.pi: 
            self.kf.x[3] -= np.pi * 2
        while self.kf.x[3] < -np.pi: 
            self.kf.x[3] += np.pi * 2

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # 使用CTRA模型进行状态转移
        state_mat = np.mat(self.DFM_to_CTRA(self.kf.x))
        predicted_state = self.model.stateTransition(state_mat)
        predicted_state = self.CTRA_to_DFM(predicted_state)
        # 更新状态
        self.kf.x = predicted_state.A.flatten()
        
        # 确保预测后的角度仍在[-π, π]范围内
        if self.kf.x[3] >= np.pi: 
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: 
            self.kf.x[3] += np.pi * 2
            
        # 更新协方差矩阵
        F = self.model.getTransitionF(state_mat)
        self.kf.P = F * self.kf.P * F.T + self.kf.Q
        
        return self.kf.x[:7].flatten()
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].flatten()
