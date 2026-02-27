"""
motion model of trajectory, notice that objects of different categories often exhibit various motion patterns.
Five implemented motion models, including
- Two linear model: Constant Acceleration(CA), Constant Velocity(CV)
- Three non-linear model: Constant Turn Rate and Acceleration(CTRA), Constant Turn Rate and Velocity(CTRV), Bicycle Model
"""
import abc
import pdb
import numpy as np
from typing import Tuple
from pyquaternion import Quaternion

# origin from utils.math import warp_to_pi
def expand_dims(array: np.array, expand_len: int, dim: int) -> np.array:
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)

def warp_to_pi(yaw: float) -> float:
    """warp yaw to [-pi, pi)

    Args:
        yaw (float): raw angle

    Returns:
        float: raw angle after warping
    """
    while yaw >= np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw


# origin from data.script.NUSC_CONSTANT import *

# metrics with two return values
METRIC = ['iou_3d', 'giou_3d']
FAST_METRIC = ['giou_3d', 'giou_bev']

# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = {'bicycle': 0, 'bus': 1, 'car': 2, 'motorcycle': 3, 'pedestrian': 4, 'trailer': 5, 'truck': 6}
CLASS_STR_TO_SEG_CLASS = {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'pedestrian', 5: 'trailer', 6: 'truck'}

# math
PI, TWO_PI = np.pi, 2 * np.pi

# init EKFP for different non-linear motion model
CTRA_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    # 'car': [4, 4, 4, 1, 1, 1, 1000, 10, 10, 10],
    'car': [4, 4, 4, 1, 1, 1, 1000, 1, 1, 0.01],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
CTRV_INIT_EFKP = {
    # [x, y, z, w, l, h, v, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 1, 1, 1, 1000, 1, 0.01],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
BIC_INIT_EKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'bicycle': [10, 10, 10, 10, 10, 10, 10000, 10, 10, 10],
    'motorcycle': [4, 4, 4, 4, 4, 4, 100, 4, 4, 1],
}



class ABC_MODEL(abc.ABC):
    """interface of all motion models
    """
    def __init__(self) -> None:
        self.SD = self.MD = -1
    
    @abc.abstractmethod
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet

        Args:
            det_infos (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }

        Returns:
            np.mat: [state dim, 1], state vector
        """
        pass
    
    @abc.abstractmethod
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov.

        Args:
            cls_label (int): set init errorcov category-specific. 

        Returns:
            np.mat: [state dim, state dim], Initialized covariance matrix
        """
        pass
    
    @abc.abstractmethod
    def getProcessNoiseQ(self) -> np.mat:
        """get process noise matrix. The value set is somewhat arbitrary

        Returns:
            np.mat: process noise matrix(fix)
        """
        pass
    
    @abc.abstractmethod
    def getTransitionF(self) -> np.mat:
        """get state transition matrix.
        obtain matrix in the motion_module/script/
        Returns:
            np.mat: [state dim, state dim], state transition matrix
        """
        pass
    
    @abc.abstractmethod
    def getMeaNoiseR(self) -> np.mat:
        """get measurement noise matrix. The value set is also somewhat arbitrary
        Returns:
            np.mat: measure noise matrix(fix)
        """
        pass
    
    @abc.abstractmethod
    def getMeaStateH(self) -> np.mat:
        """get state to measure transition matrix.
        obtain matrix in the motion_module/script
        Returns:
            np.mat: [measure dim, state dim], state to measure transition matrix
        """
        pass

    @abc.abstractmethod
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [12(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4)]
        """
        pass
    
    def getStateDim(self) -> int:
        return self.SD
    
    def getMeasureDim(self) -> int:
        return self.MD
     

class CA(ABC_MODEL): 
    """Constant Acceleration Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 13
        self.MD = 9 if self.has_velo else 7
    
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Acceleration and velocity on the z-axis are both set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set x, y, z, w, l, h, (vx, vy, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6:8] = det[6:8]
        
        # set yaw
        init_state[-1] = det_box.yaw
        
        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw
        
        return np.mat(init_state).T
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. Generally, the CA model can converge quickly, 
        so not particularly sensitive to initialization
        """
        return np.mat(np.eye(self.SD)) * 0.01
    
    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        """
        return np.mat(np.eye(self.SD)) * 100
    
    def getMeaNoiseR(self) -> np.mat:
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.MD)) * 0.001
    
    def getTransitionF(self) -> np.mat:
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        """
        dt = self.dt
        F = np.mat([[1, 0, 0, 0, 0, 0, dt,  0, 0, 0.5*dt**2,         0,  0, 0],
                    [0, 1, 0, 0, 0, 0,  0, dt, 0,         0, 0.5*dt**2,  0, 0],
                    [0, 0, 1, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 1, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 1, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 1,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  1,  0, 0,        dt,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  1, 0,         0,        dt,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         1,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         1,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 1]])
        return F
    
    def getMeaStateH(self) -> np.mat:
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        """
        if self.has_velo:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        else:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return H

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-1, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)
    
    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res
    
    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state
    
class CV(ABC_MODEL):
    """Constant Velocity Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """

    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7

    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Velocity on the z-axis are set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']

        # set x, y, z, w, l, h, (vx, vy, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6:8] = det[6:8]

        # set yaw
        init_state[-1] = det_box.yaw

        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw

        return np.mat(init_state).T

    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. Generally, the CV model can converge quickly,
        so not particularly sensitive to initialization
        """
        return np.mat(np.eye(self.SD)) * 0.01

    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        """
        return np.mat(np.eye(self.SD)) * 100

    def getMeaNoiseR(self) -> np.mat:
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.MD)) * 0.001

    def getTransitionF(self) -> np.mat:
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        dt = self.dt
        F = np.mat([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return F

    def getMeaStateH(self) -> np.mat:
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        if self.has_velo:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        else:
            H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return H

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-1, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)

    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state

        
class CTRA(ABC_MODEL): 
    """Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]
        Measure vector: [x, y, z, w, l, h, (vx, vz, optional), ry]
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7
        
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Acceleration and yaw(turn) rate are both set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set x, y, z, w, l, h, (v, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6] = np.hypot(det[6], det[7])
        
        # set yaw
        init_state[-2] = det_box.yaw
        
        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw
        
        return np.mat(init_state).T
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. In general, when the speed is observable, 
        the CTRA model can converge quickly, but when the speed is not measurable, 
        we need to carefully set the initial covariance to help the model converge
        """
        if not self.has_velo:
            cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
            vector_p = CTRA_INIT_EFKP[cls_name] if cls_name in CTRA_INIT_EFKP else CTRA_INIT_EFKP['car']
        else:
            vector_p = CTRA_INIT_EFKP['car']
        
        return np.mat(np.diag(vector_p))
    
    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        """
        # 自定义 Q 矩阵
        q_diag = [1.0, 1.0, 1.0,    # x, y, z (位置相对准)
                0.01, 0.01, 0.01, # w, l, h (尺寸几乎不变)
                # 1.0, 0.01,         # v, a (速度加速度允许变化)
                # 1.0, 0.01]         # ry (角度), omega (角速度噪声要大一点还是小一点取决于你相信模型还是测量)
                1.0, 0.05,         # v, a (速度加速度允许变化)
                1.0, 0.05]         # ry (角度), omega (角速度噪声要大一点还是小一点取决于你相信模型还是测量)
        # 对于 has_velo=False，建议限制 omega 的噪声，防止乱转
        return np.mat(np.diag(q_diag))
    
    def getMeaNoiseR(self) -> np.mat:
        """
        set measure noise(fix)
        Standard LiDAR detectors (like VirConv/PointRCNN) usually have:
        - Position error: ~0.1 - 0.3m
        - Size error: ~0.2 - 0.5m
        - Angle error: ~0.05 - 0.2 rad (3 - 10 degrees)
        """
        # [x, y, z, w, l, h, (v), ry] 
        # 注意：如果不含速度，维度是7；含速度是9。这里假设 has_velo=False，维度=7
        
        # r_diag = [
        #     0.1, 0.1, 0.1,   # x, y, z: 设为 0.1，表示极其信任检测器的位置 (跟得紧!)
        #     1.0, 1.0, 1.0,   # w, l, h: 尺寸多大无所谓，设大点让它稳定，不要乱变
        #     0.1              # ry: 设为 0.1 (约5.7度)。必须比 1.0 小！
        #                      # 这样一旦车头转了，轨迹能立刻跟过去。
        # ]
        r_diag = [0.001, 0.001, 0.001, 1.0, 1.0, 1.0, 0.001] 
        # 兼容性处理：防止 self.MD 维度变化导致报错
        if self.has_velo:
             # 如果有速度观测，通常速度噪声也比较大，不如位置准
             r_diag = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1] 

        # 截取对应维度的对角阵
        R = np.mat(np.diag(r_diag[:self.MD]))
        return R
    
    def stateTransition(self, state: np.mat) -> np.mat:
        """state transition, 
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        dt = self.dt
        x, y, z, w, l, h, v, a, ry, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(ry), np.cos(ry)
        next_v, next_ry = v + a * dt, ry + omega * dt
        
        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        # MARK：此处kitti数据集上检测器的角度(rect坐标系下)与CTRA模型定义的角度不同
        # CTRA模型角度定义示意图:
        #  ↑ Y
        #  |
        #  |    V
        #  |   /|
        #  |  / |
        #  | /  | Vy = V*sin(θ)
        #  |/ θ |
        #  +----+----→ X
        #  O    Vx = V*cos(θ)
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x + displacement * yaw_cos,
                             y,
                             z + displacement * yaw_sin,
                             w, l, h, 
                             next_v, a, 
                             next_ry, omega]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
                             y,
                             z + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
                             w, l, h,
                             next_v, a, 
                             next_ry, omega]
        
        return np.mat(predict_state).T  
        
        
    
    def StateToMeasure(self, state: np.mat) -> np.mat:
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, (vx, vz, optional), ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        if self.has_velo:
            state_info = [x, y, z,
                          w, l, h,
                          v * np.cos(theta), # vx (Rect X)
                          v * np.sin(theta), # vz (Rect Z)
                          theta]
        else:
            state_info = [x, y, z,
                          w, l, h,
                          theta]
        
        return np.mat(state_info).T
         
        
    
    def getTransitionF(self, state: np.mat) -> np.mat:
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.dt
        _, _, _, _, _, _, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)

        # Initialize Identity
        F = np.eye(self.SD)
        # Helper indices
        IDX_X, IDX_Y, IDX_Z = 0, 1, 2
        IDX_V, IDX_A, IDX_THETA, IDX_OMEGA = 6, 7, 8, 9

        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            
            # dX / d...
            F[IDX_X, IDX_V] = dt * yaw_cos
            F[IDX_X, IDX_A] = 0.5 * dt**2 * yaw_cos
            F[IDX_X, IDX_THETA] = -displacement * yaw_sin
            
            # dZ / d... (Old Y logic moves here)
            F[IDX_Z, IDX_V] = dt * yaw_sin
            F[IDX_Z, IDX_A] = 0.5 * dt**2 * yaw_sin
            F[IDX_Z, IDX_THETA] = displacement * yaw_cos
            
            # dv/da
            F[IDX_V, IDX_A] = dt
            # dTheta / dOmega
            F[IDX_THETA, IDX_OMEGA] = dt
            
        else:
            ry_rate_inv = 1.0 / omega
            ry_rate_inv_square = ry_rate_inv * ry_rate_inv
            ry_rate_inv_cube = ry_rate_inv_square * ry_rate_inv
            
            next_v = v + a * dt
            next_ry = theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            
            # --- Row 0: X (Lateral) --- 
            # Logic corresponds to "x" in standard CTRA (cos terms)
            F[IDX_X, IDX_V] = -ry_rate_inv*(yaw_sin-next_yaw_sin) # Simplifies
            F[IDX_X, IDX_A] = -ry_rate_inv_square*(yaw_cos-next_yaw_cos)+ry_rate_inv*dt*next_yaw_sin
            F[IDX_X, IDX_THETA] = ry_rate_inv_square*a*(yaw_sin-next_yaw_sin)+ry_rate_inv*(next_v*next_yaw_cos-v*yaw_cos) # approx check
            F[IDX_X, IDX_OMEGA] = ry_rate_inv_cube*2*a*(yaw_cos-next_yaw_cos)+ry_rate_inv_square*(v*yaw_sin-v*next_yaw_sin-2*a*dt*next_yaw_sin)+ry_rate_inv*dt*next_v*next_yaw_cos
            # --- Row 2: Z (Longitudinal) ---
            # Logic corresponds to "y" in standard CTRA (sin terms)
            # ... (Z行有值)
            F[IDX_Z, IDX_V] = ry_rate_inv*(yaw_cos-next_yaw_cos)
            F[IDX_Z, IDX_A] = -ry_rate_inv_square*(yaw_sin-next_yaw_sin)-ry_rate_inv*dt*next_yaw_cos
            F[IDX_Z, IDX_THETA] = ry_rate_inv_square*a*(-yaw_cos+next_yaw_cos)+ry_rate_inv*(next_v*next_yaw_sin-v*yaw_sin)
            F[IDX_Z, IDX_OMEGA] = ry_rate_inv_cube*2*a*(yaw_sin-next_yaw_sin)+ry_rate_inv_square*(v*next_yaw_cos-v*yaw_cos+2*a*dt*next_yaw_cos)+ry_rate_inv*dt*next_v*next_yaw_sin
            
            # dv/da
            F[IDX_V, IDX_A] = dt
            # dTheta / dOmega
            F[IDX_THETA, IDX_OMEGA] = dt
            
            
        return np.mat(F)
    
    def getMeaStateH(self, state: np.mat) -> np.mat:
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """
        
        if self.has_velo:
            _, _, _, _, _, _, v, _, theta, _ = state.T.tolist()[0]
            yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
            
            H = np.zeros((9, 10))
            # Identity blocks for x,y,z,w,l,h
            H[0:6, 0:6] = np.eye(6)
            
            # Row 6: vx = v * cos(theta)
            H[6, 6] = yaw_cos        # dvx/dv
            H[6, 8] = -v * yaw_sin   # dvx/dtheta
            
            # Row 7: vz = v * sin(theta)
            H[7, 6] = yaw_sin        # dvz/dv
            H[7, 8] = v * yaw_cos    # dvz/dtheta
            
            # Row 8: ry
            H[8, 8] = 1
            
        else:
            H = np.zeros((7, 10))
            H[0:6, 0:6] = np.eye(6)
            H[6, 8] = 1
            
        return np.mat(H)

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-2, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)
    
    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res
    
    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state


class CTRV(ABC_MODEL):
    """
    Constant Turn Rate and Velocity Motion Model
    Adapted to match the coordinate system and math logic of the CTRA model provided.
    
    Basic info:
        State vector: [x, y, z, w, l, h, v, ry, omega]  (9 dims)
        Measure vector: [x, y, z, w, l, h, (vx, vz, optional), ry]
        
    Coordinate System:
        Right-handed: X (Right), Y (Down/Height), Z (Front)
        Motion is on X-Z plane. Y is constant.
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 9
        self.MD = 9 if self.has_velo else 7

    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet
        Yaw(turn) rate are set to 0.
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']

        # set x, y, z, w, l, h, (v, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo: init_state[6] = np.hypot(det[6], det[7])

        # set yaw
        init_state[-2] = det_box.yaw

        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw

        return np.mat(init_state).T

    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. 
        """
        # 使用与 CTRA 类似的逻辑，但不需要加速度 a
        # 建议参考我们讨论过的优化参数，特别是 omega 给小一点
        if not self.has_velo:
            cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
            vector_p = CTRV_INIT_EFKP[cls_name] if cls_name in CTRV_INIT_EFKP else CTRV_INIT_EFKP['car']
        else:
            vector_p = CTRV_INIT_EFKP['car']

        return np.mat(np.diag(vector_p))

    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        Based on the Optimized Exp 6 logic:
        High Pos/Angle Noise (Response), Low Dimension/Omega Noise (Stability)
        """
        q_diag = [
            1.0, 1.0, 1.0,    # x, y, z (位置跟得紧)
            0.01, 0.01, 0.01, # w, l, h (尺寸稳定)
            1.0,              # v (速度允许变化)
            1.0,              # ry (角度跟得紧)
            0.01             # omega (角速度压制，防止乱转)
        ]
        return np.mat(np.diag(q_diag))

    def getMeaNoiseR(self) -> np.mat:
        """set measure noise(fix)
        Based on Optimized R-Opt logic
        """
        # 针对无速度观测优化的 R 矩阵
        r_diag = [
            0.1, 0.1, 0.1,   # x, y, z (相信检测器位置)
            1.0, 1.0, 1.0,   # w, l, h
            0.1              # ry (相信检测器角度)
        ]
        
        if self.has_velo:
             r_diag = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1] 

        R = np.mat(np.diag(r_diag[:self.MD]))
        return R

    def stateTransition(self, state: np.mat) -> np.mat:
        """state transition
        Derived from CTRA by setting a = 0.
        Motion is on X-Z plane (Indices 0 and 2). Index 1 (Y) is Height.
        """
        assert state.shape == (9, 1), "state vector number in CTRV must equal to 9"

        dt = self.dt
        # 注意这里的解包顺序，没有 a
        x, y, z, w, l, h, v, theta, omega = state.T.tolist()[0]
        
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v = v # 恒定速度
        next_ry = theta + omega * dt # 恒定转率

        # corner case(tiny yaw rate)
        if abs(omega) < 0.001:
            displacement = v * dt
            # 线性模型：沿着当前角度直线运动
            # 注意：参考 CTRA，cos 对应 x，sin 对应 z
            predict_state = [x + displacement * yaw_cos,
                             y, # Y 不变
                             z + displacement * yaw_sin,
                             w, l, h,
                             next_v,
                             next_ry, omega]
        else:
            # 非线性模型 (圆弧运动)
            # 公式推导自 CTRA，令 a=0
            ry_rate_inv = 1.0 / omega
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            
            predict_state = [
                # X: x + (v/w) * (sin(next) - sin(curr))
                x + ry_rate_inv * (v * next_yaw_sin - v * yaw_sin),
                y, # Y 不变
                # Z: z + (v/w) * (cos(curr) - cos(next))
                # 注意 CTRA 中 Z轴对应项是: -next_v*omega*next_yaw_cos + v*omega*yaw_cos
                # 除以 omega^2 后 -> (v/w) * (cos(curr) - cos(next))
                z + ry_rate_inv * (v * yaw_cos - v * next_yaw_cos),
                w, l, h,
                next_v,
                next_ry, omega
            ]

        return np.mat(predict_state).T

    def StateToMeasure(self, state: np.mat) -> np.mat:
        """get state vector in the measure space
        """
        assert state.shape == (9, 1), "state vector number in CTRV must equal to 9"

        x, y, z, w, l, h, v, theta, _ = state.T.tolist()[0]
        
        if self.has_velo:
            state_info = [x, y, z,
                          w, l, h,
                          v * np.cos(theta), # vx
                          v * np.sin(theta), # vz (注意这里改为 vz)
                          theta]
        else:
            state_info = [x, y, z,
                          w, l, h,
                          theta]

        return np.mat(state_info).T

    def getTransitionF(self, state: np.mat) -> np.mat:
        """
        Jacobian Matrix F calculation.
        Derived strictly from CTRA's Jacobian by removing Row/Col 7 (acceleration 'a')
        and setting a = 0 in all remaining formulas.
        """
        dt = self.dt
        x, y, z, w, l, h, v, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)

        # Initialize Identity
        F = np.eye(self.SD)
        
        # Indices map for CTRV (9 dims)
        # x, y, z, w, l, h, v, theta, omega
        # 0, 1, 2, 3, 4, 5, 6, 7,     8
        IDX_X, IDX_Z = 0, 2
        IDX_V, IDX_THETA, IDX_OMEGA = 6, 7, 8

        if abs(omega) < 0.001:
            # Linear approximation derivatives
            # dX / d...
            F[IDX_X, IDX_V] = dt * yaw_cos
            F[IDX_X, IDX_THETA] = -v * dt * yaw_sin
            
            # dZ / d...
            F[IDX_Z, IDX_V] = dt * yaw_sin
            F[IDX_Z, IDX_THETA] = v * dt * yaw_cos
            
            # dTheta / dOmega
            F[IDX_THETA, IDX_OMEGA] = dt
            
        else:
            ry_rate_inv = 1.0 / omega
            ry_rate_inv_square = ry_rate_inv * ry_rate_inv
            
            next_ry = theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            
            # --- Row 0: X (Lateral) ---
            # From CTRA: F[0, V] = -ry_rate_inv*(yaw_sin-next_yaw_sin)
            F[IDX_X, IDX_V] = -ry_rate_inv * (yaw_sin - next_yaw_sin)
            
            # From CTRA (a=0): F[0, Theta] = ry_rate_inv*(v*next_yaw_cos - v*yaw_cos)
            F[IDX_X, IDX_THETA] = ry_rate_inv * v * (next_yaw_cos - yaw_cos)
            
            # From CTRA (a=0): F[0, Omega]
            # Term: ry_rate_inv_square*(v*yaw_sin - v*next_yaw_sin) + ry_rate_inv*dt*next_v*next_yaw_cos
            F[IDX_X, IDX_OMEGA] = ry_rate_inv_square * v * (yaw_sin - next_yaw_sin) + \
                                  ry_rate_inv * dt * v * next_yaw_cos

            # --- Row 2: Z (Longitudinal) ---
            # From CTRA: F[2, V] = ry_rate_inv*(yaw_cos-next_yaw_cos)
            F[IDX_Z, IDX_V] = ry_rate_inv * (yaw_cos - next_yaw_cos)
            
            # From CTRA (a=0): F[2, Theta] = ry_rate_inv*(v*next_yaw_sin - v*yaw_sin)
            F[IDX_Z, IDX_THETA] = ry_rate_inv * v * (next_yaw_sin - yaw_sin)
            
            # From CTRA (a=0): F[2, Omega]
            # Term: ry_rate_inv_square*(v*next_yaw_cos - v*yaw_cos) + ry_rate_inv*dt*next_v*next_yaw_sin
            F[IDX_Z, IDX_OMEGA] = ry_rate_inv_square * v * (next_yaw_cos - yaw_cos) + \
                                  ry_rate_inv * dt * v * next_yaw_sin
            
            # dTheta / dOmega
            F[IDX_THETA, IDX_OMEGA] = dt

        return np.mat(F)

    def getMeaStateH(self, state: np.mat) -> np.mat:
        """
        H Matrix. Matches CTRA logic.
        """
        if self.has_velo:
            _, _, _, _, _, _, v, theta, _ = state.T.tolist()[0]
            yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
            
            H = np.zeros((9, 9))
            # Identity blocks for x,y,z,w,l,h
            H[0:6, 0:6] = np.eye(6)
            
            # Row 6: vx = v * cos(theta)
            H[6, 6] = yaw_cos        # dvx/dv
            H[6, 7] = -v * yaw_sin   # dvx/dtheta
            
            # Row 7: vz = v * sin(theta)
            H[7, 6] = yaw_sin        # dvz/dv
            H[7, 7] = v * yaw_cos    # dvz/dtheta
            
            # Row 8: ry
            H[8, 7] = 1
            
        else:
            H = np.zeros((7, 9))
            H[0:6, 0:6] = np.eye(6)
            H[6, 7] = 1
            
        return np.mat(H)

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        """
        # 索引 -2 是 theta
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-2, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)
    
    # 静态方法保持不变
    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state


class BICYCLE(ABC_MODEL):
    """Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x_gra, y_gra, z_geo, w, l, h, v, a, ry, sigma]
        Measure vector: [x_geo, y_geo, z_geo, w, l, h, (vx, vy, optional), ry]
    Important assumptions:
        1. Although the acceleration interface is reserved, 
        we still think that the velocity is constant when 
        beta is large. In other cases, the object is considered 
        to be moving in a straight line with uniform acceleration.
        2. Based on experience, we set two hyperparameters here, 
        wheelbase ratio is set to 0.8, rear tire ratio is set to 0.5.
        3. the steering angle(sigma) is also considered to be constant.
        4. x, y described in the filter are gravity center, however, 
        the infos we measure are geometric center. Transformation 
        process is required.
        5. We don't intergrate variable 'length' to the jacobian matrix
        for simple and fast solution
    """
    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7
        self.w_r, self.lf_r = 0.8, 0.5
        
    def getInitState(self, det_infos: dict) -> np.mat:
        """from detection init tracklet, we set some assumptions in 
        the BICYCLE model for easy calculation
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos['np_array'], det_infos['nusc_box']
        
        # set gravity center x, y
        init_state[:2] = self.geoCenterToGraCenter(geo_center=[det[0], det[1]],
                                                   theta=det_box.yaw,
                                                   length=det[4])
        
        # set z, w, l, h, (v, if velo is valid)
        init_state[2:6] = det[2:6]
        if self.has_velo: init_state[6] = np.hypot(det[6], det[7])
        
        # set yaw
        init_state[-2] = det_box.yaw
        
        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw
        
        return np.mat(init_state).T
    
    def getInitCovP(self, cls_label: int) -> np.mat:
        """init errorcov. In the BICYCLE motion model, we apply 
        same errorcov initialization strategy as the CTRA model.
        """
        if not self.has_velo:
            cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
            vector_p = BIC_INIT_EKFP[cls_name] if cls_name in BIC_INIT_EKFP else BIC_INIT_EKFP['bicycle']
        else:
            vector_p = BIC_INIT_EKFP['bicycle'] if cls_label == 0 else [10, 10, 10, 10, 10, 10, 1000, 10, 10, 10]
        
        return np.mat(np.diag(vector_p))
    
    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        """
        return np.mat(np.eye(self.SD)) * 1
    
    def getMeaNoiseR(self) -> np.mat:
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.MD)) * 1
    
    def getTransitionF(self, state: np.mat) -> np.mat:
        """obtain matrix in the motion_module/script/BIC_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        
        dt = self.dt
        _, _, _, _, l, _, v, a, theta, sigma = state.T.tolist()[0]
        beta, _, lr = self.getBicBeta(l, sigma)
        
        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)
        
        # corner case, tiny beta
        if abs(beta) < 0.001:
            displacement = a*dt**2/2 + dt*v
            F = np.mat([[1, 0, 0, 0, 0, 0,  dt*cos_yaw,  dt**2*cos_yaw/2,        -displacement*sin_yaw, 0],
                        [0, 1, 0, 0, 0, 0,  dt*sin_yaw,  dt**2*sin_yaw/2,         displacement*cos_yaw, 0],
                        [0, 0, 1, 0, 0, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 1, 0, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 1, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 0, 1,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           1,               dt,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           0,                1,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           0,                0,                            1, 0],
                        [0, 0, 0, 0, 0, 0,           0,                0,                            0, 1]])
        else:
            next_yaw, sin_beta = theta + v / lr * np.sin(beta) * dt, np.sin(beta)
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            F = np.mat([[1, 0, 0, 0, 0, 0, dt*np.cos(next_v_yaw), 0, -lr*np.cos(v_yaw)/sin_beta + lr*np.cos(next_v_yaw)/sin_beta, 0],
                        [0, 1, 0, 0, 0, 0, dt*np.sin(next_v_yaw), 0, -lr*np.sin(v_yaw)/sin_beta + lr*np.sin(next_v_yaw)/sin_beta, 0],
                        [0, 0, 1, 0, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 1, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 1, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 1,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,                     1, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,        dt/lr*sin_beta, 0,                                                           1, 0],
                        [0, 0, 0, 0, 0, 0,                     0, 0,                                                           0, 1]])
        return F
    
    def getMeaStateH(self, state: np.mat) -> np.mat:
        """obtain matrix in the motion_module/script/BIC_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """
        _, _, _, _, l, _, v, _, theta, sigma = state.T.tolist()[0]
        
        geo2gra_dist, lr2l = self.graToGeoDist(l), self.w_r * (0.5 - self.lf_r)
        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)
        
        if self.has_velo:
            beta, _, _ = self.getBicBeta(l, sigma)
            v_yaw = beta + theta
            sin_v_yaw, cos_v_yaw = np.sin(v_yaw), np.cos(v_yaw)
            H = np.mat([[1, 0, 0, 0, -lr2l*cos_yaw, 0,               0, 0,  geo2gra_dist*sin_yaw, 0],
                        [0, 1, 0, 0, -lr2l*sin_yaw, 0,               0, 0, -geo2gra_dist*cos_yaw, 0],
                        [0, 0, 1, 0,             0, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 1,             0, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             1, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             0, 1,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             0, 0,       cos_v_yaw, 0,          -v*sin_v_yaw, 0],
                        [0, 0, 0, 0,             0, 0,       sin_v_yaw, 0,           v*cos_v_yaw, 0],
                        [0, 0, 0, 0,             0, 0,               0, 0,                     1, 0]])
        else:
            H = np.mat([[1, 0, 0, 0, -lr2l*cos_yaw, 0,               0, 0,  geo2gra_dist*sin_yaw, 0],
                        [0, 1, 0, 0, -lr2l*sin_yaw, 0,               0, 0, -geo2gra_dist*cos_yaw, 0],
                        [0, 0, 1, 0,             0, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 1,             0, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             1, 0,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             0, 1,               0, 0,                     0, 0],
                        [0, 0, 0, 0,             0, 0,               0, 0,                     1, 0]])
        
        return H
    
    def stateTransition(self, state: np.mat) -> np.mat:
        """State transition based on model assumptions
        """
        assert state.shape == (10, 1), "state vector number in BICYCLE must equal to 10"
        
        dt = self.dt
        x_gra, y_gra, z, w, l, h, v, a, theta, sigma = state.T.tolist()[0]
        beta, _, lr = self.getBicBeta(l, sigma)
        
        # corner case, tiny yaw rate
        if abs(beta) > 0.001:
            next_yaw = theta + v / lr * np.sin(beta) * dt
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            predict_state = [x_gra + (lr * (np.sin(next_v_yaw) - np.sin(v_yaw))) / np.sin(beta),  # x_gra
                             y_gra - (lr * (np.cos(next_v_yaw) - np.cos(v_yaw))) / np.sin(beta),  # y_gra
                             z, w, l, h,
                             v, 0, next_yaw, sigma]
        else:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x_gra + displacement * np.cos(theta),
                             y_gra + displacement * np.sin(theta),
                             z, w, l, h,
                             v + a * dt, a, theta, sigma]
        return np.mat(predict_state).T  
    
    def StateToMeasure(self, state: np.mat) -> np.mat:
        """get state vector in the measure space
        """
        assert state.shape == (10, 1), "state vector number in BICYCLE must equal to 10"
        
        x_gra, y_gra, z, w, l, h, v, _, theta, sigma = state.T.tolist()[0]
        
        beta, _, _ = self.getBicBeta(l, sigma)
        geo2gra_dist = self.graToGeoDist(l)
        
        if self.has_velo:
            meas_state = [x_gra - geo2gra_dist * np.cos(theta),
                          y_gra - geo2gra_dist * np.sin(theta),
                          z, w, l, h,
                          v * np.cos(theta + beta),
                          v * np.sin(theta + beta),
                          theta]
        else:
            meas_state = [x_gra - geo2gra_dist * np.cos(theta),
                          y_gra - geo2gra_dist * np.sin(theta),
                          z, w, l, h,
                          theta]
        
        return np.mat(meas_state).T
    
    def getBicBeta(self, length: float, sigma: float) -> Tuple[float, float, float]:
        """get the angle between the object velocity and the 
        X-axis of the coordinate system

        Args:
            length (float): object length
            sigma (float): the steering angle, radians

        Returns:
            float: the angle between the object velocity and X-axis, radians
        """
        
        lf, lr = length * self.w_r * self.lf_r, length * self.w_r * (1 - self.lf_r)
        beta = np.arctan(lr / (lr + lf) * np.tan(sigma))
        return beta, lf, lr
        
    
    def geoCenterToGraCenter(self, geo_center: list, theta: float, length: float) -> np.ndarray:
        """from geo center to gra center

        Args:
            geo_center (list): object geometric center
            theta (float): object heading yaw
            length (float): object length

        Returns:
            np.ndarray: object gravity center
        """
        geo2gra_dist = self.graToGeoDist(length)
        gra_center = [geo_center[0] + geo2gra_dist * np.cos(theta),
                      geo_center[1] + geo2gra_dist * np.sin(theta)]
        return np.array(gra_center)
    
    def graCenterToGeoCenter(self, gra_center: list, theta: float, length: float) -> np.ndarray:
        """from gra center to geo center

        Args:
            gra_center (list): object gravity center
            theta (float): object heading yaw
            length (float): object length

        Returns:
            np.ndarray: object geometric center
        """
        geo2gra_dist = self.graToGeoDist(length)
        geo_center = [gra_center[0] - geo2gra_dist * np.cos(theta),
                      gra_center[1] - geo2gra_dist * np.sin(theta)]
        return np.array(geo_center)
    
    def graToGeoDist(self, length: float) -> float:
        """get gra center to geo center distance

        Args:
            length (float): object length

        Returns:
            float: gra center to geo center distance
        """
        return length * self.w_r * (0.5 - self.lf_r)

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """

        rotation = Quaternion(axis=(0, 0, 1), radians=state[-2, 0]).q
        geo_center = self.graCenterToGeoCenter(gra_center=[state[0, 0], state[1, 0]],
                                               theta=state[-2, 0],
                                               length=state[4, 0])
        list_state = geo_center.tolist() + state.T.tolist()[0][2:8] + rotation.tolist()
        return np.array(list_state)
    
    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res
    
    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, v, a, ry, sigma]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state

    
        
    
    
        
    