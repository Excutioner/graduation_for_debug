# Author: wangxy
# Emial: 1393196999@qq.com
import os

import numpy as np
from tracking.detection import Detection_3D_Fusion, Detection_3D_only, Detection_2D
from tracking.tracker import Tracker
from utils.kitti_oxts import load_oxts, load_poses_matrix
from utils.cmcs import load_cmcs

class DeepFusionMOT():
    def __init__(self, cfg, category):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.min_frames = cfg[category].min_frames
        self.tracker = Tracker(cfg, category)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0
        self.use_cmc = cfg.if_cmc

    def get_calib_p2(self, calib_file):
        """
        从标定文件中读取 P2 矩阵 (3x4)
        """
        if not os.path.exists(calib_file):
            return None
            
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('P2:'):
                    # 提取数字 P2: a b c ...
                    data = line.split()[1:]
                    P2 = np.array([float(x) for x in data]).reshape(3, 4)
                    return P2
        return None

    def update(self, dets_3d_fusion, dets_2d_high, dets_2d_low, dets_3d_only, cfg, frame, seq_id):
        # 1. 初始化依赖变量
        calib_file = None
        oxts_file = None
        cmc_file = None
        imu_poses = None
        cmc_transforms = None
        calib_p2 = None

        # ==============================================================================
        # 修改点 1: 只有在非 nuScenes 模式下，才去读取 KITTI 专属的 txt 标定和位姿文件
        # ==============================================================================
        if cfg.dataset != 'nuscenes':
            calib_file = os.path.join(cfg.dataset_path, cfg.spilt, 'calib' + "/" + str(seq_id).zfill(4) + '.txt')
            oxts_file = os.path.join(cfg.dataset_path, cfg.spilt, 'oxts' + "/" + str(seq_id).zfill(4) + '.txt')
            cmc_file = os.path.join(cfg.dataset_path, cfg.spilt, 'cmc_folder', cfg.ex_cfg, str(seq_id).zfill(4) + '.txt')
            
            if os.path.exists(oxts_file):
                imu_poses = load_oxts(oxts_file)
            if os.path.exists(cmc_file):
                cmc_transforms = load_cmcs(cmc_file)
            calib_p2 = self.get_calib_p2(calib_file)
        
        # 2. 格式化检测输入
        dets_3d_fusion_camera = np.array(dets_3d_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(dets_3d_fusion['dets_3d_fusion_info'])
        dets_3d_only_camera = np.array(dets_3d_only['dets_3d_only'])
        dets_3d_only_info = np.array(dets_3d_only['dets_3d_only_info'])

        # -------------- [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h] ---------------
        if len(dets_3d_fusion_camera) == 0:
            dets_3d_fusion_camera = dets_3d_fusion_camera
        else:
            dets_3d_fusion_camera = dets_3d_fusion_camera[:, self.reorder]
            
            if dets_3d_fusion_camera.shape[1] >= 9:
                dets_3d_fusion_camera = np.concatenate((dets_3d_fusion_camera, dets_3d_fusion_camera[:, 7:9]), axis=1)
            else:
                dets_3d_fusion_camera = dets_3d_fusion_camera
                
        if len(dets_3d_only_camera) == 0:
            dets_3d_only_camera = dets_3d_only_camera
        else:
            dets_3d_only_camera = dets_3d_only_camera[:, self.reorder]
            if dets_3d_only_camera.shape[1] >= 9:
                dets_3d_only_camera = np.concatenate((dets_3d_only_camera, dets_3d_only_camera[:, 7:9]), axis=1)
            else:
                dets_3d_only_camera = dets_3d_only_camera

        dets_3d_fusion_camera = [Detection_3D_Fusion(det_fusion, dets_3d_fusion_info[i]) for i, det_fusion in enumerate(dets_3d_fusion_camera)]
        dets_3d_only_camera = [Detection_3D_only(det_only, dets_3d_only_info[i]) for i, det_only in enumerate(dets_3d_only_camera)]
        dets_2d_high_obj = [Detection_2D(det_2d) for i, det_2d in enumerate(dets_2d_high)]
        dets_2d_low_obj = [Detection_2D(det_2d) for i, det_2d in enumerate(dets_2d_low)]

        # 执行预测
        self.tracker.predict_2d()
        self.tracker.predict_3d()

        # ==============================================================================
        # 修改点 2: 彻底屏蔽 nuScenes 的自车运动补偿 (CMC)
        # ==============================================================================
        if cfg.dataset != 'nuscenes':
            # --------------------3D Ego Motion Compensation (KITTI Only) --------------
            if (frame > 0) and (calib_file is not None) and (imu_poses is not None):
                self.tracker.ego_motion_compensation_3d(frame, calib_file, imu_poses)
                
            # --------------------2D Ego Motion Compensation-LETNET (KITTI Only) -------
            if (self.use_cmc == "True") and (frame > 0) and (cmc_transforms is not None):
                self.tracker.ego_motion_compensation_2d(frame, cmc_transforms)
            
        # ------------------------- Track update ------------------------
        # 注意: 这里对于 nuScenes 传入的 calib_p2 为 None，我们在融合层不再依赖它，
        # 因为我们已经在 main.py 中投影出了 dets_3dto2d_image
        self.tracker.update(dets_3d_fusion_camera, dets_3d_only_camera, dets_2d_high_obj, dets_2d_low_obj, calib_p2=calib_p2)
        
        # --------------------------- Outputs ----------------------------
        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            # --------------- Only outputs trajectory with confirmed status --------------
            if track.is_confirmed() or self.frame_count <= self.min_frames:
                bbox = np.array(track.pose[self.reorder_back])# bbox[h,w,l,x,y,z,rot_y]
                outputs.append(np.concatenate(([track.track_id_3d], bbox.flatten(), track.additional_info)).reshape(1, -1))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs