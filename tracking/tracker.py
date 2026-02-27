# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np
from tracking.cost_function import iou_2d, sdiou_2d 

from tracking.matching import associate_dets_to_trks_fusion, associate_2D_to_3D_tracking, linear_assignment
from tracking.track_2d import Track_2D
from tracking.track_3d import Track_3D

DELETE_2D = True

class Tracker():
    def __init__(self, cfg, category):
        self.cfg = cfg
        self.cost_3d = self.cfg[category].metric_3d
        self.cost_2d = self.cfg[category].metric_2d
        self.threshold_3d = self.cfg[category]["cost_function"][self.cost_3d]
        self.threshold_2d = self.cfg[category]["cost_function"][self.cost_2d]
        self.max_age = self.cfg[category].max_ages
        self.min_frames = self.cfg[category].min_frames
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0   # The id of 3D track is represented by an even number.
        self.track_id_2d = 1   # The id of 3D track is represented by an odd number.
        self.unmatch_tracks_3d = []
        self.kfstate_2d = cfg.kfstate_2d
        self.motion_model = cfg.motion_model
        self.cmc_with_rect2imu = cfg.cmc_with_rect2imu
        self.use_fga = cfg["use_fga"]
        cost_opts = self.cfg[category].get('cost_options', {})
        self.ro_gdiou_params = cost_opts.get('ro_gdiou_3d', {}) 
        self.iou_2d_c_params = cost_opts.get('iou_2d_c', {}) 
        # [MCTrack Params]
        self.use_rv_match = cfg[category].get('use_rv_match', "False")
        self.rv_metric = cfg[category].get('rv_metric', 'sdiou_2d')
        self.rv_threshold = cfg[category].get('rv_threshold', 0.5)
        
        # [微创新配置读取]
        self.micro_cfg = self.cfg[category].get('micro_innovation', {})
        self.cg_akf_cfg = {
            'use_cg_akf': self.micro_cfg.get('use_cg_akf', False),
            # 原参数 {'alpha': 5.0} 修改为论文推荐的 mu=1.0, tau=1.0
            'cg_akf_params': self.micro_cfg.get('cg_akf_params', {'mu': 1.0, 'tau': 1.0})
        }
        self.dist_aware_cfg = {
            'use_dist_aware': self.micro_cfg.get('use_dist_aware', False),
            'dist_aware_params': self.micro_cfg.get('dist_aware_params', {'far_dist_thresh': 45.0, 'far_iou_thresh': 0.25})
        }
        
        # [新增] 2D CG-AKF 独立配置
        self.cg_akf_2d_cfg = {
            'use_cg_akf_2d': self.micro_cfg.get('use_cg_akf_2d', False),
            # 2D 部分参数可视情况调整，这里保持结构一致
            'cg_akf_2d_params': self.micro_cfg.get('cg_akf_2d_params', {'mu': 1.0, 'tau': 1.0})
        }
        
        # 3. [新增] APN-CTRA (自适应过程噪声)
        self.apn_ctra_cfg = {
            'use_apn_ctra': self.micro_cfg.get('use_apn_ctra', False),
            'apn_params': self.micro_cfg.get('apn_ctra_params', {'maneuver_factor_omega': 2.0, 'maneuver_factor_accel': 0.5})
        }
        # 4. [新增] AW-Ro-GDIoU (各向异性代价)
        # 逻辑：如果开关开启，则修改 self.ro_gdiou_params 中的 depth_weight
        # 如果开关关闭，强制设为 1.0 (各向同性)
        use_aw = self.micro_cfg.get('use_aw_ro_gdiou', False)
        aw_params = self.micro_cfg.get('aw_ro_gdiou_params', {'depth_weight': 0.3})
        
        if use_aw:
            # 覆盖/添加 depth_weight
            self.ro_gdiou_params['depth_weight'] = aw_params.get('depth_weight', 0.3)
        else:
            # 强制为默认值
            self.ro_gdiou_params['depth_weight'] = 1.0
            
        if self.kfstate_2d == "ltrb":
            from tracking import kalman_filter_2d_ltrb
            self.kf_2d = kalman_filter_2d_ltrb.KalmanFilter()
        elif self.kfstate_2d == "ltrbc":
            from tracking import kalman_filter_2d_ltrbc
            self.kf_2d = kalman_filter_2d_ltrbc.KalmanFilter()
        elif self.kfstate_2d == "xyah":
            from tracking import kalman_filter_2d
            self.kf_2d = kalman_filter_2d.KalmanFilter()
        else :
            raise ValueError("kfstate_2d must be ltrb、ltrbc or xyah")
        
        if self.motion_model == "CTRA":
            from tracking.extend_kalman_fileter_3d_ctra import KalmanBoxTracker
        elif self.motion_model == "CTRV":
            from tracking.extend_kalman_fileter_3d_ctrv import KalmanBoxTracker
        elif self.motion_model == "CV":
            from tracking.kalman_fileter_3d import  KalmanBoxTracker
        elif self.motion_model == "CA":  # [新增] CA 模型支持
            from tracking.kalman_filter_3d_ca import KalmanBoxTracker
        else :
            raise ValueError("motion_model must be CTRA、CA、CTRV or CV")
        self.KalmanBoxTracker_Class = KalmanBoxTracker
    def project_track_to_2d(self, track, calib_p2):
        """
        利用 3D 预测位置计算当前的 2D 框 (x1, y1, x2, y2)
        """
        # 1. 获取 3D 状态 [x, y, z, rot_y, l, w, h]
        # 注意：DeepFusionMOT 中 pose 的顺序通常被重排为 [x, y, z, rot_y, l, w, h]
        pose = track.pose
        x, y, z = pose[0], pose[1], pose[2]
        ry = pose[3]
        l, w, h = pose[4], pose[5], pose[6]

        # 2. 构建 3D Bounding Box 的 8 个角点 (Camera Coordinate: x-right, y-down, z-forward)
        # 假设 (x,y,z) 是底部中心 (KITTI 标准)
        c = np.cos(ry)
        s = np.sin(ry)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # 3D 框的 8 个角点 (相对于中心)
        # x: +/- l/2, y: 0 to -h, z: +/- w/2
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        
        # 旋转并平移
        corners_3d = np.dot(R, corners_3d)
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z

        # 3. 投影到 2D 图像平面
        # 扩展为齐次坐标 (4, 8)
        corners_3d_hom = np.vstack((corners_3d, np.ones((1, 8))))
        
        # 应用投影矩阵 P2 (3, 4)
        corners_2d = np.dot(calib_p2, corners_3d_hom)
        
        # 归一化 (x/z, y/z)
        # 防止除以0
        epsilon = 1e-5
        corners_2d[2, :] = np.maximum(corners_2d[2, :], epsilon) 
        
        corners_2d[0, :] /= corners_2d[2, :]
        corners_2d[1, :] /= corners_2d[2, :]
        
        # 4. 获取 2D 包围盒 (Min-Max)
        min_x = np.min(corners_2d[0, :])
        min_y = np.min(corners_2d[1, :])
        max_x = np.max(corners_2d[0, :])
        max_y = np.max(corners_2d[1, :])
        
        # 边界保护 (简单的非负约束)
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        
        return np.array([min_x, min_y, max_x, max_y])
    def predict_3d(self):
        # tracks_3d的定义：self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info))
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d, apn_cfg=self.apn_ctra_cfg)

    def predict_2d(self):
        for track in self.tracks_2d:
            if self.kfstate_2d == 'ltrb':
                track.ltbr_predict_2d(self.kf_2d)
            elif self.kfstate_2d == 'ltrbc':
                track.ltbrc_predict_2d(self.kf_2d)
            elif self.kfstate_2d == 'xyah':
                track.predict_2d(self.kf_2d)
            else :
                raise ValueError("kfstate_2d must be ltrb、ltrbc or xyah")

    def ego_motion_compensation_3d(self, frame, calib_file, oxts):
        for track in self.tracks_3d:
            track.ego_motion_compensation_3d(frame, calib_file, oxts, self.motion_model, self.cmc_with_rect2imu)
    def ego_motion_compensation_2d(self, frame, cmc_transforms):
        for track in self.tracks_2d:
            track.ego_motion_compensation_2d(frame, cmc_transforms)
        for track in self.tracks_3d:
            if not track.compensated_2d:
                track.ego_motion_compensation_2d(frame, cmc_transforms)
                track.compensated_2d = True

    def ego_motion_compensation_2d_imu(self, frame, calib_file, oxts):
        for track in self.tracks_2d:
            track.ego_motion_compensation_2d_imu(frame, calib_file, oxts)
    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_high, dets_2d_low, calib_p2=None):
        # =========================================================
        # 1st Level: Fusion Match (高质量3D + 2D)
        # =========================================================
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d', cost_params=self.ro_gdiou_params)
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx], cg_akf_cfg=self.cg_akf_cfg)            
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        # =========================================================
        # 2nd Level: 3D Only Match (含 MCTrack RV 逻辑)
        # =========================================================
        # 找出一阶段未匹配的3D轨迹
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        # --- Stage 1: 原始 BEV/3D 匹配 ---
        matched_only_idx, unmatched_dets_only_idx, unmatched_trks_only_idx_local = associate_dets_to_trks_fusion(
            dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d', 
            cost_params=self.ro_gdiou_params,
            dist_aware_cfg=self.dist_aware_cfg)
        # --- Stage 2: MCTrack RV (2D) 补救匹配 ---
        # 只有在开关开启、P2存在、且有残余匹配项时才执行
        if (self.use_rv_match == "True") and (calib_p2 is not None) and \
           (len(unmatched_trks_only_idx_local) > 0) and (len(unmatched_dets_only_idx) > 0):

            # 1. 准备 Stage 1 剩下的 Candidates
            candidate_trks = [self.unmatch_tracks_3d1[i] for i in unmatched_trks_only_idx_local]
            candidate_dets = [dets_3d_only[i] for i in unmatched_dets_only_idx]

            # 2. 构建 Cost Matrix (RV 空间)
            num_dets = len(candidate_dets)
            num_trks = len(candidate_trks)
            cost_matrix_rv = np.zeros((num_dets, num_trks), dtype=np.float32)

            # 预计算 Tracks 的 2D 投影 (减少循环内计算)
            trks_2d_boxes = [self.project_track_to_2d(t, calib_p2) for t in candidate_trks]
            
            # Dets 的 2D 框直接获取 (无需投影)
            # 根据 main.py，additional_info 索引 2:6 是 [x1, y1, x2, y2]
            dets_2d_boxes = [d.additional_info[2:6] for d in candidate_dets]

            # 计算矩阵
            for d in range(num_dets):
                for t in range(num_trks):
                    if self.rv_metric == 'sdiou_2d':
                        score = sdiou_2d(dets_2d_boxes[d], trks_2d_boxes[t])
                    else:
                        score = iou_2d(dets_2d_boxes[d], trks_2d_boxes[t])
                    cost_matrix_rv[d, t] = score

            # 3. 匈牙利匹配 (注意取反，因为 linear_assignment 求最小代价)
            matched_indices_rv = linear_assignment(-cost_matrix_rv)

            # 4. 整合匹配结果
            rv_matched_det_local_indices = [] # 记录在 RV 阶段被匹配掉的局部索引

            for m in matched_indices_rv:
                d_idx, t_idx = m[0], m[1]
                score = cost_matrix_rv[d_idx, t_idx]
                
                if score >= self.rv_threshold:
                    # 映射回原始索引
                    # Detection 原始索引
                    original_det_idx = unmatched_dets_only_idx[d_idx]
                    # Track 原始索引 (在 self.unmatch_tracks_3d1 中的索引)
                    original_trk_idx = unmatched_trks_only_idx_local[t_idx]
                    
                    # 添加到总匹配列表
                    matched_only_idx = np.vstack((matched_only_idx, [original_det_idx, original_trk_idx]))
                    
                    # 标记该 Detection 已被 RV 阶段抢救
                    rv_matched_det_local_indices.append(original_det_idx)

            # 5. 更新 Unmatched Dets 列表 (剔除 RV 阶段匹配成功的)
            # 注意：unmatched_trks 不需要显式更新，因为后续代码是通过 matched_only_idx 来更新 Track 状态的
            new_unmatched_dets = []
            for det_idx in unmatched_dets_only_idx:
                if det_idx not in rv_matched_det_local_indices:
                    new_unmatched_dets.append(det_idx)
            unmatched_dets_only_idx = np.array(new_unmatched_dets, dtype=int)
            
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(dets_3d_only[detection_idx], cg_akf_cfg=self.cg_akf_cfg)
                    index_to_delete.append(track_idx)
                    break
        # 找出二阶段未匹配的3D轨迹(未匹配)
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        for detection_idx in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        # 找出新初始化的3D轨迹(待确认)
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # =========================================================
        # 3rd Level: FGA / 2D Match 
        # =========================================================
        if self.use_fga == "True":
            # >>>>>>>>>> 模式 A: LGTrack 级联匹配 >>>>>>>>>>
            
            # 3.1: 高分匹配 (特征/IoU)
            matched_high, unmatch_trks_high_idx, unmatch_dets_high_idx = \
                associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_high, self.cost_2d, self.threshold_2d, metric='match_2d', cost_params=self.iou_2d_c_params)
            
            for track_idx, detection_idx in matched_high:
                self._update_2d_track_state(track_idx, dets_2d_high[detection_idx])

            # 3.2: 低分挽救 (仅 IoU)
            candidates_for_rescue = [self.tracks_2d[i] for i in unmatch_trks_high_idx]
            unmatch_trks_final_idx = unmatch_trks_high_idx # 默认全部未匹配

            if len(candidates_for_rescue) > 0 and len(dets_2d_low) > 0:
                # 关键：这里强制使用 iou_2d，你需要确认 associate 函数支持
                matched_low_local, _, _ = associate_dets_to_trks_fusion(
                    candidates_for_rescue, dets_2d_low, self.cost_2d, self.threshold_2d, metric='match_2d', cost_params=self.iou_2d_c_params)
                
                rescued_global_indices = []
                for local_track_idx, detection_idx in matched_low_local:
                    track_obj = candidates_for_rescue[local_track_idx]
                    global_track_idx = self.tracks_2d.index(track_obj)
                    self._update_2d_track_state(global_track_idx, dets_2d_low[detection_idx])
                    rescued_global_indices.append(global_track_idx)
                
                # 重新计算最终未匹配的轨迹
                unmatch_trks_final_idx = [idx for idx in unmatch_trks_high_idx if idx not in rescued_global_indices]

            # 处理最终未匹配轨迹
            for track_idx in unmatch_trks_final_idx:
                self.tracks_2d[track_idx].mark_missed()

            # 3.3: 仅初始化高分框
            for detection_idx in unmatch_dets_high_idx:
                self.initiate_trajectory_2d(dets_2d_high[detection_idx])

        else:
            # >>>>>>>>>> 模式 B: 原始 DeepFusionMOT 逻辑 >>>>>>>>>>
            
            # 合并两个列表 (为了稳健性，防止外面传错了非空 low list 进来)
            dets_2d_all = dets_2d_high + dets_2d_low
            
            matched, unmatch_trks, unmatch_dets = \
                associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_all, self.cost_2d, self.threshold_2d, metric='match_2d', cost_params=self.iou_2d_c_params)
            
            for track_idx, detection_idx in matched:
                self._update_2d_track_state(track_idx, dets_2d_all[detection_idx])
            
            for track_idx in unmatch_trks:
                self.tracks_2d[track_idx].mark_missed()
            
            for detection_idx in unmatch_dets:
                self.initiate_trajectory_2d(dets_2d_all[detection_idx])

        # 清理已删除轨迹
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        #  4th Level of Association
        #  未匹配轨迹应由3D转向2D(出区域)，待确认轨迹应由2D转向3D(入区域)
        #  似乎不存在未匹配轨迹3D转2D的过程，只有待确认轨迹2D转3D的过程
        #  确认最终输出(也是可视化的结果)是3D轨迹
        matched_track_2d, unmatch_tracks_2d = associate_2D_to_3D_tracking(self.tracks_2d, self.unmatch_tracks_3d, self.threshold_2d)
        if DELETE_2D:
            index_to_delete2 = []
        for track_idx_2d, track_idx_3d in matched_track_2d:
            if not DELETE_2D:
                if self.tracks_2d[track_idx_2d].time_since_update > self.cfg.remain_threshold_2d:
                    continue
            for i in range(len(self.tracks_3d)):
                # 打印触发的次数
                if self.tracks_3d[i].track_id_3d == self.unmatch_tracks_3d[track_idx_3d].track_id_3d:
                    self.tracks_3d[i].age = self.tracks_2d[track_idx_2d].age + 1
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        if DELETE_2D:
                            self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                        else :
                            self.tracks_3d[i].hits += 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            if DELETE_2D:
                index_to_delete2.append(track_idx_2d)
        if DELETE_2D:
            self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]
        else:
            self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

    def initiate_trajectory_3d(self, detection):
        # [x,y,z,rot_y，l,w,h]
        self.kf_3d = self.KalmanBoxTracker_Class(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0) # self.kf_3d.kf.x[:7] = [x,y,z,theta,l,w,h]
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info))
        self.track_id_3d += 2

    def initiate_trajectory_2d(self, detection):
        if self.kfstate_2d == 'ltrb':
            mean, covariance = self.kf_2d.initiate(detection.to_x1y1x2y2()) # in: [x1,y1,x2,y2], out: [x,y,a,h]
        elif self.kfstate_2d == 'ltrbc':
            measurement = np.append(detection.to_x1y1x2y2(), detection.get_confidence())# in: [x1,y1,x2,y2,c], out: [x,y,a,h,c]
            mean, covariance = self.kf_2d.initiate(measurement)
        elif self.kfstate_2d == 'xyah':
            mean, covariance = self.kf_2d.initiate(detection.to_xyah())
        else :
            raise ValueError("kfstate_2d must be ltrb、ltrbc or xyah")
        self.tracks_2d.append(Track_2D(mean, covariance, self.track_id_2d, self.min_frames, self.max_age))
        self.track_id_2d += 2
        
    def _update_2d_track_state(self, track_idx, detection):
        if self.kfstate_2d == 'ltrb':
            # 传入 self.cg_akf_2d_cfg
            self.tracks_2d[track_idx].ltbr_update_2d(self.kf_2d, detection, self.cg_akf_2d_cfg)
        elif self.kfstate_2d == 'ltrbc':
            self.tracks_2d[track_idx].ltbrc_update_2d(self.kf_2d, detection, self.cg_akf_2d_cfg)
        elif self.kfstate_2d == 'xyah':
            self.tracks_2d[track_idx].update_2d(self.kf_2d, detection, self.cg_akf_2d_cfg)
        else:
            raise ValueError("kfstate_2d must be ltrb、ltrbc or xyah")