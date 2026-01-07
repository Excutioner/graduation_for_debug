# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np

from tracking.matching import associate_dets_to_trks_fusion, associate_2D_to_3D_tracking
from tracking.track_2d import Track_2D
from tracking.track_3d import Track_3D


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
        self.use_rv_match = cfg[category].get('use_rv_match', "False")
        self.rv_metric = cfg[category].get('rv_metric', 'sdiou_2d')
        self.rv_threshold = cfg[category].get('rv_threshold', 0.5)
        
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
        else :
            raise ValueError("motion_model must be CTRA、CTRV or CV")
        self.KalmanBoxTracker_Class = KalmanBoxTracker

    def predict_3d(self):
        # tracks_3d的定义：self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info))
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

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
    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_high, dets_2d_low):
        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d', cost_params=self.ro_gdiou_params)
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        #  2nd Level of Association
        # 找出一阶段未匹配的3D轨迹
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
            dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d', cost_params=self.ro_gdiou_params)
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(dets_3d_only[detection_idx])
                    index_to_delete.append(track_idx)
                    break
        # 找出二阶段未匹配的3D轨迹(未匹配)
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        for detection_idx in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        # 找出新初始化的3D轨迹(待确认)
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # 3rd Level of Association
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
        # index_to_delete2 = []
        for track_idx_2d, track_idx_3d in matched_track_2d:
            if self.tracks_2d[track_idx_2d].time_since_update > self.cfg.remain_threshold_2d:
                continue
            for i in range(len(self.tracks_3d)):
                # 打印触发的次数
                # print(self.tracks_3d[i].track_id_3d)
                if self.tracks_3d[i].track_id_3d == self.unmatch_tracks_3d[track_idx_3d].track_id_3d:
                    self.tracks_3d[i].age = self.tracks_2d[track_idx_2d].age + 1
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        # self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                        self.tracks_3d[i].hits += 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            # index_to_delete2.append(track_idx_2d)
        # self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]
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
            self.tracks_2d[track_idx].ltbr_update_2d(self.kf_2d, detection)
        elif self.kfstate_2d == 'ltrbc':
            self.tracks_2d[track_idx].ltbrc_update_2d(self.kf_2d, detection)
        elif self.kfstate_2d == 'xyah':
            self.tracks_2d[track_idx].update_2d(self.kf_2d, detection)
        else:
            raise ValueError("kfstate_2d must be ltrb、ltrbc or xyah")