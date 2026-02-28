# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from tracking.cost_function import (
    iou_2d, giou_2d, sdiou_2d, diou_2d, giou_3d, dist_3d, iou_2d_c, 
    ro_gdiou_3d, dist_bev_nusc, ro_gdiou_3d_nusc
)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def kitti_cost(dets, trks, iou_threshold, iou_matrix, cost_func, cost_params=None, dist_aware_cfg=None):
    if cost_params is None: cost_params = {}
    matched_indices, _ = cost_calculate(dets, trks, iou_matrix, iou_threshold, cost_func, cost_params)
    if min(iou_matrix.shape) > 0:
        # 默认掩码 (硬阈值)
        valid_mask = (iou_matrix > iou_threshold).astype(np.int32)
        
        # 创新点: 距离感知的各向异性匹配 (Distance-Aware Matching)
        if dist_aware_cfg and dist_aware_cfg.get('use_dist_aware', False):
            params = dist_aware_cfg.get('dist_aware_params', {})
            far_dist = params.get('far_dist_thresh', 40.0)
            far_iou = params.get('far_iou_thresh', 0.25) # 远距离放宽要求
            
            for t, trk in enumerate(trks):
                # 计算距离 (兼容 KITTI 和 nuScenes)
                try:
                    if 'nusc' in cost_func:
                        dist = np.sqrt(trk.pose[0]**2 + trk.pose[1]**2) # nuScenes (X-Y)
                    else:
                        dist = np.sqrt(trk.pose[0]**2 + trk.pose[2]**2) # KITTI (X-Z)
                except:
                    continue
                
                # 如果是远距离目标，使用更宽松的阈值
                if dist > far_dist:
                    valid_mask[:, t] = (iou_matrix[:, t] > far_iou).astype(np.int32)

        # 2. 执行匹配 (Linear Assignment)
        masked_iou_matrix = iou_matrix.copy()
        masked_iou_matrix[valid_mask == 0] = -1000.0 # 使用一个足够小的负数，防止阈值设为负数时被误匹配
        
        matched_indices = linear_assignment(-masked_iou_matrix)
        
        # 3. 二次校验 (Double Check)
        final_matches = []
        for m in matched_indices:
            d_idx, t_idx = m[0], m[1]
            
            # 获取对应的阈值
            current_thresh = iou_threshold
            if dist_aware_cfg and dist_aware_cfg.get('use_dist_aware', False):
                try:
                    trk = trks[t_idx]
                    if 'nusc' in cost_func:
                        dist = np.sqrt(trk.pose[0]**2 + trk.pose[1]**2)
                    else:
                        dist = np.sqrt(trk.pose[0]**2 + trk.pose[2]**2)
                        
                    if dist > dist_aware_cfg['dist_aware_params']['far_dist_thresh']:
                        current_thresh = dist_aware_cfg['dist_aware_params']['far_iou_thresh']
                except:
                    pass
            
            if iou_matrix[d_idx, t_idx] > current_thresh:
                final_matches.append(m)
        
        matched_indices = np.array(final_matches)
        if len(matched_indices) == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
        matched_indices = np.empty(shape=(0, 2))
        
    return matched_indices, iou_matrix


def cost_calculate(dets, trks, iou_matrix, iou_threshold, cost_func, cost_params=None):
    if cost_params is None: cost_params = {}
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            if cost_func == 'iou_2d':
                iou_matrix[d, t] = iou_2d(det, trk)  # det: 8 x 3, trk: 8 x 3
            elif cost_func == 'iou_2d_c':
                iou_matrix[d, t] = iou_2d_c(det, trk, **cost_params)
            elif cost_func == 'giou_2d':
                iou_matrix[d, t] = giou_2d(det, trk)
            elif cost_func == 'sdiou_2d':
                iou_matrix[d, t] = sdiou_2d(det, trk)
            elif cost_func == 'diou_2d':
                iou_matrix[d, t] = diou_2d(det, trk)
            elif cost_func == 'giou_3d' or cost_func == 'iou_3d':
                iou_matrix[d, t] = giou_3d(det, trk, cost_func)
            elif cost_func == 'dist_3d':
                iou_matrix[d, t] = dist_3d(det, trk)
            elif cost_func == 'ro_gdiou_3d':
                # 默认权重 w1=1, w2=1，对应论文中两个框相距很远时趋向于 -2
                iou_matrix[d, t] = ro_gdiou_3d(det, trk, **cost_params)
            # ========================================================
            # [新增] 路由到 nuScenes 专用的 BEV 距离度量
            # ========================================================
            elif cost_func == 'dist_bev_nusc':
                iou_matrix[d, t] = dist_bev_nusc(det, trk, **cost_params)
            elif cost_func == 'ro_gdiou_3d_nusc':
                iou_matrix[d, t] = ro_gdiou_3d_nusc(det, trk, **cost_params)
                
    return [], iou_matrix


def associate_dets_to_trks_fusion(dets, trks, cost_func, cost_threshold, metric='match_3d', cost_params=None, dist_aware_cfg=None):
    if cost_params is None:
        cost_params = {}
    if (len(trks) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), []
    if (len(dets) == 0):
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks))
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if metric == 'match_3d':
        matched_indices, _ = kitti_cost(dets, trks, cost_threshold, iou_matrix, cost_func, cost_params, dist_aware_cfg)
    elif metric == 'match_2d':
        if cost_func == 'iou_2d_c':
            # 如果使用带置信度的IoU，需要提取 x1,y1,x2,y2,conf
            # Detection_2D 需要实现 to_x1y1x2y2c() 或类似方法
            # Track_2D 需要实现 to_x1y1x2y2c()
            dets_array = np.array([d.to_x1y1x2y2c() for d in dets]) 
            trks_array = np.array([t.to_x1y1x2y2c() for t in trks])
            matched_indices, _ = kitti_cost(dets_array, trks_array, cost_threshold, iou_matrix, cost_func, cost_params)        
        else:
            # 传统逻辑，只取坐标
            dets_array = np.array([d.to_x1y1x2y2() for d in dets])
            trks_array = np.array([t.to_x1y1x2y2() for t in trks])
            matched_indices, _ = kitti_cost(dets_array, trks_array, cost_threshold, iou_matrix, cost_func, cost_params)
    return is_matched(dets, trks, matched_indices, iou_matrix, cost_threshold)


def trackfusion2Dand3D(trks_2d, trks_3Dto2D_image, iou_threshold):
    trk_indices = list(range(len(trks_2d)))  # 跟踪对象索引
    det_indices = list(range(len(trks_3Dto2D_image)))  # 检测对象索引
    matches = []
    if len(trk_indices) == 0 or len(det_indices) == 0:
        return [], trk_indices, det_indices  # Nothing to match.

    iou_matrix = np.zeros((len(trks_2d), len(trks_3Dto2D_image)), dtype=np.float32)
    for t, trk in enumerate(trks_2d):
        for d, det in enumerate(trks_3Dto2D_image):
            iou_matrix[t, d] = iou_2d(trk.to_x1y1x2y2(), det)  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_dets = []
    for d, det in enumerate(trks_3Dto2D_image):
        if d not in matched_indices[:, 1]:
            unmatched_dets.append(d)

    unmatched_trks_2d = []
    for t, trk in enumerate(trks_2d):
        if t not in matched_indices[:, 0]:
            unmatched_trks_2d.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[1])
            unmatched_trks_2d.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trks_2d), np.array(unmatched_dets)


def associate_2D_to_3D_tracking(trks_2d, trks_3d, iou_threshold):
    trks_3Dto2D_image = [list(i.additional_info[2:6]) for i in trks_3d]
    matched_trks_2d, unmatch_trks_2d, _ = trackfusion2Dand3D(trks_2d, trks_3Dto2D_image, iou_threshold)
    return matched_trks_2d, unmatch_trks_2d


def is_matched(dets, trks, matched_indices, iou_matrix, iou_threshold):
    matches, unmatched_dets, unmatched_trks = [], [], []
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    for t, trk in enumerate(trks):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
