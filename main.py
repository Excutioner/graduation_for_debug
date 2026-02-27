import argparse
import os, tqdm
import shutil
import time
from os.path import join

# 限制底层库的并行线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0) # 禁止 OpenCV 多线程
cv2.ocl.setUseOpenCL(False) # 禁止 OpenCL
import numpy as np

# 在 import cv2 附近添加
from nuscenes.nuscenes import NuScenes

from datasets.coordinate_transformation import convert_x1y1x2y2c_to_tlwhc
from datasets.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from tracking.DeepFusionMOT import DeepFusionMOT
from utils.config import Config
from evaluation.KITTI.evaluation_HOTA.scripts.run_kitti import eval_kitti
from utils.combine_trk_cat import combine_category_result
from datasets.data_fusion import data_fusion
from utils.save_results import save_results

# 以下为debug专用
# import debugpy
# # 保证host与container的端口一致，listen可以只设置端口，则为localhost，否则设置成（host，port）
# debugpy.listen(12345)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("Debugger attached")

def tracking(cfg):
    spilt = cfg.spilt
    seq_list = cfg.tracking_seqs
    total_time, total_frames = 0, 0
    
    # --- [新增] 初始化 nuScenes (如果是 nuScenes 数据集) ---
    nusc = None
    if cfg.dataset == 'nuscenes':
        # 假设 cfg.dataset_path 指向 nuScenes 的 root (如 /datav/.../nuscenes/test)
        # version 根据你的实际情况填写，通常是 'v1.0-test' 或 'v1.0-trainval'
        nusc = NuScenes(version='v1.0-test', dataroot=cfg.dataset_path, verbose=False)
        print("NuScenes initialized for image loading.")
    # -----------------------------------------------------

    for category in cfg.cat_list:
        junk_conf = cfg[category]["fga"]["junk_conf"]
        high_conf = cfg[category]["fga"]["high_conf"]
        loc_thresh = cfg[category]["fga"]["loc_thresh"] # LGTrack 新增参数
        for seq_id in tqdm.trange(len(seq_list)):
        # for seq_id in tqdm.trange(seq_list):
            # ----------------------------- Initialize tracker -------------------------
            tracker = DeepFusionMOT(cfg, category)
            seq_name = str(seq_id).zfill(4)
            dets_path_3d = os.path.join(cfg.dets_path_3d, cfg.detector_3d, spilt, category) + "/" + str(seq_id).zfill(4) + '.txt'
            dets_path_2d = os.path.join(cfg.dets_path_2d, cfg.detector_2d, spilt, category) + "/" + str(seq_id).zfill(4) + '.txt'
            
            if cfg.dataset == 'nuscenes':
                # 1. 找到对应的 Scene
                # 注意：假设你的 seq_id 是 614，对应的 scene name 是 'scene-0614'
                # 如果你的 seq_list 里面已经是 [614, ...]，则需要拼接 'scene-'
                target_scene_name = f"scene-{str(seq_id).zfill(4)}"
                
                # 在 nusc.scene 中查找
                try:
                    scene = next(s for s in nusc.scene if s['name'] == target_scene_name)
                except StopIteration:
                    print(f"[Error] Scene {target_scene_name} not found in nuScenes!")
                    continue

                # 2. 遍历该 Scene 的所有 Sample 获取 CAM_FRONT 的路径
                image_filenames = []
                current_token = scene['first_sample_token']
                while current_token:
                    sample = nusc.get('sample', current_token)
                    cam_token = sample['data']['CAM_FRONT'] # 默认使用前视相机
                    cam_data = nusc.get('sample_data', cam_token)
                    
                    # 获取绝对路径
                    img_path = os.path.join(nusc.dataroot, cam_data['filename'])
                    image_filenames.append(img_path)
                    
                    current_token = sample['next']
            else:
                # [原有逻辑] KITTI 格式
                image_02_path = os.path.join(cfg.dataset_path, spilt, 'image_02') + "/" + str(seq_id).zfill(4)
                if os.path.exists(image_02_path):
                    filenames = os.listdir(image_02_path)
                    sorted_filenames = sorted(filenames)
                    image_filenames = [join(image_02_path, x) for x in sorted_filenames]
                else:
                    image_filenames = []
                    print(f"[Warning] Image path not found: {image_02_path}")
            
            # print(image_filenames)
            dets_3d = np.loadtxt(dets_path_3d, delimiter=',')  # load 3D detections, N x 15
            dets_2d = np.loadtxt(dets_path_2d, delimiter=',')

            #----------------- Remove 3D detections of low confidence -------------------
            # det_scores = seq_dets_3d[:, 6]
            # mask = det_scores > cfg.input_score
            # seq_dets_3d = seq_dets_3d[mask]

            #----------------- Remove 2D detections of low confidence -------------------
            # if dets_2d.any():
            #     det_scores_2d = dets_2d[:, 5]
            #     mask_2d = det_scores_2d > 0.4
            #     dets_2d = dets_2d[mask_2d]


            min_frame, max_frame = 0, len(image_filenames)
            for frame in tqdm.trange(max_frame):
                img0_path = image_filenames[frame]
                img_0 = cv2.imread(img0_path)
                dets_3d_camera = dets_3d[dets_3d[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)

                ori_array = dets_3d[dets_3d[:, 0] == frame, -1].reshape((-1, 1)) # alpha
                other_array = dets_3d[dets_3d[:, 0] == frame, 1:7] # 3D检测器中的 type + 2D BBOX + score
                additional_info = np.concatenate((ori_array, other_array), axis=1)
                dets_3dto2d_image = dets_3d[dets_3d[:, 0] == frame, 2:6] # 3D检测器中的 2D BBOX

                frame_mask = (dets_2d[:, 0] == frame) & (dets_2d[:, 5] > junk_conf)
                current_dets = dets_2d[frame_mask]
                dets_high = np.empty((0, 7))
                dets_low_valid = np.empty((0, 7))
                if len(current_dets) > 0:
                    scores_final = current_dets[:, 5]
                    # 集合 1: 高分检测
                    mask_high = scores_final > high_conf
                    dets_high = current_dets[mask_high]
                    # 集合 2: 低分但定位准 (仅在 LGTrack 模式下真正有用，但先计算出来)
                    if  cfg["use_fga"] == "True":
                        # 确保输入数据有第6列(loc_score)，否则回退
                        if current_dets.shape[1] > 6:
                            scores_loc = current_dets[:, 6]
                            mask_low_score = (scores_final < high_conf) & (scores_final > junk_conf)
                            mask_loc_valid = scores_loc > loc_thresh
                            dets_low_valid = current_dets[mask_low_score & mask_loc_valid]
                        else:
                            print("[Warning] Dets_2d missing Loc_Score column! Cannot use LGTrack logic.")
                if cfg["use_fga"] == "True":
                    # 融合模式：高分 + LGTrack挽救的低分
                    if len(dets_high) > 0 and len(dets_low_valid) > 0:
                        dets_2d_combined = np.concatenate((dets_high, dets_low_valid), axis=0)
                    elif len(dets_high) > 0:
                        dets_2d_combined = dets_high
                    elif len(dets_low_valid) > 0:
                        dets_2d_combined = dets_low_valid
                    else:
                        dets_2d_combined = np.empty((0, 7))
                else:
                    # 传统模式：仅大于high_conf 的检测
                    dets_2d_combined = dets_high
                    
                dets_2d_input = dets_2d_combined
                # -------------------- The fusion of 3D detections and 2D detections -------------
                dets_3d_fusion, dets_3d_only, dets_2d_only_list = \
                    data_fusion(dets_3d_camera, dets_2d_input, dets_3dto2d_image, additional_info)

                dets_2d_high_tlwhc = []
                dets_2d_low_tlwhc = []

                if len(dets_2d_only_list) > 0:
                    dets_2d_only_array = np.array(dets_2d_only_list) # 转回 numpy 方便操作
                    
                    # 再次利用 high_conf 进行拆分
                    # 注意：dets_2d_only_array 每一行依然是 [frame, x1, y1, x2, y2, score, loc_score]
                    scores = dets_2d_only_array[:, 5]
                    
                    # 拆分
                    high_mask = scores >= high_conf
                    # 低分框自然是那些分数低但依然存在于列表中的(说明它是合法的low_valid)
                    low_mask = ~high_mask 
                    
                    raw_high = dets_2d_only_array[high_mask]
                    raw_low = dets_2d_only_array[low_mask]

                    # 转换为 TLWH 格式供 Tracker 使用
                    if len(raw_high) > 0:
                        dets_2d_high_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_high])
                    if len(raw_low) > 0:
                        dets_2d_low_tlwhc = np.array([convert_x1y1x2y2c_to_tlwhc(row[1:6]) for row in raw_low])
                
                # 转换为 numpy array 防止报错
                if len(dets_2d_high_tlwhc) == 0: dets_2d_high_tlwhc = np.empty((0, 5))
                if len(dets_2d_low_tlwhc) == 0: dets_2d_low_tlwhc = np.empty((0, 5))


                start_time = time.time()
                # 传统模式下dets_2d_low_tlwhc为空
                trackers = tracker.update(dets_3d_fusion,
                                          dets_2d_high_tlwhc,  # Stage 3.1 主力匹配
                                          dets_2d_low_tlwhc,   # Stage 3.2 挽救匹配 (LGTrack)
                                          dets_3d_only,
                                          cfg,
                                          frame,
                                          seq_id)
                # trackers为3D轨迹时，
# [track.track_id_3d](3D轨迹为偶数，2D轨迹为奇数), bbox([h,w,l,x,y,z,rot_y]), track.additional_info(alpha + 3D检测器中的 type + 2D BBOX + score)
                cycle_time = time.time() - start_time
                total_time += cycle_time
                total_frames += 1
                save_results(trackers, cfg, seq_name, frame, category, img_0)

    print('--------------The total time is {}s --------------'.format(total_time))
    print('--------------FPS = {} --------------'.format(total_frames / total_time))


if __name__ == '__main__':
    # file_path = 'results'
    # try:
    #     shutil.rmtree(file_path)
    # except OSError as e:
    #     print("Error: %s - %s." % (e.filename, e.strerror))

    parser = argparse.ArgumentParser(description='DeepFusionMOT')
    parser.add_argument('--cfg', type=str, default='./config/kitti.yaml', help='data')
    args = parser.parse_args()
    cfg, _ = Config(args.cfg)

    tracking(cfg)
    combine_category_result(cfg)

    # print("--------------Starting Evaluation-------------")
    # results = eval_kitti(cfg)