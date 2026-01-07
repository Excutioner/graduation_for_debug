import os, cv2
import random
import numpy as np

from utils.file_operation.file import mkdir_if_inexistence

def compute_color_for_trk(trk_id):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (trk_id ** 2 - trk_id + 1)) % 255) for p in palette]
    return tuple(color)

def save_vis_results(dets_2d_frame, cfg, seq_name, frame, category, image, ry_data=None, line_thickness = 3):
    save_image_dir = os.path.join(cfg.vis_save_path, category, "image", seq_name); mkdir_if_inexistence(save_image_dir)
    if len(dets_2d_frame) > 0:
        for idx, d in enumerate(dets_2d_frame):
            bbox2d = d
            img_id = str(frame).zfill(6)
            assert image.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
            tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
            color = (0, 255, 0)  # 绿色框
            c1, c2 = (int(bbox2d[0]), int(bbox2d[1])), (int(bbox2d[2]), int(bbox2d[3]))
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            
            # 如果提供了ry数据，则在框上方显示
            if ry_data is not None and len(ry_data) > idx:
                ry = ry_data[idx]
                label = f'{ry:.2f}'
                # 减小字体大小
                font_scale = tl / 6  # 更小的字体比例
                tf = 2
                # 绘制标签文字
                cv2.putText(image, str(label), (c1[0], c1[1] - 10), 0, font_scale, [255, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

            cv2.imwrite(save_image_dir + "/" + "{}.png".format(img_id), image)

def save_gt_results(gts_2d_frame, cfg, seq_name, frame, category, image, line_thickness = 3):
    save_image_dir = os.path.join(cfg.gt_vis_save_path, category, "image", seq_name); mkdir_if_inexistence(save_image_dir)
    img_id = str(frame).zfill(6)
    assert image.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    if len(gts_2d_frame) > 0:
        for t in gts_2d_frame:
            bbox2d = t
            id_tmp = int(bbox2d[0])
            label = f'{id_tmp} {"car"}'

            color = compute_color_for_trk(int(bbox2d[0]))
            c1, c2 = (int(bbox2d[1]), int(bbox2d[2])), (int(bbox2d[3]), int(bbox2d[4]))
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            # label
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, str(label), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(save_image_dir + "/" + "{}.png".format(img_id), image)