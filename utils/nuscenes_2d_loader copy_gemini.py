import os
import json
from nuscenes.nuscenes import NuScenes

class NuScenes2DLoader:
    def __init__(self, nusc_path, json_dir):
        """
        初始化 2D 检测框加载器
        nusc_path: nuScenes 数据集根目录
        json_dir: mmdetection_cascade_x101 文件夹中 val 或 test 的路径
        """
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_path, verbose=False)
        self.json_dir = json_dir
        
        # 预加载场景字典，方便通过 scene_token 找到对应的 json 文件
        self.scene_token_to_json = {}
        for f_name in os.listdir(json_dir):
            if f_name.endswith('.json'):
                scene_token = f_name.split('_')[0]
                self.scene_token_to_json[scene_token] = os.path.join(json_dir, f_name)

    def load_aligned_2d_boxes(self, scene_name):
        """
        根据场景名称（例如 'scene-0003'）加载并对齐该场景下所有的 2D 框
        返回格式:
        {
            'sample_token_1': {
                'CAM_FRONT': [ {'bbox': [x1,y1,x2,y2], 'score': s, 'class_label': 'car'}, ... ],
                'CAM_FRONT_LEFT': [ ... ],
                ...
            },
            'sample_token_2': { ... }
        }
        """
        # 1. 根据 scene_name 获取 scene_token
        scene_token = None
        for scene in self.nusc.scene:
            if scene['name'] == scene_name:
                scene_token = scene['token']
                break
        
        if not scene_token:
            raise ValueError(f"在 nuScenes 数据库中找不到场景: {scene_name}")

        json_path = self.scene_token_to_json.get(scene_token)
        if not json_path:
            raise FileNotFoundError(f"找不到场景 {scene_name} ({scene_token}) 对应的 JSON 检测结果文件")

        # 2. 加载 JSON 数据
        with open(json_path, 'r') as f:
            scene_data = json.load(f)

        aligned_2d_data = {}

        # 3. 解析并重组数据 (遵循 EagerMOT 的四层结构)
        # frame_token 即为我们需要对齐 CenterPoint 的 sample_token
        for sample_token, cam_dict in scene_data.items():
            aligned_2d_data[sample_token] = {}
            
            for cam_data_token, class_dict in cam_dict.items():
                # 查询 nuScenes 数据库，把 cam_data_token 转换为具体的相机名称
                try:
                    cam_info = self.nusc.get('sample_data', cam_data_token)
                    cam_channel = cam_info['channel']  # 例如 'CAM_FRONT'
                except Exception:
                    # 如果找不到，可以跳过或做异常处理
                    continue
                
                aligned_2d_data[sample_token][cam_channel] = []
                
                # 提取该相机下的所有检测框
                for class_label, det_list in class_dict.items():
                    # 在 EagerMOT 中，class_label 通常对应 nuImages 的分类，你可以在这里做过滤
                    for det in det_list:
                        xmin, ymin, xmax, ymax, score = det
                        # 你可以在这里设置置信度阈值过滤，例如 score > 0.3
                        if score > 0.3:
                            aligned_2d_data[sample_token][cam_channel].append({
                                'bbox': [xmin, ymin, xmax, ymax],
                                'score': score,
                                'class_label': class_label 
                            })
                            
        return aligned_2d_data

# ================= 测试代码 =================
if __name__ == '__main__':
    # 替换为你实际的路径
    NUSC_PATH = '/datav/DeepFusionMOT/data/nuscenes/datasets'
    JSON_DIR = '/datav/DeepFusionMOT/data/detections/2D/mmdetection_cascade_x101/val'
    
    print("正在初始化加载器...")
    loader = NuScenes2DLoader(NUSC_PATH, JSON_DIR)
    
    # 我们知道 json 里的场景是验证集中的某个场景，假设我们测试 'scene-0003' 
    # (你可以根据你的实际验证集 scene name 列表来查)
    test_scene_name = 'scene-0003' # 请换成一个你验证集中真实存在的 scene name
    
    try:
        print(f"正在抽取并对齐 {test_scene_name} 的 2D 结果...")
        aligned_data = loader.load_aligned_2d_boxes(test_scene_name)
        
        # 随便取一个关键帧展示
        sample_token = list(aligned_data.keys())[0]
        print(f"\n✅ 成功对齐！关键帧 {sample_token} 包含的 2D 框视角分布：")
        for cam_name, boxes in aligned_data[sample_token].items():
            print(f" - {cam_name}: {len(boxes)} 个目标")
            
    except Exception as e:
        print(e)