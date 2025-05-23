# from pycocotools import mask as maskUtils
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy
import pickle

from PublicCore.FileControl.FileFunction import CheckDirPath, GetFileNameWithoutExtension

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
      
class CocoDataConverter:
    def __init__(self, image_dir, output_dir):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_id = 1
        self.annotation_id = 1
        self.coco_data = {
            "images": [],
            "annotations": [],
            # 其他必要的字段，如 "categories" 等
        }
        CheckDirPath(self.image_dir)
        CheckDirPath(self.output_dir)


    def boolean_matrix_to_coco_segmentation(self, binary_mask):
        pixel_list = np.argwhere(binary_mask)
    
        # 将坐标成对拆分，转换为COCO segmentation格式
        segmentation = [coord for point in pixel_list for coord in point]

        return [segmentation]  # 返回包含一个segmentation列表的列表


    def add_image(self, file_name, height, width, license):
        image_data = {
            "id": self.image_id,
            "file_name": file_name,
            "height": height,
            "width": width,
            "license": license
        }
        self.coco_data["images"].append(image_data)
    

    def add_annotation(self, category_id, binary_mask):
        segmentation = self.boolean_matrix_to_coco_segmentation(binary_mask)
         
        image_id = self.image_id 
        # area = maskUtils.area(segmentation)
        # bbox = maskUtils.toBbox(segmentation)

        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            # "area": area,
            # "bbox": bbox.tolist(),
            "iscrowd": 0,
            "segmentation": segmentation
        }
        self.coco_data["annotations"].append(annotation)




    def increment_ids(self):
        self.annotation_id += 1  # 增加标注ID
        


        
        
    def save_coco_data(self, file_name):
        full_path = os.path.join(self.output_dir, file_name)
        
        with open(full_path, "w") as f:
            print(file_name)
            print()
            # print(self.coco_data)
            json.dump(self.coco_data, f, cls=NpEncoder)

    
    def save_white_image(self, white_image, filename):
        # Construct the file path for saving
        file_path = os.path.join(self.image_dir, filename)
        # Save the image to the specified directory
        save_image = cv2.cvtColor(white_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, save_image)
        # white_image.save(file_path)

        

# 示例用法
if __name__ == "__main__":
    imgPath = "Picture\\6.jpg"
    folder_path = "pkl"
    file_name = GetFileNameWithoutExtension(imgPath)

    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    plt.ion()
    plt.show()
    
    
    # 添加图像信息
    # binary_mask = np.array([[False, False, False, True, True, False, False],
    #                         [False, False, True, True, True, True, False],
    #                         [False, True, True, True, True, True, True],
    #                         [True, True, True, True, True, True, True],
    #                         [True, True, True, True, True, True, True]])
    converter = CocoDataConverter(image_dir="./images", output_dir="./output")
     
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as file:
            pkl_file_var = pickle.load(file)
        print(f"Loaded data from {pkl_file}:")
        print(pkl_file_var["binary_mask"])
    
        #white_image
        white_image = np.ones((pkl_file_var["height"], pkl_file_var["width"], 3), dtype=np.uint8) * 255
        white_image[pkl_file_var["binary_mask"]] = image[pkl_file_var["binary_mask"]]
        white_image = Image.fromarray(white_image)  # 将 NumPy 数组转换为 PIL 图像对象
        converter.save_white_image(white_image, pkl_file_var["partial_file_name"]+ ".png")
        
        # save_data
        converter.add_annotation(1, pkl_file_var["binary_mask"])
        converter.increment_ids()  # 增加ID
        partial_converter = copy.deepcopy(converter)
        partial_converter.save_coco_data(f"{file_name}.json")
        
        plt.imshow(white_image)
        plt.ion()
        plt.show()
        plt.pause(0.1)
        plt.close()

    converter.add_image(pkl_file_var["file_name"], pkl_file_var["height"], pkl_file_var["width"], 1)
    converter.save_coco_data(f"{file_name}.json")
    










