import os
import sys
import copy
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import urllib.request
from PublicCore.FileControl.FileFunction import GetFileNameWithoutExtension
from SegmantationConvertCoco import CocoDataConverter
import torch
from IPython.display import clear_output


# [segment_anything]
# pip install segment-anything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# [CLIP]
# pip install git+https://github.com/openai/CLIP.git
import clip

# [YOLO]
# pip install ultralytics
# import ultralytics
# ultralytics.checks()
# from ultralytics import YOLO


imgPath = r"Picture\2000301.jpg"
DatasetFolderPath = "Dataset"
area_threshold_min = 5 #0.5
area_threshold_max = 100 #15

# sam_model_type = "vit_l"
# sam_checkpoint = "sam_vit_l_8a9f7a.pth"
sam_model_type = "vit_h"
sam_filename = "sam_vit_h_4b8939.pth"
sam_checkpoint = sam_filename
url = f"https://dl.fbaipublicfiles.com/segment_anything/{sam_filename}"

CLIP_device = "cuda" if torch.cuda.is_available() else "cpu"
SAM_device = "cuda" if torch.cuda.is_available() else "cpu"




def show_anns(anns):
  if len(anns) == 0:
      return
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)

  img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
  img[:,:,3] = 0
  for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [0.35]])
      img[m] = color_mask
  ax.imshow(img)
  return img

def show_anns_bbox(anns, image):
  if len(anns) == 0:
    return
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)

  img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
  img[:,:,3] = 0

  # 初始化物體計數
  object_count = 0

  for ann in sorted_anns:
    m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask


    # 使用 numpy.where 找到 True 值的座標位置
    true_indices = np.where(m)
    row_indices, col_indices = true_indices
    x_center = int(np.mean(row_indices))
    y_center = int(np.mean(col_indices))
    print(x_center, y_center)
    # 在圖像上標記物體數字
    # object_count += 1
    # cv2.putText(img, str(object_count), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    bbox = ann['bbox']
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    print(y_min, y_max, x_min, x_max)
    bbox_region = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    bbox_region = Image.fromarray(bbox_region)  # 将 NumPy 数组转换为 PIL 图像对象

    plt.imshow(bbox_region)
    plt.axis('off')
    plt.show(block=False)
    # plt.pause(0.1)
    CLIP_predict(bbox_region)


    # if object_count == 50:
    #   break;

  ax.imshow(img)

def save_matrix(matrix, filename):
    with open(filename, 'wb') as file:
        pickle.dump(matrix, file)

def load_matrix(filename):
    with open(filename, 'rb') as file:
        matrix = pickle.load(file)
    return matrix

def binarize_image(image, threshold=254):
    """
    將圖像進行二值化處理。

    參數:
    image (numpy.ndarray): 要處理的圖像。
    threshold (int): 二值化閾值。

    返回:
    numpy.ndarray: 二值化後的圖像。
    """
    # 將圖像轉換為灰階
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 應用二值化
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def extract_borders(binary_image):
    """
    從二值化圖像中提取邊界。

    參數:
    binary_image (numpy.ndarray): 二值化後的圖像。

    返回:
    numpy.ndarray: 只有邊界為白色的圖像。
    """
    # 尋找輪廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 創建一個全黑的圖像
    border_image = np.zeros_like(binary_image)
    # 繪製白色輪廓
    cv2.drawContours(border_image, contours, -1, (255), 1)
    return border_image

def overlay_borders(image_paths):
    """
    將多張圖片上的線條疊加在一起。

    參數:
    image_paths (list): 包含圖片路徑的列表。

    返回:
    numpy.ndarray: 線條疊加後的圖像。
    """
    # 初始化一個全黑的圖像作為基礎
    overlay_image = None
    for path in image_paths:
        # 讀取圖片
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if overlay_image is None:
            overlay_image = np.zeros_like(image)
        # 將當前圖片的線條疊加到基礎圖像上
        overlay_image = cv2.bitwise_or(overlay_image, image)
    return overlay_image


# YoloAutoLabel
def YoloAutoLabel(image):
      
  results = model_yolo.predict(source=image, conf=0.25)
  print(len(results[0]))
  if len(results[0]) == 0:
    return None, None

  xyxyList = []
  confidenceList = []
  classList = []

  for box in results[0].boxes:
    xyxyList.append(box.xyxy)
    confidenceList.append(box.conf[0])
    classList.append(box.cls)

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label_text = f"Class: {model_yolo.names[int(box.cls[0])]}, Confidence: {box.conf[0]:.2f}"
    print(label_text)
    # plt.imshow(image)
    # plt.show()
    # plt.pause(0.1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label_text, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  
  detections = [{'xyxy': xyxy, 'confidence': confidence, 'class': cls} for xyxy, confidence, cls in zip(xyxyList, confidenceList, classList)]
  best_detection = max(detections, key=lambda x: x['confidence'])
  
  best_box = best_detection['xyxy']
  best_confidence = best_detection['confidence']
  best_class_index = int(best_detection['class'])
  best_class_name = model_yolo.names[best_class_index]
  result_str = f"{best_class_name}_{best_confidence:.5f}"
  print(f"Best Detection - Class: {result_str}")


  return best_box, result_str

# CLIP_predict
def CLIP_predict(image, target=None):
  status = False
  result_dict = {}

  # 对图像进行预处理
  image_Pil = Image.fromarray(image)  #NumPy 轉為 PIL圖像對象
  image_input = preprocess(image_Pil).unsqueeze(0).to(CLIP_device)

  # 运行模型
  with torch.no_grad():
      image_features = model.encode_image(image_input)

  # class_labels
  # class_labels = ['cat', 'dog', 'flower', 'food', 'car']
  # class_labels = ["herring bone ladder", "trestle ladder", "ladder","People"]
  # class_labels = ["ladder", "person", "people", "Employee", "man", "tool", "item"]
  # class_labels = ["ladder", "person", "other"]
  # class_labels = ["people", "other", "wood ladder"]
  # class_labels = ["Worker","person", "other", "wood ladder"]
  # class_labels = ["Worker","person", "other", "Aluminum ladder"]
  # class_labels = ["person", "Aluminum ladder", "wood ladder"]
  # class_labels = ["person", "Aluminum ladder", "wood ladder", "object"]
  # class_labels = ["Worker","person", "other", "Black ladder"]
  # class_labels = ["Aggressive objects", "Offensive objects", "weapons","drone", "robot", "object"]
  class_labels = ["drone", "bird", "airplane", "robot"]
  class_labels = ["drone", "bird", "tree"]
  # class_labels = ["person", "ladder", "helmet"]
  class_labels = ["gray","green", "River", "Road"]  
  class_labels = ["River", "Road"]
  class_labels = ["gray","green"]

  # 加载类别描述
  class_descriptions = clip.tokenize(class_labels).to(CLIP_device)

  # 计算图像与类别描述之间的相似度
  logits_per_image, logits_per_text = model(image_input, class_descriptions)
  probas = logits_per_image.softmax(dim=-1)
  sorted_indices = probas.argsort(descending=True)

  # 输出预测结果
  first_iteration = True
  for i in sorted_indices[0]:
    class_label = class_labels[i]
    prob = round(float(probas[0][i]), 5)
    print(f"{class_label}: {prob}")

    if first_iteration:
      result_dict["class_label"] = class_label
      result_dict["prob"] = prob
      first_iteration = False

    if target != None:
      if (class_label == target) and (prob >=0.7):
        print(f"****************************")
        status = True

  return status, result_dict

# show_anns_white
def show_anns_white(anns, image, threshold_min, threshold_max, file_name, output_dir_name):
  print("anns:", len(anns))
  if len(anns) == 0:
    return
  
  current_fig = None
  height, width, _ = image.shape
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  
  prob_max_1 = 0
  prob_max_2 = 0
  prob_max_3 = 0
  kernel_dilate = np.ones((3,3), np.uint8)  # 膨胀操作的结构元素
  for ann in sorted_anns:
    #areaPercent
    area_percentage = round((ann['area'] / ann['segmentation'].size) * 100, 2)
    if area_percentage < threshold_min or area_percentage > threshold_max:
          continue
    print("area:",area_percentage)
        
    # white_image
    binary_mask = ann['segmentation']
    # print(binary_mask)
    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    white_image[binary_mask] = image[binary_mask]


    # predict
    clear_output(wait=True)
    status, result_dict = CLIP_predict(white_image)
    #bbox, result_dict = YoloAutoLabel(white_image)
    
    class_label = result_dict["class_label"]
    prob = result_dict["prob"]     
    result_str =  f"{class_label}_{prob:.2f}"
    if result_dict == {}:
      continue    
    # if prob < 0.75:
    #   continue
    # if ("person" not in class_label) and ("ladder" not in class_label):
    #   continue

    
    # save_data
    partial_file_name = f"{file_name}_{result_str}_{area_percentage}"
    print("-------------------------------------")
    partial_converter = CocoDataConverter(image_dir=f"./images_{output_dir_name}", output_dir=f"./output_{output_dir_name}")
    partial_converter.add_annotation(1, binary_mask)
    partial_converter.add_image(file_name, height, width, 1)
    partial_converter.save_coco_data(partial_file_name+ ".json")
    partial_converter.save_white_image(white_image, partial_file_name + ".png")
    if current_fig != None:
          plt.close(current_fig)
    current_fig = plt.figure()
    # plt.axis('off')
    plt.imshow(white_image)
    plt.show(block=False)
    plt.pause(0.1)
    
    binary_image = binarize_image(white_image)
    border_image = extract_borders(binary_image)
    border_image = cv2.dilate(border_image, kernel_dilate, iterations=1)
    plt.imshow(border_image,cmap='gray')
    plt.show(block=False)
    plt.pause(0.1)
    partial_converter.save_white_image(border_image, partial_file_name + "_binary.png")
    
    # save to DatasetPredict
    # if ("Road" in class_label) and  ("_RO_" in file_name) and (prob > prob_max_1) :
    #   prob_max_1 = prob
    # elif ("River" in class_label) and  ("_RI_" in file_name) and (prob > prob_max_2) :
    #   prob_max_2 = prob
    if ("gray" in class_label)and (prob > prob_max_3) :
       prob_max_3 = prob
    else:
      continue
    
    save_image = cv2.cvtColor(border_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'DatasetPredict/{file_name}.png', save_image)

    
  #   converter.add_annotation(1, binary_mask)
  #   converter.add_image(file_name, height, width, 1)
  #   converter.increment_ids()  # 增加ID
    
  # converter.save_coco_data(f"{file_name}.json")


#Test_Dataset
def Test_Dataset(imgPath):
  current_date = datetime.now().strftime("%Y%m%d")
  file_name = GetFileNameWithoutExtension(imgPath)
  output_dir_name = f"{file_name}_{current_date}"
  converter = CocoDataConverter(image_dir=f"./images_{output_dir_name}", output_dir=f"./output_{output_dir_name}")

  #image
  image = cv2.imread(imgPath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.ion()
  plt.imshow(image)
  plt.axis('off')
  plt.show(block=False)
  plt.pause(0.1)
  converter.save_white_image(image, file_name + ".png")

  # CLIP_predict
  status, result_dict = CLIP_predict(image)

  # mask_generator
  torch.cuda.empty_cache()
  masks2 = mask_generator_2.generate(image)
  len(masks2) #0.9=174  #0.5=306 #0.1=307 #1=10

  # plt.imshow(image)
  # image_mask = show_anns(masks2)
  # plt.axis('off')
  # plt.show(block=False)
  # plt.pause(0.1)

  class_label = result_dict["class_label"]
  prob = result_dict["prob"]
  result_str =  f"{class_label}_{prob:.4f}"
  partial_file_name = f"{file_name}_{result_str}"
  converter.save_white_image(image, partial_file_name + ".png")


  # show_anns_white
  show_anns_white(masks2, image, area_threshold_min, area_threshold_max, file_name, output_dir_name)

def TestFlow(dataset_folder_path):
  torch.cuda.empty_cache()
  files = os.listdir(dataset_folder_path)

  for file_name in files:
    if file_name.endswith((".jpg", ".jpeg", ".png")):
      file_path = os.path.join(dataset_folder_path, file_name)
      print("picture:", file_path)
      plt.close('all')
      Test_Dataset(file_path)



if __name__ == "__main__":
    
    if not os.path.isfile(sam_filename):
        urllib.request.urlretrieve(url, sam_filename)

    # [YOLO]
    # model_yolo = YOLO(f'yolov8x.pt')
    
    # [CLIP]
    model, preprocess = clip.load('RN50x4', device=CLIP_device)
    # CLIP-ViT-B/32: 400 million parameters
    # CLIP-ViT-B/16: 400 million parameters
    # CLIP-RN50: 400 million parameters
    # CLIP-RN101: 500 million parameters
    # CLIP-RN50x4: 1.6 billion parameters
   
    #[SAM]
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=SAM_device)
    
    # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
    # * `segmentation` : the mask
    # * `area` : the area of the mask in pixels
    # * `bbox` : the boundary box of the mask in XYWH format
    # * `predicted_iou` : the model's own prediction for the quality of the mask
    # * `point_coords` : the sampled input point that generated this mask
    # * `stability_score` : an additional measure of mask quality
    # * `crop_box` : the crop of the image used to generate this mask in XYWH format
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)
    # print(len(masks))
    # # print(masks[0].keys())
      
    # ## Automatic mask generation options
    # There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. Here is an example configuration that samples more masks:
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=2,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,  # Requires open-cv to run post-processing
    )

    # mask_generator_2 = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )



    # RUN
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = GetFileNameWithoutExtension(imgPath)
    output_dir_name = f"{file_name}_{current_date}"
    converter = CocoDataConverter(image_dir=f"./images_{output_dir_name}", output_dir=f"./output_{output_dir_name}")
    
    # image
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.ion()
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(0.1)
    converter.save_white_image(image, file_name + ".png")
   
    # CLIP_predict
    status, result_dict = CLIP_predict(image)


    # mask_generator
    torch.cuda.empty_cache()
    masks2 = mask_generator_2.generate(image)
    len(masks2) #0.9=174  #0.5=306 #0.1=307 #1=10

    plt.figure()
    plt.imshow(image)
    image_mask = show_anns(masks2)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.1)
  
    class_label = result_dict["class_label"]
    prob = result_dict["prob"]
    result_str =  f"{class_label}_{prob:.4f}"
    partial_file_name = f"{file_name}_{result_str}"
    converter.save_white_image(image, partial_file_name + ".png")
      
      
      
    # show_anns_white
    show_anns_white(masks2, image, area_threshold_min, area_threshold_max, file_name, output_dir_name)

    # [rectangle putText]
    # xyxyList = []]
    # confidenceList = []
    # classList = []
    # results = model.predict(source=image, conf=0.25)

    # for box in results[0].boxes:
    #     xyxyList.append(box.xyxy)
    #     confidenceList.append(box.conf)
    #     classList.append(box.cls)

    # x1, y1, x2, y2 = map(int, box.xyxy[0])
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # label_text = f"Class: {model.names[int(box.cls[0])]}, Confidence: {box.conf[0]:.2f}"
    # cv2.putText(image, label_text, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #TestFlow DatasetFolderPath
    TestFlow(DatasetFolderPath)