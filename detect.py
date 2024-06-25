import cv2
import torch
from super_gradients.training import models
import pandas as pd
import os
import shutil
import json
import time
from datetime import datetime
from tqdm import tqdm
import argparse

def detect_init(img_root_path, confidence):
    start = time.time()
    # 클래스 정보를 포함한 엑셀 파일 경로
    OBJ_CLASS_XLSX = "./제품클래스.xlsx"
    if isinstance(img_root_path, str):
        img_root_paths = [img_root_path]
    elif isinstance(img_root_path, list):
        img_root_paths = img_root_path
    else:
        print("경로 내에 폴더가 존재하지 않습니다.")
        return

    if not all(path.startswith("./") for path in img_root_paths):
        img_root_paths = ["./" + path if not path.startswith("./") else path for path in img_root_paths]

    # 장치 설정 (GPU 또는 CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(torch.cuda.get_device_name(0)

    # 클래스 이름 로드
    obj_names = {k: c[1] for k, c in enumerate(pd.read_excel(OBJ_CLASS_XLSX).values.tolist())}
    print(obj_names)
    print(type(obj_names[0]))

    # 모델 로드
    model = models.get("yolo_nas_s", num_classes=len(obj_names), checkpoint_path="./obj_ckpt_best.pth").to(device)
    detected_count = 0
    instance_count = 0
    total_image_count = 0

    for img_root_path in img_root_paths:
        img_folder = os.listdir(img_root_path)

        # tqdm을 사용하여 진행 상태 표시
        for i in tqdm(img_folder, desc="Folders"):
            img_folder_path = os.path.join(img_root_path, i)
            detected_path = os.path.join(img_folder_path, "detected")
            none_path = os.path.join(img_folder_path, "none")
            json_folder_path = os.path.join(img_folder_path, "json")
            if os.path.exists(detected_path):
                shutil.rmtree(detected_path)
            if os.path.exists(none_path):
                shutil.rmtree(none_path)
            if os.path.exists(json_folder_path):
                shutil.rmtree(json_folder_path)
            os.makedirs(detected_path)
            os.makedirs(none_path)
            os.makedirs(json_folder_path)
            img_file_names = os.listdir(img_folder_path)

            for o in tqdm(img_file_names, desc="Images", leave=False):
                if o.endswith(".jpg"):
                    total_image_count += 1
                    # 클래스별 탐지된 객체 개수 초기화
                    class_counts = {}
                    img_file = os.path.join(img_folder_path, o)
                    json_file = os.path.join(json_folder_path, o.split(".")[0] + ".json")
                    image = cv2.imread(img_file)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = model.predict(image_rgb, conf=confidence, fuse_model=False)
                    predictions = results.prediction
                    num_detections = len(predictions.bboxes_xyxy)
                    # print(f"탐지된 객체수: {num_detections}")
                    for label in predictions.labels:
                        class_name = obj_names[label]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    if num_detections > 0:
                        shutil.copyfile(img_file, os.path.join(detected_path, o))
                        data = {
                            "탐지된 인스턴스 객체수": num_detections,
                            "탐지된 클래스명": [obj_names[label] for label in predictions.labels],
                            "클래스별 탐지된 개수": class_counts
                        }
                        with open(json_file, "w", encoding='UTF-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        detected_count += 1
                        instance_count += num_detections
                        # print(f"{o}: 이 파일은 객체가 탐지되어 detected 폴더로 파일을 복사했습니다.")
                    else:
                        shutil.copyfile(img_file, os.path.join(none_path, o))
                        # print(f"{o}: 이 파일은 객체가 탐지되지 않아 none 폴더로 파일을 복사했습니다.")

    time_consumed = round(time.time() - start, 4)

    result = {
        "총 이미지 개수": total_image_count,
        "총 객체가 탐지된 이미지 개수": detected_count,
        "총 탐지된 인스턴스 객체수": instance_count,
        "총 작업 소요 시간": time_consumed
    }
    with open(f'{img_root_path}_result.json', "w", encoding='UTF-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection script")
    parser.add_argument('-p', '--path', type=str, default="./", help="이미지 폴더명 입력. 미입력 시 자동으로 폴더를 탐색하여 작업")
    parser.add_argument('-c', '--confidence', type=float, default=0.7, help="탐지 기준 설정용. 기본값=0.7")
    args = parser.parse_args()
    if args.path == "./":
        from pathlib import Path
        p = Path("./")
        img_root_path = [entry.name for entry in p.iterdir() if entry.is_dir()]
    else:
        img_root_path=args.path
    detect_init(img_root_path=img_root_path, confidence=args.confidence)

# if __name__ == '__main__':
#     import sys
#     try:
#         img_root_path = sys.argv[1]
#     except:
#         from pathlib import Path
#         p = Path("./")
#         img_root_path = [entry.name for entry in p.iterdir() if entry.is_dir()]
#     try:
#         confidence = float(sys.argv[2])
#     except:
#         confidence = 0.7
#     detect_init(img_root_path=img_root_path, confidence=confidence)



# import os
#
# product = '립스틱' #제품명
# folders = os.listdir('C:/Users/user/PycharmProjects/pythonProject1/detect/lotion') #폴더경로
# total = 0
# for folder in folders:
#       print(folder)
#       files = os.listdir('C:/Users/user/PycharmProjects/pythonProject1/detect/lotion'+ '/' + folder)
#       files = [file for file in files if file.endswith(".jpg" or "png")]
#       #print(len(files))
#       total += len(files)
# print(total)#작업 완료 이미지 개수