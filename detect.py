import cv2
import math
import os
from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8 

yolov8_detector = YOLOv8('best.onnx')
classNames = [chr(i) for i in range(65, 91)] 
detection_sign='' 
def process_detection(img, conf_threshold=0.45):
    resized_img = cv2.resize(img, (640, 640))
    pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    detections = yolov8_detector(pil_img, size=640, conf_thres=conf_threshold, iou_thres=0.5)
    predicted_sign = None
    for detection in detections:
        bbox = detection['bbox']          
        conf = detection['score']         
        cls = detection['class_id']       
        
        if conf > conf_threshold:
            x1, y1, x2, y2 = map(int, bbox)  
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            predicted_sign = classNames[cls]
            label = f'{predicted_sign} {conf:.2f}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return predicted_sign, img

def image_detection(image_path):
    img = cv2.imread(image_path)
    predicted_sign, processed_img = process_detection(img, 0.45)
    processed_image_path = os.path.join('static/files', 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, processed_img)
    return predicted_sign, processed_image_path

def video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    global detection_sign
    while True:
        success, img = cap.read()
        if not success:
            break  # Exit loop if no more frames
        detection_sign, processed_img = process_detection(img, 0.45)
        
        # Yield processed frame and detected sign
        yield processed_img, detection_sign
