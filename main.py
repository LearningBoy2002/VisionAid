import cv2
import numpy as np
import time
import os
from gtts import gTTS
from playsound import playsound

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
frame_no = 0
inc = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    class_ids = []
    confidences = []
    boxes = []
    
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    voice = ""

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"{label}: {confidence*100:.2f}%")
            voice = f"{label} in front of you"

            file_path = f'voice{inc}.mp3'
            inc += 1
            sound = gTTS(text=voice, lang='en')
            sound.save(file_path)
            playsound(file_path)
            os.remove(file_path)
    else:
        playsound('no_obj.mp3')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"FPS: {1/elapsed:.2f}")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
