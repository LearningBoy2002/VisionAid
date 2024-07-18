from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import threading

app = Flask(__name__)


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def generate_frames():
    cap = cv2.VideoCapture(0)
    inc = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.6:
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
                    voice = f"{label} in front of you"
                    file_path = f'voice{inc}.mp3'
                    inc += 1
                    sound = gTTS(text=voice, lang='en')
                    sound.save(file_path)
                    playsound(file_path)
                    os.remove(file_path)
            else:
                playsound('no_obj.mp3')

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
