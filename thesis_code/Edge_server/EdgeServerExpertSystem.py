import socket
import threading
import time
import cv2
import numpy as np
import struct
import pickle
import clipspy_car as expert_system
from easyocr import Reader

# expert system loading
expert_system.load_clp()
# ocr setup
ocr_reader = Reader(['en'])
# yolo setup
labels = open('model/tsd.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv2.dnn.readNet("model/custom-yolov4-tiny-detector_final.weights", "model/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# distance configuration
FOCAL_LENGTH = 666  # in pixel
SIGN_WIDTH = 9  # in cm

# client-server setup
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Connection oriented, IPV
s.bind((socket.gethostname(), 8097))  # Ip address information, port
s.listen(5)
print(s.getsockname())
connections = []  # Connection added to this list every time a client connects
speed = 1

# steering prediction model converted to omnx
net_steer = cv2.dnn.readNetFromONNX('steering_model.onnx')

payload_size = struct.calcsize(">L")

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


def accept_connection():
    while True:
        clientsocket, address = s.accept()
        connections.append(clientsocket)
        threading.Thread(target=drive_vehicle, args=(clientsocket, address,)).start()


def pre_process(frame):
    frame = frame[54:120, :, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.resize(frame, (200, 66))
    frame = frame / 255
    return frame


# start ocr
def image_to_ocr(img, box):
    w = box[2]
    h = box[3]
    x = box[0]
    y = box[1]
    crop_img = img[y:y + h, x:x + w]
    detection_txt = ocr_reader.readtext(crop_img)
    if len(detection_txt) > 0 and detection_txt[0][1].isnumeric() > 0:
        print(detection_txt[0][1])
        return str(detection_txt[0][1])
    else:
        return ''


def draw_predictions(image, classes, scores, boxes):
    distance = 0
    height = 0
    class_name = ''
    obj_box = ''
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        if classid[0] == 0:  # TL
            continue
        width = box[2]
        height = box[3]
        obj_box = box
        class_name = labels[classid[0]]
        distance = (SIGN_WIDTH * FOCAL_LENGTH) / width
        distance = round(distance, 2)
        label = "%s : %f Dis: %s" % (class_name, score, distance)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, distance, height, class_name, obj_box


def get_steering_prediction(img):
    img = cv2.resize(img, (240, 120))  # resize is important because we collected packet in this form during training
    img = np.asarray(img)
    img = pre_process(img)
    img = np.array([img])
    net_steer.setInput(img)
    steering = float(net_steer.forward())
    return round(steering, 2)


def draw_steering_line(image):
    height = image.shape[0]
    width = image.shape[1]
    centerX = int(width / 2)
    centerY = int(height / 2)


def drive_vehicle(clientsocket, address):
    print(f"Connection from {address} has been established.")
    packet = b""
    frame_count = 0
    t = time.time() + 3  # to count average fps
    speed_limit_OCR_checked = time.time()
    while True:
        try:
            while len(packet) < payload_size:
                packet += clientsocket.recv(4096)
            packed_msg_size = packet[:payload_size]
            packet = packet[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(packet) < msg_size:
                packet += clientsocket.recv(4096)
            frame_data = packet[:msg_size]
            packet = packet[msg_size:]
        except ConnectionError:
            print(f"Connection from {address} has been lost.")
            if clientsocket in connections:
                connections.remove(clientsocket)
            return

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame_count += 1

        classes, scores, boxes = model.detect(img, 0.7, 0.3)  # threshold and confidence
        image, obj_distance, obj_height, obj_name, obj_box = draw_predictions(img, classes, scores, boxes)

        # steering prediction
        steering = get_steering_prediction(img)

        t2 = time.time()
        fps = "FPS :" + str(round(frame_count / (t2 - t), 1))

        cv2.putText(image, fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('ImageWindow', image)
        global speed
        if 'speedlimit' in obj_name and speed_limit_OCR_checked < time.time() - 3:
            ocr_res = str(image_to_ocr(image, obj_box))
            if ocr_res != '':
                detected_speed_limit = ocr_res
                print(detected_speed_limit)
                speed_limit_OCR_checked = time.time()
        print("distance: %i : height: %i class %s" % (obj_distance, obj_height, obj_name))
        distance_range = ''
        if obj_distance < 100:  # if object 100 cm far from the car
            distance_range = 1
        speed = expert_system.get_decision(obj_name, distance_range, 1)
        actuate_cmd = str(speed) + "," + str(steering)

        cv2.waitKey(1)
        for connection in connections:
            try:
                connection.send(bytes(str(actuate_cmd), encoding='utf-8'))
            except ConnectionError:
                print(f"Unable to reach client with socket {connection}")
                if connection in connections:
                    connections.remove(connection)


print("Edge-server Started")
accept_connection()
