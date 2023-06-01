import socket
import threading
import time
import cv2
import numpy as np
import struct
import pickle

# yolo setup
labels = open('model/tsd.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv2.dnn.readNet("model/custom-yolov4-tiny-detector_final.weights", "model/custom-yolov4-tiny-detector.cfg")
# net = cv2.dnn.readNet("model/tiny_v4_mosaic.weights", "model/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# distance configuration
FOCAL_LENGTH = 666  # in pixel
SIGN_WIDTH = 9  # in cm

# client-server setup
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Connection oriented, IPV
s.bind((socket.gethostname(), 9092))  # Ip address information, port
s.listen(5)
print(s.getsockname())
connections = []  # Connection added to this list every time a client connects
speed = 0.5
last_sign = ""

# steering prediction model converted to omnx
# net_steer = cv2.dnn.readNetFromONNX('steering_model.onnx')
net_steer = cv2.dnn.readNetFromONNX('model/lab.onnx')

payload_size = struct.calcsize(">L")
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

stop_sign = cv2.imread('speed_30.jpg')
stop_sign = cv2.resize(stop_sign, (640, 480))
model.detect(stop_sign, 0.7, 0.6)

def accptconnection():
    while True:
        clientsocket, address = s.accept()
        connections.append(clientsocket)  # adds the clients information to the connections array
        threading.Thread(target=car_heart, args=(clientsocket, address,)).start()

# function to pre-process image before feeding to the steering angle predection model
def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


def draw_predictions(image, classes, scores, boxes):
    distance = 0
    height = 0
    class_name = ''
    obj_box = ''
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        if classid[0] == 0:  # TL #todo would be 0
            continue
        width = box[2]
        height = box[3]
        obj_box = box
        class_name = labels[classid[0]]
        distance = (SIGN_WIDTH * FOCAL_LENGTH) / width
        distance = round(distance, 2)
        label = "%s : %.2f Dis: %s" % (class_name, score, distance)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, distance, height, class_name, obj_box

# this function forward an image to the steering prediction model and return the final predicted steering angle value
def get_steering_prediction(img):
    img = cv2.resize(img, (240, 120))  # resize is important because we collected data in this form during training
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    net_steer.setInput(img)
    steering = float(net_steer.forward())
    return round(steering, 2)

# utility method to rescale the frame size
# this method can be used just to show the video in small or big resulation
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 150)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# This method connects to the client and perform all necessary function calling and return the result to the car client
def car_heart(clientsocket, address):
    print(f"Connection from {address} has been established.")
    data = b""
    frame_count = 0
    t = time.time() # to count average fps
    print("Vehicle connect at " + str(time.time()))
    while True:
        try:
            while len(data) < payload_size:
                data += clientsocket.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(data) < msg_size:
                data += clientsocket.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
        except ConnectionError:
            print(f"Connection from {address} has been lost.")
            if clientsocket in connections:
                connections.remove(clientsocket)
            return

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # save a frame for test
        # cv2.imwrite("D:\pythonProject\self-driving\client_video\image1.jpg", frame)
        frame_count += 1
        # img = frame
        img = cv2.flip(frame, 0)  # needed for raspberry pi because raspberry pi upside down
        image = cv2.flip(img, 1) # we are filping again since it has mirror view
        # img = cv2.flip(frame, 0)
        startNN = time.time()
        classes, scores, boxes = model.detect(image, 0.7, 0.6)  # threshold and confidence
        image, obj_distance, obj_height, obj_name, obj_box = draw_predictions(image, classes, scores, boxes)
        #print(obj_name, obj_distance)

        # steering prediction
        steering = get_steering_prediction(img) # we are providing only vertical flipped because the data was collected
        # and save this way. webcam module is only flipping once

        # print(steering)
        t2 = time.time()
        fps = "FPS : " + str(round(frame_count / (t2 - t), 1))
        # print(f"Time: {t2 - t}")

        # steerToShow = int(steering * 100)
        cv2.putText(image, fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "NN Latency : " + str("{:.3f}".format((t2 - startNN)) + " ms"), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.line(image, (centerX + steerToShow, centerY + 40), (centerX, height), (255, 0, 0), 4)

        # detectedImg = np.concatenate((image, stop_sign), axis=1)
        global speed
        global last_sign
        # #print(class_name+ ": "+str(w))
        if obj_name == "stop" and obj_distance < 70:  # 81,75 ()
            global speed
            speed = 0
            last_sign = "stop"
        elif obj_name == "speedlimit" and obj_distance < 100:
            speed = 0.3
            last_sign = "speedlimit"
        elif obj_name == "crosswalk" and obj_distance < 100:
            last_sign = "crosswalk"
            speed = 0.5
        # elif last_sign == "speedlimit":  #if we want to keep the last speed
        #     speed = 0.3
        else:
            speed = 1 #todo full speed =1

        cv2.putText(image, "Throttle : " + str(speed*100) + " %", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Est. Distance of sign : " + str("{:.0f}".format(obj_distance)) + " CM", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Steering Value : " + str("{:.1f}".format(steering)), (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        actuate_cmd = str(speed) + "," + str(steering)
        # frame150 = rescale_frame(image, percent=200)
        # cv2.imshow('frame150', frame150)
        cv2.imshow('ImageWindow1', image)

        cv2.waitKey(1)
        # for connection in connections:  # iterates through the connections array and sends message to each one
        try:
            clientsocket.send(bytes(str(actuate_cmd), encoding='utf-8'))
        except ConnectionError:
            print(f"Unable to reach vehicle with socket {clientsocket}")
            if clientsocket in connections:
                connections.remove(clientsocket)


print("Edge Server Started")
accptconnection()
