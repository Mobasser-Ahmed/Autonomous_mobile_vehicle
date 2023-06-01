import cv2
import io
import socket
import struct
import time
import pickle
import zlib
# import ThesisCarControll_v1 as mM
import threading
import logging
import socket
import numpy as np


net_steer = cv2.dnn.readNetFromONNX('model/lab.onnx')

SERVER_IP = '192.168.1.101'
#SERVER_IP = '192.168.29.64'
SERVER_PORT = 9092

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70] # originally it was 90
server_available = False
cam = cv2.VideoCapture(0)

isLocal = False

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def get_steering_prediction(img):
    img = cv2.resize(img, (240, 120))  # resize is important because we collected data in this form during training
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    net_steer.setInput(img)
    steering = float(net_steer.forward())
    if steering > 1:
        steering = 1
    if steering < -1:
        steering = -1
    return round(steering, 2)

def is_socket_open():
    while True:
        time.sleep(0.2)
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.settimeout(2)  # 2 Second Timeout
        result = client_sock.connect_ex((SERVER_IP, SERVER_PORT))
        if result == 0:
            client_sock.shutdown(socket.SHUT_RDWR)
            client_sock.close()
            print('port OPEN')
            time.sleep(0.2)
            global server_available
            server_available = True
            break
        else:
            #print('port CLOSED, connect_ex returned: ' + str(result))
            server_available = False
# def video_stream():


def client_local():
    global isLocal
    if isLocal:
        return
    isLocal = True
    print("LOCAL")
    time.sleep(0.1)
    #cam = cv2.VideoCapture(0)
    check_thread = threading.Thread(target=is_socket_open)
    check_thread.start()
    while True:
        if not server_available:
            ret_feed, img = cam.read()
            cv2.imshow('Input', img)
            c = cv2.waitKey(1)
            if c == 27:
                break
            
            ret, frame = cam.read()
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            img = cv2.flip(frame, 0)
            
            #image = cv2.flip(img, 1)
            steering = get_steering_prediction(img)  # we are providing only vertical flipped because the data was collected
            print("locally :"+ str(steering))
            # mM.move(1,steering)
            

        else:
            #cam.release()
            isLocal = False
            cv2.destroyAllWindows()
            client_server()

#to increase the fps
def client_recv(client_socket):
    fps= 0
    time.sleep(0.3)
    print("RCV")
    try:
        while True:
            start_time = time.time()
            msg_received = client_socket.recv(1024)
            rcved_time = time.time()
            latency = float("{:.2f}".format(rcved_time-start_time))
            #print(str(latency) + " sec ");
            obj = msg_received.decode()
            if len(obj) < 9 and len(obj) > 0:
                print("run")
            #     mM.move(float(obj.split(",")[0]),float(obj.split(",")[1]))
                # print(obj.split(","))
    except Exception:
        #cam.release()
        print("Disconnected during rcv")
        return
    
def client_server():
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2) # required for the rpi otherwise it waits
        client_socket.connect((SERVER_IP, SERVER_PORT))
        rcv = threading.Thread(target=client_recv,args=[client_socket])
        rcv.start()
        global server_available
        server_available = True
        print("clint connect at " + str(time.time()))
        while True:
            try:
                ret, frame = cam.read()
                #frame = cv2.resize(frame,(800,800))# have to remove, testing part
                result, frame = cv2.imencode('.jpg', frame, encode_param)
                # data = zlib.compress(pickle.dumps(frame, 0))
                data = pickle.dumps(frame, 0)
                size = len(data)
                client_socket.send(struct.pack(">L", size) + data)
                # msg_received = client_socket.recv(1024)
                # obj = msg_received.decode()
                # if len(obj) < 9 and len(obj) > 0:
                #     print(obj.split(","))

            except ConnectionError:
                #socket close and shutdown
                print(f"Connection from server has been lost during transmit.")
                return
                #cam.release()
    except socket.timeout:
        print(f"No server found")
        client_socket.shutdown(socket.SHUT_RDWR)
        client_socket.close()
        return
    except Exception:
        print("exception")
    finally:
        print("final")
        time.sleep(0.1)
        global isLocal
        if not isLocal:
            client_local()

if __name__ == '__main__':
    client_server()