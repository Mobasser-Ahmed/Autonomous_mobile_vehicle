import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import ThesisCarController as mM
import threading

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.105', 9090))  # ip address and port of the edge-server
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]

startTime = time.time()


def handle_rcv():
    try:
        msg_received = client_socket.recv(1024)
        obj = msg_received.decode();
        if 9 > len(obj) > 0:
            mM.move(float(obj.split(",")[0]), float(obj.split(",")[1]))

    except ConnectionError:
        print(f"Connection from the edge-server has been lost.")
        mM.move(0, 0)


print("AMV started")
mM.move(0, 0)

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    try:
        client_socket.send(struct.pack(">L", size) + data)
    except ConnectionError:
        mM.move(0, 0)
        break
    handle_rcv()

cam.release()
