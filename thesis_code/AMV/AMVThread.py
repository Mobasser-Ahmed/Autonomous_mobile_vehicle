import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import ThesisCarController as mM
import threading
import logging
import socket

SERVER_IP = '192.168.1.105'
SERVER_PORT = 9090

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
server_available = False


def is_socket_open():
    while True:
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.settimeout(3)  # 2 Second Timeout
        result = client_sock.connect_ex((SERVER_IP, SERVER_PORT))
        if result == 0:
            client_sock.shutdown(socket.SHUT_RDWR)
            client_sock.close()
            print('port OPEN')
            time.sleep(1)
            global server_available
            server_available = True
            break
        else:
            print('port CLOSED, connect_ex returned: ' + str(result))
            server_available = False


def local_pilot():
    time.sleep(1)
    cam = cv2.VideoCapture(0)
    check_thread = threading.Thread(target=is_socket_open)
    check_thread.start()
    while True:
        if server_available == False:
            ret_feed, img = cam.read()
            cv2.imshow('Input', img)
            c = cv2.waitKey(1)
            if c == 27:
                break
            print("Driving locally")
            mM.move(0, 0)  # the local pilot can be human or an autonomous system
            # The local pilot evaluation is not included in the thesis scope.
            # Thus, the vehicle holds it's position during local driving.
        else:
            cam.release()
            cv2.destroyAllWindows()
            online_pilot()


def client_recv(client_socket):
    time.sleep(1)
    while True:
        msg_received = client_socket.recv(1024)
        obj = msg_received.decode()
        if 9 > len(obj) > 0:
            print(obj.split(","))
            mM.move(float(obj.split(",")[0]), float(obj.split(",")[1]))


def online_pilot():
    try:
        cam = cv2.VideoCapture(0)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(6)  # required for the rpi otherwise it waits
        client_socket.connect((SERVER_IP, SERVER_PORT))
        rcv = threading.Thread(target=client_recv, args=[client_socket])
        rcv.start()
        global server_available
        server_available = True
        while True:
            try:
                ret, frame = cam.read()
                result, frame = cv2.imencode('.jpg', frame, encode_param)
                # data = zlib.compress(pickle.dumps(frame, 0))
                data = pickle.dumps(frame, 0)
                size = len(data)
                print(size)
                client_socket.send(struct.pack(">L", size) + data)
                msg_received = client_socket.recv(1024)
                obj = msg_received.decode()
                if 9 > len(obj) > 0:
                    print(obj.split(","))
                    mM.move(float(obj.split(",")[0]), float(obj.split(",")[1]))

            except ConnectionError:
                # socket close and shutdown
                print(f"Connection from server has been lost during recv.")
                cam.release()
                local_pilot()
    except Exception as e:
        print(e)
        print(f"No server found")
        cam.release()
        local_pilot()


if __name__ == '__main__':
    online_pilot()
