import cv2
import socket
import sys
import time
import pickle
import struct

host = '192.168.1.107'  # the ip address of the server machine
port = 9090
buffersize = 1024
server_address = (host, port) 

N = 1000
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]

def benchmark_streaming():
    cam = cv2.VideoCapture(0)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    print("Benchmark streaming...")
    while True:
        duration = 0.0
        for i in range(0, N):
            start = time.time()
            
            ret, frame = cam.read()
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(frame, 0)
            size = len(data)
            client_socket.send(struct.pack(">L", size) + data)
                
            datarcv= client_socket.recv(buffersize)
            datarcv = int(datarcv.decode())
            duration += time.time() - start
            
            if size != datarcv:
                print("frame is lost")

        print(duration*pow(10, 6)/N, "µs for TCP")
    client_socket.close()
    
benchmark_streaming()    