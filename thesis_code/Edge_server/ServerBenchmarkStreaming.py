import pickle
import socket
import struct

import cv2

HOST = ''
PORT = 9090

edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
edge_socket.bind((HOST, PORT))
edge_socket.listen(2)

conn, addr = edge_socket.accept()

packet = b''
payload_size = struct.calcsize("L")

while True:

    while True:
        while len(packet) < payload_size:
            packet += conn.recv(4096)

        packed_msg_size = packet[:payload_size]
        packet = packet[payload_size:]
        msg_length = struct.unpack(">L", packed_msg_size)[0]
        while len(packet) < msg_length:
            packet += conn.recv(4096)
        frame_data = packet[:msg_length]
        packet = packet[msg_length:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        cv2.imshow('stream', frame)
        try:
            conn.send(bytes(str(msg_length), encoding='utf-8'))
        except ConnectionError:
            print(f"connection failed during sending to the client {conn}")
        cv2.waitKey(1)
