import socket
import sys
import time

host = '192.168.1.107'  # the ip address of the server machine
port = 9090
buffersize = 1024
server_address = (host, port) 

N = 1000

def benchmark_TCP_client():
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect(server_address)

    print("Starting benchmark_TCP_client ...")
    while True:
        duration = 0.0
        for i in range(0, N):
            b = bytes("a"*buffersize, "utf-8")
            start = time.time()
            socket_client.sendall(b)
            data = socket_client.recv(buffersize)
            duration += time.time() - start

            if data != b:
                print("Error: packet lost!")

        print(duration*pow(10, 6)/N, "µs took for TCP")
    socket_client.close()
    
benchmark_TCP_client()    