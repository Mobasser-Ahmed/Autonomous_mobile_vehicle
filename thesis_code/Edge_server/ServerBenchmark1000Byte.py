import socket
import sys

port = 9090
buffersize = 1024


def start_edge_server():
    edge_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    edge_server.bind((socket.gethostname(), port))
    print(edge_server.getsockname())
    edge_server.listen(5)

    print("Edge server socket is running...")

    while True:
        client, client_address = edge_server.accept()

        while True:
            data = client.recv(buffersize)
            if not data: break
            client.sendall(data)

        client.close()


start_edge_server()





