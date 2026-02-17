#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <thread>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

#define PORT 54000
#define BUFFER_SIZE 1024

mutex consoleMutex;

void receiveMessages(SOCKET sock)
{
    char buffer[BUFFER_SIZE];
    int bytesReceived;

    while (true)
    {
        bytesReceived = recv(sock, buffer, BUFFER_SIZE - 1, 0);
        if (bytesReceived <= 0)
            break;

        buffer[bytesReceived] = '\0';

        lock_guard<mutex> lock(consoleMutex);
        cout << "\n[Message] " << buffer << endl;
        cout << "> " << flush;
    }
}

int main()
{
    WSADATA wsData;
    WSAStartup(MAKEWORD(2, 2), &wsData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);

    inet_pton(AF_INET, "172.20.10.3", &server.sin_addr);

    if (connect(sock, (sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
    {
        cout << "Connection failed\n";
        return 1;
    }

    cout << "Connected to server\n";

    thread(receiveMessages, sock).detach();

    while (true)
    {
        string msg;

        {
            lock_guard<mutex> lock(consoleMutex);
            cout << "> ";
        }

        getline(cin, msg);
        if (msg == "exit")
            break;

        send(sock, msg.c_str(), msg.size(), 0);
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
