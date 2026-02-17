#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <thread>
#include <vector>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")

using namespace std;

#define PORT 54000
#define BUFFER_SIZE 1024

vector<SOCKET> clients;
mutex clientsMutex;

void handleClient(SOCKET clientSocket)
{
    char buffer[BUFFER_SIZE];
    int bytesReceived;

    while (true)
    {
        bytesReceived = recv(clientSocket, buffer, BUFFER_SIZE - 1, 0);

        if (bytesReceived <= 0)
        {
            cout << "Client disconnected\n";
            closesocket(clientSocket);

            lock_guard<mutex> lock(clientsMutex);
            for (auto it = clients.begin(); it != clients.end(); ++it)
            {
                if (*it == clientSocket)
                {
                    clients.erase(it);
                    break;
                }
            }
            break;
        }

        buffer[bytesReceived] = '\0';

        lock_guard<mutex> lock(clientsMutex);
        for (SOCKET client : clients)
        {
            if (client != clientSocket)
            {
                send(client, buffer, bytesReceived, 0);
            }
        }
    }
}

int main()
{
    WSADATA wsData;
    WSAStartup(MAKEWORD(2, 2), &wsData);

    SOCKET listening = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    sockaddr_in serverHint{};
    serverHint.sin_family = AF_INET;
    serverHint.sin_port = htons(PORT);
    serverHint.sin_addr.s_addr = INADDR_ANY;

    bind(listening, (sockaddr *)&serverHint, sizeof(serverHint));
    listen(listening, SOMAXCONN);

    cout << "Server started on port " << PORT << endl;

    while (true)
    {
        sockaddr_in client;
        int clientSize = sizeof(client);

        SOCKET clientSocket = accept(listening, (sockaddr *)&client, &clientSize);
        if (clientSocket == INVALID_SOCKET)
            continue;

        {
            lock_guard<mutex> lock(clientsMutex);
            clients.push_back(clientSocket);
        }

        thread(handleClient, clientSocket).detach();
    }

    closesocket(listening);
    WSACleanup();
    return 0;
}
