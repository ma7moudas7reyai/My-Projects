#include <windows.h>
#include <iostream>
#include <thread>
#include <cstring>

using namespace std;

#define SHM_NAME "Local\\ChatSharedMemory"
#define SEM_NAME "Local\\ChatSemaphore"

struct ChatBuffer
{
    char username[32];
    char message[256];
    bool hasNewMessage;
};

ChatBuffer *buffer = nullptr;
HANDLE hMapFile = NULL;
HANDLE hSemaphore = NULL;
bool running = true;

void readerThread(const string &self)
{
    while (running)
    {
        if (!hSemaphore)
            continue;

        WaitForSingleObject(hSemaphore, INFINITE);

        if (buffer->hasNewMessage && strcmp(buffer->username, self.c_str()) != 0)
        {
            cout << "\n[" << buffer->username << "]: " << buffer->message << endl;

            buffer->hasNewMessage = false;
        }

        ReleaseSemaphore(hSemaphore, 1, NULL);
        Sleep(100);
    }
}

void writerThread(const string &self)
{
    string msg;

    while (running)
    {
        getline(cin, msg);

        if (msg == "exit")
        {
            running = false;
            break;
        }

        if (!hSemaphore)
            continue;

        WaitForSingleObject(hSemaphore, INFINITE);

        strcpy(buffer->username, self.c_str());
        strcpy(buffer->message, msg.c_str());
        buffer->hasNewMessage = true;

        ReleaseSemaphore(hSemaphore, 1, NULL);
    }
}

int main()
{
    string username;
    cout << "Enter your username: ";
    getline(cin, username);

    hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(ChatBuffer),
        SHM_NAME);

    if (hMapFile == NULL)
    {
        cout << "Failed to create shared memory\n";
        return 1;
    }

    buffer = (ChatBuffer *)MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(ChatBuffer));

    if (buffer == NULL)
    {
        cout << "Failed to map shared memory\n";
        return 1;
    }

    if (GetLastError() != ERROR_ALREADY_EXISTS)
    {
        buffer->hasNewMessage = false;
        strcpy(buffer->username, "");
        strcpy(buffer->message, "");
    }

    hSemaphore = CreateSemaphoreA(
        NULL,
        1,
        1,
        SEM_NAME);

    if (hSemaphore == NULL)
    {
        cout << "Failed to create semaphore\n";
        return 1;
    }

    cout << "Shared Memory Chat started\n";
    cout << "Type messages (type 'exit' to quit)\n";

    thread reader(readerThread, username);
    thread writer(writerThread, username);

    reader.join();
    writer.join();

    UnmapViewOfFile(buffer);
    CloseHandle(hMapFile);
    CloseHandle(hSemaphore);

    return 0;
}
