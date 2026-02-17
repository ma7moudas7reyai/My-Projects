#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <string>
#include <thread>

#pragma comment(lib, "ws2_32.lib")

#define PORT 54000
#define BUFFER_SIZE 1024

HWND hEditChat, hEditInput, hButtonSend;
SOCKET sock;

void AppendText(HWND hEdit, const std::wstring &text)
{
    int len = GetWindowTextLengthW(hEdit);
    SendMessageW(hEdit, EM_SETSEL, len, len);
    SendMessageW(hEdit, EM_REPLACESEL, 0, (LPARAM)text.c_str());
}

void ReceiveMessages()
{
    char buffer[BUFFER_SIZE];

    while (true)
    {
        int bytes = recv(sock, buffer, BUFFER_SIZE - 1, 0);
        if (bytes <= 0)
            break;

        buffer[bytes] = '\0';

        wchar_t wbuffer[BUFFER_SIZE];
        MultiByteToWideChar(CP_ACP, 0, buffer, -1, wbuffer, BUFFER_SIZE);

        AppendText(hEditChat, L"Friend: ");
        AppendText(hEditChat, wbuffer);
        AppendText(hEditChat, L"\r\n");
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{

    switch (msg)
    {

    case WM_CREATE:
        hEditChat = CreateWindowW(
            L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY,
            10, 10, 360, 200,
            hwnd, NULL, NULL, NULL);

        hEditInput = CreateWindowW(
            L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | WS_TABSTOP,
            10, 220, 260, 25,
            hwnd, NULL, NULL, NULL);

        hButtonSend = CreateWindowW(
            L"BUTTON", L"Send",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP,
            280, 220, 90, 25,
            hwnd, (HMENU)1, NULL, NULL);

        SetFocus(hEditInput);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == 1)
        {
            wchar_t wmsg[BUFFER_SIZE];
            GetWindowTextW(hEditInput, wmsg, BUFFER_SIZE);

            if (wcslen(wmsg) > 0)
            {
                char msg[BUFFER_SIZE];
                WideCharToMultiByte(CP_ACP, 0, wmsg, -1, msg, BUFFER_SIZE, NULL, NULL);

                send(sock, msg, strlen(msg), 0);

                AppendText(hEditChat, L"Me: ");
                AppendText(hEditChat, wmsg);
                AppendText(hEditChat, L"\r\n");

                SetWindowTextW(hEditInput, L"");
                SetFocus(hEditInput);
            }
        }
        break;

    case WM_DESTROY:
        closesocket(sock);
        WSACleanup();
        PostQuitMessage(0);
        break;
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{

    WSADATA wsData;
    WSAStartup(MAKEWORD(2, 2), &wsData);

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);

    inet_pton(AF_INET, "172.20.10.3", &server.sin_addr);

    connect(sock, (sockaddr *)&server, sizeof(server));

    std::thread(ReceiveMessages).detach();

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"ChatWindow";

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowW(
        L"ChatWindow", L"Socket Chat GUI",
        WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME,
        CW_USEDEFAULT, CW_USEDEFAULT, 400, 320,
        NULL, NULL, hInstance, NULL);

    ShowWindow(hwnd, nCmdShow);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return 0;
}
