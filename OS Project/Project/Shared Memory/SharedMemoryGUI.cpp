#include <windows.h>
#include <thread>
#include <cstring>
#include <cstdio>

#define SHM_NAME "Local\\ChatSharedMemory"
#define SEM_NAME "Local\\ChatSemaphore"

struct ChatBuffer
{
    char username[32];
    char message[256];
    bool hasNewMessage;
};

ChatBuffer *buffer = nullptr;
HANDLE hMapFile = NULL, hSemaphore = NULL;

HWND hUserEdit, hChatBox, hMsgEdit, hSendBtn;
WNDPROC oldEditProc = nullptr;

char username[32] = "";

HFONT hFont = CreateFontA(
    18, 0, 0, 0,
    FW_NORMAL,
    FALSE, FALSE, FALSE,
    ANSI_CHARSET,
    OUT_DEFAULT_PRECIS,
    CLIP_DEFAULT_PRECIS,
    DEFAULT_QUALITY,
    DEFAULT_PITCH | FF_SWISS,
    "Segoe UI");

void sendMessage()
{
    char msgText[256] = {0};
    GetWindowTextA(hMsgEdit, msgText, 256);
    GetWindowTextA(hUserEdit, username, 32);

    if (strlen(msgText) == 0 || strlen(username) == 0)
        return;

    WaitForSingleObject(hSemaphore, INFINITE);

    strcpy(buffer->username, username);
    strcpy(buffer->message, msgText);
    buffer->hasNewMessage = true;

    ReleaseSemaphore(hSemaphore, 1, NULL);

    SetWindowTextA(hMsgEdit, "");
    SetFocus(hMsgEdit);
}

LRESULT CALLBACK EditProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_KEYDOWN && wParam == VK_RETURN)
    {
        sendMessage();
        return 0;
    }
    return CallWindowProc(oldEditProc, hwnd, msg, wParam, lParam);
}

void receiveThread()
{
    static char lastMsg[256] = "";

    while (true)
    {
        WaitForSingleObject(hSemaphore, INFINITE);

        if (buffer->hasNewMessage &&
            strcmp(buffer->message, lastMsg) != 0)
        {

            char line[320];
            sprintf(line, "[%s]: %s\r\n",
                    buffer->username, buffer->message);

            SendMessageA(hChatBox, EM_SETSEL, -1, -1);
            SendMessageA(hChatBox, EM_REPLACESEL, 0, (LPARAM)line);

            strcpy(lastMsg, buffer->message);
        }

        ReleaseSemaphore(hSemaphore, 1, NULL);
        Sleep(100);
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {

    case WM_ERASEBKGND:
    {
        HDC hdc = (HDC)wParam;
        RECT rc;
        GetClientRect(hwnd, &rc);
        HBRUSH brush = CreateSolidBrush(RGB(230, 240, 255));
        FillRect(hdc, &rc, brush);
        DeleteObject(brush);
        return 1;
    }

    case WM_DRAWITEM:
    {
        DRAWITEMSTRUCT *dis = (DRAWITEMSTRUCT *)lParam;
        if (dis->CtlID == 1)
        {
            HBRUSH br = CreateSolidBrush(RGB(0, 120, 215));
            FillRect(dis->hDC, &dis->rcItem, br);
            SetTextColor(dis->hDC, RGB(255, 255, 255));
            SetBkMode(dis->hDC, TRANSPARENT);
            DrawTextA(dis->hDC, "Send", -1,
                      &dis->rcItem,
                      DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            DeleteObject(br);
            return TRUE;
        }
        break;
    }

    case WM_COMMAND:
        if (LOWORD(wParam) == 1)
        {
            sendMessage();
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow)
{

    hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL,
        PAGE_READWRITE, 0,
        sizeof(ChatBuffer),
        SHM_NAME);

    buffer = (ChatBuffer *)MapViewOfFile(
        hMapFile, FILE_MAP_ALL_ACCESS,
        0, 0, sizeof(ChatBuffer));

    hSemaphore = CreateSemaphoreA(NULL, 1, 1, SEM_NAME);

    WNDCLASSA wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "SharedMemChat";

    RegisterClassA(&wc);

    HWND hwnd = CreateWindowA(
        "SharedMemChat",
        "Shared Memory Chat",
        WS_OVERLAPPEDWINDOW,
        200, 200, 520, 520,
        NULL, NULL, hInst, NULL);

    hUserEdit = CreateWindowA("EDIT", "",
                              WS_CHILD | WS_VISIBLE | WS_BORDER,
                              10, 10, 480, 28,
                              hwnd, NULL, hInst, NULL);

    hChatBox = CreateWindowA("EDIT", "",
                             WS_CHILD | WS_VISIBLE | WS_BORDER |
                                 ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                             10, 50, 480, 340,
                             hwnd, NULL, hInst, NULL);

    hMsgEdit = CreateWindowA("EDIT", "",
                             WS_CHILD | WS_VISIBLE | WS_BORDER,
                             10, 405, 380, 32,
                             hwnd, NULL, hInst, NULL);

    hSendBtn = CreateWindowA(
        "BUTTON", "",
        WS_CHILD | WS_VISIBLE | BS_OWNERDRAW,
        400, 405, 90, 32,
        hwnd, (HMENU)1, hInst, NULL);

    SendMessage(hUserEdit, WM_SETFONT, (WPARAM)hFont, TRUE);
    SendMessage(hChatBox, WM_SETFONT, (WPARAM)hFont, TRUE);
    SendMessage(hMsgEdit, WM_SETFONT, (WPARAM)hFont, TRUE);

    oldEditProc = (WNDPROC)SetWindowLongPtr(
        hMsgEdit, GWLP_WNDPROC, (LONG_PTR)EditProc);

    ShowWindow(hwnd, nCmdShow);
    SetFocus(hMsgEdit);

    std::thread recv(receiveThread);
    recv.detach();

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    return 0;
}
