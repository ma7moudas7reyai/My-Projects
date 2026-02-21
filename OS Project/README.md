# 💬 Operating Systems Chat System  
### Shared Memory & Socket Programming Project

## 📌 Overview

This project demonstrates core Operating Systems concepts including:

- Inter-Process Communication (IPC)
- Shared Memory
- Semaphores
- Multithreading
- Socket Programming (TCP)
- Client-Server Architecture

The system consists of two main implementations:

1️⃣ Shared Memory Chat (Local IPC)  
2️⃣ Socket-Based Chat (Network Communication)

Both console and GUI versions are implemented using Windows API.

---

# 🧠 Part 1 – Shared Memory Chat (IPC)

## 🔹 Concepts Implemented

- Windows Shared Memory (CreateFileMapping)
- Memory Mapping (MapViewOfFile)
- Semaphore Synchronization
- Reader-Writer Model
- Multithreading
- WinAPI GUI

## 🔹 How It Works

- Processes communicate using a shared memory buffer.
- A named semaphore ensures synchronized access.
- Messages are written and read safely without race conditions.
- GUI version implemented using Win32 API.

---

# 🌐 Part 2 – Socket-Based Chat System

## 🔹 Concepts Implemented

- TCP Socket Programming
- Multi-client server
- Thread-per-client handling
- Mutex for shared resource protection
- Winsock2 API
- Client-Server architecture
- GUI and Console versions

## 🔹 Server Features

- Listens on port 54000
- Accepts multiple clients
- Broadcasts messages to all connected clients
- Removes disconnected clients safely

## 🔹 Client Features

- Connects to server
- Sends and receives messages
- Threaded receive handling
- GUI interface using WinAPI

---

# 🛠 Technologies Used

- C++
- Windows API
- Winsock2
- Multithreading (std::thread)
- Synchronization (Semaphore & Mutex)
- TCP Networking

---

# 🧩 Operating Systems Concepts Covered

- Inter-Process Communication (IPC)
- Process Synchronization
- Critical Section Management
- Thread Management
- Shared Resources Protection
- Networking in OS
- Client-Server Model

---

# 🎯 Key Features

- Named Shared Memory Communication
- Named Semaphore Synchronization
- Multi-threaded Message Handling
- Multi-client TCP Server
- GUI-based Chat Interfaces
- Safe concurrent access control

---

# 📚 What I Learned

- Difference between IPC and Networking communication
- Handling race conditions
- Synchronization using semaphores and mutex
- Multi-threaded server design
- Windows memory management
- Implementing real-time communication systems

---

# 🚀 Future Improvements

- Add encryption for messages
- Add authentication system
- Implement message history storage
- Cross-platform socket version (Linux)
- Add asynchronous I/O model

---

## 👤 Author
Mahmoud Ashrey  
Computer Science Student
