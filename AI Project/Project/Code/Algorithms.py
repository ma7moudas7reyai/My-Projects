# Depth First Search (DFS)

# Graph representation
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

def dfs(node, goal, visited):
    # Goal Test
    if node == goal:
        print("Goal reached:", node)
        return True

    visited.add(node)
    print("Visiting:", node)

    for neighbour in graph[node]:
        if neighbour not in visited:
            if dfs(neighbour, goal, visited):
                return True

    return False


# تشغيل DFS
visited = set()
dfs('A', 'E', visited)

# Greedy Best-First Search (GBFS)
import heapq

# Graph
graph = {
    1: [2, 3],
    2: [4, 5],
    3: [5],
    4: [6],
    5: [6],
    6: [7],
    7: []
}

# Heuristic (costs)
heuristic = {
    1: 8,
    2: 7,
    3: 5,
    4: 4,
    5: 3,
    6: 0,
    7: 2
}

def greedy_best_first_search(start, goal):
    visited = set()
    priority_queue = []

    # (heuristic value, node)
    heapq.heappush(priority_queue, (heuristic[start], start))

    while priority_queue:
        _, current = heapq.heappop(priority_queue)

        if current == goal:
            print("Goal reached:", current)
            return  

        if current in visited:
            continue

        print("Visiting:", current)
        visited.add(current)

        for neighbour in graph[current]:
            if neighbour not in visited:
                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbour], neighbour)
                )

    print("Goal not found")


# تشغيل Greedy Search
greedy_best_first_search(1, 6)
