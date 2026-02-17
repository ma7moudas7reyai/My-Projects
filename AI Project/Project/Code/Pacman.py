import heapq

# 0 = path, 1 = wall
maze = [
    [1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,1],
    [1,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1],
]

ROWS = len(maze)
COLS = len(maze[0])


class Game:
    def __init__(self, level):
        self.level = level
        self.reset()

    def reset(self):
        self.pacman = [1, 1]
        self.ghost = [7, 7]
        self.score = 0
        self.game_over = False

        self.food = {(i, j) for i in range(ROWS) for j in range(COLS)
                     if maze[i][j] == 0 and (i, j) != (1, 1)}

        if self.level == "easy":
            self.ghost_delay = 3
        elif self.level == "medium":
            self.ghost_delay = 2
        else:  
            self.ghost_delay = 1

        self.tick = 0

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance

    def astar(self, start, goal):
        open_set = [(0, start)]
        came = {}
        g = {tuple(start): 0}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == goal:
                path = []
                while tuple(cur) in came:
                    path.append(cur)
                    cur = came[tuple(cur)]
                return path[::-1]

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cur[0]+dx, cur[1]+dy 
                if maze[nx][ny] == 0:
                    ng = g[tuple(cur)] + 1
                    if (nx, ny) not in g or ng < g[(nx, ny)]:
                        came[(nx, ny)] = cur
                        g[(nx, ny)] = ng
                        f = ng + self.heuristic([nx, ny], goal)
                        heapq.heappush(open_set, (f, [nx, ny]))
        return []

    def move_pacman(self, dx, dy):
        if self.game_over:
            return
        nx, ny = self.pacman[0]+dx, self.pacman[1]+dy
        if maze[nx][ny] == 0:
            self.pacman = [nx, ny]
            if (nx, ny) in self.food:
                self.food.remove((nx, ny))
                self.score += 10

    def move_ghost(self):
        if self.game_over:
            return

        self.tick += 1
        if self.tick % self.ghost_delay != 0:
            return

        path = self.astar(self.ghost, self.pacman)
        if path:
            self.ghost = path[0]

    def update(self):
        self.move_ghost()
        if self.pacman == self.ghost:
            self.game_over = True
        if not self.food:
            self.game_over = True

    def win(self):
        return not self.food

    def lose(self):
        return self.pacman == self.ghost
