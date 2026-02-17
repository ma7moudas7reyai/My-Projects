import pygame
from Pacman import Game, maze, ROWS, COLS

pygame.init()

CELL = 60
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + 120

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pacman")

font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 64)
clock = pygame.time.Clock()

# States
MENU = "menu"
PLAYING = "playing"
GAME_OVER = "game_over"

state = MENU
game = None
level = None

# Colors
BLACK = (0,0,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
RED = (255,0,0)
WHITE = (255,255,255)
GRAY = (180,180,180)

GREEN = (0,200,0)
DARK_GREEN = (0,150,0)
ORANGE = (255,165,0)
DARK_ORANGE = (200,130,0)
DARK_RED = (180,0,0)


def draw_button(text, x, y, w, h, color, hover, mouse):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, hover if rect.collidepoint(mouse) else color,
                     rect, border_radius=12)
    label = font.render(text, True, BLACK)
    screen.blit(label, (x + w//2 - label.get_width()//2,
                         y + h//2 - label.get_height()//2))
    return rect


def draw_menu():
    screen.fill(BLACK)
    mouse = pygame.mouse.get_pos()

    title = big_font.render("PACMAN", True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 40))

    sub = font.render("Choose Difficulty: ", True, GRAY)
    screen.blit(sub, (WIDTH//2 - sub.get_width()//2, 120))

    easy = draw_button("EASY", WIDTH//2 - 120, 200, 240, 55,
                       GREEN, DARK_GREEN, mouse)
    medium = draw_button("MEDIUM", WIDTH//2 - 120, 270, 240, 55,
                         ORANGE, DARK_ORANGE, mouse)
    hard = draw_button("HARD", WIDTH//2 - 120, 340, 240, 55,
                       RED, DARK_RED, mouse)

    pygame.display.flip()
    return easy, medium, hard


def draw_game():
    screen.fill(BLACK)

    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == 1:
                pygame.draw.rect(screen, BLUE,
                                 (c*CELL, r*CELL, CELL, CELL), 2)
            elif (r, c) in game.food:
                pygame.draw.circle(screen, WHITE,
                                   (c*CELL+CELL//2, r*CELL+CELL//2), 5)

    pygame.draw.circle(screen, YELLOW,
        (game.pacman[1]*CELL+CELL//2, game.pacman[0]*CELL+CELL//2),
        CELL//2-6)

    pygame.draw.circle(screen, RED,
        (game.ghost[1]*CELL+CELL//2, game.ghost[0]*CELL+CELL//2),
        CELL//2-6)

    score = font.render(f"Score: {game.score}", True, WHITE)
    screen.blit(score, (10, HEIGHT-90))

    pygame.display.flip()


def draw_game_over():
    screen.fill(BLACK)
    msg = "YOU WIN!" if game.win() else "GAME OVER"
    color = YELLOW if game.win() else RED
    text = big_font.render(msg, True, color)
    screen.blit(text, (WIDTH//2 - text.get_width()//2, 200))

    hint = font.render("Press R to return to Menu", True, WHITE)
    screen.blit(hint, (WIDTH//2 - hint.get_width()//2, 280))

    pygame.display.flip()


running = True
while running:
    clock.tick(8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == MENU:
            if event.type == pygame.MOUSEBUTTONDOWN:
                easy, medium, hard = draw_menu()
                if easy.collidepoint(event.pos):
                    level = "easy"
                elif medium.collidepoint(event.pos):
                    level = "medium"
                elif hard.collidepoint(event.pos):
                    level = "hard"

                if level:
                    game = Game(level)
                    state = PLAYING

        elif state == PLAYING:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    game.move_pacman(-1,0)
                elif event.key == pygame.K_s:
                    game.move_pacman(1,0)
                elif event.key == pygame.K_a:
                    game.move_pacman(0,-1)
                elif event.key == pygame.K_d:
                    game.move_pacman(0,1)

        elif state == GAME_OVER:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = MENU
                game = None
                level = None

    if state == PLAYING:
        game.update()
        if game.win() or game.lose():
            state = GAME_OVER

    if state == MENU:
        draw_menu()
    elif state == PLAYING:
        draw_game()
    else:
        draw_game_over()

pygame.quit()
