import pygame
import numpy as np
from stable_baselines3 import DQN
pygame.init()

width = 480
height = 800

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong")

paddle_width = 100
paddle_height = 15

player_x = (width - paddle_width) // 2
player_y = height - 40
bob_x = (width - paddle_width) // 2
bob_y = 25

ball_size = 12

ball_x = (width - ball_size) // 2
ball_y = (height - ball_size) // 2

player_speed = 7
bob_speed = 6

ball_speed_x = 3
ball_speed_y = 5

player_score = 0
bob_score = 0

game_over = False
winner = ""
winner_font = pygame.font.Font(None, 65)
restart_font = pygame.font.Font(None, 30)

score_font = pygame.font.Font(None, 30)

running = True

clock = pygame.time.Clock()

model = DQN.load("bob_dqn_v5")

while running:
    screen.fill((10, 10, 25))
    pygame.draw.rect(screen, (0, 220, 255), (player_x, player_y, paddle_width, paddle_height))
    pygame.draw.rect(screen, (100, 0, 255), (bob_x, bob_y, paddle_width, paddle_height))
    pygame.draw.circle(screen, (255, 255, 255), (ball_x + ball_size // 2, ball_y + ball_size // 2), ball_size // 2)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                player_score = 0
                bob_score = 0
                game_over = False
                winner = ""

                player_x = (width - paddle_width) // 2
                bob_x = (width - paddle_width) // 2
                ball_x = (width - ball_size) // 2
                ball_y = (height - ball_size) // 2
                ball_speed_x = 3
                ball_speed_y = 5
    if game_over:
        winner_text = winner_font.render(
            f"{winner} Wins!",
            True,
            (255, 255, 255)
        )
        restart_text = restart_font.render(
            "Press R to restart or ESC to quit",
            True,
            (180, 180, 180)
        )
        screen.blit(
            winner_text,
            winner_text.get_rect(center=(width // 2, height // 2 - 30))
        )
        screen.blit(
            restart_text,
            restart_text.get_rect(center=(width // 2, height // 2 + 30))
        )
        pygame.display.flip()
        clock.tick(60)
        continue
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
    if keys[pygame.K_RIGHT]:
        player_x += player_speed
    if player_x < 0:
        player_x = 0
    if player_x > width - paddle_width:
        player_x = width - paddle_width
    if ball_speed_y < 0:
        predicted_x = ball_x
        predicted_speed_x = ball_speed_x
        frames = int(np.ceil(
            (ball_y - 40) / abs(ball_speed_y)
        ))
        for _ in range(max(0, frames)):
            predicted_x += predicted_speed_x
            if predicted_x <= 0:
                predicted_x = 0
                predicted_speed_x *= -1
            elif predicted_x >= width - ball_size:
                predicted_x = width - ball_size
                predicted_speed_x *= -1
        obs = np.array([
            ball_x,
            ball_y,
            ball_speed_x,
            ball_speed_y,
            bob_x,
            predicted_x
        ], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        if action == 0:
            bob_x -= bob_speed
        elif action == 2:
            bob_x += bob_speed

    bob_x = max(0, min(width - paddle_width, bob_x))
    ball_x += ball_speed_x
    ball_y += ball_speed_y


    if ball_x <= 0 or ball_x >= width - ball_size:
        ball_speed_x *= -1
    ball_rect = pygame.Rect(ball_x, ball_y, ball_size, ball_size)
    player_rect = pygame.Rect(player_x, player_y, paddle_width, paddle_height)
    bob_rect = pygame.Rect(bob_x, bob_y, paddle_width, paddle_height)
    if ball_rect.colliderect(player_rect) and ball_speed_y > 0:
        hit_position = ( ball_x + ball_size / 2) - (player_x + paddle_width / 2)
        hit_offset = hit_position / (paddle_width / 2)
        hit_offset = max(-1, min(1, hit_offset))
        ball_speed_x = hit_offset * 6
        ball_speed_y = -abs(ball_speed_y)
    if ball_rect.colliderect(bob_rect) and ball_speed_y < 0:
        hit_position = (ball_x + ball_size / 2) - (bob_x + paddle_width / 2)
        hit_offset = hit_position / (paddle_width / 2)
        hit_offset = max(-1, min(1, hit_offset))
        ball_speed_x = hit_offset * 6
        ball_speed_y = abs(ball_speed_y)
    if ball_y < 0:
        player_score += 1
        if player_score >= 7:
            game_over = True
            winner = "You"
        ball_x = (width - ball_size) // 2
        ball_y = (height - ball_size) // 2
        ball_speed_x = 3
        ball_speed_y = 5
    if ball_y > height:
        bob_score += 1
        if bob_score >= 7:
            game_over = True
            winner = "Bob"
        ball_x = (width - ball_size) // 2
        ball_y = (height - ball_size) // 2
        ball_speed_x = 3
        ball_speed_y = -5
    bob_text = score_font.render(f"Bob: {bob_score}", True, (180, 0, 225))
    screen.blit(bob_text, (20, 65))
    player_text = score_font.render(f"You: {player_score}", True, (0, 220, 255))
    screen.blit(player_text, (20, height - 110))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()