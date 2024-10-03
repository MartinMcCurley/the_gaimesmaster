import gymnasium as gym
import pygame
import numpy as np

def init_game():
    pygame.init()
    env = gym.make("ALE/Blackjack-v5", render_mode="rgb_array")
    screen = pygame.display.set_mode((160*3, 210*3))
    pygame.display.set_caption("Atari Blackjack")
    font = pygame.font.Font(None, 36)
    return env, screen, font

def handle_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                return 1  # FIRE
            elif event.key == pygame.K_UP:
                return 2  # UP
            elif event.key == pygame.K_DOWN:
                return 3  # DOWN
            elif event.key == pygame.K_ESCAPE:
                return None  # Quit
    return 0  # NOOP

def play_blackjack():
    env, screen, font = init_game()
    observation, info = env.reset()
    
    clock = pygame.time.Clock()
    running = True
    action_text = ""
    
    while running:
        action = handle_input()
        if action is None:
            running = False
            break
        
        action_mapping = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "DOWN"
        }
        action_text = action_mapping.get(action, "Unknown")

        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the game
        frame = env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        
        # Display action text
        text_surface = font.render(f"Action: {action_text}", True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print("Game Over!")
            observation, info = env.reset()
        
        clock.tick(60)  # Limit to 60 FPS
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_blackjack()