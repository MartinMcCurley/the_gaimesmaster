import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from collections import deque
import random
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Script started")

# Set up the environment
env = gym.make("ALE/Blackjack-ram-v5")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(f"Environment created. State size: {state_size}, Action size: {action_size}")

# Define the neural network model
def build_model(state_size, action_size):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_size,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Model built")
    return model

# Initialize the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.update_target_model()
        print("DQNAgent initialized")

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("Target model updated")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state, verbose=0)[0]))
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        print(f"Model loaded from {name}")

    def save(self, name):
        self.model.save_weights(name)
        print(f"Model saved to {name}")


# live plotting
def plot_learning_curve(episodes, scores, avg_scores, epsilon):
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, scores, alpha=0.2, color='b')
    plt.plot(episodes, avg_scores, color='r')
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Average Score'])
    ax2 = plt.twinx()
    ax2.plot(episodes, epsilon, color='g')
    ax2.set_ylabel('Epsilon', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.savefig('training_progress.png')
    plt.close()
    
# Training function
def interpret_state(state):
    # For RAM-based observation (128 bytes)
    # Note: This is a simplified interpretation and may need further refinement
    player_score = state[3]  # Assuming player's score is stored in the 4th byte
    dealer_card = state[5]   # Assuming dealer's visible card is stored in the 6th byte
    has_usable_ace = state[7] > 0  # Assuming usable ace information is in the 8th byte
    
    return player_score, dealer_card, has_usable_ace

# Update the environment creation to use the RAM-based observation
env = gym.make("ALE/Blackjack-ram-v5")
state_size = env.observation_space.shape[0]  # Should be 128
action_size = env.action_space.n  # Should be 4

# Update the train_dqn function to use the new interpret_state function
def train_dqn(episodes, checkpoint_interval=100, max_steps=100):
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    scores = []
    avg_scores = []
    epsilon_values = []
    episodes_list = []
    start_time = time.time()
    best_avg_score = float('-inf')
    
    print("Starting training loop")
    for e in range(episodes):
        print(f"Starting episode {e}")
        state = env.reset()[0]
        print(f"Environment reset. Initial state shape: {state.shape}")
        score = 0
        step = 0
        
        while step < max_steps:
            print(f"Episode {e}, Step {step}")
            action = agent.act(state)
            print(f"Action chosen: {action}")
            next_state, reward, done, _, info = env.step(action)
            print(f"Raw next state shape: {next_state.shape}")
            player_score, dealer_card, has_usable_ace = interpret_state(next_state)
            print(f"Interpreted state - Player score: {player_score}, Dealer card: {dealer_card}, Usable ace: {has_usable_ace}")
            print(f"Step result - Reward: {reward}, Done: {done}")
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            step += 1
            
            if done:
                print(f"Episode {e} finished after {step} steps with score {score}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                print("Replay performed")
        
        scores.append(score)
        mean_score = np.mean(scores[-100:])
        avg_scores.append(mean_score)
        epsilon_values.append(agent.epsilon)
        episodes_list.append(e)
        
        if e % 10 == 0 or e == episodes - 1:
            elapsed_time = time.time() - start_time
            print(f"Episode: {e}/{episodes}, Score: {score}, Avg Score: {mean_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}, Time: {elapsed_time:.2f}s")
            plot_learning_curve(episodes_list, scores, avg_scores, epsilon_values)
        
        # Save checkpoint
        if e % checkpoint_interval == 0:
            agent.save(f"blackjack_dqn_checkpoint_{e}.h5")
            print(f"Checkpoint saved at episode {e}")
        
        # Save best model
        if mean_score > best_avg_score:
            best_avg_score = mean_score
            agent.save("blackjack_dqn_best.h5")
            print(f"New best model saved with average score: {best_avg_score:.2f}")
    
    print("\nTraining completed.")
    return agent, scores, avg_scores, epsilon_values

# summary plotting
def plot_summary(scores, avg_scores, epsilon_values):
    plt.figure(figsize=(12, 8))
    episodes = range(len(scores))
    plt.plot(episodes, scores, alpha=0.2, color='b')
    plt.plot(episodes, avg_scores, color='r')
    plt.title('DQN Training Summary')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Average Score'])
    ax2 = plt.twinx()
    ax2.plot(episodes, epsilon_values, color='g')
    ax2.set_ylabel('Epsilon', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.savefig('training_summary.png')
    plt.show()

# Evaluation function
def evaluate_model(model_path, num_episodes=100):
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # Use only exploitation (no random actions)

    total_score = 0
    for e in range(num_episodes):
        state = env.reset()[0]
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            score += reward

        total_score += score
        print(f"Episode {e}: Score = {score}")

    avg_score = total_score / num_episodes
    print(f"Average score over {num_episodes} episodes: {avg_score}")

# Main execution
if __name__ == "__main__":
    episodes = 1000
    print("Starting training...")
    agent, scores, avg_scores, epsilon_values = train_dqn(episodes, checkpoint_interval=100)
    agent.save("blackjack_dqn_final.h5")
    print("Final model saved.")
    
    print("Plotting summary...")
    plot_summary(scores, avg_scores, epsilon_values)
    
    print("\nEvaluating best model...")
    evaluate_model("blackjack_dqn_best.h5")
    
    print("\nEvaluating final model...")
    evaluate_model("blackjack_dqn_final.h5")