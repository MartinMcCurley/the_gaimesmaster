import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def run(episodes, is_training=False, render=True):
    print("Initializing environment...")
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True,
                   render_mode='human' if render else None)
    goal_state_index = np.argwhere(env.unwrapped.desc == b'G')[0][0] * env.desc.shape[1] + \
                       np.argwhere(env.unwrapped.desc == b'G')[0][1]
    print(f"Goal state index: {goal_state_index}")

    if is_training:
        print("Starting training...")
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
        alpha = 0.9  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 1.0
        epsilon_decay_rate = 0.0001  # Linear decay
        rng = np.random.default_rng()
    else:
        print("Loading trained Q-table...")
        with open('frozen_lake_8x8_q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
        epsilon = 0.05  # Small exploration rate during testing
        rng = np.random.default_rng()

    rewards_per_episode = []
    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        # Adjust epsilon and alpha
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            if epsilon == 0:
                alpha = 0.0001  # Reduce learning rate when exploration stops

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # Standard Q-learning update
                q_table[state, action] += alpha * (
                    reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
                )

            state = next_state
            total_reward += reward

            if render and not is_training:
                env.render()
                time.sleep(0.5)

        rewards_per_episode.append(total_reward)

        if is_training and (episode + 1) % 10000 == 0:
            average_reward = np.mean(rewards_per_episode[-10000:])
            success_rate = np.mean(rewards_per_episode[-10000:]) * 100
            elapsed_time = time.time() - start_time
            print(f"Episode {episode + 1}, Average Reward: {average_reward:.4f}, "
                  f"Success Rate: {success_rate:.2f}%, Epsilon: {epsilon:.4f}, "
                  f"Time Elapsed: {elapsed_time:.2f} seconds")
            start_time = time.time()

    env.close()

    if is_training:
        print("Training complete. Saving Q-table...")
        with open("frozen_lake_8x8_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
        # Plot cumulative successes
        sum_rewards = np.cumsum(rewards_per_episode)
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Successes')
        plt.title('Cumulative Successes Over Episodes')
        plt.show()
        visualize_policy(q_table, env)
    else:
        return rewards_per_episode

def visualize_policy(q_table, env):
    action_mapping = {
        0: '←',  # Left
        1: '↓',  # Down
        2: '→',  # Right
        3: '↑'   # Up
    }
    policy = np.array([action_mapping[np.argmax(q_table[s])] if np.max(q_table[s]) > 0 else ' ' 
                       for s in range(env.observation_space.n)])
    policy = policy.reshape(env.desc.shape)
    print("Derived Policy:")
    for row in policy:
        print(' '.join(row))

if __name__ == '__main__':
    try:
        # To train
        # run(100000, is_training=True, render=False)

        # To test
        # Uncomment the line below to test the trained agent
        run(5, is_training=False, render=True)
    except Exception as e:
        print(f"An error occurred: {e}")
