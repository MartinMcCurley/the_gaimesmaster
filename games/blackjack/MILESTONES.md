## Milestone 1: Environment Setup
[x] 1.1: Create a new virtual environment and activate it.
[x] 1.2: Install all necessary dependencies, including OpenAI Gymnasium and the Atari Blackjack environment.
[x] 1.3: Verify installations by importing libraries in a Python script without errors.

## Milestone 2: Human Interaction with the Game
[] 2.1: Write a Python script to launch the Blackjack game environment.
[] 2.2: Implement functionality to allow a human player to interact with the game using keyboard inputs.
[] 2.3: Play several rounds manually to ensure the game functions correctly and understand gameplay mechanics.

## Milestone 3: Understanding the Game Environment
[] 3.1: Explore and document the observation space (e.g., player's hand, dealer's hand, usable ace).
[] 3.2: Explore and document the action space (e.g., hit, stand).
[] 3.3: Identify all relevant game variables, including bet sizes and rules variations.

## Milestone 4: Agent Interaction with Random Actions
[] 4.1: Write a Python script that allows an agent to interact with the game using random actions.
[] 4.2: Run the script to ensure the agent can play multiple episodes without errors.
[] 4.3: Collect and log basic gameplay data for analysis (e.g., win/loss outcomes).

## Milestone 5: Mapping Inputs and Outputs
[] 5.1: Define how inputs (actions) are mapped in the game environment.
[] 5.2: Implement a function or class method that translates agent decisions into game inputs.
[] 5.3: Ensure that the agent can perform all valid actions in the game (e.g., hit, stand) correctly.

## Milestone 6: Accessing and Utilizing Game State Information
[] 6.1: Modify the agent to interpret and utilize key game state information from observations.
[] 6.2: Ensure the agent can access and understand player hand, dealer hand, and bet size.
[] 6.3: Validate that the agent's decisions can be based on this information.

## Milestone 7: Implementing a Basic Strategy Agent
[] 7.1: Code a hardcoded agent that follows a basic Blackjack strategy chart.
[] 7.2: Run simulations to compare the basic strategy agent's performance against the random agent.
[] 7.3: Analyze results to confirm the basic strategy agent performs better than random.

## Milestone 8: Setting Up Reinforcement Learning Framework
[] 8.1: Choose an appropriate RL algorithm (e.g., Q-Learning, Deep Q-Network).
[] 8.2: Implement the RL algorithm in a new Python script or module.
[] 8.3: Define the neural network architecture (if using deep learning) and hyperparameters.

## Milestone 9: Training the Agent
[] 9.1: Integrate the RL agent with the game environment.
[] 9.2: Start the training process, allowing the agent to learn from interactions.
[] 9.3: Implement checkpointing to save the model at regular intervals.

## Milestone 10: Monitoring Training Progress
[] 10.1: Create visualizations (in separate Python files) to track training metrics (e.g., average reward per episode, loss values).
[] 10.2: Log training data to files for later analysis.
[] 10.3: Adjust visualization scripts to update in real-time if possible.

## Milestone 11: Evaluating the Trained Agent
[] 11.1: After training, test the agent over a significant number of episodes without learning (evaluation mode).
[] 11.2: Collect performance metrics such as win rate, average reward, and compare them to benchmarks (random agent, basic strategy agent).
[] 11.3: Analyze whether the agent meets or exceeds the known optimal performance percentage.

## Milestone 12: Optimization and Hyperparameter Tuning
[] 12.1: Experiment with different hyperparameters to improve performance (e.g., learning rate, exploration rate).
[] 12.2: Implement advanced techniques if necessary (e.g., experience replay, target networks).
[] 12.3: Retrain and reevaluate the agent after each significant change.

## Milestone 13: Final Evaluation and Project Completion
[] 13.1: Confirm that the agent achieves the target performance metric (e.g., optimal win rate).
[] 13.2: Generate final visualizations showing the agent's performance over time.
[] 13.3: Document all findings, including successes, challenges, and potential improvements.

## Milestone 14: Code Refactoring and Documentation
[] 14.1: Organize code into modules and separate Python files as appropriate.
[] 14.2: Add comments and docstrings to explain code functionality.
[] 14.3: Prepare a README file with instructions on how to run the scripts and reproduce results.

## Milestone 15: Project Presentation
[] 15.1: Prepare a presentation or report summarizing the project goals, methodology, results, and conclusions.
[] 15.2: Include visualizations and code snippets to illustrate key points.
[] 15.3: Review the entire project to ensure completeness and accuracy before considering it finished.