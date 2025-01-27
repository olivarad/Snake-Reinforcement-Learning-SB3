from SnakeEnvironment import SnakeEnvironment

environment = SnakeEnvironment()
episodes = 5000

for episode in range(episodes):
    done = False
    observation = environment.reset()
    while not done:
        random_action = environment.action_space.sample()
        print(f"Action: {random_action}")
        observation, reward, done, _, info = environment.step(random_action)
        print(f"Reward: {reward}")