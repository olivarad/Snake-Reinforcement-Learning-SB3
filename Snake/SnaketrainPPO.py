import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

# Registering the SnakeEnvironment
register(
    id='SnakeEnvironment-v0',
    entry_point='SnakeEnvironment:SnakeEnvironment',
    max_episode_steps=10000,
)

# Config for training
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "SnakeEnvironment-v0",
}

# Initialize WandB run
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Create the environment and wrap it with monitor
def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # Monitor the environment for episode stats (e.g., reward)
    return env

env = DummyVecEnv([make_env])  # Wrap the environment in DummyVecEnv for compatibility

# Initialize PPO model
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

# Add custom WandBCallback to log episode reward
class CustomWandbCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_reward = 0  # Initialize cumulative episode reward

    def _on_step(self) -> bool:
        # Retrieve current reward and done flag
        reward = self.locals.get('rewards', 0)  # Current reward for the step
        done = self.locals.get('dones', [False])[0]  # Check if episode is done (done is a list)

        # Update cumulative episode reward
        self.episode_reward += reward

        # When the episode ends (done is True), log both episode_reward and episode_apples
        if done:
            # Assuming the environment has a method or attribute to get the current score (e.g., env.score or env.get_score())
            score = self.locals['env'].get_attr('score')[0]  # Get the score from the environment
            min_apple_distance = self.locals['env'].get_attr('min_apple_distance')[0]
            max_apple_distance = self.locals['env'].get_attr('max_apple_distance')[0]

            # Log both "episode_apples" and "episode_reward"
            wandb.log({
                'episode_apples': score,  # Log the score as "episode apples"
                'episode_reward': self.episode_reward,  # Log the sum of rewards for the episode
                'min_apple_distance': min_apple_distance,
                'max_apple_distance': max_apple_distance
            })

            self.episode_reward = 0  # Reset episode reward for the next episode

        return True

# Train the model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CustomWandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

# Finish WandB run after training
run.finish()