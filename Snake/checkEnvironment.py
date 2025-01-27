from stable_baselines3.common.env_checker import check_env
from SnakeEnvironment import SnakeEnvironment

environment = SnakeEnvironment()
check_env(environment)