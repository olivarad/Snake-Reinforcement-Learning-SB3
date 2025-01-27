import gymnasium as gym  # Change this to gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10, random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnvironment(gym.Env):

    def __init__(self):
        super(SnakeEnvironment, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float32)
        self.np_random = None
        self.render_mode = 'rgb_array'

    def step(self, action):
        self.reward = 0
        self.prev_actions.append(action)
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                    (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
        
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]),
                        (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        if action == 1: # Right
            self.snake_head[0] += 10
        elif action == 0: # Left
            self.snake_head[0] -= 10
        elif action == 2: # Up
            self.snake_head[1] += 10
        elif action == 3: # Down
            self.snake_head[1] -= 10
        if (self.snake_head[0] == self.snake_position[1][0] and self.snake_head[1] == self.snake_position[1][1]):
                self.reward = -100000

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('a', self.img)
            self.reward += -100
            self.done = True
        else:
            # Reward for moving closer to apple (distance-based)
            distance_to_apple = np.sqrt(self.apple_delta_x**2 + self.apple_delta_y**2)
            reward_for_closeness = int(707 / distance_to_apple) / 10 # 707 is about max distance

            # Reward for eating apple (length of snake + bonus)
            if self.snake_head == self.apple_position:
                self.reward = 1
            else:
                self.reward = reward_for_closeness

        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        self.apple_delta_x = self.apple_position[0] - head_x
        self.apple_delta_y = self.apple_position[1] - head_y
        self.calc_min_and_max_apple_distances()

        # Create observation:
        observation = [head_x, head_y, self.apple_delta_x, self.apple_delta_y, snake_length, action] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float32)

        # Return five values: obs, reward, terminated, truncated, info
        return observation, self.reward, self.done, False, info

    def reset(self, seed=None, options=None):
        # Set the seed if provided
        self.np_random, seed = seeding.np_random(seed)
        
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        
        # Use np_random.integers instead of np_random.randint
        self.apple_position = [self.np_random.integers(1, 50) * 10, self.np_random.integers(1, 50) * 10]
        
        self.score = 0
        self.reward = 0
        self.min_apple_distance = -1
        self.max_apple_distance = -1
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        self.apple_delta_x = self.apple_position[0] - head_x
        self.apple_delta_y = self.apple_position[1] - head_y
        self.calc_min_and_max_apple_distances()

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [head_x, head_y, self.apple_delta_x, self.apple_delta_y, snake_length, 0] + list(self.prev_actions)

        # Ensure the observation is of dtype float32
        observation = np.array(observation, dtype=np.float32)

        # Return the observation and an empty dictionary (info)
        return observation, {}
    
    def render(self):
        return self.img
    def calc_min_and_max_apple_distances(self):
        distance_to_apple = np.sqrt(self.apple_delta_x**2 + self.apple_delta_y**2)
        if (self.min_apple_distance == -1): # initial calculation
            self.min_apple_distance = distance_to_apple
            self.max_apple_distance = distance_to_apple
        else:
            if distance_to_apple > self.max_apple_distance:
                self.max_apple_distance = distance_to_apple
            if distance_to_apple < self.min_apple_distance:
                self.min_apple_distance = distance_to_apple