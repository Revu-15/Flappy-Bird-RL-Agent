
import numpy as np

class FlappyBirdEnvSimple:
    def __init__(self):
        self.gravity = 0.5
        self.jump_strength = -8
        self.bird_y = 200
        self.bird_velocity = 0

        self.pipe_x = 400
        self.pipe_gap = 120
        self.pipe_width = 60
        self.pipe_height = np.random.randint(50, 250)

        self.action_space_n = 2  # 0 = do nothing, 1 = jump

    def reset(self):
        self.bird_y = 200
        self.bird_velocity = 0

        self.pipe_x = 400
        self.pipe_height = np.random.randint(50, 250)

        return self._get_state()

    def step(self, action):
        # Action: flap
        if action == 1:
            self.bird_velocity = self.jump_strength

        # Apply physics
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.pipe_x -= 3

        # Reset pipe
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = 400
            self.pipe_height = np.random.randint(50, 250)

        # Collision detection
        done = False
        reward = 1  # small reward for staying alive

        if self.bird_y <= 0 or self.bird_y >= 400:
            done = True
            reward = -100

        if self.pipe_x < 50 < self.pipe_x + self.pipe_width:
            if not (self.pipe_height < self.bird_y < self.pipe_height + self.pipe_gap):
                done = True
                reward = -100

        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_height])
