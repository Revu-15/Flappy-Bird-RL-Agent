import numpy as np

class BaseAgent:
    """
    Abstract Base class for RL agents (RL-Glue style).
    You may extend or replace this with your course-provided version.
    """

    def agent_init(self, agent_info={}):
        raise NotImplementedError

    def agent_start(self, observation):
        raise NotImplementedError

    def agent_step(self, reward, observation):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        return None


class FlappyAgent(BaseAgent):
    """
    Flappy Bird Agent supporting:
    - Q-Learning
    - Expected SARSA
    """

    def agent_init(self, agent_info={}):
        self.alpha = agent_info.get("alpha", 0.1)
        self.gamma = agent_info.get("gamma", 0.99)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.algorithm = agent_info.get("algorithm", "Q-Learning")

        self.num_actions = agent_info.get("num_actions", 2)

        # Q-table dictionary
        self.Q = {}

    def _epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon or state not in self.Q:
            return np.random.randint(self.num_actions)

        return np.argmax(self.Q[state])

    def agent_start(self, observation):
        state = tuple(np.round(observation, 1))
        self.Q.setdefault(state, np.zeros(self.num_actions))

        action = self._epsilon_greedy(state)

        self.last_state = state
        self.last_action = action

        return action

    def agent_step(self, reward, observation):
        next_state = tuple(np.round(observation, 1))
        self.Q.setdefault(next_state, np.zeros(self.num_actions))

        a = self.last_action
        s = self.last_state
        s_next = next_state

        # -------- Q-Learning Update --------
        if self.algorithm == "Q-Learning":
            td_target = reward + self.gamma * np.max(self.Q[s_next])
            self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

        # -------- Expected SARSA Update --------
        else:
            policy = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
            best = np.argmax(self.Q[s_next])
            policy[best] += 1 - self.epsilon

            expected_value = np.dot(policy, self.Q[s_next])

            td_target = reward + self.gamma * expected_value
            self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

        # choose next action
        action = self._epsilon_greedy(s_next)

        self.last_state = s_next
        self.last_action = action

        return action

    def agent_end(self, reward):
        s = self.last_state
        a = self.last_action

        td_target = reward  # terminal state

        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    def agent_message(self, message):
        if message == "get-Q":
            return self.Q
        return None
