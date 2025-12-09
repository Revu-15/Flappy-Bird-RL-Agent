import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from flappy_bird_env_simple import FlappyBirdEnvSimple
from plot_utils import plot_FB_score

# ---------------------------
# Q-Learning and Expected SARSA Functions
# ---------------------------
def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon or state not in Q:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

def expected_sarsa_update(Q, state, action, reward, next_state, alpha, gamma, epsilon):
    policy = np.ones(2) * (epsilon / 2)
    policy[np.argmax(Q[next_state])] += 1 - epsilon
    expected_value = np.dot(policy, Q[next_state])
    Q[state][action] += alpha * (reward + gamma * expected_value - Q[state][action])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Flappy Bird RL with Q-Learning & Expected SARSA")

episodes = st.sidebar.slider("Number of Episodes", 100, 5000, 1000)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.5, 1.0, 0.99)
epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.01, 1.0, 0.1)
agent_type = st.sidebar.selectbox("Agent Type", ["Q-Learning", "Expected SARSA"])

# ---------------------------
# Training
# ---------------------------
def train_agent(agent_type, episodes, alpha, gamma, epsilon):
    env = FlappyBirdEnvSimple()
    Q = {}
    rewards = []

    for ep in range(episodes):
        state = tuple(np.round(env.reset(), 1))
        Q.setdefault(state, np.zeros(env.action_space_n))

        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, env.action_space_n)
            next_obs, reward, done = env.step(action)

            next_state = tuple(np.round(next_obs, 1))
            Q.setdefault(next_state, np.zeros(env.action_space_n))

            if agent_type == "Q-Learning":
                q_learning_update(Q, state, action, reward, next_state, alpha, gamma)
            else:
                expected_sarsa_update(Q, state, action, reward, next_state, alpha, gamma, epsilon)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards

# ---------------------------
# UI Actions
# ---------------------------
if st.button("Train Agent"):
    rewards = train_agent(agent_type, episodes, alpha, gamma, epsilon)

    st.success(f"{agent_type} Training Completed!")

    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title(f"{agent_type} - Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    st.pyplot(fig)
