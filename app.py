import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from flappy_bird_env_simple import FlappyBirdEnvSimple
from plot_utils import plot_FB_score

# ---------------------------
# Q-Learning and Expected SARSA Functions
# ---------------------------
def epsilon_greedy(Q, state, epsilon, n_actions):
    """Choose action using epsilon-greedy policy"""
    if np.random.rand() < epsilon or state not in Q:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    """Q-Learning update"""
    if next_state not in Q:
        Q[next_state] = np.zeros(2)
    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

def expected_sarsa_update(Q, state, action, reward, next_state, alpha, gamma, epsilon):
    """Expected SARSA update"""
    if next_state not in Q:
        Q[next_state] = np.zeros(2)
    policy = np.ones(2) * epsilon / 2
    policy[np.argmax(Q[next_state])] += 1 - epsilon
    expected_value = np.dot(policy, Q[next_state])
    Q[state][action] += alpha * (reward + gamma * expected_value - Q[state][action])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Flappy Bird RL with Q-Learning & Expected SARSA")

# Sidebar controls
episodes = st.sidebar.slider("Number of Episodes", 100, 5000, 1000)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.5, 1.0, 0.99)
epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.01, 1.0, 0.1)
agent_type = st.sidebar.selectbox("Agent Type", ["Q-Learning", "Expected SARSA"])

# ---------------------------
# Training Function
# ---------------------------
def train_agent(agent_type, episodes, alpha, gamma, epsilon):
    env = FlappyBirdEnvSimple()
    Q = {}
    rewards = []

    for ep in range(episodes):
        state = tuple(np.round(env.reset(), 1))
        if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, env.action_space.n)
            next_obs, reward, done, _ = env.step(action)
            next_state = tuple(np.round(next_obs, 1))
            if next_state not in Q:
                Q[next_state] = np.zeros(env.action_space.n)

            if agent_type == "Q-Learning":
                q_learning_update(Q, state, action, reward, next_state, alpha, gamma)
            else:
                expected_sarsa_update(Q, state, action, reward, next_state, alpha, gamma, epsilon)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return rewards

# ---------------------------
# Run Training
# ---------------------------
if st.button("Train Agent"):
    rewards = train_agent(agent_type, episodes, alpha, gamma, epsilon)

    st.success(f"{agent_type} Training Completed!")

    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title(f"{agent_type} - Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    st.pyplot(fig)
