import matplotlib.pyplot as plt

def plot_FB_score(rewards):
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    return fig
