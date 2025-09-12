#Note: Below functions have been modified using the base functions borrowed from the supporting .py files provided
#for 3MD4120: Reinforcement Learning at CentraleSupélec


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'figure.figsize': [10,5]})


#plots scores/reward of an agent based on the sleected hyperparameters
def plot_FB_HP_score(score_history, score_name,agent_name,window=100):
    assert score_name in score_history.keys()
    plt.figure(figsize = (20,10))
    
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(score_history[score_name].keys())))))
    for pair in score_history[score_name].keys():
        plt.plot(pd.Series(score_history[score_name][pair]).rolling(500).mean(), 
                 label = f"step size:{pair[0]}, epsilon start: {pair[1]}")
            
    plt.title(f"{agent_name} Hyper Parameter Tuning");
    plt.xlabel('Number of Episodes')
    plt.ylabel(f"Moving Average {score_name}")
    plt.legend()
    plt.show()
    
#plots the value function of a select agent    
def plot_FB_values(agent,agent_name):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    def get_Z(x, y, agent):
        if (x,y) in agent.state_dict.keys():
            return np.max(agent.q[agent.state_dict[(x,y)]])
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 13)
        y_range = np.arange(-11, 12)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,agent) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
        ax.set_xlabel('Player\'s X Distance from Pipe')
        ax.set_ylabel('Player\'s Y Distance from Pipe')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(agent_name + ' State Value Plot')
    get_figure(ax)
    
#plots the moving average scores/reward per episode   
def plot_FB_score(score_history, score_name,agent_name,window=100):
    assert score_name in score_history.keys()
    
    moving_average = []
    for i in range(len(score_history[score_name]) - window + 1):
         moving_average.append(np.mean(score_history[score_name][i:i+window]))
           
    plt.plot(moving_average);
    plt.title(f"{agent_name} {score_name} - {window} Moving Average");
    plt.xlabel('Number of Episodes')
    plt.ylabel(f"Moving Average {score_name}")
    plt.show()

#plots the optimal policy found by an agent based on their qvalues
def plot_policy(agent,agent_name):
    
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    def get_Z_with_unexplored(x, y,agent):
        if (x,y) in agent.state_dict.keys():
            return np.argmax(agent.q[agent.state_dict[(x,y)]])
        else:
            return -1 #define a value for Z to visualize unexplored areas
    
    def get_figure(ax):
        x_range = np.arange(0, 13)
        y_range = np.arange(-12, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z_with_unexplored(x,y, agent) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape) 
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2'))
        plt.xticks(x_range) 
        plt.yticks(np.arange(23), range(-11, 12, 1)) 
        plt.gca().invert_xaxis() 
        ax.set_xlabel('X distance from Pipe') 
        ax.set_ylabel('Y distance from Pipe') 

        ax.grid(color='w', linestyle='-', linewidth=1) 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(surf, boundaries=[-1.5,-0.5,0.5,1.5], ticks=[-1,0,1], cax=cax) 
        cbar.ax.set_yticklabels(['Unexplored Region', 'Idle','Flap'])

    
    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.add_subplot(111)
    ax.set_title(agent_name + ' Policy')
    get_figure(ax)
    plt.show() # show the plot