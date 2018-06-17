import gym
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, clear_output
from gym.envs.box2d.lunar_lander import heuristic

import keras
import tensorflow as tf
from tensorflow.contrib.distributions import Beta
from keras.models import Sequential, Model, model_from_json, load_model, model_from_yaml
from keras.layers import Dense, Lambda, Input, Dropout
from keras.optimizers import Adam
from keras import backend as K

import time, datetime
import dill

seed = 44
np.random.seed(seed)
tf.set_random_seed(seed*2)



# ### Gym Setup
# 
# Here we load the Reinforcement Learning environments from Gym (both the continuous and discrete versions).
# 
# We limit each episode to 500 steps so that we can train faster. 

gym.logger.setLevel(logging.ERROR)
discrete_env = gym.make('LunarLander-v2')
discrete_env._max_episode_steps = 500
discrete_env.seed(seed*3)
continuous_env = gym.make('LunarLanderContinuous-v2')
continuous_env._max_episode_steps = 500
continuous_env.seed(seed*4)
gym.logger.setLevel(logging.WARN)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams["animation.html"] = "jshtml"


# ### Utilities
# 
# We include a function that lets you visualize an "episode" (i.e. a series of observations resulting from the actions that the agent took in the environment).
# 
# As well, we will use the "Results" class (a wrapper around a python dictionary) to store, save, load and plot your results. You can save your results to disk with results.save('filename') and reload them with Results(filename='filename'). Use results.pop(experiment_name) to delete an old experiment.


def AddValue(output_size, value):
    return Lambda(lambda x: x + value, output_shape=(output_size,))

# this could be used in the ipython version of the code
def render(episode, env):
    
    fig = plt.figure()
    img = plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')

    def animate(i):
        img.set_data(episode[i])
        return img,

    anim = FuncAnimation(fig, animate, frames=len(episode), interval=24, blit=True)
    html = HTML(anim.to_jshtml())
    
    plt.close(fig)
    get_ipython().system('rm None0000000.png')
    
    return html

class Results(dict):
    
    def __init__(self, *args, **kwargs):
        if 'filename' in kwargs:
            data = np.load(kwargs['filename'])
            super().__init__(data)
        else:
            super().__init__(*args, **kwargs)
        self.new_key = None
        self.plot_keys = None
        self.ylim = None
        
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.new_key = key

    def plot(self, window):
        clear_output(wait=True)
        for key in self:
            #Ensure latest results are plotted on top
            if self.plot_keys is not None and key not in self.plot_keys:
                continue
            elif key == self.new_key:
                continue
            self.plot_smooth(key, window)
        if self.new_key is not None:
            self.plot_smooth(self.new_key, window)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc='lower right')
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.show()
        
    def plot_smooth(self, key, window):
        if len(self[key]) == 0:
            plt.plot([], [], label=key)
            return None
        y = np.convolve(self[key], np.ones((window,))/window, mode='valid')
        x = np.linspace(window/2, len(self[key]) - window/2, len(y))
        plt.plot(x, y, label=key)
        
    def save(self, filename='results'):
        np.savez(filename, **self)


def run_experiment(RLAgent_es, experiment_name, env, num_episodes, learning_rate=0.001, baseline=None, old_params=None,
                   graph=True, results_to_use=None):
    
    if results_to_use is not None:
        results = results_to_use
    else:
        print('New Results!')
        results = Results()
    rewards = []
    startin_reward_len = 0
    
    # Initiate the learning agent
    if old_params:
        agent = old_params[0]
        rewards = old_params[1]
        startin_reward_len = len(rewards)
    else:
        agent = RLAgent_es(n_obs=env.observation_space.shape[0], action_space=env.action_space,
                           learning_rate=learning_rate, discount=0.99, baseline=baseline)

    all_episode_frames = []
    step = 0
    finish = False
    for episode in range(1, num_episodes + 1):
        if finish:
            break
        # Update results plot and occasionally store an episode movie
        episode_frames = None
        if episode % 10 == 0:
            results[experiment_name] = np.array(rewards)
            if graph:
                results.plot(10)
        if episode % 500 == 0 or episode == num_episodes:
            episode_frames = []

        # Reset the environment to a new episode
        observation = env.reset()
        episode_reward = 0
        while True:  # in every episode there is a full trip to the moon

            if episode_frames is not None:
                episode_frames.append(env.render(mode='rgb_array'))

            # 1. Decide on an action based on the observations
            action = agent.decide(observation)

            # 2. Take action in the environment
            next_observation, reward, done, info = env.step(action)
            episode_reward += reward

            # 3. Store the information returned from the environment for training
            agent.observe(observation, action, reward)

            # 4. When we reach a terminal state ("done"), use the observed episode to train the network
            if done:
                rewards.append(episode_reward)
                if not graph:
                    a = 0#print('episode number:', episode + startin_reward_len, 'reward:', episode_reward)
                if episode_frames is not None:
                    all_episode_frames.append(episode_frames)
                agent.train()  # in this way I'm training every episode, at the end of the episode!
                break

            # Reset for next step
            observation = next_observation
            step += 1

    return all_episode_frames, agent, rewards


# ## The Agent
# 
# Here we give the outline of a python class that will represent the reinforcement learning agent (along with its decision-making network). We'll modify this class to add additional methods and functionality throughout the course of the miniproject.
# 
# NOTE: We have set up this class to implement new functionality as we go along using keyword arguments. If you prefer, you can instead subclass RLAgent for each question.

# In[10]:


class RLAgent(object):
    
    def __init__(self, n_obs, action_space, learning_rate, discount, baseline = None):

        #We need the state and action dimensions to build the network
        self.n_obs = n_obs
        #We'll treat the continuous case a bit differently
        self.continuous = 'Discrete' not in str(action_space)
        if self.continuous:
            self.n_act = action_space.shape[0]
            self.act_low = action_space.low
            self.act_range = action_space.high - action_space.low
        else:
            self.n_act = action_space.n
        self.lr = learning_rate
        self.gamma = discount
        
        self.moving_baseline = None
        self.use_baseline = False
        self.use_adaptive_baseline = False
        if baseline == 'adaptive':
            self.use_baseline = True
            self.use_adaptive_baseline = True
        elif baseline == 'simple':
            self.use_baseline = True

        #These lists stores the cumulative observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        self.model_policy = None
        self.model_baseline = None
        #Build the keras network
        self._build_network()

    def observe(self, state, action, reward):
        """ This function takes the observations the agent received from the environment and stores them
            in the lists above. If necessary, preprocess the action here for the network. You may also get 
            better results clipping or normalizing the reward to limit its range for training."""
        raise NotImplementedError
        
    def decide(self, state):
        """ This function feeds the observed state to the network, which returns a distribution
            over possible actions. Sample an action from the distribution and return it."""
        raise NotImplementedError

    def train(self):
        """ When this function is called, the accumulated observations, actions and discounted rewards from the
            current episode should be fed into the network and used for training. Use the _get_returns function 
            to first turn the episode rewards into discounted returns. """
        raise NotImplementedError

    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode, then optionally apply a baseline. Hint: work backwards."""
        raise NotImplementedError

    def _build_network(self):
        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""
        raise NotImplementedError
    
    def update_model(self, model_policy, model_baseline = None):
        self.model_policy = model_policy
        if model_baseline:
            print('Loaded model baseline')
            self.model_baseline = model_baseline



class RLAgent_Ex1(RLAgent):

    def __init__(self, n_obs, action_space, learning_rate, discount, baseline=None):
        super(RLAgent_Ex1, self).__init__(n_obs, action_space, learning_rate, discount, baseline)
        self.epsilon = 0.1
        print('RLAgent 1')

    def observe(self, state, action, reward):
        self.episode_observations.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def decide(self, state):
        state = np.expand_dims(state, axis=0)

        actions = self.model_policy.predict(state)[0]
        action = np.random.choice(range(actions.shape[0]), p=actions.ravel())
        return action

    def train(self):
        episode_steps = len(self.episode_observations)
        num_actions = 4
        inputs = np.asarray(self.episode_observations)
        targets = np.zeros((episode_steps, num_actions))
        moving_avarage_value, moving_avarage_index = [], 0
        G = 0
        for k in range(episode_steps):
            G += pow(self.gamma, k) * self.episode_rewards[k]

        Gs = []
        for t in range(0, episode_steps):
            if t!=0:
                G = (G - self.episode_rewards[t - 1]) / self.gamma
                
            if self.use_adaptive_baseline:
                state_reshaped = np.expand_dims(self.episode_observations[t], axis=0)
                G_reshaped = np.expand_dims(G, axis=0)
                _, _ = self.model_baseline.train_on_batch(state_reshaped, G_reshaped)
                adaptive_baseline = self.model_baseline.predict(state_reshaped)[0]
                G1 = G - adaptive_baseline
                Gs.append(G1 * pow(self.gamma, t))
            elif self.use_baseline:
                avg_period = 20
                if moving_avarage_index < avg_period:
                    moving_avarage_index += 1
                    moving_avarage_value.append(G)
                else:
                    moving_avarage_value.pop(0)
                    moving_avarage_value.append(G)
                G1 = G - (sum(moving_avarage_value)) / moving_avarage_index
                Gs.append(G1 * pow(self.gamma, t))
            else:
                Gs.append(G * pow(self.gamma, t))


        Gs -= np.mean(Gs)
        Gs /= np.std(Gs)
        for t in range(episode_steps):
            targets[t, self.episode_actions[t]] = Gs[t]
            
        loss, _ = self.model_policy.train_on_batch(inputs, targets)

        # These lists stores the cumulative observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

    def _build_network(self):
        print(self.use_baseline, self.use_adaptive_baseline)
        optimizer_adam = Adam(lr=self.lr)
        model = Sequential()
        model.add(Dense(10, activation='relu', input_dim=8))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=optimizer_adam,
                      loss='categorical_crossentropy', metrics=['acc'])
        self.model_policy = model

        if self.use_adaptive_baseline:
            optimizer_adam = Adam(lr=self.lr)
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=8))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=optimizer_adam,
                          loss='MSE',
                          metrics=['accuracy'])
            self.model_baseline = model



learning_rate_testd = [0.01, 0.001, 0.0001] # only the selected one for more epochs
learning_rate = 0.001
d_agent_rew = []

discrete_results = Results()

name = 'REINFORCE'
episodes, recurrent_agent, rewards= run_experiment(RLAgent_Ex1, name, discrete_env, 100, learning_rate, 
                                                   graph=True,
                                                  results_to_use=discrete_results)


name = "REINFORCE (with baseline)"
episodes, recurrent_agent, rewards= run_experiment(RLAgent_Ex1, name, discrete_env,10, learning_rate, 
                                                   baseline='simple',
                                                   graph=True,
                                                  results_to_use=discrete_results)



name =  "REINFORCE (adaptive baseline)"
agent, rewards = get_agent_reward(name)
episodes, adaptive_agent, rewards= run_experiment(RLAgent_Ex1, name, discrete_env, 30, learning_rate, 
                                                   baseline='adaptive',
                                                   graph=True,
                                                  results_to_use=discrete_results)





class RLAgent_Ex4(RLAgent):
    
    def __init__(self, n_obs, action_space, learning_rate, discount, baseline = None):
        super(RLAgent_Ex4, self).__init__(n_obs, action_space, learning_rate, discount, baseline)
        self.epsilon = 0.1

    def observe(self, state, action, reward):
        self.episode_observations.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
    def decide(self, state):
        state = np.expand_dims(state, axis=0)
        
        output = self.model_policy.predict(state)
        action1_value = np.random.beta(output[0][0][0], output[0][0][1])* 2 - 1
        action2_value = np.random.beta(output[1][0][0], output[1][0][1])* 2 - 1
        return np.array([action1_value, action2_value])
        
    def train(self):        
        episode_steps = len(self.episode_observations)
        num_actions = 2
        inputs = np.asarray(self.episode_observations)
        targets1 = np.zeros((episode_steps, num_actions+1 - 1))
        targets2 = np.zeros((episode_steps, num_actions+1 - 1))
        moving_avarage_value, moving_avarage_index = [], 0
        
        G = 0
        for k in range(episode_steps):
            G += pow(self.gamma, k) * self.episode_rewards[k]

        for t in range(0, episode_steps):
            if t!=0:
                G = (G - self.episode_rewards[t - 1]) / self.gamma
                
            if self.use_adaptive_baseline:
                state_reshaped = np.expand_dims(self.episode_observations[t], axis=0)
                G_reshaped = np.expand_dims(G, axis=0)
                _, _ = self.model_baseline.train_on_batch(state_reshaped, G_reshaped)
                adaptive_baseline = self.model_baseline.predict(state_reshaped)
                G1 = G - adaptive_baseline
                
            elif self.use_baseline:
                avg_period = 20
                if moving_avarage_index < avg_period:
                    moving_avarage_index += 1
                    moving_avarage_value.append(G)
                else:
                    moving_avarage_value.pop(0)
                    moving_avarage_value.append(G)
                G1 = G - (sum(moving_avarage_value)) / moving_avarage_index
                
            else:
                G1 = G

            
            targets1[t] = (self.episode_actions[t][0]+1)/2, (pow(self.gamma, t) * G1)
            targets2[t] = (self.episode_actions[t][1]+1)/2, (pow(self.gamma, t) * G1)
        loss = self.model_policy.train_on_batch(inputs, {'final_output1': targets1, 'final_output2':targets2})
        
        #These lists stores the cumulative observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        

    def _build_network(self):
        optimizer_adam = Adam(lr=self.lr)
        state_input = Input(shape=(8,))

        h1 = Dense(20, activation='relu')(state_input)
        h2 = Dense(10, activation='relu')(h1)
        h3 = Dense(10, activation='relu')(h2)
        output_action1 = Dense(2, activation='softplus')(h3)
        output_action2 = Dense(2, activation='softplus')(h3)
        final_output1 = Lambda(lambda x: x + 1, output_shape=(output_action1.shape[0],), name='final_output1')(output_action1)
        final_output2 = Lambda(lambda x: x + 1, output_shape=(output_action2.shape[0],), name='final_output2')(output_action2)
        model = Model(input=state_input, output=[final_output1, final_output2])
        adam  = Adam(lr=0.001)
        model.compile(loss={'final_output1': beta_loss, 'final_output2':beta_loss}, optimizer=adam)
        self.model_policy = model
        
        
        if self.use_adaptive_baseline:
            optimizer_adam = Adam(lr=self.lr)
            model = Sequential()
            model.add(Dense(20, activation='relu', input_dim=8))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=optimizer_adam,
                      loss='MSE',
                      metrics=['accuracy'])
            self.model_baseline = model

def beta_loss(target, output):
    action1_prob = Beta(output[:, 0], output[:, 1])
    a = action1_prob.log_prob(target[:, 0])
    result = tf.reduce_sum(a * target[:, 1], axis=-1)
    return -result
    


results_continuous_simple = Results()
results_continuous_adaptive = Results()


# # Implementation Notes

# I have decided to implement a network with two separate outputs, similary to what we did in HW2 with pitches and durations, to better training both of them. I have created a custom loss function called 'beta_loss' with a negative return since I'm using adam optimizer.
# 
# I have obtained better results using adaptive baseline. However I haven't enough resources to train all of them as much as I wanted so I decided to focus on adaptive ones, with different learing rates as requested. (I had problems with model saving)

# In[114]:


agent.model_policy.summary()


# In[121]:


learning_rates = [0.001, 0.0001, 0.01]
c_agents_rew = []
results_continuous_adaptive.plot_keys = []
results_continuous_simple.plot_keys = []
for lr in learning_rates:
    for baseline in [ 'simple', 'adaptive']:
        if baseline == 'simple':
            results_to_use = results_continuous_simple
        else:
            results_to_use=results_continuous_adaptive
        experiment_name = ("Continuous REINFORCE {0} (learning rate: {1})".format(baseline, str(lr)))
        results_to_use.plot_keys.append(experiment_name)
        episodes, agent, rewards = run_experiment(RLAgent_Ex4, experiment_name, continuous_env, 11, lr, 
                                                  baseline=baseline, 
                                                  graph=False,
                                                 results_to_use=results_to_use)
        c_agents_rew.append([agent, sum(rewards[-100:]) / len(rewards[-100:]), episodes])



