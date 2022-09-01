import numpy as np

from rl_utils import running_mean, discount_rwds

# set seed for pseudorandom number generation -- make sure our trajectories look the same
np.random.seed(80)


class Q_Agent(object):
    def __init__(self, nstates, nactions, 
                 learning_rate=0.1, discount=0.9, epsilon=1): 
        
        self.num_actions = nactions
        self.num_states = nstates
        self.action_space = np.arange(self.num_actions)

        # this agent selects actions from a table of state,action values which we initalize randomly
        #self.q_table = np.random.uniform(low=-1, high=1, size=(nstates, nactions))
        self.q_table = np.zeros((nstates,nactions))

        # parameters for learning
        self.epsilon       = epsilon
        self.learning_rate = learning_rate
        self.discount      = discount # gamma
        
    def choose_action(self, state):
        # this agent uses epsilon-greedy action selection, meaning that it selects 
        # the greedy (highest value) action most of the time, but with epsilon probability
        # it will select a random action -- this helps encourage the agent to explore
        # unseen trajectories
        
        ## TO DO -- write action selection for an epsilon-greedy policy 
        if np.random.random()>self.epsilon:
            # take the action which corresponds to the highest value in the q table at that row (state)
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def update_q_table(self, current_state, current_action, reward, new_state):
        # this function describes how the Q table gets updated so the agent can make 
        # better choices based on what it has experienced from the environment 
        current_q = self.q_table[ current_state, current_action]
        max_future_q = np.max(self.q_table[new_state,:])
        
        ## TO DO -- write the Q value update using Q-learning
        new_q = current_q + self.learning_rate*(reward+self.discount*max_future_q-current_q)
        self.q_table[current_state, current_action] = new_q
        
    def navigate(self, gw_obst, num_episodes, random_start=False, start=0):
        # set how we will decay the randomness of action selection over the course of training
        start_eps_decay = 1
        end_eps_decay = num_episodes//2
        epsilon_decay_value = self.epsilon/(end_eps_decay-start_eps_decay)
        snapshot=[]

        # initialize empty list for keeping track of rewards achieved per episode
        reward_tracking=[]
        max_steps= 1000

        for episode in range(num_episodes):
            gw_obst.reset()
            # initalize reward counter
            total_reward=0

            # get first state
            if random_start:
                state=gw.obst.state
            else:
                state=start

            for step in range(max_steps):
                action = self.choose_action(state)
                # take a step in the environment
                next_state, reward, done, _ = gw_obst.step(action)

                total_reward+=reward

                if not done:
                    self.update_q_table(state, action, reward, next_state)
                else:
                    self.q_table[state, action] = 0
                    break
                state=next_state

            reward_tracking.append(total_reward)
            snapshot.append(self.q_table.copy())

            if end_eps_decay >= episode >= start_eps_decay:
                self.epsilon -= epsilon_decay_value

        return reward_tracking,snapshot

class Dyna_Q_Agent(object):
    def __init__(self, nstates, nactions, 
                 learning_rate=0.1, discount=0.9, epsilon=1): 
        
        self.num_actions = nactions
        self.num_states  = nstates
        self.action_space = np.arange(self.num_actions)
        
        self.dyna_q_table = np.zeros((nstates,nactions))
        self.model        = {}
        
        # parameters for learning
        self.epsilon       = epsilon
        self.learning_rate = learning_rate
        self.discount      = discount # gamma
        
    def choose_action(self, state):
        # this agent uses epsilon-greedy action selection, meaning that it selects 
        # the greedy (highest value) action most of the time, but with epsilon probability
        # it will select a random action -- this helps encourage the agent to explore
        # unseen trajectories
    
        ## TO DO -- write action selection for an epsilon-greedy policy 
        if np.random.random()>self.epsilon:
            # take the action which corresponds to the highest value in the q table at that row (state)
            action = np.argmax(self.dyna_q_table[state])
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def planning(self,n_steps):  
        for i in range(n_steps):
            (state,action) = np.random.choice(self.model.keys()) 
            experience_list=self.model[(state,action)]
            (next_state,reward)= np.random.choice(experience_list)
            self.update_dyna_q_table(state,action,reward,next_state)
         
                 
    def update_dyna_q_table(self, current_state, current_action, reward, new_state):
        current_dyna_q    = self.dyna_q_table[ current_state, current_action]
        max_future_dyna_q = np.max(self.dyna_q_table[new_state,:])    
        
        #value update for Dyna_Q using Q learning
        new_dyna_q        = current_dyna_q + self.learning_rate*(reward+self.discount*max_future_dyna_q-current_dyna_q)
        self.dyna_q_table[current_state, current_action] = new_dyna_q        
    
    def navigate(self, gw_obst, num_episodes, random_start=False, start=0):
        # set how we will decay the randomness of action selection over the course of training
        start_eps_decay = 1
        end_eps_decay = num_episodes//2
        epsilon_decay_value = self.epsilon/(end_eps_decay-start_eps_decay)
        snapshot=[]
        # initialize empty list for keeping track of rewards achieved per episode
        reward_tracking=[]
        max_steps= 1000

        for episode in range(num_episodes):
            gw_obst.reset()
            # initialize reward counter
            total_reward=0
                 
            #get first stage
            if random_start:
                 state=gw_obst.state
            else: # take a step in the environment
                state=start
                gw_obst.state = start
                 
            for step in range (max_steps):
                action = self.choose_action(state)
                # take a step in the environment
                next_state, reward, done, _ = gw_obst.step(action)
                total_reward+=reward
                 
                if (state,action) not in self.model.keys():
                    self.model[(state,action)]=[]
                # storing transition in the model
                self.model[(state,action)].append((next_state,reward))
                
                if not done:
                    self.update_dyna_q_table(state, action, reward, next_state)
                else:
                    self.dyna_q_table[state, action] = 0
                    break
                state=next_state

            reward_tracking.append(total_reward)
            snapshot.append(self.dyna_q_table.copy())

            if end_eps_decay >= episode >= start_eps_decay:
                self.epsilon -= epsilon_decay_value

        return reward_tracking,snapshot