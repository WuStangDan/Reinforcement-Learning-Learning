# Implement Q Learning where the state is discretized.

import gym
import numpy as np

class StateDiscretization:
    def __init__(self):
        # Discretize each state into 10 bins.
        self.cart_pos_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_vel_bins = np.linspace(-2, 2, 9)
        # Pole angle is -15 degrees to 15 degrees
        self.pole_ang_bins = np.linspace(-0.26, 0.26, 9)
        self.pole_vel_bins = np.linspace(-3.5, 3.5, 9)


    def StateToIndex(self, state):
        # Read in individual state values.
        cart_pos, cart_vel, pole_ang, pole_vel = state
        
        # Get index for each value within discretized states.
        cart_pos_bin_index = np.digitize(cart_pos, self.cart_pos_bins)
        cart_vel_bin_index = np.digitize(cart_vel, self.cart_vel_bins)
        pole_ang_bin_index = np.digitize(pole_ang, self.pole_ang_bins)
        pole_vel_bin_index = np.digitize(pole_vel, self.pole_vel_bins)

        # Combine indices into one number.
        temp_str = str(cart_pos_bin_index)
        temp_str += str(cart_vel_bin_index)
        temp_str += str(pole_ang_bin_index)
        temp_str += str(pole_vel_bin_index)
        
        # Return state index.
        return int(temp_str)


class Agent:
    def __init__(self, update, state_num, action_num, discretized_states):
        
        self.discretized_states = discretized_states
        self.state_num = state_num
        self.action_num = action_num

        # Matrix of state action pairs.
        # Returns expected future rewards (discounted) by
        # taking action (a) when in state (s) and following
        # optimal policy from then on.
        # Matrix size is num_states x num_actions.
        self.Q = np.random.uniform(low=-1, high=1, size=(state_num, action_num))
     
        # Place holder for Q that achives highest average reward over
        # 100 episodes.
        self.Q_best = self.Q

        self.alpha = update


    def ExploitAction(self, state):
        # Perform the best action (highest expected return)
        # for the given state.
        s_i = self.discretized_states.StateToIndex(state)
        # Expected returns of taking each action in
        # current state.
        action_values = self.Q[s_i]
        exploit_action_index = np.argmax(action_values)

        return exploit_action_index

    def ExploreAction(self):
        # Perform a random action.
        return np.random.choice(self.action_num,1)[0]


    def UpdateQ(self, state, a_i, G):
        # Given state action pair, and sampled return (G)
        # update Q for that state action pair.
        s_i = self.discretized_states.StateToIndex(state)

        self.Q[s_i, a_i] += self.alpha * (G -  self.Q[s_i, a_i])

    def GetMaxQ(self, state):
        # Return the max Q value for a specific state.
        s_i = self.discretized_states.StateToIndex(state)
        return np.max(self.Q[s_i])
        
    
        

def PlayEpisode(env, agent, epsilon, gamma):
    # Reset playing environment.
    s_t0 = env.reset()

    total_episode_reward = 0
    time_steps = 0
    episode_over = False

    while (not over):     # and (time_steps < 5000):
        # Determine whether to explore or exploit.
        if (np.random.random() < epsilon):
            # Explore.
            a_t0 = agent.ExploreAction()
        else:
            # Exploit.
            a_t0 = agent.ExploitAction(s_t0)
            
        # Perform action and move to next state.
        s_t1, reward, episode_over, info = env.step(a_t0)


        total_episode_reward += reward

        # Check if episode is over, if yes and episode hasn't reached
        # max time steps, apply negative reward.
        if episode_over and (time_steps < 199):
            reward -= 300

        # Calculate return and update Q.
        G = reward + gamma*agent.GetMaxQ(s_t1)
        agent.UpdateQ(s_t0, a_t0, G)

        time_steps += 1


    return total_episode_reward


if __name__ == '__main__':
    environment = gym.make('CartPole-v1')
    states_dis = StateDiscretization()

    # Set future rewards discount rate.
    gamma = 0.9

    
    rl_agent = Agent(0.001, 10**4, 2, states_dis)

    episode_num = 1000
    all_episode_r = np.zeros(episode_num)

    for i in range(episode_num):
        # Reduce epsilon over time.
        eps = 1.0/np.sqrt(n+1)

        # Play episode.
        ep_reward = PlayEpisode(environment, rl_agent, eps, gamma)
        all_episode_r[i] = ep_reward

        
        
        
    
    



        

       

            
                

            
            
        
        


   
