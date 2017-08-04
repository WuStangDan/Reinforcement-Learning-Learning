# Implement Q Learning where the state is discretized.


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
        action_values = self.Q[s_i]
        exploit_action_index = np.argmax(action_values)

        return exploit_action_index


    def UpdateQ(self, state, a_i, G):
        # Given state action pair, and sampled return (G)
        # update Q for that state action pair.
        s_i = self.discretized_states.StateToIndex(state)

        self.Q[s_i, a_i] += self.alpha * (G -  self.Q[s_i, a_i])
        
    
        

        
        


   
