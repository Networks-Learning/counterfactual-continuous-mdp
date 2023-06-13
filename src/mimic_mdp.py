import numpy as np
import pickle
from src.models import SCM
import torch
import json
import heapq
import logging
import click
import time
import logging


class Node:
    def __init__(self, s, l, t, rwd, parent=None, action=None):
        self.s = s
        self.l = l
        self.t = t
        self.rwd = rwd
        self.parent = parent
        self.action = action

    def __eq__(self, other):
        if isinstance(other, Node):
            return (torch.all(torch.eq(self.s, other.s))) and (self.l == other.l) and (self.t == other.t)
        else:
            return False
    
    def __hash__(self):
        return hash((self.s, self.l, self.t))

    def __lt__(self, other):
        return self.rwd > other.rwd

# Saves configuration and results to a JSON file
def generate_summary(pid, k, seed, anchor_method, anchor_samples, hidden_layers, hidden_units, lipschitz_loc, lipschitz_scale, prior_type, T, actions, states, survived,\
                        reward, cf_actions, cf_states, cf_reward, visited, added, anchor_runtime, astar_runtime, ebf, anchor_set_size, algo):

    summary = {}
    # configuration
    summary['pid'] = pid
    summary['k'] = k
    summary['seed'] = seed
    summary['anchor_method'] = anchor_method
    summary['anchor_samples'] = anchor_samples
    summary['hidden_layers'] = hidden_layers
    summary['hidden_units'] = hidden_units
    summary['lipschitz_loc'] = lipschitz_loc
    summary['lipschitz_scale'] = lipschitz_scale
    summary['prior_type'] = prior_type
    summary['algo'] = algo
    
    # results
    summary['actions'] = actions.tolist()
    summary['states'] = states.tolist()
    summary['survived'] = survived
    summary['reward'] = reward
    summary['cf_actions'] = cf_actions.tolist()
    summary['cf_states'] = cf_states.tolist()
    summary['cf_reward'] = cf_reward
    summary['visited'] = visited
    summary['added'] = added
    summary['anchor_runtime'] = anchor_runtime
    summary['astar_runtime'] = astar_runtime
    summary['ebf'] = ebf
    summary['anchor_set_size'] = anchor_set_size
    summary['horizon'] = T
         
    return summary

def compute_ebf(target_nodes, depth, precision=0.00001):
    
    # perform binary search from 0+precision to 10.0 for b
    left = 1.0
    right = 10.0
    ebf = (left + right)/2.0
    nodes = (ebf ** (depth+1) - 1)/(ebf - 1.0) - 1 

    while right-left > precision:
        if nodes > target_nodes:
            right = ebf
        else:
            left = ebf
        ebf = (left + right)/2.0
        nodes = (ebf ** (depth+1) - 1)/(ebf - 1.0) - 1
        
    return float(np.round(ebf, decimals=3))


class MimicMDP():
    
    def __init__(self, model_filename, temp_output_directory, data_directory, experiment_directory, patient_id, device='cpu', seed=42):
        
        # fix the random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.trajectory_filename = ''.join([temp_output_directory, 'trajectory_patient_', str(patient_id), '.pkl'])
        self.model_filename = model_filename
        self.temp_output_directory = temp_output_directory
        self.experiment_directory = experiment_directory
        self.action_dictionary_filename = ''.join([data_directory, 'action_dictionary.json'])
        self.device = device

        # read the json file with the action dictionary
        with open(self.action_dictionary_filename, 'r') as f:
            self.action_dictionary = json.load(f)

        # read the trajectory dictionary
        with open(self.trajectory_filename, 'rb') as f:
            self.trajectory = pickle.load(f)

        self.a_dim = 3
        self.num_of_features = self.trajectory['states'].shape[1]
        self.c_dim = 3
        
        # read model parameters from filename
        params = self.model_filename.split('/')[-1].split('_')
        self.hidden_layers = int(params[3])
        self.hidden_units = int(params[5])
        self.lipschitz_loc = float(params[11])
        self.lipschitz_scale = float(params[13])
        self.prior_type = params[15].split('.')[0]

        # initialize an SCM object
        self.scm = SCM(self.num_of_features, self.hidden_layers, self.hidden_units, a_dim=self.a_dim, c_dim=self.c_dim, lipschitz_loc=self.lipschitz_loc, lipschitz_scale=self.lipschitz_scale, prior_type=self.prior_type, device=device).to(device)
        # load its trainable parameters from file
        self.scm.load_state_dict(torch.load(self.model_filename, map_location=device))
        self.scm.eval()

        # prepare the observed episode tensors
        states = self.trajectory['states']
        actions = self.trajectory['actions']
        
        self.s_all = torch.tensor(states, dtype=torch.float32, device=self.device)

        s = self.s_all[:-1,:]
        s_prime = self.s_all[1:,:]
        self.a = torch.tensor(actions[:-1,:], dtype=torch.float32, device=self.device)
        self.T = self.s_all.shape[0]
        with torch.no_grad():
            self.u, _ = self.scm.backward(s, self.a, s_prime)

        # reward from t to end
        self.rewards = (-self.s_all[:, -1]).flip(dims=[0]).cumsum(dim=0).flip(dims=[0])


    def solve_facility_location(self, size, logger):
        
        ###############################################################
        # solve the facility location problem as described in the paper
        ###############################################################
        s_all = []
        for patient_id in self.trajectories:
            s_all.append(torch.tensor(self.trajectories[patient_id]['states'], dtype=torch.float32, device=self.device))
        s_all = torch.cat(s_all, dim=0)
        s_all = s_all[:, self.c_dim:]
        
        s_anchor = s_all[np.random.randint(s_all.shape[0]),:].reshape(1,-1)
        
        while s_anchor.shape[0] < size:
            
            dists = torch.cdist(s_all, s_anchor)
            vals = dists.min(dim=1).values
            max_val, furthest_id = vals.max(dim=0)
            s_anchor = torch.cat((s_anchor, s_all[furthest_id,:].reshape(1,-1)), dim=0)

            if s_anchor.shape[0] % 500 == 0:
                logger.info('Current size is {siz} and max. distance is {val}'.format(siz=s_anchor.shape[0], val=max_val))
            if s_anchor.shape[0] % 5000 == 0:
                with open(''.join([self.temp_output_directory, 'anchor_set_size_', str(s_anchor.shape[0]), '.pkl']), 'wb') as f:
                    pickle.dump(s_anchor.cpu(), f)

        return s_anchor

    def compute_exact_anchor_set(self, k, s_all, a, u, lipschitz, method='montecarlo-proportional', anchor_samples=1000):
        
        ################################################################################################################################
        # perform monte carlo rollouts of action sequences differing in at most k actions and add the resulting states in the anchor set
        ################################################################################################################################

        if method == 'facility-location':
            try:
                with open(''.join([self.temp_output_directory, 'facility_anchor_set_size_', str(anchor_samples), '.pkl']), 'rb') as f:
                    s_anchor_reduced = pickle.load(f)
            except:
                raise Exception('Facility location solution of size {siz} not found!'.format(siz=anchor_samples))
            
            s_anchor = torch.zeros((s_anchor_reduced.shape[0], s_all.shape[1]), device=self.device)
            s_anchor[:,self.c_dim:] = s_anchor_reduced
            s_anchor[:, :self.c_dim] = s_all[0,:self.c_dim].expand((s_anchor.shape[0], -1))
            s_anchor = torch.cat((s_anchor, s_all), dim=0)

            return s_anchor

        else:

            s_anchor = s_all.clone()
            
            s = s_all[:-1,:]

            # s_cf = torch.zeros((s_all.shape[0], anchor_samples, s_all.shape[1]), device=self.device)
            s_cf = torch.zeros_like(s_all, device=self.device)
            a_cf = torch.zeros_like(a, device=self.device)

            anchor_set_size = s_anchor.shape[0] - s_all.shape[0]
            while anchor_set_size < anchor_samples:
                s_cf[0,:] = s[0,:]
                a_cf = a.clone()
                changes = np.random.randint(1, k+1)
                if method == 'montecarlo-uniform':
                    # only the first two actions are changed, the third (mechvent) is kept the same
                    idx = np.random.choice(a_cf.shape[0], changes, replace=False) # choose time steps to change the action uniformly at random
                    a_cf[idx,:2] = torch.randint(-2, 3, (changes, a_cf.shape[1]-1), dtype=torch.float32, device=self.device) / 4
                elif method == 'montecarlo-proportional':
                    # only the first two actions are changed, the third (mechvent) is kept the same
                    probs = (lipschitz[:-1] / lipschitz[:-1].sum()).cpu().numpy()
                    idx = np.random.choice(a_cf.shape[0], changes, replace=False, p=probs)
                    a_cf[idx,:2] = torch.randint(-2, 3, (changes, a_cf.shape[1]-1), dtype=torch.float32, device=self.device) / 4
            
                with torch.no_grad():
                    for t in range(a_cf.shape[0]):
                        s_cf[t+1,:] = self.scm.forward(s_cf[t,:].reshape(1,-1), a_cf[t,:].reshape(1,-1), u[t,:].reshape(1,-1))

                s_anchor = torch.cat((s_anchor, s_cf), dim=0)
                
                # keep only one copy of each state if it appears more than once in s_anchor
                s_anchor = torch.unique(s_anchor, dim=0)
                anchor_set_size = s_anchor.shape[0] - s_all.shape[0]

            return s_anchor

    def compute_anchor_set(self, k, s_all, a, u, lipschitz, method='montecarlo-proportional', anchor_samples=1000):
        
        ################################################################################################################################
        # perform monte carlo rollouts of action sequences differing in at most k actions and add the resulting states in the anchor set
        ################################################################################################################################

        if method == 'facility-location':
            try:
                with open(''.join([self.temp_output_directory, 'anchor_set_size_', str(anchor_samples), '.pkl']), 'rb') as f:
                    s_anchor_reduced = pickle.load(f)
            except:
                raise Exception('Facility location solution of size {siz} not found!'.format(siz=anchor_samples))
            
            s_anchor = torch.zeros((s_anchor_reduced.shape[0], s_all.shape[1]), device=self.device)
            s_anchor[:,self.c_dim:] = s_anchor_reduced
            s_anchor[:, :self.c_dim] = s_all[0,:self.c_dim].expand((s_anchor.shape[0], -1))
            s_anchor = torch.cat((s_anchor, s_all), dim=0)

            return s_anchor


        s_anchor = s_all.clone()
        
        if k>0:
            s = s_all[:-1,:]

            s_cf = torch.zeros((s_all.shape[0], anchor_samples, s_all.shape[1]), device=self.device)
            a_cf = torch.zeros((a.shape[0], anchor_samples, a.shape[1]), device=self.device)
            
            for i in range(anchor_samples):
                s_cf[0,i,:] = s[0,:].reshape(1,-1)
                a_cf[:,i,:] = a
                changes = np.random.randint(1, k+1)
                if method == 'montecarlo-uniform':
                    # only the first two actions are changed, the third (mechvent) is kept the same
                    idx = np.random.choice(a_cf.shape[0], changes, replace=False) # choose time steps to change the action uniformly at random
                    a_cf[idx,i,:2] = torch.randint(-2, 3, (changes, a_cf.shape[2]-1), dtype=torch.float32, device=self.device) / 4
                elif method == 'montecarlo-proportional':
                    # only the first two actions are changed, the third (mechvent) is kept the same
                    probs = (lipschitz[:-1] / lipschitz[:-1].sum()).cpu().numpy()
                    idx = np.random.choice(a_cf.shape[0], changes, replace=False, p=probs)
                    a_cf[idx,i,:2] = torch.randint(-2, 3, (changes, a_cf.shape[2]-1), dtype=torch.float32, device=self.device) / 4
            
            with torch.no_grad():
                for t in range(a_cf.shape[0]):
                    s_cf[t+1,:,:] = self.scm.forward(s_cf[t,:,:], a_cf[t,:,:], u[t,:].expand((anchor_samples,-1)))

            s_cf = s_cf.flatten(0,1)
            s_anchor = torch.cat((s_anchor, s_cf), dim=0)
            
        # keep only one copy of each state if it appears more than once in s_anchor
        s_anchor = torch.unique(s_anchor, dim=0)

        return s_anchor

    def compute_upper_bounds(self, k, logger, anchor_samples=100, anchor_method='montecarlo-proportional', nologging=False, exact=False):
        
        # location lipschitz constant
        with torch.no_grad():
            lip_layer_0 = torch.linalg.norm(self.scm.location_model.net[0].s_linear.weight, ord=2)
            mult_0 = self.scm.location_model.net[1].constant
            lip_layer_1 = torch.linalg.norm(self.scm.location_model.net[3].weight, ord=2)
            mult_1 = self.scm.location_model.net[4].constant
            loc_lipschitz = lip_layer_0 * mult_0.item() * lip_layer_1 * mult_1.item()

            # scale lipschitz constant
            lip_layer_0 = torch.linalg.norm(self.scm.scale_model.net[0].s_linear.weight, ord=2)
            mult_0 = self.scm.scale_model.net[1].constant
            lip_layer_1 = torch.linalg.norm(self.scm.scale_model.net[3].weight, ord=2)
            mult_1 = self.scm.scale_model.net[4].constant
            scale_lipschitz = lip_layer_0 * mult_0.item() * lip_layer_1 * mult_1.item()

        s = self.s_all[:-1,:]
        
        self.lipschitz = torch.zeros(self.T, device=self.device)
        self.lipschitz[self.T-1] = 1.0
        for t in range(self.T-2, -1, -1):
            K_t = loc_lipschitz + scale_lipschitz * self.u[t].abs().max()
            self.lipschitz[t] = 1 + K_t*self.lipschitz[t+1]

        if exact:
            s_anchor = self.compute_exact_anchor_set(k=k, s_all=self.s_all, a=self.a, u=self.u, lipschitz=self.lipschitz, method=anchor_method, anchor_samples=anchor_samples)
        else:
            s_anchor = self.compute_anchor_set(k=k, s_all=self.s_all, a=self.a, u=self.u, lipschitz=self.lipschitz, method=anchor_method, anchor_samples=anchor_samples)
        root_idx = torch.where(torch.all(s_anchor==s[0,:], dim=1))[0][0].item()

        # initialize upper bound matrix
        upper_bounds = torch.zeros((s_anchor.shape[0], k+1, self.T), device=self.device)
        upper_bounds[:,:,self.T-1] = (-s_anchor[:,-1]).expand((k+1,-1)).T # -1 is the SOFA column

        # main loop
        if not nologging:
            logger.info('Computing upper bounds...')
        for t in range(self.T-2, -1, -1):
            for l in range(k, -1, -1):
                
                available_actions = self._get_actions(t, l, k)

                Vs = torch.zeros((available_actions.shape[0], s_anchor.shape[0]), device=self.device)

                # Get the number of s_anchor samples and available actions
                num_s_anchor = s_anchor.shape[0]
                num_actions = available_actions.shape[0]

                # Broadcast self.s_anchor to have the same shape as available_actions for each sample
                s_anchor_expanded = s_anchor.unsqueeze(0).expand(num_actions, -1, -1)

                # Broadcast available_actions to have the same shape as self.s_anchor for each action
                available_actions_expanded = available_actions.unsqueeze(1).expand(-1, num_s_anchor, -1)

                # Apply transitions
                with torch.no_grad():
                    s_a = self.scm.forward(
                        s_anchor_expanded.reshape(-1, s_anchor_expanded.shape[-1]),
                        available_actions_expanded.reshape(-1, available_actions_expanded.shape[-1]),
                        self.u[t, :].expand(num_s_anchor * num_actions, -1)
                    ).reshape(num_actions, num_s_anchor, -1)

                # Calculate l_a
                l_a = torch.where(torch.all(available_actions_expanded[:,:,:2] == self.a[t, :2], dim=-1), l, l + 1) # :2 because we consider the mechvent action fixed
                
                # Compute V
                for i in range(num_actions):
                    V_a = torch.cdist(s_a[i,:,:], s_anchor_expanded[i,:,:], p=2)
                    V_a *= self.lipschitz[t + 1]
                    V_a += upper_bounds[:, l_a[i], t + 1].T
                    Vs[i,:] = -s_anchor_expanded[i,:, -1] + V_a.min(dim=1).values

                upper_bounds[:, l, t] = Vs.max(dim=0).values
        
        if not nologging:
            logger.info('Best estimate: {b}'.format(b=upper_bounds[root_idx, 0, 0].item()))

        return upper_bounds, s_anchor, root_idx

    def _action_id_to_vector(self, action_id):
        
        mechvent = action_id // 25
        vaso = (action_id % 25) // 5
        iv = action_id % 5
        
        x = torch.tensor([vaso-2, iv-2, mechvent], device=self.device)
        x[2] = 4*x[2] - 2
        x = x/4

        return x

    def _action_vector_to_id(self, action_vector):
            
            mechvent = int(action_vector[2] + 0.5)
            vaso = int(4*(action_vector[0] + 0.5))
            iv = int(4*(action_vector[1] + 0.5))
            
            action_id = 25*mechvent + 5*vaso + iv
    
            return action_id

    def _get_actions(self, t, l, k):
        
        if l == k:
            available_actions = self.a[t].reshape((1,-1))
        else:
            
            x = torch.meshgrid(torch.arange(-2, 3, device=self.device), torch.arange(-2, 3, device=self.device), indexing='ij')
            x = torch.stack((x[0].reshape(-1), x[1].reshape(-1)), dim=1)
            x = x/4
            available_actions = torch.cat((x, self.a[t, 2].reshape((1,-1)).expand(x.shape[0], -1)), dim=1)
            
        return available_actions
    
    def heuristic_batch(self, nodes, k, upper_bounds, s_anchor):
        num_nodes = len(nodes)
        t = nodes[0].t

        s = torch.stack([node.s for node in nodes])
        l = torch.tensor([node.l for node in nodes], device=self.device)
        available_actions_list = [self._get_actions(t, node.l, k) for node in nodes]

        unique_action_counts = {actions.shape[0] for actions in available_actions_list}

        if len(unique_action_counts) == 1:
            num_actions = unique_action_counts.pop()
            available_actions_stacked = torch.stack(available_actions_list)

            R_a = -s[:, -1]

            # Case (i): 25 actions for all nodes
            if num_actions == 25:
                s_expanded = s.unsqueeze(1).expand(-1, num_actions, -1)
                with torch.no_grad():
                    s_a = self.scm.forward(s_expanded.reshape(-1, s_expanded.shape[-1]), available_actions_stacked.reshape(-1, available_actions_stacked.shape[-1]), self.u[t, :].expand(num_nodes * num_actions, -1)).reshape(num_nodes, num_actions, -1)
                l_a = torch.where(torch.any(available_actions_stacked[:, :, :2] != self.a[t, :2], dim=-1), l.unsqueeze(1) + 1, l.unsqueeze(1))

                dists = self.lipschitz[t + 1] * torch.cdist(s_a, s_anchor, p=2)
                V = R_a + (upper_bounds[:, l_a.squeeze(1), t+1].permute(1,2,0) + dists).min(dim=2).values.max(dim=1).values
                return V
            # Case (ii): 1 action for all nodes
            elif num_actions == 1:
                with torch.no_grad():
                    s_a = self.scm.forward(s, available_actions_stacked.squeeze(1), self.u[t, :].expand(num_nodes, -1))
                l_a = torch.where(torch.any(available_actions_stacked.squeeze(1)[:, :2] != self.a[t, :2], dim=-1), l + 1, l)

                dists = self.lipschitz[t + 1] * torch.cdist(s_a, s_anchor, p=2)
                V = R_a + (upper_bounds[:, l_a, t+1].T + dists).min(dim=1).values.max(dim=0).values
                return V
        
        else:
            max_num_actions = max(unique_action_counts)

            node_idx_with_25_actions = [idx for idx, actions in enumerate(available_actions_list) if actions.shape[0] == max_num_actions][0]
            node_with_25_actions = available_actions_list[node_idx_with_25_actions]
            del available_actions_list[node_idx_with_25_actions]

            single_action_nodes = torch.stack(available_actions_list)

            with torch.no_grad():
                s_a_single_action = self.scm.forward(torch.cat([s[:node_idx_with_25_actions], s[node_idx_with_25_actions+1:]]), single_action_nodes.squeeze(1), self.u[t, :].expand(num_nodes-1, -1))
            
            l_a_single_action = torch.cat([l[:node_idx_with_25_actions], l[node_idx_with_25_actions+1:]])

            R_a = -s[:, -1]
            dists = self.lipschitz[t + 1] * torch.cdist(s_a_single_action, s_anchor, p=2)
            V_single_action = torch.cat((R_a[:node_idx_with_25_actions], R_a[node_idx_with_25_actions+1:])) + (upper_bounds[:, l_a_single_action, t+1].T + dists).min(dim=1).values

            s_expanded = s[node_idx_with_25_actions].unsqueeze(0).expand(max_num_actions, -1)
            with torch.no_grad():
                s_a_node_with_25_actions = self.scm.forward(s_expanded, node_with_25_actions, self.u[t, :].expand(max_num_actions, -1))

            l_a_node_with_25_actions = torch.where(torch.any(node_with_25_actions[:, :2] != self.a[t, :2], dim=-1), l[node_idx_with_25_actions] + 1, l[node_idx_with_25_actions])

            dists = self.lipschitz[t + 1] * torch.cdist(s_a_node_with_25_actions, s_anchor, p=2)
            V_node_with_25_actions = R_a[node_idx_with_25_actions] + (upper_bounds[:, l_a_node_with_25_actions, t+1].T + dists).min(dim=1).values.max(dim=0).values

            V = torch.cat([V_single_action[:node_idx_with_25_actions], V_node_with_25_actions.unsqueeze(0), V_single_action[node_idx_with_25_actions:]], dim=0)
            return V


    def get_neighbors_with_heuristic(self, node, k, upper_bounds, s_anchor):
        # return the neighbors of a given node along with the value of the heuristic function for each neighbor node

        if node.t == self.T-1:
            # the only neighbor is the goal node with the terminating action (42) and the heuristic is 0.0
            neighbors = [(Node(node.s, node.l, self.T, node.rwd - node.s[-1].item(), node, 42), 0.0)]
            return neighbors
        
        else:
        
            # the neighbors depend on the available actions but their time step is T-1, so the heuristic can be easily computed
            s, l, t = node.s, node.l, node.t
            R_s = -s[-1].item()
            available_actions = self._get_actions(t, l, k)
            num_actions = available_actions.shape[0]

            s_expanded = s.expand(num_actions, -1)

            # Apply transitions
            with torch.no_grad():
                s_child = self.scm.forward(s_expanded, available_actions, self.u[node.t, :].expand(num_actions, -1))
            
            l_child = torch.where(torch.all(available_actions[:,:2] == self.a[node.t, :2], dim=-1), l, l + 1)   # :-2 because we consider the mechvent action fixed
            
            if node.t == self.T-2:    
                neighbors = [(Node(s_child[i], l_child[i].item(), self.T-1, node.rwd + R_s, node, self._action_vector_to_id(available_actions[i])), -s_child[i,-1].item()) for i in range(num_actions)]
                return neighbors
            else:
                node_list = [Node(s_child[i], l_child[i].item(), t+1, node.rwd + R_s, node, self._action_vector_to_id(available_actions[i])) for i in range(num_actions)]
                V_a = self.heuristic_batch(node_list, k, upper_bounds, s_anchor)
                neighbors = [(node_list[i], V_a[i].item()) for i in range(num_actions)]
                return neighbors

    def reconstruct_path(self, node):
        # reconstructs the action sequence given by the A* algorithm
        actions = []
        states = []
        cf_reward = node.rwd
        while node is not None:
            actions.append(node.action)
            states.append(node.s)
            node = node.parent
        
        actions.reverse()
        states.reverse()
        actions = actions[1:-1]
        actions = torch.stack([self._action_id_to_vector(a) for a in actions])
        states = torch.stack(states[:-1])
        
        return actions, states, cf_reward
    
    def random_maximize(self, k):
        
        s_cf = self.s_all.clone()
        a_cf = self.a.clone()
        u = self.u

        available_actions = self._get_actions(0, 0, k)
        # select k rows of a and replace them with k random actions
        idx = torch.randperm(a_cf.shape[0])[:k]
        new_action_idx = torch.randint(0, available_actions.shape[0], (k,))
        a_cf[idx,:2] = available_actions[new_action_idx,:2]

        # compute the counterfactual states
        with torch.no_grad():
            for t in range(0, self.T-1):
                s_cf[t+1] = self.scm.forward(s_cf[t].unsqueeze(0), a_cf[t].unsqueeze(0), u[t].unsqueeze(0))
        
        # compute the counterfactual reward
        cf_reward = -s_cf[:, -1].sum().item()

        return a_cf, s_cf, cf_reward, 0, 0
    
    def topk_maximize(self, k):
        
        s_cf = self.s_all.clone()
        a_cf = self.a.clone()
        u = self.u

        if k>0:
            available_actions = self._get_actions(0, 0, k)
            num_available_actions = available_actions.shape[0]

            for t in range(0, self.T-1):

                # Expand action and state tensors to include all available actions for each time step
                a_cf_exp = a_cf.unsqueeze(1).expand(-1, num_available_actions, -1).clone()
                s_cf_exp = s_cf.unsqueeze(1).expand(-1, num_available_actions, -1).clone()

                # Replace actions at the current time step with all available actions
                a_cf_exp[t, :, :2] = available_actions[:, :2]

                # Compute the counterfactual states
                with torch.no_grad():
                    for t_inner in range(t, self.T-1):
                        s_cf_exp[t_inner+1] = self.scm.forward(
                            s_cf_exp[t_inner], a_cf_exp[t_inner], u[t_inner].unsqueeze(0).expand(num_available_actions, -1)
                        )

                # Compute the counterfactual rewards
                cf_rewards = -s_cf_exp[:, :, -1].sum(dim=0).squeeze()

                if t == 0:
                    all_rewards = cf_rewards.unsqueeze(1)
                else:
                    all_rewards = torch.concat((all_rewards, cf_rewards.unsqueeze(1)), dim=1)

            # Get the indices of the top k rewards
            top_k_indices = torch.topk(all_rewards.flatten(), k)[1]

            # Calculate the corresponding actions and time steps
            top_k_actions = available_actions[top_k_indices // (self.T-1), :2]
            top_k_time_steps = top_k_indices % (self.T-1)

            # Replace the actions in the action tensor with the top k actions
            a_cf[top_k_time_steps,:2] = top_k_actions

            # Compute the counterfactual states
            with torch.no_grad():
                for t in range(0, self.T-1):
                    s_cf[t+1] = self.scm.forward(s_cf[t].unsqueeze(0), a_cf[t].unsqueeze(0), u[t].unsqueeze(0))
        
        # Compute the counterfactual reward
        cf_reward = -s_cf[:, -1].sum().item()

        return a_cf, s_cf, cf_reward, 0, 0

    def greedy_maximize(self, k):

        s_cf = self.s_all.clone()
        a_cf = self.a.clone()
        u = self.u

        available_actions = self._get_actions(0, 0, k)
        max_reward = self.rewards[0]

        for _ in range(k):
            best_t = None
            best_action = None

            for t in range(0, self.T-1):
                # Replace actions at the current time step with all available actions
                a_cf_exp = a_cf.unsqueeze(0).expand(available_actions.size(0), -1, -1).clone()
                a_cf_exp[:, t, :2] = available_actions[:, :2]

                # Compute the counterfactual states
                s_cf_exp = s_cf.unsqueeze(0).expand(available_actions.size(0), -1, -1).clone()
                with torch.no_grad():
                    for t_inner in range(t, self.T-1):
                        s_cf_exp[:, t_inner+1] = self.scm.forward(
                            s_cf_exp[:, t_inner], a_cf_exp[:, t_inner], u[t_inner].unsqueeze(0).expand(available_actions.size(0), -1)
                        )

                # Compute the counterfactual rewards
                cf_rewards = -s_cf_exp[:, :, -1].sum(dim=1).squeeze()

                # Find the best action for the current time step
                best_action_idx = torch.argmax(cf_rewards)
                best_reward_t = cf_rewards[best_action_idx]

                # Update the best action and time step if the reward is higher
                if best_reward_t > max_reward:
                    max_reward = best_reward_t
                    best_t = t
                    best_action = available_actions[best_action_idx]

            # Update the action sequence with the best action found (if any)
            if best_t is not None:
                a_cf[best_t, :2] = best_action[:2]

                # Update the state sequence with the new action
                with torch.no_grad():
                    for t in range(best_t, self.T-1):
                        s_cf[t+1] = self.scm.forward(s_cf[t].unsqueeze(0), a_cf[t].unsqueeze(0), u[t].unsqueeze(0))
        
        # Compute the counterfactual reward
        cf_reward = -s_cf[:, -1].sum().item()

        return a_cf, s_cf, cf_reward, 0, 0


    def maximize(self, k, upper_bounds, s_anchor, root_idx, logger, nologging=False):

        root = Node(s=s_anchor[root_idx], l=0, t=0, rwd=0, parent=None, action=None)
        queue = [(0, root)]
        visited = set()

        added = 1
        num_visited = 0
        while queue:
            val, current = heapq.heappop(queue)
            num_visited += 1
            if num_visited % 500 == 0:
                if not nologging:
                    logger.info('Visited {n}, added {l} to the queue, current tentative reward is {rwd}'.format(n=num_visited, l=added, rwd=str(-val)))

            if current.t == self.T:
                cf_actions, cf_states, cf_reward = self.reconstruct_path(current)
                return cf_actions, cf_states, cf_reward, num_visited, added

            visited.add(current)

            neighbors_with_heur = self.get_neighbors_with_heuristic(current, k, upper_bounds, s_anchor)
            for neighbor, heur in neighbors_with_heur:
                if neighbor in visited:
                    continue
                
                added += 1
                tentative_rwd = neighbor.rwd + heur
                heapq.heappush(queue, (-tentative_rwd, neighbor))

        return None

    def evaluate(self, cf_actions, logger, nologging=False):
        # Compute the counterfactual states
        s_cf = self.s_all.clone()
        a_cf = cf_actions
        u = self.u

        with torch.no_grad():
            for t in range(0, self.T-1):
                s_cf[t+1] = self.scm.forward(s_cf[t].unsqueeze(0), a_cf[t].unsqueeze(0), u[t].unsqueeze(0))
    
        # Compute the counterfactual reward
        cf_reward = -s_cf[:, -1].sum().item()
        if not nologging:
            logger.info('Optimal counterfactual reward is {rwd}'.format(rwd=str(cf_reward)))

        return a_cf.cpu().numpy(), s_cf.cpu().numpy(), cf_reward

    def get_trajectory_actions(self):
        actions = self.trajectory['actions']
        return actions
    
    def get_trajectory_states(self):
        states = self.trajectory['states']
        return states


@click.command()
@click.option('--model_filename', type=str, required=True, help='file containing the trained SCM')
@click.option('--temp_directory', type=str, required=True, help='directory of temporary outputs')
@click.option('--processed_data_directory', type=str, required=True, help='directory to read the data from')
@click.option('--experiment_directory', type=str, required=True, help='directory of final outputs')
@click.option('--device', type=str, default='cpu', help='device to run the experiment on')
@click.option('--pid', type=int, default=11, help='patient ID to run the experiment on')
@click.option('--k', type=int, default=2, help='number of actions to change')
@click.option('--seed', type=int, default=42, help='random seed')
@click.option('--anchor_method', type=str, default='montecarlo-proportional', help='strategy to select the anchor set')
@click.option('--anchor_samples', type=int, default=1000, help='number of anchor points')
@click.option('--facility', type=int, default=None, help='if an integer is given, it computes facility location anchor sets up to that size')
@click.option('--algo', type=str, default='astar', help='algorithm to use for finding the optimal counterfactual action sequence (astar, greedy, topk, random)')
@click.option('--nologging', is_flag=True, default=False, help='flag to not print the logs')
@click.option('--exact', is_flag=True, default=False, help='flag to set an (almost) exact size for the anchor set')
def experiment(model_filename, temp_directory, processed_data_directory, experiment_directory, device, pid, k, seed, anchor_method, anchor_samples, facility, algo, nologging, exact):
    
    logging.basicConfig(level=logging.INFO)
    mdp = MimicMDP(model_filename=model_filename,temp_output_directory=temp_directory,\
                        data_directory=processed_data_directory, experiment_directory=experiment_directory, patient_id=pid, device=device, seed=seed)
    
    # prepare logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log.log')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    if facility is not None:
        if not nologging:
            logger.info('Computing facility location anchor sets...')
        s_anchor = mdp.solve_facility_location(size=facility, logger=logger)
        return

    if not nologging:
        logger.info('\n=============================')
        logger.info('Analyzing patient #{id}'.format(id=str(pid)))
        logger.info('=============================')

    if algo == 'astar':
        # compute the upper bounds for the anchor set
        start_time = time.time()
        upper_bounds, s_anchor, root_idx = mdp.compute_upper_bounds(k=k, anchor_samples=anchor_samples, anchor_method=anchor_method, logger=logger, nologging=nologging, exact=exact)    
        end_time = time.time()
        anchor_runtime = end_time - start_time
        if not nologging:
            logger.info('Anchor set size is {size}'.format(size=str(s_anchor.shape[0])))
            logger.info('Horizon is {horizon}'.format(horizon=str(mdp.T)))
            logger.info('Observed reward is {rwd}'.format(rwd=mdp.rewards[0].item()))

        # compute the optimal counterfactual trajectory
        start_time = time.time()
        cf_actions, cf_states, cf_reward, visited, added = mdp.maximize(k=k, upper_bounds=upper_bounds, s_anchor=s_anchor, root_idx=root_idx, logger=logger, nologging=nologging)
        end_time = time.time()
        astar_runtime = end_time - start_time

        ebf = compute_ebf(added, mdp.T)
        anchor_set_size = s_anchor.shape[0]
    else:
        anchor_runtime = 0.0
        ebf=0.0
        anchor_set_size = 0
        if not nologging:
            logger.info('Horizon is {horizon}'.format(horizon=str(mdp.T)))
            logger.info('Observed reward is {rwd}'.format(rwd=mdp.rewards[0].item()))

        if algo == 'greedy':
            start_time = time.time()
            cf_actions, cf_states, cf_reward, visited, added = mdp.greedy_maximize(k=k)
            end_time = time.time()
            astar_runtime = end_time - start_time
        elif algo == 'topk':
            start_time = time.time()
            cf_actions, cf_states, cf_reward, visited, added = mdp.topk_maximize(k=k)
            end_time = time.time()
            astar_runtime = end_time - start_time
        elif algo == 'random':
            start_time = time.time()
            cf_actions, cf_states, cf_reward, visited, added = mdp.random_maximize(k=k)
            end_time = time.time()
            astar_runtime = end_time - start_time
        else:
            raise ValueError('Unknown algorithm {algo}'.format(algo=algo))

    cf_actions, cf_states, cf_reward = mdp.evaluate(cf_actions, logger, nologging=nologging)

    if not nologging:
        logger.info('Saving results...')
    actions = mdp.get_trajectory_actions()
    states = mdp.get_trajectory_states()
    reward = mdp.rewards[0].item()
    survived = mdp.trajectory['survived']

    summary = generate_summary(pid, k, seed, anchor_method, anchor_samples, mdp.hidden_layers, mdp.hidden_units, mdp.lipschitz_loc, mdp.lipschitz_scale, mdp.prior_type, mdp.T,\
                                    actions, states, survived, reward, cf_actions, cf_states, cf_reward, visited, added,\
                                        anchor_runtime, astar_runtime, ebf, anchor_set_size, algo)

    filename = 'patient_{pid}_k_{k}_seed_{seed}_anchor_{anchor}_anchor_samples_{anchor_samples}_hl_{hl}_hu_{hu}_lipschitzloc_{lipschitzloc}_lipschitzscale_{lipschitzscale}_prior_{prior}_algo_{algo}'.format(\
                    pid=pid, k=k, seed=seed, anchor=anchor_method, anchor_samples=anchor_samples, hl=mdp.hidden_layers, hu=mdp.hidden_units, lipschitzloc=mdp.lipschitz_loc,\
                        lipschitzscale=mdp.lipschitz_scale, prior=mdp.prior_type, algo=algo)

    output = ''.join([experiment_directory, filename])
    with open('{output}.json'.format(output=output), 'w') as outfile:
        json.dump(summary, outfile)

    return

if __name__ == '__main__':
    experiment()
    # experiment(model_filename='outputs/models/mimic_transitions_hl_1_hu_100_lr_0.001_bs_256_lipschitzloc_1.2_lipschitzscale_0.01_prior_laplace_maxepochs_100.pt',
    #            temp_directory='outputs/temp_outputs/', processed_data_directory='data/processed/', experiment_directory='outputs/experiments/', device='cpu',
    #            pid=11, k=3, seed=42, anchor_method='facility-location', anchor_samples=10000, facility=None, algo='astar', nologging=False, exact=True)