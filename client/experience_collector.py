'''This script emulates a game client that uses the trained models to collect new experience. The client sends data to the cluster;
to this end, the client frequently downloads newer versions of the model to keep generating more substantial game records'''

# External dependencies
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Internal dependencies
from connector import ClusterConnector
from envwrapper import EnvWrapper
import utils

class Memory():
    def __init__(self):
        self.episodes = []
        self.steps = []
        self.states = []
        self.actions = []
        self.action_probs = []
        self.action_logps = []
        self.vf_preds = []
        self.value_targets = []
        self.advantages = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.episodes[:]
        del self.steps[:]
        del self.states[:]
        del self.actions[:]
        del self.action_probs[:]
        del self.action_logps[:]
        del self.vf_preds[:]
        del self.value_targets[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.dones[:]

class MovingAverage():
    def __init__(self, window_size):
        self.reward_window = []
        self.window_size = window_size

    def push(self,rewards):
        if isinstance(rewards, list):
            self.reward_window += rewards
        else:
            self.reward_window.append(rewards)
        while len(self.reward_window) > self.window_size:
            del self.reward_window[0]

    def average(self):
        return np.mean(self.reward_window)
    
    def sum(self):
        return np.sum(self.reward_window)

class Agent():
    def __init__(self, client, deterministic=False, device='cpu'):
        self.client = client
        self.device = device
        self.deterministic = deterministic
        self.load_model()
        self.model.to(self.device)
        self.properties = ['episode', 'step', 'state', 'action', 'reward', 'next_state', 'done'] # SAC algorithm

    def forward(self, state):
        input = {'obs':state.to(self.device)}
        action_logits = self.model.action_model(input)[0] # [action_size]
        action_probs = F.softmax(action_logits)
        values = self.model.get_q_values(input)
        if not self.deterministic:
            m = Categorical(action_probs)
            action = m.sample()
            #action_logp = m.log_prob(action)
        else:
            action = torch.argmax(action_probs, dim=1, keepdim=False)
        return action, values[0][0,action]

    def parse_batch(self,batch):
        records = []
        for i in range(len(batch.states)):
            if i<len(batch.states)-1:
                record = (batch.episodes[i],batch.steps[i], 
                    batch.states[i],batch.actions[i], # (s,a,r,s')
                    batch.rewards[i],batch.states[i+1],
                    batch.dones[i])
            else:
                record = (batch.episodes[i],batch.steps[i],  #TODO: get next state from final state
                    batch.states[i],batch.actions[i], # (s,a,r,s')
                    batch.rewards[i],batch.states[0],
                    batch.dones[i])
            records.append(record)
        return records
    
    def load_model(self):
        self.client.download_model()
        self.model = torch.load(self.client.weight_local_path, map_location=self.device)

def state_to_tensor(state):
    return torch.from_numpy(state).unsqueeze(0)

def main(): # Code start here!
    # Parameters and objects
    opt_file = 'config.json'
    opt = utils.load_json(opt_file)
    device = 'cuda:{}'.format(opt['device']) if opt['device'] >= 0 and torch.cuda.is_available() else 'cpu'
    utils.mkdir(os.path.join(opt['output_dir'], opt['experiment_name']))

    # Initialization
    cluster_client = ClusterConnector(opt)
    env =  EnvWrapper(opt)
    agent = Agent(cluster_client, opt['agent']['deterministic'], device)
    memory = Memory()
    reward_window = MovingAverage(opt["window_size"])
    plot_reward, plot_count, iters = [], 0, 0

    # Experience collection loop
    for episode in range(opt['total_episodes']):
        state = env.reset()
        is_done=False
        state_tensor = state_to_tensor(state)
        episode_reward = 0.0
        step = 0
        # Episode loop
        while not is_done:
            # Evaluation of the policy for current state
            action = agent.forward(state_tensor)[0]
            state, reward, is_done = env.step(action.cpu().detach().numpy()[0])
            state_tensor = state_to_tensor(state)

            # Storing the records
            memory.episodes.append(episode)
            memory.steps.append(step)
            memory.states.append(state)
            memory.actions.append(action.cpu().detach().numpy()[0])
            memory.rewards.append(reward)
            memory.dones.append(is_done)
            episode_reward+=reward
            step += 1 
            iters += 1
            if opt['env']['render']:
                env.render()
        reward_window.push(episode_reward)
        # Send the experience collected to the cluster
        if episode % opt['upload_freq'] == 0:
            batch = agent.parse_batch(memory)
            cluster_client.upload_batch(batch, agent.properties)
            memory.clear()
            time.sleep(7.5)
        # Updating the model by downloadinf from s3 bucket
        if episode % opt['download_freq'] == 0:
            agent.load_model()
        # Plotting performance
        if episode % opt['plot_freq'] == 0:
            plot_reward.append(reward_window.average())
            plot_count += 1
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)
            ax.plot(range(plot_count),plot_reward)
            ax.set_xlabel("Iters")
            ax.set_ylabel("Reward")
            ax.set_title("Average reward {}".format(opt['env']['env_name']))
            fig.savefig(os.path.join(opt['output_dir'], 
                                    opt['experiment_name'],
                                    'plot_reward.png'))
            plt.cla()
            plt.close(fig)

if __name__ == "__main__":
    main()
    