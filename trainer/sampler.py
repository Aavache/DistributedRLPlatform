"""https://github.com/anyscale/academy/tree/main/ray-rllib/multi-armed-bandits"""

import numpy as np
import random
import boto3
import cassandra
from cassandra.cluster import Cluster
from ray.rllib.policy.sample_batch import SampleBatch

# https://docs.ray.io/en/master/tune/tutorials/tune-serve-integration-mnist.html
class CassandraSampler(object):
    def __init__ (self, config={}): 
        self.batch_size = config['batch_size']
        self.minibatch_size = config['minibatch_size']
        self.iters_per_batch = config['iters']
        self.iters = 0

        # Connect to cassandra through boto3
        self.query = "SELECT * FROM {} LIMIT {}".format(config['table'], self.batch_size)
        try:
            db_conn = Cluster([config['address']], port=int(config['port']))
            self.session = db_conn.connect(config['keyspace'])
            print("The connection the database was succesful")
        except Exception as e:
            raise Exception("Connection to cassandra failed: {}".format(str(e))) 

        # Get initial batch
        self.batch_def = {'eps_id':[],'unroll_id':[],'obs':[],
                        'actions':[],'rewards':[],'new_obs':[],
                        'dones':[],'agent_index':[], 'weights':[]} # TODO: make it general for other algorithms
        self.batch = self.batch_def.copy()
        self.last_episode = -1

    def get_batch(self):
        '''This methods returns a mini-batch sampled from the cassandra database where the experience records 
        are stored. If the batch is None, there is still not new available data and should be requested later'''
        if len(self.batch['obs']) == 0 or (self.iters >= self.iters_per_batch and self.iters!=0):
            valid = self._pull_batch()
            self.iters=0
            if not valid:
                self._clear_batch()
                return None
        sample_batch =  SampleBatch(self._sample_batch()) # Encapsulating the dictionary with rllib wrapper
        self.iters += 1
        return sample_batch

    def _pull_batch(self):
        '''This methods queries a sorted batch from the cluster db. The table should be defined in such a way the 
        episodes and steps are already sorted.'''
        try:
            table = self.session.execute(self.query)
            if len(table._current_rows) > 0:
                self._clear_batch()
                self._parse_table(table)
                return self._validate_batch()
            else:
                return False
        except Exception as e:
            print("Error: failed to execute the fetching query, {}".format(str(e)))
            return False

    def _parse_table(self, table):
        for row in table:
            self.batch['eps_id'].append(row.episode)
            self.batch['unroll_id'].append(row.step)
            self.batch['obs'].append(row.state)
            self.batch['actions'].append(row.action)
            self.batch['rewards'].append(row.reward)
            self.batch['new_obs'].append(row.next_state)
            self.batch['dones'].append(row.done)
            # Secondary fields
            self.batch['agent_index'].append(0)
            self.batch['weights'].append(1)

    def _sample_batch(self):
        '''This method samples a minibatch from the fetched batch for training'''
        minibatch = {}
        batch_size = len(self.batch['obs'])
        idx = random.randint(0,batch_size-self.minibatch_size-1)
        for key in list(self.batch_def.keys()):
            # Reversed is applied because the table is descently ordered by episode however
            minibatch[key] = list(reversed(self.batch[key][idx:idx+self.minibatch_size]))
        return minibatch

    def _clear_batch(self):
        del self.batch
        self.batch = self.batch_def.copy()

    def _validate_batch(self):
        '''Checking whether more data has been added to the database'''
        current_last_episode = int(self.batch['eps_id'][0])
        if current_last_episode > self.last_episode:
            self.last_episode = current_last_episode
            return True
        else:
            return False

    def _create_fake_batch(self):
        '''
        This is for testing purposes, let us evaluate the Cartpole environemnt definition of state-action
        Observation space is defined as:
            0       Cart Position             -4.8                    4.8
            1       Cart Velocity             -Inf                    Inf
            2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
            3       Pole Angular Velocity     -Inf                    Inf
        Action space is defined as:
            0     Push cart to the left
            1     Push cart to the right
        '''
        states, actions, action_probs, action_logps, \
                 action_dist_inputs, vf_preds, value_targets, advantages, rewards, dones = ([] for i in range(10))
        steps = list(range(128+1))
        # Generating random batch
        for step in steps:
            states.append(np.array([1.2,5.0,0.1,5.0]))
            actions.append(0)
            action_probs.append(np.array([0.8]))
            action_logps.append(np.array([-0.096]))
            #action_dist_inputs.append(np.array([0.8,0.2])) # PPO
            #vf_preds.append(2.0) # PPO
            #value_targets.append(2.0) # PPO
            #advantages.append(0.1) # PPO
            rewards.append(0.1)
            dones.append(False)

        # PPO/_SAC
        batch_dict = {"obs": states[:-1], "actions": actions[:-1], "rewards": rewards[:-1], 
                        "new_obs": states[1:], "dones": dones[:-1], "weights": np.ones(len(states)-1), 
                        "eps_id":np.zeros(len(states)-1),"unroll_id": steps[:-1], 
                        "agent_index": np.zeros(len(states)-1), #"action_prob": action_probs[:-1],
                        #"action_logp": action_logps[:-1], #"action_dist_inputs":action_dist_inputs[:-1],
                        #"vf_preds":vf_preds[:-1], "advantages":advantages[:-1],"value_targets":value_targets[:-1]
        }
        return SampleBatch(batch_dict)     