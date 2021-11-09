"""Example of a custom training workflow. Run this for a demo.
This example shows:
  - using Tune trainable functions to implement custom training workflows
You can visualize experiment results in ~/ray_results using TensorBoard.

Examples by the author: https://docs.ray.io/en/master/rllib-examples.html
Interesting course about RLLIB: https://github.com/anyscale/academy/tree/main/ray-rllib/multi-armed-bandits
"""
# External libs
import argparse
import os
import time
import torch
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer 
from sampler import CassandraSampler
import boto3
import copy
# Internal libs
import utils

parser = argparse.ArgumentParser()
# parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", default=True, action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--localdir", default= "./results/", type=str)
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)

def save_checkpoint(agent, conn, datapath, bucket_name):
    '''This method saves the pytorch policy into an s3 bucket, we can connect to the s3 by mounting the s3
    in the ec2 instance or using boto3 library for easy transfer.
        - Mounting s3: https://www.youtube.com/watch?v=FFTxUlW8_QQ&ab_channel=ValaxyTechnologies
        - boto3 transfer:
    '''
    # Using boto3 transfering
    torch.save(agent.get_policy().model, datapath) # Storing the file in the local storage
    # Transfering the model's weights with boto3 to s3
    conn.upload_file(datapath, bucket_name, datapath) # args: [source_path, bucket_name, target_path]

def my_train_fn(config, reporter):
    '''
        Training function, this method updates the algorithm by pulling batches 
        from the database.
    '''
    # Unpacking parameters
    model_name = config['my_train_fn_config']['model']
    env_name = config['my_train_fn_config']['env_name']
    saving_freq = config['my_train_fn_config']['saving_freq']
    bucket_name = config['my_train_fn_config']['bucket_name']
    weight_name = config['my_train_fn_config']['weight_name']
    cassandra_config = config['my_train_fn_config']['cassandra'].copy()
    del config['my_train_fn_config'] # This info is not acceptable for the ray trainer

    # Loading desired model
    if model_name == 'sac':
        agent = SACTrainer(env=env_name, config=config) # new models can be defined, custom env would require custom trainers
    elif model_name == 'ppo':
        agent = PPOTrainer(env=env_name, config=config)
    else:
        raise NotImplementedError("Model {} is not implemented".format(model_name))

    sampler = CassandraSampler(cassandra_config)
    bucket_connector = boto3.client('s3')

    save_checkpoint(agent, bucket_connector, weight_name, bucket_name) # Saving the init model, so that it can be used by the clients
    update_count = 0
    while True: # Training loop
        # Sampling a new batch
        batch = sampler.get_batch()
        if batch is not None:
            # Updating parameters
            agent.get_policy().learn_on_batch(batch)
            update_count+=1
            # Checkpointing the model after several updates
            if update_count % saving_freq == 0:
                save_checkpoint(agent, bucket_connector, weight_name, bucket_name)
                print("New model checkpoint has been succesfully transfered to the bucket")
        else:
            print('WAIT: There is no new available records...')
            time.sleep(3.0)

if __name__ == "__main__":
    # ray.init()
    # ray.init(address="auto")
    ray.init(local_mode=True) # For debugging
    print("ray is initialized!")
    args = parser.parse_args()

    # Creating dirs
    utils.mkdirs(["./results", "./checkpoints"])

    # Defining the training configuration
    config = {
        "lr":0.001,
        "num_gpus": 0,
        "num_workers": 0,
        "framework": "torch" if args.torch else "tf",
        "train_batch_size": 128,
        "rollout_fragment_length": 16,
        'explore':  False,
        'my_train_fn_config':{
                "model":"sac", # ppo
                'bucket_name': 'weightstorage',
                'weight_name': 'weights_def.pt',
                "env_name":"CartPole-v0",
                "saving_freq": 20,
                'cassandra':{   
                        'batch_size': 128,
                        "minibatch_size": 8,
                        "iters": 64,
                        'address': '3.36.133.139',
                        'port': 9042,
                        'keyspace': 'test',
                        'table': 'cartpole_sac',
                        }
                }
    } # it can also be loaded from a external file

    results = tune.run(my_train_fn, config=config, local_dir=args.localdir)

    ray.shutdown()
    print("ray is shutdown!")
    