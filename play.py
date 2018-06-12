'''
The main script.Let the agent to play a specific game.
'''
import tensorflow as tf
import random
from brain_dqn import *
import utils
from collections import namedtuple
import game
import multiprocessing
import threading

def play(env,session,timesteps_num):
    def stopping_criterion(env, t):
        return env.stop_criterion(t)
    ##########################
    # learning rate schedule #
    ##########################
    iterations_num=float(timesteps_num)/4.0
    lr_multiplier=1.0
    lr_schedule=utils.PiecewiseSchedule([
        (0 , 1e-4*lr_multiplier),
        (iterations_num/10 , 1e-4*lr_multiplier),
        (iterations_num/2 , 5e-5*lr_multiplier)
    ],outside_value=5e-5*lr_multiplier)
    #################
    # set optimizer #
    #################
    OptimizerSepc = namedtuple('OptimizerSpec', ['constructor', 'kwargs', 'lr_schedule'])
    optimizer=OptimizerSepc(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )
    ########################
    # exploration schedule #
    ########################
    exploration_schedule=utils.PiecewiseSchedule([
        (0 , 1.0),
        (1e6 , 0.1),
        (iterations_num/2 , 0.01)
    ],outside_value=0.01)
    #################
    # play the game #
    #################
    '''
    worker_max_num=multiprocessing.cpu_count()
    numworkers=10
    assert numworkers<worker_max_num
    for i in range(numworkers):
        t = threading.Thread(target=work, args=(str(i)))
        t.start()
    '''
    dqn_worker(env=env,
                 name='dqn_worker',
                 optimizer_spec=optimizer,
                 session=session,
                 exploration=exploration_schedule,
                 replay_buffer_size=1000000,
                 batch_size=32,
                 gamma=0.99,
                 learn_start=50000,
                 learn_freq=4,
                 history_frames_num=4,
                 target_update_freq=10000,
                 grad_norm_clipping=10,
                 stop_criterion=stopping_criterion
                 )



def set_global_seeds(i):
    tf.set_random_seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    session=tf.Session(config=tf_config)
    #print('AVAILABLE GPUS:',utils.get_available_gpus())
    return session

if __name__ == '__main__':

    #seed = 0
    env = game.qiuqiu_env()
    session = get_session()
    play(env, session, timesteps_num=10000000)