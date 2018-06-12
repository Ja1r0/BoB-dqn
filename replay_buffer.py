'''
The structure of replay buffer.
'''
import numpy as np
import random

class Replay_buffer:
    def __init__(self,size,history_frames):
        '''
        :param size: the capacity of replay buffer
        :param history_frames: the number of recent frames to stack
        '''
        self.size=size
        self.history_frames=history_frames
        self.next_idx=0
        self.num_in_buffer=0
        self.obs=None
        self.act=None
        self.rew=None
        self.end=None

    def can_sample(self,batch_size):
        '''
        :param batch_size: the number of samples per batch
        :return: True if num_in_buffer > batch_size
        '''
        return batch_size+1<=self.num_in_buffer

    def sample(self,batch_size):
        '''
        :param batch_size: the number of transitions to sample
        :return:
        obs_batch
        =========
            -{ndarray}
            -(batch_size , img_h , img_w , img_c*history_frames)
            -np.unint8
        act_batch
        =========
            -{ndarray}
            -(batch_size,)
            -np.int32
        rew_batch
        =========
            -{ndarray}
            -(batch_size,)
            -np.float32
        next_obs_batch
        ==============
            -{ndarray}
            -(batch_size , img_h , img_w , img_c*history_frames)
            -np.uint8
        done_batch
        ==========
            -{ndarray}
            -(batch_size,)
            -np.float32
        '''
        assert self.can_sample(batch_size)
        idxes=sample_n_unique(lambda: random.randint(0,self.num_in_buffer-2),batch_size)
        return self._stack_sample(idxes)

    def store_frame(self,frame):
        '''
        push the current frame into the buffer in the next available index,
        delete the oldest frame if the buffer is filled.
        :param: frame
            -{ndarray}
            -(img_h,img_w,img_c)
            -np.uint8
        :return: idx
            the index at where the frame is stored.
            -{int}
        '''
        if self.obs is None:
            self.obs=np.empty([self.size]+list(frame.shape),dtype=np.uint8)
            self.act=np.empty([self.size],dtype=np.int32)
            self.rew=np.empty([self.size],dtype=np.float32)
            self.end=np.empty([self.size],dtype=np.bool)
        self.obs[self.next_idx]=frame
        idx_of_frame=self.next_idx
        self.next_idx=(self.next_idx+1)%self.size
        self.num_in_buffer=min(self.size,self.num_in_buffer+1)
        return idx_of_frame

    def store_transition(self,idx,action,reward,done):
        '''
        :param idx: {int}
        :param action: {int}
        :param reward: {float}
        :param done: {bool}
        :return: store the transition sample (a,r,done) to buffer
        '''
        self.act[idx]=action
        self.rew[idx]=reward
        self.end[idx]=done

    def stack_recent_obs(self):
        '''
        :return: observations
            -{ndarray}
            -(img_h,img_w,img_c*history_frames)
            -np.uint8
            stack recent observations of length history_frames
        '''
        assert self.num_in_buffer>0
        return self._stack_obs((self.next_idx-1)%self.size)

    def _stack_sample(self,idxes):
        obs_batch=np.concatenate([self._stack_obs(idx)[None] for idx in idxes],0)
        act_batch=self.act[idxes]
        rew_batch=self.rew[idxes]
        next_obs_batch=np.concatenate([self._stack_obs(idx+1)[None] for idx in idxes],0)
        done_batch=np.array([1.0 if self.end[idx] else 0.0 for idx in idxes],dtype=np.float32)
        return obs_batch,act_batch,rew_batch,next_obs_batch,done_batch

    def _stack_obs(self,idx):
        end_idx=idx+1
        start_idx=end_idx-self.history_frames
        if start_idx<0 and self.num_in_buffer!=self.size:
            start_idx=0
        for idx in range(start_idx,end_idx-1):
            if self.end[idx%self.size]:
                start_idx=idx+1
        missing_context=self.history_frames-(end_idx-start_idx)
        if start_idx<0 or missing_context>0:
            frames=[np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx,end_idx):
                frames.append(self.obs[idx%self.size])
            return np.concatenate(frames,2)
        else:
            img_h,img_w=self.obs.shape[1],self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1,2,0,3).reshape(img_h,img_w,-1)
