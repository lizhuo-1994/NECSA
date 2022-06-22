import os,sys
import copy
import numpy as np
import pickle, joblib, time
from .interfaces import grid_abs_analysis
from .interfaces import Grid
from multiprocessing import Process  
import scipy.stats as stats
from multiprocessing import Queue
import json

class ScoreInspector:
    
    def __init__(self, order, grid_num, state_dim, state_min, state_max, action_dim, action_min, action_max, mode):

        self.order = order
        self.grid_num = grid_num
        self.state_dim = state_dim
        self.state_min = state_min
        self.state_max = state_max
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.basic_states = None
        self.basic_states_times = None
        self.basic_states_scores = None
        self.basic_states_proceeds = None
        self.mode = mode
        
        self.score_avg = None
        self.pcaModel = None
        self.performance_list = []
        self.avg_performance_list = []

        
        #self.QUEUE_LEN
        self.s_token = Queue(10)
        self.r_token = Queue(10)
        
        self.setup()

    
    def setup(self):

        if self.mode == 'state':
            self.min_state = np.array([self.state_min for i in range(self.state_dim)])
            self.max_state = np.array([self.state_max for i in range(self.state_dim)])
        elif self.mode == 'state_action':
            self.min_state = np.array([self.state_min for i in range(self.state_dim)] + [self.action_min for i in range(self.action_dim)])
            self.max_state = np.array([self.state_max for i in range(self.state_dim)] + [self.action_max for i in range(self.action_dim)])

        self.min_avg_proceed = 0
        self.max_avg_proceed = 10

        #self.scores = scores
        self.score_avg = 0
        
        #self.states_info = self.setup_score_dict(states, times, proceeds, scores, values)
        self.states_info = dict()
        
        #self.pcaModel = joblib.load(config.PCA_MODEL_PATH)
        self.grid = Grid(self.min_state, self.max_state, self.grid_num)   


    def discretize_states(self, con_states):
        abs_states = self.grid.state_abstract(con_states)
        return abs_states
    
    def inquery(self, pattern):
        if pattern in self.states_info.keys():
            return self.states_info[pattern]['score'], self.states_info[pattern]['time'] 
        else:
            return None, None

    def sync_scores(self):
        if self.s_token.qsize() > 0:

            new_states_info, min_avg_proceed, max_avg_proceed = self.s_token.get()
            
            if min_avg_proceed < self.min_avg_proceed:
                self.min_avg_proceed = min_avg_proceed
            if max_avg_proceed > self.max_avg_proceed:
                self.max_avg_proceed = max_avg_proceed

            self.states_info.update(new_states_info)
            self.score_avg = np.mean([self.states_info[abs_state]['score'] for abs_state in self.states_info.keys()])
            
            
            print('############################################################')
            #print('Abstract states :\t', self.states_info)
            print('Abstract states number :\t', len(self.states_info.keys()))
            print('Average states score :\t', self.score_avg)
            print('Queue size :\t',self.s_token.qsize())
            print('min and max proceed', self.min_avg_proceed, self.max_avg_proceed)
            print('############################################################')
            
            
    
    def start_pattern_abstract(self, con_states, rewards):

        con_states = np.array(con_states)

        if self.mode == 'state':
            con_states = con_states[:,:self.state_dim]
        elif self.mode == 'state_action':
            con_states = con_states[:,:self.state_dim + self.action_dim]

        t = Process(target = self.pattern_abstract, args = (con_states, rewards))
        t.daemon = True
        t.start()

    def pattern_abstract(self, con_states, rewards):

        abs_states = self.discretize_states(con_states)
        min_avg_proceed = self.min_avg_proceed
        max_avg_proceed = self.max_avg_proceed

        new_states_info = dict()
        normal_scale = self.max_avg_proceed - self.min_avg_proceed

        proceed = sum(rewards)
        for i in range(len(abs_states)):
            if i + self.order >= len(abs_states):
                break
                
            
            if proceed < self.min_avg_proceed:
                min_avg_proceed = proceed
            if proceed > self.max_avg_proceed:
                max_avg_proceed = proceed
            pattern = abs_states[i:i+self.order]
            pattern = '-'.join(pattern)

            if pattern in self.states_info.keys():
                new_states_info[pattern] = self.states_info[pattern]
                new_states_info[pattern]['proceed'] += proceed
                new_states_info[pattern]['time'] += 1
                average_proceed = new_states_info[pattern]['proceed'] / new_states_info[pattern]['time']
                score = (average_proceed - self.min_avg_proceed)  / normal_scale
                score = np.clip(score, 0, 1)
                new_states_info[pattern]['score'] =  score
            else:
                new_states_info[pattern] = {}
                new_states_info[pattern]['proceed'] = proceed
                new_states_info[pattern]['time'] = 1
                score = (proceed - self.min_avg_proceed) / normal_scale
                score = np.clip(score, 0, 1)
                new_states_info[pattern]['score'] =  score

        self.s_token.put((new_states_info, min_avg_proceed, max_avg_proceed))

    


class Abstracter:
    
    def __init__(self, order, decay):
        self.con_states = []
        self.con_values = []
        self.con_reward = []
        self.con_dones  = []
        self.order = order
        self.decay = decay
        self.inspector = None

        
        
    def append(self, con_state, reward, done):
        self.con_states.append(con_state)
        self.con_reward.append(reward)
        self.con_dones.append(done)

        if done:
            self.inspector.start_pattern_abstract(self.con_states, self.con_reward)
            self.clear()
    
    def clear(self):
        self.con_states = []
        self.con_reward = []
        self.con_dones  = []
    
    def handle_pattern(self,con_states,rewards):
        
        abs_pattern = self.inspector.discretize_states(con_states)
        
        if len(abs_pattern) != self.order:
            return rewards[0]
        pattern = '-'.join(abs_pattern)
        score, time = self.inspector.inquery(pattern)
        
        if score != None:
            if  time > 1:
                delta = (score - self.inspector.score_avg) * self.decay
                rewards[0] += delta
                #self.inspector.states_info[pattern]['score'] = self.inspector.states_info[pattern]['score'] * 0.99

        return rewards[0]



    def reward_shaping(self, state_list, reward_list):

        shaping_reward_list = copy.deepcopy(reward_list)

        for i in range(len(state_list) - self.order):

            target_states = state_list[i:i+self.order]
            target_rewards = reward_list[i:i+self.order]

            shaped_reward = self.handle_pattern(target_states, target_rewards)
            shaping_reward_list[i] = shaped_reward
        
        shaping_reward_list = np.array(shaping_reward_list)
        return shaping_reward_list