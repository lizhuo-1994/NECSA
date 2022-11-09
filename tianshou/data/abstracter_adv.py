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
    
    def __init__(self, step, grid_num, raw_state_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, mode, reduction):

        self.step = step
        self.grid_num = grid_num
        self.raw_state_dim = raw_state_dim
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
        self.reduction = reduction

        self.performance_list = []
        self.avg_performance_list = []

        
        #self.QUEUE_LEN
        self.s_token = Queue(10)
        self.r_token = Queue(10)
        
        self.setup()


    
    def setup(self):

        self.min_state = np.array([self.state_min for i in range(self.raw_state_dim)])
        self.max_state = np.array([self.state_max for i in range(self.raw_state_dim)])
        self.min_action = np.array([self.action_min for i in range(self.action_dim)])
        self.max_action = np.array([self.action_max for i in range(self.action_dim)])
            
        self.min_avg_proceed = 0
        self.max_avg_proceed = 1000

        #self.states_info = self.setup_score_dict(states, times, proceeds, scores, values)
        self.states_info = dict()
        
        self.state_grid = Grid(self.min_state, self.max_state, self.grid_num)   
        self.action_grid = Grid(self.min_action, self.max_action, self.grid_num) 

    def save(self, env_name):

        with open(env_name + '.json', 'w') as f:
            json.dump(self.states_info, f)

    def discretize_states(self, con_states):
        abs_states = self.state_grid.state_abstract(con_states)
        return abs_states

    def discretize_actions(self, con_actions):
        abs_actions = self.action_grid.state_abstract(con_actions)
        return abs_actions
    
    def average_score(self, pattern_info):
        sum_score = 0
        count = 0
        for action in pattern_info.keys():
            sum_score += pattern_info[action]['score']
            count += 1
        return sum_score/count


    def inquery(self, abs_pattern, abs_action, baseline = False):

        if abs_pattern in self.states_info.keys():
            if baseline:
                return self.average_score(self.states_info[abs_pattern])
            if abs_action in self.states_info[abs_pattern].keys():
                return self.states_info[abs_pattern][abs_action]['score']
        
        return None

    def sync_scores(self):
        if self.s_token.qsize() > 0:

            new_states_info, min_avg_proceed, max_avg_proceed = self.s_token.get()
            
            if min_avg_proceed < self.min_avg_proceed:
                self.min_avg_proceed = min_avg_proceed
            if max_avg_proceed > self.max_avg_proceed:
                self.max_avg_proceed = max_avg_proceed

            self.states_info.update(new_states_info)
            
            
            # print('############################################################')
            # #print('Abstract states :\t', self.states_info)
            # print('Abstract states number :\t', len(self.states_info.keys()))
            # print('Average states score :\t', self.score_avg)
            # print('Queue size :\t',self.s_token.qsize())
            # print('min and max proceed', self.min_avg_proceed, self.max_avg_proceed)
            # print('############################################################')
            
            
    
    def start_pattern_abstract(self, con_states, con_actions, rewards):

        con_states = np.array(con_states)
        con_actions = np.array(con_actions) 

        t = Process(target = self.pattern_abstract, args = (con_states, con_actions, rewards))
        t.daemon = True
        t.start()

    def pattern_abstract(self, con_states, con_actions, rewards):

        abs_pattern = self.discretize_states(con_states)
        abs_action = self.discretize_actions(con_actions)
        min_avg_proceed = self.min_avg_proceed
        max_avg_proceed = self.max_avg_proceed

        new_states_info = dict()
        normal_scale = self.max_avg_proceed - self.min_avg_proceed

        proceed = sum(rewards)
        for i in range(len(abs_pattern)):
            if i + self.step >= len(abs_pattern):
                break
                
            
            if proceed < self.min_avg_proceed:
                min_avg_proceed = proceed
            if proceed > self.max_avg_proceed:
                max_avg_proceed = proceed
            abs_pattern = abs_pattern[i:i+self.step]
            abs_action = abs_action[i+self.step -1:i+self.step]
            abs_pattern = '-'.join(abs_pattern)
            abs_action = '-'.join(abs_action)

            if abs_pattern in self.states_info.keys():
                new_states_info[abs_pattern] = self.states_info[abs_pattern]
                if abs_action in new_states_info[abs_pattern].keys():
                    new_states_info[abs_pattern][abs_action]['proceed'] += proceed
                    new_states_info[abs_pattern][abs_action]['time'] += 1
                    average_proceed = new_states_info[abs_pattern][abs_action]['proceed'] / new_states_info[abs_pattern][abs_action]['time']
                    score = (average_proceed - self.min_avg_proceed)  / normal_scale
                    score = np.clip(score, 0, 1)
                    new_states_info[abs_pattern][abs_action]['score'] =  score
                else:
                    new_states_info[abs_pattern][abs_action] = {}
                    new_states_info[abs_pattern][abs_action]['time'] = 1
                    new_states_info[abs_pattern][abs_action]['proceed'] = proceed
                    score = (proceed - self.min_avg_proceed) / normal_scale
                    score = np.clip(score, 0, 1)
                    new_states_info[abs_pattern][abs_action]['score'] =  score
            else:
                new_states_info[abs_pattern] = {}
                new_states_info[abs_pattern][abs_action] = {}
                new_states_info[abs_pattern][abs_action]['time'] = 1
                new_states_info[abs_pattern][abs_action]['proceed'] = proceed
                score = (proceed - self.min_avg_proceed) / normal_scale
                score = np.clip(score, 0, 1)
                new_states_info[abs_pattern][abs_action]['score'] =  score

        self.s_token.put((new_states_info, min_avg_proceed, max_avg_proceed))

    


class Abstracter:
    
    def __init__(self, step, epsilon):
        self.con_states = []
        self.con_actions = []
        self.con_values = []
        self.con_reward = []
        self.con_dones  = []
        self.step = step
        self.epsilon = epsilon
        self.inspector = None
        
    def append(self, con_state, con_action, reward, done):

        self.con_states.append(con_state)
        self.con_actions.append(con_action)
        self.con_reward.append(reward)
        self.con_dones.append(done)

        if done:
            self.inspector.start_pattern_abstract(self.con_states, self.con_actions, self.con_reward)
            self.clear()
    
    def clear(self):
        self.con_states = []
        self.con_actions = []
        self.con_reward = []
        self.con_dones  = []
    
    def handle_pattern(self,con_states,con_actions,rewards):

        abs_pattern = self.inspector.discretize_states(con_states)
        abs_action = self.inspector.discretize_actions(con_actions)
        
        if len(abs_pattern) != self.step:
            return rewards[0]
        abs_pattern = '-'.join(abs_pattern)
        abs_action = '-'.join(abs_action)
        score = self.inspector.inquery(abs_pattern, abs_action)
        baseline = self.inspector.inquery(abs_pattern, abs_action, baseline = True)
        
        if score != None and baseline != None:
            delta = (score - baseline) * self.epsilon * 10
            #print(abs_pattern, score, self.inspector.score_avg, rewards[0], rewards[0] + delta)
            rewards[0] += delta
                
        return rewards[0]



    def reward_shaping(self, state_list, action_list, reward_list):
        
        shaping_reward_list = copy.deepcopy(reward_list)

        for i in range(len(state_list) - self.step):

            target_states = state_list[i:i+self.step]
            target_action = action_list[i+self.step-1: i+self.step]
            target_rewards = reward_list[i:i+self.step]

            shaped_reward = self.handle_pattern(target_states, target_action, target_rewards)
            shaping_reward_list[i] = shaped_reward
        
        shaping_reward_list = np.array(shaping_reward_list)
        return shaping_reward_list