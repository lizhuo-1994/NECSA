import os
import numpy as np
import joblib
from collections import OrderedDict
import numpy as np
from sklearn.mixture import GaussianMixture
import time
from collections import defaultdict
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

class AbstractModel():
    def __init__(self):
        self.initial = []
        self.final = []
        

class Reduction(ABC):
    @abstractmethod
    def do_reduction(self, data):
        pass
    

class PCA_R(Reduction):
    def __init__(self, top_components):
        self.top_components = top_components
        self.pca = None
        self.pca_min = None
        self.pca_max = None
        self.explained_variance_ratio = None
        
    def create_pca(self, all_observations):
        assert(len(all_observations) > 0)
        if self.top_components >= all_observations[0].shape[-1]:
            self.pca = None
            self.pca_min = np.min(all_observations, axis=0)
            self.pca_max = np.max(all_observations, axis=0)
            return all_observations, self.pca_min, self.pca_max
        else:
            print("build a PCA model...")
            print(">>>original:{}".format(all_observations.shape))
            start_time = time.time()
            self.pca = PCA(n_components=self.top_components)
            self.pca.fit(all_observations)
            pca_data = self.pca.transform(all_observations)
            self.pca_min = np.around(np.min(pca_data,axis=0),4)
            self.pca_max = np.around(np.max(pca_data,axis=0),4)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            #print(">>>pca_data",pca_data.shape)
            #print(">>>explained_variance_ratio",np.sum(self.pca.explained_variance_ratio_))
            #print("PCA data min:",self.pca_min)
            #print("PCA data max:",self.pca_max)
            #print(">>>it takes {} seconds.".format(time.time()-start_time))
            return pca_data, self.pca_min, self.pca_max
        
    def do_reduction(self,data):
        if self.pca is None:
            return data
            assert False,"please create a pca model based on training data..."
        else:
            return self.pca.transform(data)
        
class Grid(AbstractModel):
    '''
    Multiple DTMCs from a set of sets of traces
    traces: a set of sets of traces
    '''
    def __init__(self, min_val, max_val, grid_num, clipped=True):
        super().__init__()
        self.min = min_val
        self.max = max_val
        self.k = grid_num
        self.dim = max_val.shape[0]
        self.total_states = pow(grid_num,self.dim)
        self.unit = (max_val - min_val) / self.k
        self.clipped = clipped

    def state_abstract(self, con_states):
        con_states = con_states
        lower_bound = self.min
        upper_bound = self.max
        unit = (upper_bound - lower_bound)/self.k
        abs_states = np.zeros(con_states.shape[0],dtype=np.int8)
        
        #print(lower_bound)
        #print(upper_bound)
        indixes = np.where(unit == 0)[0]
        unit[indixes] = 1
        #print('unit:\t', unit)
        
        tmp = ((con_states-self.min)/unit).astype(int)
        if self.clipped:
            tmp = np.clip(tmp, 0, self.k-1)
            
        dims = tmp.shape[1]
        for i in range(dims):
            abs_states = abs_states + tmp[:,i]*pow(self.k, i)
#         abs_states = np.expand_dims(abs_states,axis=-1)
            
        abs_states = [str(item) for item in abs_states]
        return abs_states
    
    def extract_abs_trace(self,dones,abs_states,abs_rewards, abs_values):

        end_idx = np.where(np.abs(dones)==1)[0]
        all_traces = []
        all_rewards = []
        all_values = []

        start = 0
        for cur_end in end_idx:
            all_rewards.append(abs_rewards[start : cur_end + 1])
            all_values.append(abs_values[start : cur_end + 1])
            all_traces.append(abs_states[start : cur_end + 1])
            start = cur_end + 1
           
        return all_traces,all_rewards,all_values

    
    
    
    def profiling_all_traces(self, all_traces):
        state_dic = defaultdict(list) # {state:[fail_stat, success_stat]}
        edge_dic = defaultdict(list)
        for cur_trace in all_traces:
            isSuccess =  cur_trace[-1]=='S'
            for idx in range(len(cur_trace)-1):
                cur_state = cur_trace[idx]
                next_state = cur_trace[idx+1]
                cur_edge = (cur_state,next_state)
                if cur_state not in state_dic:
                    state_dic[cur_state] = [0,0]
                state_dic[cur_state][isSuccess] += 1
                if cur_edge not in edge_dic:
                    edge_dic[cur_edge] = [0,0]
                edge_dic[cur_edge][isSuccess] += 1
        return state_dic,edge_dic

def traj_stat_analysis(trajectory_dic, keep=2):

    trajectory_dic = joblib.load(trajectory_dic)
    #all_observations = trajectory_dic['all_obvs']
    #all_actions = trajectory_dic['all_acts']
    all_dones = trajectory_dic['all_dones'].flatten()
    all_rewards = trajectory_dic['all_rwds'].flatten()
    all_states = trajectory_dic['all_states']
    all_values = trajectory_dic['all_values']
    avg_performace = trajectory_dic['avg_performace']
    last_idx = np.where(np.abs(all_dones)==1)[0][-1] + 1
    all_dones,all_rewards,all_states,all_values = all_dones[:last_idx],all_rewards[:last_idx],all_states[:last_idx],all_values[:last_idx]
    #print(all_observations.shape)
    #print(all_actions.shape)
    #print(all_dones.shape)
    #print(all_rewards.shape)
    #print(all_hidden.shape)
    #print("min obversations:",np.around(np.min(all_observations,axis=0),keep))
    #print("max obversations:",np.around(np.max(all_observations,axis=0),keep))
    #print("min actions:",np.around(np.min(all_actions,axis=0),keep))
    #print("max actions:",np.around(np.max(all_actions,axis=0),keep))

    return (all_dones,all_rewards,all_states,all_values, avg_performace)
    


def pca_analysis(n_components, pcaModelPath, all_observations, save=False):
    if os.path.exists(pcaModelPath):
        pcaModel = joblib.load(pcaModelPath)
        pca_min, pca_max = pcaModel.pca_min, pcaModel.pca_max
        pca_data = pcaModel.do_reduction(all_observations)
        #print("Load saved pca model successfully!")
    else:
        pcaModel = PCA_R(top_components=n_components)
        pca_data, pca_min, pca_max = pcaModel.create_pca(all_observations)
        if save:
            joblib.dump(pcaModel, pcaModelPath)
            #print("Save the pca model to {} successfully".format(pcaModelPath))
    pca_dic = {'pca_data':pca_data, 'pca_min':pca_min, 'pca_max':pca_max }
    return pcaModel, pca_dic


def grid_abs_analysis( pca_dic,all_rewards, all_values, all_dones,grid_num,abs_profiling_file):
    
    pca_min, pca_max, pca_data = pca_dic['pca_min'],pca_dic['pca_max'],pca_dic['pca_data']

    dtmc = DTMC(pca_min, pca_max, grid_num)
    abs_states = dtmc.state_abstract(con_states = pca_data)
    abs_traces, rewards, values= dtmc.extract_abs_trace(all_dones,abs_states,all_rewards, all_values)
    profiling_dic = {
        'dtmc'       : dtmc,
        'abs_states'  : abs_states,
        'abs_traces' : abs_traces,
        'all_rewards':rewards,
        'all_values':values
    }
    joblib.dump(profiling_dic,abs_profiling_file)
    #print("Save profiling results to ",abs_profiling_file)
    
    
    return profiling_dic
    
    
    '''
    #print(pca_min, pca_max, pca_data)
    if os.path.exists(abs_profiling_file):
        profiling_dic = joblib.load(abs_profiling_file)
        # dtmc, abs_states = profiling_dic['dtmc'], profiling_dic['abs_states'] 
        # abs_traces, tracesLen, results = profiling_dic['abs_traces'], profiling_dic['tracesLen'], profiling_dic['results']
        #print("Load profiling results from {} successfully!".format(abs_profiling_file))
    else:
        dtmc = DTMC(pca_min, pca_max, grid_num)
        abs_states = dtmc.state_abstract(con_states = pca_data)
        abs_traces, rewards, values= dtmc.extract_abs_trace(all_dones,abs_states,all_rewards, all_values)
        profiling_dic = {
            'dtmc'       : dtmc,
            'abs_states'  : abs_states,
            'abs_traces' : abs_traces,
            'all_rewards':rewards,
            'all_values':values
        }
        joblib.dump(profiling_dic,abs_profiling_file)
        #print("Save profiling results to ",abs_profiling_file)
    
    
    return profiling_dic
    '''

def con_2_dis( obv_list):
    _, pca_dic = pca_analysis(config.n_components, config.pcaModelPath, obv_list)
    abs_state_list = []
    for data in pca_dic['pca_data']:
        abs_tuple = concrete_to_abstract(pca_dic, None, config.grid_num,np.array([data,data]))
        abs_state_list.append(abs_tuple[0])
    
    return abs_state_list

def concrete_to_abstract(pca_dic,all_rewards,grid_num,concrete_state):
    pca_min, pca_max, pca_data = pca_dic['pca_min'],pca_dic['pca_max'],pca_dic['pca_data']
    dtmc = DTMC(pca_min, pca_max, grid_num)
    abs_states = dtmc.state_abstract(con_states = concrete_state)
    return abs_states
    
    

def calc_coverage(abs_states,grid_num,n_componenct):
    uniq = len(np.unique(np.array(abs_states)))
    total = pow(grid_num, n_componenct)
    #print("The number of unique states:{}".format(uniq))
    #print("The number of total states:{}".format(total))
    #print("Coverage:{:.2f}".format((uniq*1.0)/total))

from collections import defaultdict

# abs2con：记录抽象状态编号和具体状态下表的索引关系；
# state_dic：记录经过状态s,失败的trace数量和胜利trace的数量
# transistion_dic：记录状态s的迁移关系，即下一个状态
def analyze_abstraction(abs_states, abs_traces):
    abs2con = defaultdict(list)
    for idx,key in enumerate(abs_states):
        abs2con[key].append(idx)
    state_dic = defaultdict(list)
    transistion_dic = defaultdict(list)
    counts = []
    for curTrace in abs_traces:
        isSuccess = curTrace[-1]=='S'
        state_set = set()
        counts.append(len(curTrace)-1)
        for idx in range(len(curTrace)-1):
            curState = curTrace[idx]
            nextState = curTrace[idx+1]
            transistion_dic[curState].append(nextState)
            state_set.add(curState)
        state_set.add(curTrace[-2])
        for state in state_set:
            if state not in state_dic:
                state_dic[state] = [0,0]
            state_dic[state][isSuccess] += 1
    avgFreq = np.sum(np.array(counts))/len(state_dic.keys())
    return avgFreq, abs2con, state_dic, transistion_dic
    

def fetchCriticalState(state_dic, lowest, threshold, savepath):
    
    if savepath is not None and os.path.exists(savepath):
        print("Load critical states info from {} successfully!".format(savepath))
        
        critical_dic = joblib.load(savepath)
        good_list, bad_list = critical_dic['good_list'], critical_dic['bad_list']
        print('good state:',len(good_list))
        print('bad state:',len(bad_list))

        return critical_dic['critical'], good_list, bad_list
    
    critical = []
    good_list = []
    bad_list = []
    for key,val in state_dic.items():
        total = val[0]+val[1]
        failRate = (1.0*val[0])/total
        if total < lowest:
            continue
        if failRate<=threshold:
            good_list.append(key)
            critical.append([key, val[0],val[1],failRate])
        if failRate >= 1-threshold:
            bad_list.append(key)
            critical.append([key, val[0],val[1],failRate])
    print("lowest:",lowest)
    print("failRate threshold:",threshold)
    print('good state:',len(good_list))
    print('bad state:',len(bad_list))
    critical = sorted(critical,key=lambda x:x[-1],reverse=True)
    critical_dic = {
    'critical': critical,
    'good_list': good_list,
    'bad_list': bad_list,
    'lowest': lowest,
    'threshold': threshold
    }
    if savepath is not None:
        joblib.dump(critical_dic,savepath)
        print("Save critical states info to ",savepath)
    return critical, good_list, bad_list

import numpy as np
# 以抽象状态中所有具体状态的平均值作为该抽象状态的中心
def get_abs_center( concrete_states , abstract_dic, state_list):
    abs_state_centers = []
    for state in state_list:
        conIdx = abstract_dic[state]
        abs_state = np.mean(concrete_states[conIdx],axis=0)
        abs_state_centers.append(abs_state)
    return np.array(abs_state_centers)

# 计算当前抽象状态集合在抽象状态空间的coverage
def calc_coverage(abs_states,grid_num,n_componenct):
    uniq = len(np.unique(np.array(abs_states)))
    total = pow(grid_num, n_componenct)
    print("The number of unique states:{}".format(uniq))
    print("The number of total states:{}".format(total))
    print("Coverage:{:.2f}".format((uniq*1.0)/total))

# 经过抽象状态s，胜利的trace数量大于失败的trace数量; 记录一个trace出现/不出现某个状态
def abs_state_eval(abs_traces):
    state_dic = defaultdict(list) # {state:[fail_stat, success_stat]}
    edge_dic = defaultdict(list)
    for cur_trace in abs_traces:
        isSuccess =  cur_trace[-1]=='S'
        state_set = set()
        edge_set = set()
        for idx in range(len(cur_trace)-1):
            cur_state = cur_trace[idx]
            next_state = cur_trace[idx+1]
            cur_edge = (cur_state,next_state)
            if cur_state not in state_dic:
                state_dic[cur_state] = [0,0]
            state_set.add(cur_state)
            if cur_edge not in edge_dic:
                edge_dic[cur_edge] = [0,0]
            edge_set.add(cur_edge)
        
        for state in state_set:
            state_dic[state][isSuccess] += 1
        for edge in edge_set:
            edge_dic[edge][isSuccess] += 1
    
    return state_dic, edge_dic

def fetchStateOrder(abs_traces):
    success_traces = [item for item in abs_traces if item[-1]=='S']
    fail_traces = [item for item in abs_traces if item[-1]=='F']

    goodStateFeqDic = defaultdict(int)
    badStateFeqDic = defaultdict(int)
    for trace in success_traces:
        for cur in set(trace):
            goodStateFeqDic[cur] += 1
    good_list = sorted(goodStateFeqDic.items(),key=lambda x:x[1],reverse=True)

    for trace in fail_traces:
         for cur in set(trace):
            badStateFeqDic[cur] += 1
    bad_list = sorted(badStateFeqDic.items(),key=lambda x:x[1],reverse=True)

    return good_list,bad_list

# 抽象状态s(在所有胜利的trace中出现的次数)大于(在所有失败的trace中出现的次数+margin)。记录所有成功或失败race中状态s出现的次数
def abs_state_eval2(abs_traces):
    state_dic = defaultdict(list) # {state:[fail_stat, success_stat]}
    edge_dic = defaultdict(list)
    for cur_trace in abs_traces:
        isSuccess =  cur_trace[-1]=='S'
        for idx in range(len(cur_trace)-1):
            cur_state = cur_trace[idx]
            next_state = cur_trace[idx+1]
            cur_edge = (cur_state,next_state)
            if cur_state not in state_dic:
                state_dic[cur_state] = [0,0]
            state_dic[cur_state][isSuccess] += 1
            if cur_edge not in edge_dic:
                edge_dic[cur_edge] = [0,0]
            edge_dic[cur_edge][isSuccess] += 1
    return state_dic,edge_dic

# 获取同一类型（胜利/失败）下抽象状态的排序（依照出现的频率）
def fetchStateOrder2(abs_traces):
    success_traces = [item for item in abs_traces if item[-1]=='S']
    fail_traces = [item for item in abs_traces if item[-1]=='F']

    goodStateFeqDic = defaultdict(int)
    badStateFeqDic = defaultdict(int)
    for trace in success_traces:
        for idx in range(len(trace)-1):
            goodStateFeqDic[trace[idx]] += 1
    good_list = sorted(goodStateFeqDic.items(),key=lambda x:x[1],reverse=True)

    for trace in fail_traces:
        for idx in range(len(trace)-1):
            badStateFeqDic[trace[idx]] += 1
    bad_list = sorted(badStateFeqDic.items(),key=lambda x:x[1],reverse=True)

    return good_list,bad_list

