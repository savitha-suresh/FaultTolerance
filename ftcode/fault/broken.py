import numpy as np
from ftcode.fault.registry import FAULTS
from ftcode.fault.nofault import NoFault
import random
import copy


@FAULTS.register
class Broken(NoFault):
    def __init__(self, args, env, alg_controller, cl_controller):
        super().__init__(args, env, alg_controller, cl_controller)
        self.cl_controller = cl_controller
        self.alg_controller = alg_controller

        self.fault_probs = args.fault_probs
        self.fault_id = -1
        self.episode_cnt = 0
        self.fault_time_start = args.fault_time
        self.fault_time = -1
        self.time_step = -1

    def add_fault(self, time_step, obs_n):
        self.time_step = time_step
        # if no agent is failed
        if time_step == 0:
            self.fault_time = self.cl_controller.get_fault_time(self.episode_cnt)
        if not any(self.fault_list):
            if time_step >= self.fault_time:
                self.fault_id = random.choices(self.agent_list, self.fault_probs)[0]
                self.fault_list[self.fault_id] = True
                self.agents[self.fault_id].fault = True
                self.agents[self.fault_id].movable = False
                self.agents[self.fault_id].collide = False
                self.fault_change = True
            else:
                self.fault_change = False
        else:
            self.fault_change = False

    def obs_fault(self, obs_n):
        if self.fault_id >= 0:
            self.alg_controller.obs_fault_modify(obs_n, self.fault_list)

    def actors_nofault(self, actors, obs_n):
        good_list = list(range(self.n))
        if self.fault_id >= 0:
            good_list.pop(self.fault_id)
        return [actors[i] for i in good_list], [obs_n[i] for i in good_list]

    def action_fault(self, action_n):
        if self.fault_id >= 0:
            action_n.insert(self.fault_id, np.zeros(len(action_n[0])))

    def new_obs_fault(self, new_obs_n):
        if self.fault_id >= 0:
            self.alg_controller.obs_fault_modify(new_obs_n, self.fault_list)

    def reset(self):
        super().reset()
        self.fault_time = -1
        self.fault_id = -1
        self.episode_cnt += 1

    def info(self):
        return {'fault_list': copy.deepcopy(self.fault_list), 'fault_change': copy.deepcopy(self.fault_change),
                'fault_time': self.fault_time, 'current_time': self.time_step}


