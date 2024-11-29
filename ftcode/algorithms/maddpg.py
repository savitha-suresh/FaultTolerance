import os.path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ftcode.algorithms.alg_controller import AlgController
from ftcode.algorithms.module_utils import ActorMLP, CriticMLP
from ftcode.utils.timer import timer
import numpy as np

class MADDPGLearner(AlgController):
    def __init__(self, args, env_args, ex_name):
        super().__init__(args, env_args, ex_name)
        self.model_file_dir = os.path.join('models', ex_name)

        self.actors_cur = [ActorMLP(self.obs_shape_n[i], self.action_shape_n[i], args.mlp_hidden_size).to(args.device) for i in range(self.n_agents)]
        self.actors_tar = [ActorMLP(self.obs_shape_n[i], self.action_shape_n[i], args.mlp_hidden_size).to(args.device) for i in range(self.n_agents)]
        self.critics_cur = [CriticMLP(sum(self.obs_shape_n), sum(self.action_shape_n), args.mlp_hidden_size).to(args.device) for i in range(self.n_agents)]
        self.critics_tar = [CriticMLP(sum(self.obs_shape_n), sum(self.action_shape_n), args.mlp_hidden_size).to(args.device) for i in range(self.n_agents)]
        self.optimizers_a = [optim.Adam(self.actors_cur[i].parameters(), args.lr_a) for i in range(self.n_agents)]
        self.optimizers_c = [optim.Adam(self.critics_cur[i].parameters(), args.lr_c) for i in range(self.n_agents)]

        self.update_target_networks(1.0)

    def get_models(self):
        return self.actors_cur + self.actors_tar + self.critics_cur + self.critics_tar

    def update_target_networks(self, tao):
        agents_cur = self.actors_cur + self.critics_cur
        agents_tar = self.actors_tar + self.critics_tar
        for agent_c, agent_t in zip(agents_cur, agents_tar):
            key_list = list(agent_c.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_c = agent_c.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key] * tao + (1 - tao) * state_dict_t[key]
            agent_t.load_state_dict(state_dict_t)

    def update(self, args, episode_id):
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in enumerate(zip(
                self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar, self.optimizers_a, self.optimizers_c)):
            device = self.args.device
            batch_data = self.memory.sample(self.args.batch_size)
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done, fault_info = batch_data
            normal_list = [(np.array(fault_info['fault_list'])[:, agent_idx] == False) for agent_idx in range(self.n_agents)]
            rew = torch.tensor(_rew_n, device=device, dtype=torch.float)[normal_list[agent_idx]]
            done = torch.tensor(~_done, dtype=torch.float, device=device)[normal_list[agent_idx]]
            action_cur_o = torch.from_numpy(_action_n).to(device, torch.float)[normal_list[agent_idx]]
            obs_n_o = torch.from_numpy(_obs_n_o).to(device, torch.float)[normal_list[agent_idx]]
            obs_n_n = torch.from_numpy(_obs_n_n).to(device, torch.float)[normal_list[agent_idx]]

            action_tar = torch.cat([a_t(obs_n_n[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach()
                                    for idx, a_t in enumerate(self.actors_tar)], dim=1)

            for idx in range(self.n_agents):
                action_tar[~normal_list[idx][normal_list[agent_idx]], self.action_size[idx][0]:self.action_size[idx][1]] = 0

            q = critic_c(obs_n_o, action_cur_o).reshape(-1)
            q_ = critic_t(obs_n_n, action_tar).reshape(-1)
            tar_value = q_ * self.args.gamma * done + rew[:, agent_idx]
            loss_c = torch.nn.MSELoss()(q, tar_value)
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), self.args.max_grad_norm)
            opt_c.step()

            model_out, policy_c_new = actor_c(obs_n_o[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]],
                                              model_original_out=True)

            action_cur_o[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = policy_c_new
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), self.args.max_grad_norm)
            opt_a.step()

        self.update_target_networks(args.tao)

    def policy(self, obs_n, fault_controller, training_mode=True):
        actors_cur, obs_n = fault_controller.actors_nofault(self.actors_cur, obs_n)
        if training_mode:
            action_n = [agent(torch.from_numpy(obs).to(self.args.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
        else:
            action_n = []
            for i, (actor, obs) in enumerate(zip(actors_cur, obs_n)):
                model_out, _ = actor(torch.from_numpy(obs).to(self.args.device, torch.float), model_original_out=True)
                action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())
        fault_controller.action_fault(action_n)
        return action_n

    def save_model(self, episode_id):
        path = os.path.join(self.model_file_dir, str(episode_id))
        if not os.path.exists(path):  # make the path
            os.mkdir(path)
        for agent_idx, (a_c, a_t) in enumerate(zip(self.actors_cur, self.actors_tar)):
            torch.save(a_c, os.path.join(path, 'a_c_{}.pt'.format(agent_idx)))
            torch.save(a_t, os.path.join(path, 'a_t_{}.pt'.format(agent_idx)))
        for agent_idx, (c_c, c_t) in enumerate(zip(self.critics_cur, self.critics_tar)):
            torch.save(c_c, os.path.join(path, 'c_c_{}.pt'.format(agent_idx)))
            torch.save(c_t, os.path.join(path, 'c_t_{}.pt'.format(agent_idx)))

    def load_model(self, episode_id):
        path = os.path.join(self.model_file_dir, str(episode_id))
        n_actor = len(self.actors_cur)
        n_critic = len(self.critics_cur)
        self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar = [], [], [], []
        for i in range(n_actor):
            self.actors_cur.append(torch.load(os.path.join(path, 'a_c_{}.pt'.format(i)), map_location=self.args.device))
            self.actors_tar.append(torch.load(os.path.join(path, 'a_t_{}.pt'.format(i)), map_location=self.args.device))
        for i in range(n_critic):
            self.critics_cur.append(torch.load(os.path.join(path, 'c_c_{}.pt'.format(i)), map_location=self.args.device))
            self.critics_tar.append(torch.load(os.path.join(path, 'c_t_{}.pt'.format(i)), map_location=self.args.device))

    def save_all(self, episode_id, episode_metrics_list):
        if not os.path.exists(self.model_file_dir):
            os.mkdir(self.model_file_dir)
        with open(os.path.join(self.model_file_dir, 'memory.pkl'), "wb") as f:
            pickle.dump(self.memory, f)
            f.close()

        with open(os.path.join(self.model_file_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(episode_metrics_list, f)
            f.close()
        for agent_idx, opt_a in enumerate(self.optimizers_a):
            torch.save(opt_a, os.path.join(self.model_file_dir, 'opt_a_{}.pt'.format(agent_idx)))
        for agent_idx, opt_c in enumerate(self.optimizers_c):
            torch.save(opt_c, os.path.join(self.model_file_dir, 'opt_c_{}.pt'.format(agent_idx)))

        self.save_model(episode_id)

    def load_all(self, episode_id):
        with open(os.path.join(self.model_file_dir, 'memory.pickle'), "rb") as f:
            self.memory = pickle.load(f)
            f.close()

        n_actor = len(self.optimizers_a)
        n_critic = len(self.optimizers_c)
        self.optimizers_a, self.optimizers_c = [], []
        for agent_idx in range(n_actor):
            self.optimizers_a.append(torch.load(os.path.join(self.model_file_dir, 'opt_a_{}.pt'.format(agent_idx))))
        for agent_idx in range(n_critic):
            self.optimizers_c.append(torch.load(os.path.join(self.model_file_dir, 'opt_c_{}.pt'.format(agent_idx))))

        self.load_model(episode_id)
