import wandb
from utils.timer import timer
from ftcode.logger import *


def run_episode(step_id, episode_id, env, alg_controller, fault_controller, args, att_print=None, training_mode=True):
    obs_n = env.reset()
    fault_controller.reset()
    done = False
    time_step = 0
    step_infos = []

    while not done:
        step_id[0] += 1
        fault_controller.add_fault(time_step, obs_n)
        fault_controller.obs_fault(obs_n)

        action_n = alg_controller.policy(obs_n, fault_controller, training_mode)

        # fault_controller.action_fault(action_n)

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        fault_controller.new_obs_fault(new_obs_n)
        fault_info = fault_controller.info()

        # if any(fault_info['fault_list']):
        #     print('fault')

        alg_controller.update_transition(obs_n, new_obs_n, rew_n, done_n, action_n, fault_info)
        done = all(done_n) or (time_step >= args.per_episode_max_len - 1)
        policy_updated = False

        obs_n = new_obs_n
        time_step += 1

        step_info = {'episode_id': episode_id, 'rew': rew_n, 'fault_info': fault_info}
        for k, v in info_n.items():
            step_info[k] = v
        step_infos.append(step_info)

        if args.display:
            env.render()
            print(rew_n)

        # if not training_mode and not args.display:
        #     att_print.add_critic_att(obs_n, action_n, fault_info)
        #     att_print.add_actor_att(obs_n, fault_info)

        if training_mode and episode_id >= args.learning_start_episode and step_id[0] % args.learning_fre == 0:
            alg_controller.update(args, episode_id)
    episode_info = {'rew': 0}
    episode_info2metrics(episode_info, step_infos)
    env.scenario_info2metrics(episode_info, step_infos)
    # if episode_id % 1000 == 0 and args.fault == 'broken':
    #     alg_controller.alg_info2metrics(episode_info)

    return episode_info


def run(env, controller, fault_controller, start_episode, args):
    step_id = [0]
    episode_metrics_list = []
    if start_episode > 0:
        controller.load_all(start_episode)

    # if not args.debug:
    #     for model in controller.get_models():
    #         wandb.watch(model)

    for episode_id in range(start_episode + 1, args.max_episode + 1):
        episode_metrics = run_episode(step_id, episode_id, env, controller, fault_controller, args, training_mode=True)
        episode_metrics_list.append(episode_metrics)
        # if not args.debug:
        #     wandb.log(episode_metrics)
        if episode_id % 20 == 0:
            local_print(episode_metrics_list)

        if episode_id % args.interval_save_model == 0:
            controller.save_all(episode_id, episode_metrics_list)
            print(episode_id)


def run_test(env, controller, fault_controller, episode, args):
    step_id = [0]
    controller.load_model(episode)
    att_print = AttPrint(controller, args)
    episode_infos = []
    for episode_id in range(args.test_episode):
        episode_info = run_episode(step_id, episode_id, env, controller, fault_controller, args, att_print, training_mode=False)
        episode_infos.append(episode_info)
    if not args.display:
        log_validation(args, episode_infos)
    att_print.print_critic_att()
    att_print.print_actor_att()
