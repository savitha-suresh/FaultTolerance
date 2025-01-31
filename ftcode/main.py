import wandb
from ftcode.configs.get_config import args
from ftcode.domain import make_env
from ftcode.algorithms.algorithm import make_alg
from ftcode.fault.registry import FAULTS
from ftcode.curriculums.curriculum import make_cl
from runner import run
import time
import string
from ftcode.utils.set_seed import seed_all
import os
import signal
signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号

os.environ["WANDB_API_KEY"] = '88ff66bce8b622fb5cd40bbe8c7f958d5b572e47'
os.environ["WANDB_MODE"] = "offline"

seed_all(args.seed)
if args.old_model_name is None:
    start_episode = 0
    time_now = time.strftime('%y%m_%d%H%M')
    ex_name = '{}_{}_s{}_{}_{}'.format(args.env, args.alg, args.seed, time_now, args.fault)
else:
    ex_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
    start_episode = int(args.old_model_name.split('/')[-1])


# if not args.debug:
#     wandb.init(
#         project="FaultTolerance",
#         config=args,
#         name=ex_name
#     )
#     arti_code = wandb.Artifact('algorithm', type='code')
#     arti_code.add_dir('/home/syc/Workspace/FaultTolerance/ftcode')
#     wandb.log_artifact(arti_code)
#     arti_code = wandb.Artifact('environment', type='code')
#     arti_code.add_dir('/home/syc/Workspace/FaultTolerance/mpe')
#     wandb.log_artifact(arti_code)

cl_controller = make_cl(args.cl, args)

env, kwargs = make_env(args.domain, args.env)

alg_controller = make_alg(args.alg, args, kwargs, ex_name)
fault_controller = FAULTS[args.fault](args, env, alg_controller, cl_controller)

run(env, alg_controller, fault_controller, start_episode, args)

#wandb.finish()
