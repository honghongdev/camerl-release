import argparse
import time
import threading
#
import os
from os.path import join, exists
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from flightgym import AvoidVisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy, CnnLstmPolicy
from mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO

from mav_baselines.torch.envs import vec_multi_env_wrapper as wrapper
unity_ready = False
save_finished = False

# backward-compat alias for legacy checkpoint weights
import sys
import mav_baselines
sys.modules['rpg_baselines_prev'] = mav_baselines



def rendering_thread(env):
    global unity_ready, save_finished
    time.sleep(0.1)
    while True:
        if unity_ready:
            env.render(0)
            time.sleep(0.01)
            if save_finished:
                break

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=0,    help="Random seed")
    parser.add_argument("--render",   type=int, default=1,    help="1=enable Unity render")
    parser.add_argument("--trial",    type=int, default=1,    help="PPO trial number")
    parser.add_argument("--iter",     type=int, default=100,  help="PPO iter number")
    parser.add_argument("--scene_id", type=int, default=0,    help="0=indoor, 1=outdoor")
    parser.add_argument("--rollouts", type=int, default=1000, help="Number of rollouts")
    parser.add_argument("--dir",      type=str, default="./datasets", help="Rollout directory")
    parser.add_argument("--logdir",   type=str, default="./exp_dir",  help="Log directory")
    parser.add_argument("--weight",   type=str,
                        default="./saved/ppo-final/Policy/iter_02000.pth",
                        help="Final PPO checkpoint (output of train_policy.py --retrain 1)")
    return parser

def main():
    global unity_ready, save_finished
    args = parser().parse_args()

    if args.scene_id == 0:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_new.yaml", "r"))
    else:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_new_out.yaml", "r"))

    cfg["simulation"]["num_envs"] = 1
    env = AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    env = wrapper.VisionEnvVec(env)

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir  = rsg_root + "/saved"
    new_thread = threading.Thread(target=rendering_thread, args=(env,))
    new_thread.start()

    unity_ready = env.connectUnity()
    env.getPointClouds('', 0, True)
    while not env.getSavingState():
        time.sleep(0.02)
    time.sleep(5.0)
    env.readPointClouds(0)
    while not env.getReadingState():
        time.sleep(0.02)
    time.sleep(1.0)
    save_finished = True

    weight = args.weight
    device = get_device("auto")
    saved_variables = torch.load(weight, map_location=device)
    saved_variables["data"]['shared_lstm']        = True
    saved_variables["data"]['enable_critic_lstm'] = False
    policy = MultiInputLstmPolicy(features_dim=64, **saved_variables["data"], use_rnn=True)
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)

    model = RecurrentPPO(
        tensorboard_log=log_dir,
        policy=policy,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[512, 512])],
            log_std_init=-0.5,
            use_beta=False,
        ),
        env=env,
        eval_env=env,
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=1000,
        n_seq=1,
        ent_coef=0.0,
        vf_coef=0.2,
        max_grad_norm=0.5,
        lstm_layer=1,
        batch_size=4000,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        retrain=True,
        env_cfg=cfg,
        verbose=1,
        states_dim=0,
        features_dim=64,
        if_change_maps=True,
        is_forest_env=(args.scene_id == 1),
    )

    model.setup_eval()
    ctl_iter = 2000
    print("Testing policy:", weight)
    saved_variables = torch.load(weight, map_location=device)
    model.change_policy(saved_variables["state_dict"])
    model.eval_from_outer(ctl_iter)

if __name__ == "__main__":
    main()
