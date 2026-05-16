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
from mav_baselines.torch.common.util import test_vision_policy
unity_ready = False
save_finished = False

# backward-compat alias for legacy checkpoint weights
import sys
import mav_baselines
sys.modules['rpg_baselines_prev'] = mav_baselines



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rendering_thread(env):
    global unity_ready, save_finished
    time.sleep(0.1)
    while True:
        if unity_ready:
            env.render(0)
            time.sleep(0.01)
            if save_finished:
                break

def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate schedule.
    :param progress_remaining: proportion of training remaining (1 at start, 0 at end)
    :return: learning rate as a float.
    """
    return 1e-4 * progress_remaining

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=0,  help="Random seed")
    parser.add_argument("--train",    type=int, default=1,  help="1=train, 0=evaluate")
    parser.add_argument("--render",   type=int, default=1,  help="1=enable Unity render")
    parser.add_argument("--trial",    type=int, default=1,  help="PPO trial number")
    parser.add_argument("--iter",     type=int, default=100, help="PPO iter number")
    parser.add_argument("--retrain",  type=int, default=1,  help="1=load existing weights")
    parser.add_argument("--scene_id", type=int, default=0,  help="0=indoor, 1=outdoor")
    parser.add_argument("--nocontrol",type=int, default=0,  help="1=skip actor/critic weight loading")
    parser.add_argument("--rollouts", type=int, default=1000, help="Number of rollouts")
    parser.add_argument("--dir",      type=str, default="./datasets", help="Rollout directory")
    parser.add_argument("--logdir",   type=str, default="./exp_dir",  help="Log directory")
    parser.add_argument("--weight",   type=str,
                        default="./saved/lstm/camerl-run1/Policy/iter_01950.pth",
                        help="LSTM checkpoint to load when --retrain 1 (output of train_lstm_without_env.py)")
    return parser

def main():
    args = parser().parse_args()

    if args.scene_id == 0:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_new.yaml", "r"))
    else:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_new_out.yaml", "r"))

    train_env = AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.VisionEnvVec(train_env, logdir=args.logdir)
    configure_random_seed(args.seed, env=train_env)

    old_num_envs = cfg["simulation"]["num_envs"]
    old_render   = cfg["unity"]["render"]
    cfg["simulation"]["num_envs"] = 1
    cfg["unity"]["render"] = "no"
    eval_env = wrapper.VisionEnvVec(
        AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False), logdir=args.logdir)
    cfg["simulation"]["num_envs"] = old_num_envs
    cfg["unity"]["render"] = old_render
    eval_env.wrapper.setUnityFromPtr(train_env.wrapper.getUnityPtr())

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir  = rsg_root + "/saved"

    new_thread = threading.Thread(target=rendering_thread, args=(train_env,))
    new_thread.start()

    device = get_device("auto")
    if args.render:
        global unity_ready, save_finished
        unity_ready = train_env.connectUnity()
        train_env.spawnObstacles(change_obs=True)
        while not train_env.ifSceneChanged():
            train_env.spawnObstacles(change_obs=False)
            time.sleep(0.01)
        train_env.getPointClouds('', 0, True)
        while not train_env.getSavingState():
            time.sleep(0.02)
        time.sleep(5.0)
        train_env.readPointClouds(0)
        while not train_env.getReadingState():
            time.sleep(0.02)
        time.sleep(1.0)
        eval_env.readPointClouds(0)
        while not eval_env.getReadingState():
            time.sleep(0.02)
        time.sleep(1.0)
        save_finished = True

    if args.retrain or not args.train:
        weight = args.weight
        saved_variables = torch.load(weight, map_location=device)
        # FutureWarning expected here, safe to ignore
        policy = MultiInputLstmPolicy(features_dim=64, use_rnn=True, **saved_variables["data"])
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        saved_variables["state_dict"]['log_std'] = torch.tensor([-0.0, -0.0, -0.0, -0.0], device=device)
        if args.nocontrol:
            for key in [
                'action_net.0.weight', 'action_net.0.bias',
                'value_net.weight',    'value_net.bias',
                'mlp_extractor.value_net.0.weight',  'mlp_extractor.value_net.0.bias',
                'mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias',
                'mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias',
                'mlp_extractor.value_net.2.weight',  'mlp_extractor.value_net.2.bias',
            ]:
                saved_variables["state_dict"].pop(key)
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)
    else:
        policy = "MultiInputLstmPolicy"

    if args.train:
        model = RecurrentPPO(
            tensorboard_log=log_dir,
            policy=policy,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
                use_beta=False,
                use_rnn=True,
            ),
            env=train_env,
            learning_rate=learning_rate_schedule,
            eval_env=eval_env,
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
            retrain=args.retrain,
            device=device,
            env_cfg=cfg,
            verbose=1,
            states_dim=0,
            features_dim=64,
            if_change_maps=True,
            is_forest_env=(args.scene_id == 1),
        )
        # each iteration = n_steps * n_envs
        model.learn(total_timesteps=int(8e6), log_interval=(10, 20))
    else:
        test_vision_policy(eval_env, policy)

if __name__ == "__main__":
    main()
