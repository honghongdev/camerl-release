import argparse
import math
import time
import threading
#
from PIL import Image
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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=0,    help="Random seed")
    parser.add_argument("--train",    type=int, default=1,    help="1=train, 0=evaluate")
    parser.add_argument("--render",   type=int, default=1,    help="1=enable Unity render")
    parser.add_argument("--trial",    type=int, default=1,    help="PPO trial number")
    parser.add_argument("--iter",     type=int, default=100,  help="PPO iter number")
    parser.add_argument("--retrain",  type=int, default=1,    help="1=load existing weights")
    parser.add_argument("--scene_id", type=int, default=0,    help="0=indoor, 1=outdoor")
    parser.add_argument("--rollouts", type=int, default=1000, help="Number of rollouts")
    parser.add_argument("--dir",      type=str, default="./datasets", help="Rollout directory")
    parser.add_argument("--logdir",   type=str, default="./exp_dir",  help="Log directory")
    parser.add_argument("--weight",   type=str,
                        default="./saved/ppo-init/Policy/iter_00200.pth",
                        help="Stage 1 PPO checkpoint (output of train_policy.py --retrain 0)")
    return parser

def main():
    args = parser().parse_args()

    if args.scene_id == 0:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_lstm_indoor.yaml", "r"))
    else:
        cfg = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config_lstm.yaml", "r"))

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

        if not args.retrain:
            logdir   = os.environ["AVOIDBENCH_PATH"] + "/../camerl"
            vae_file = join(logdir, 'vae_64_new', 'best.tar')
            assert exists(vae_file), "No trained VAE in the logdir..."
            state_vae = torch.load(vae_file)
            print("Loading VAE at epoch {} "
                  "with test error {}".format(state_vae['epoch'], state_vae['precision']))
        else:
            state_vae = None

    if args.retrain or not args.train:
        weight = args.weight
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        saved_variables["data"]['only_lstm_training'] = True
        policy = MultiInputLstmPolicy(
            features_dim=64,
            reconstruction_members=[False, False, True],
            reconstruction_steps=2,
            **saved_variables["data"],
            use_rnn=True)
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
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
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=500,
            n_seq=1,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lstm_layer=1,
            batch_size=500,
            n_epochs=50,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            retrain=args.retrain,
            env_cfg=cfg,
            verbose=1,
            state_vae=state_vae,
            only_lstm_training=True,
            states_dim=0,
            reconstruction_members=[False, False, True],
            save_lstm_dateset=True,
            is_forest_env=(args.scene_id == 1),
        )
        model.learn_lstm(total_timesteps=int(4e7), log_interval=(10, 10))

if __name__ == "__main__":
    main()
