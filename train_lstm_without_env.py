#!/usr/bin/env python3
import argparse
import os
from os.path import exists
import torch
import numpy as np
from stable_baselines3.common.utils import get_device
from mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy
from mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO



def configure_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    type=int, default=1,   help="1=train, 0=test")
    parser.add_argument("--retrain",  type=int, default=1,   help="1=load existing weights")
    parser.add_argument("--trial",    type=int, default=1,   help="PPO trial number")
    parser.add_argument("--iter",     type=int, default=100, help="PPO iter number")
    parser.add_argument("--recon",    nargs='+', type=int, default=[1, 1, 0],
                        help="Reconstruction targets: past now future")
    parser.add_argument("--vae-path", type=str,
                        default="./saved/vae/camerl-1/best.tar",
                        help="VAE checkpoint produced by trainvae_ca.py")
    parser.add_argument("--weight",   type=str,
                        default="./saved/ppo-init/Policy/iter_00200.pth",
                        help="Stage 1 PPO checkpoint produced by train_policy.py --retrain 0")
    parser.add_argument("--dataset",  type=str,
                        default="./saved/dataset/",
                        help="Depth rollout dataset produced by collect_data.py")
    parser.add_argument("--name",     type=str, default="camerl-run1",
                        help="Output sub-directory name under ./saved/lstm/")
    return parser

def main():
    args = parser().parse_args()
    configure_random_seed(92)
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir  = rsg_root + "/saved"

    vae_file = args.vae_path
    assert exists(vae_file), "No trained VAE in the logdir..."
    state_vae = torch.load(vae_file)
    print("Loading VAE at epoch {} "
          "with test error {}".format(state_vae['epoch'], state_vae['precision']))

    device = get_device("auto")
    weight = args.weight
    saved_variables = torch.load(weight, map_location=device)
    # saved_variables["data"] carries policy constructor kwargs (not network weights)
    saved_variables["data"]['only_lstm_training'] = True
    policy = MultiInputLstmPolicy(
        features_dim=64,
        reconstruction_members=args.recon,
        reconstruction_steps=10,
        use_rnn=True,
        **saved_variables["data"])
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)

    model = RecurrentPPO(
        tensorboard_log=log_dir,
        policy=policy,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[512, 512])],
        ),
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=1000,
        n_seq=1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lstm_layer=1,
        batch_size=500,
        n_epochs=2000,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        retrain=args.retrain,
        verbose=1,
        only_lstm_training=True,
        state_vae=state_vae,
        states_dim=0,
        reconstruction_members=args.recon,
        reconstruction_steps=10,
        train_lstm_without_env=True,
        lstm_dataset_path=args.dataset,
        lstm_weight_saved_path="lstm/" + args.name,
    )
    if args.train:
        model.train_lstm_from_dataset()
    else:
        model.test_lstm_seperate()

if __name__ == "__main__":
    main()
