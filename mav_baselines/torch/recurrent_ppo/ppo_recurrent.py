import sys
import os
import time
from copy import deepcopy
from collections import deque
from typing import Any, Dict, Optional, Type, TypeVar, Union, Tuple, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import torch as th
import torch.utils.data
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms
from gym import spaces
from ruamel.yaml import YAML
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean, configure_logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import utils

from mav_baselines.torch.recurrent_ppo.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer, LSTMDictRolloutBuffer
from mav_baselines.torch.recurrent_ppo.recurrent.policies import RecurrentActorCriticPolicy
from mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from mav_baselines.torch.recurrent_ppo.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from mav_baselines.torch.common.util import traj_rollout, lstm_rollout
SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")
from torch.utils.tensorboard.writer import SummaryWriter
import threading
from data.loaders import RolloutLSTMSequenceDataset, RosbagSequenceDataset

from PIL import Image

def save_obs_debug(obs: dict, step: int, interval: int = 100, save_dir: str = "./debug_images"):
    """
    Save obs image and state to disk every `interval` steps (for debugging).
    Saves image as PNG and state as CSV.

    """
    if step % interval != 0:
        return

    image = obs["image"]    # shape: (1, 1, H, W)
    state = obs["state"]    # shape: (1, 1, N)
    # print(f"[DEBUG] type(image): {type(image)}")

    os.makedirs(save_dir, exist_ok=True)

    img_np = image.squeeze()   # shape: (H, W)
    state_np = state.squeeze() # shape: (N,)
    image_tensor = image.squeeze(0)
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)   
    image_tensor = image_tensor.float() / 255.0
    # image_csv_path = os.path.join(save_dir, f"obs_image_{step:03d}.csv")
    # np.savetxt(image_csv_path, img_np, fmt="%d", delimiter=",")

    image_png_path = os.path.join(save_dir, f"obs_image_{step:03d}.png")
    save_image(image_tensor, image_png_path)

    # state_csv_path = os.path.join(save_dir, f"obs_state_{step:03d}.csv")
    # np.savetxt(state_csv_path, state_np.reshape(1, -1), fmt="%.4f", delimiter=",")


class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 128,
        use_tanh_act: bool = True,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        retrain: bool = False,
        lstm_layer = 1,
        n_seq = 1,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        eval_env: Union[GymEnv, str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        env_cfg: str = None,
        _init_setup_model: bool = True,
        state_vae: Optional[Dict[str, Any]] = None,
        features_dim: int = 32,
        states_dim: int = 0,
        only_lstm_training: bool = False,
        if_change_maps: bool = True,
        is_forest_env: bool = False,
        reconstruction_members: Optional[List[bool]] = [True, False, True],
        reconstruction_steps: int = 2,
        save_lstm_dateset: bool = False,
        train_lstm_without_env: bool = False,
        fine_tune_from_rosbag: bool = False,
        lstm_dataset_path: Optional[str] = None,
        lstm_weight_saved_path: Optional[str] = 'LSTM_weights',
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.use_tanh_act = use_tanh_act
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.retrain = retrain
        self.target_kl = target_kl
        self._last_lstm_states = None
        self.eval_env = eval_env
        self.env_cfg = env_cfg
        self.lstm_layer = lstm_layer
        self.n_seq = n_seq
        self.state_vae = state_vae
        self.features_dim = features_dim
        self.states_dim = states_dim
        self.only_lstm_training = only_lstm_training
        self.finished_save_pc = True
        self.if_change_maps = if_change_maps
        self.is_forest_env = is_forest_env
        self.reconstruction_members = reconstruction_members
        self.reconstruction_steps = reconstruction_steps
        self.save_lstm_dateset = save_lstm_dateset
        self.train_lstm_without_env = train_lstm_without_env
        self.lstm_dataset_path = lstm_dataset_path
        self.fine_tune_from_rosbag = fine_tune_from_rosbag

        if self.retrain:
            self.policy = policy
            # print(self.state_vae)
            if self.state_vae is not None:
                # pretrained_cnn = {
                #     'features_extractor.conv1.weight': self.state_vae['state_dict']['encoder.conv1.weight'],
                #     'features_extractor.conv1.bias': self.state_vae['state_dict']['encoder.conv1.bias'],
                #     'features_extractor.conv2.weight': self.state_vae['state_dict']['encoder.conv2.weight'],
                #     'features_extractor.conv2.bias': self.state_vae['state_dict']['encoder.conv2.bias'],
                #     'features_extractor.conv3.weight': self.state_vae['state_dict']['encoder.conv3.weight'],
                #     'features_extractor.conv3.bias': self.state_vae['state_dict']['encoder.conv3.bias'],
                #     'features_extractor.conv4.weight': self.state_vae['state_dict']['encoder.conv4.weight'],
                #     'features_extractor.conv4.bias': self.state_vae['state_dict']['encoder.conv4.bias'],
                #     'features_extractor.conv5.weight': self.state_vae['state_dict']['encoder.conv5.weight'],
                #     'features_extractor.conv5.bias': self.state_vae['state_dict']['encoder.conv5.bias'],
                #     'features_extractor.conv6.weight': self.state_vae['state_dict']['encoder.conv6.weight'],
                #     'features_extractor.conv6.bias': self.state_vae['state_dict']['encoder.conv6.bias'],
                #     'features_extractor.linear.weight': self.state_vae['state_dict']['encoder.fc_mu.weight'],
                #     'features_extractor.linear.bias': self.state_vae['state_dict']['encoder.fc_mu.bias'],
                #     'features_extractor.fc_logsigma.weight': self.state_vae['state_dict']['encoder.fc_logsigma.weight'],
                #     'features_extractor.fc_logsigma.bias': self.state_vae['state_dict']['encoder.fc_logsigma.bias'],
                # }
                pretrained_cnn = {}
                for old_key, val in self.state_vae["state_dict"].items():
                    if old_key.startswith("encoder."):
                        new_key = "features_extractor." + old_key[len("encoder."):]
                        pretrained_cnn[new_key] = val
                self.policy.load_state_dict(pretrained_cnn, strict=False)
                self.policy = self.policy.to(self.device)

        if self.train_lstm_without_env:
            self.dataset_train = RolloutLSTMSequenceDataset(self.lstm_dataset_path, self.device, train=True)
            self.dataset_test = RolloutLSTMSequenceDataset(self.lstm_dataset_path, self.device, train=False)
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=1, shuffle=False, num_workers=0)
            self.test_loader = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
            self.n_envs = 1
            self._setup_lr_schedule()
            self.set_random_seed(self.seed)
            lstm_logger = utils.configure_logger(self.verbose, self.tensorboard_log, lstm_weight_saved_path, False)
            self.set_logger(lstm_logger)
            
        elif self.fine_tune_from_rosbag:
            self.dataset_train = RosbagSequenceDataset('real_imgs', '/camera/depth/image_rect_raw', transform=None, train=True)
            self.dataset_test  = RosbagSequenceDataset('real_imgs', '/camera/depth/image_rect_raw', transform=None, train=False)
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=1, shuffle=False, num_workers=0)
            self.test_loader = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
            self.n_envs = 1
            self._setup_lr_schedule()
            self.set_random_seed(self.seed)
            lstm_logger = utils.configure_logger(self.verbose, self.tensorboard_log, lstm_weight_saved_path, False)
            self.set_logger(lstm_logger)
        else:
            if _init_setup_model:
                self._setup_model()

            new_thread = threading.Thread(target=self.rendering_thread, args=(env,))
            new_thread.start()

    def rendering_thread(self, env):
        time.sleep(0.1)
        while(True):
            if not self.finished_save_pc:
                env.render(0)
            time.sleep(0.01)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.only_lstm_training:
            buffer_cls = LSTMDictRolloutBuffer
        else:
            buffer_cls = RecurrentDictRolloutBuffer

        if not self.retrain:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                n_lstm_layers=self.lstm_layer,
                lstm_hidden_size=256,
                shared_lstm = True,
                enable_critic_lstm = False,
                states_dim = self.states_dim,
                features_dim = self.features_dim,
                only_lstm_training = self.only_lstm_training,
                reconstruction_members = self.reconstruction_members,
                reconstruction_steps = self.reconstruction_steps,
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )

            # 1) Add Tanh activation to Policy Net
            if self.use_tanh_act:
                self.policy.action_net = th.nn.Sequential(
                    self.policy.action_net, th.nn.Tanh()
                )
            if self.state_vae is not None:
            #     pretrained_cnn = {
            #         'features_extractor.conv1.weight': self.state_vae['state_dict']['encoder.conv1.weight'],
            #         'features_extractor.conv1.bias': self.state_vae['state_dict']['encoder.conv1.bias'],
            #         'features_extractor.conv2.weight': self.state_vae['state_dict']['encoder.conv2.weight'],
            #         'features_extractor.conv2.bias': self.state_vae['state_dict']['encoder.conv2.bias'],
            #         'features_extractor.conv3.weight': self.state_vae['state_dict']['encoder.conv3.weight'],
            #         'features_extractor.conv3.bias': self.state_vae['state_dict']['encoder.conv3.bias'],
            #         'features_extractor.conv4.weight': self.state_vae['state_dict']['encoder.conv4.weight'],
            #         'features_extractor.conv4.bias': self.state_vae['state_dict']['encoder.conv4.bias'],
            #         'features_extractor.conv5.weight': self.state_vae['state_dict']['encoder.conv5.weight'],
            #         'features_extractor.conv5.bias': self.state_vae['state_dict']['encoder.conv5.bias'],
            #         'features_extractor.conv6.weight': self.state_vae['state_dict']['encoder.conv6.weight'],
            #         'features_extractor.conv6.bias': self.state_vae['state_dict']['encoder.conv6.bias'],
            #         'features_extractor.linear.weight': self.state_vae['state_dict']['encoder.fc_mu.weight'],
            #         'features_extractor.linear.bias': self.state_vae['state_dict']['encoder.fc_mu.bias'],
            #         'features_extractor.fc_logsigma.weight': self.state_vae['state_dict']['encoder.fc_logsigma.weight'],
            #         'features_extractor.fc_logsigma.bias': self.state_vae['state_dict']['encoder.fc_logsigma.bias'],
            #     }
            #     self.policy.load_state_dict(pretrained_cnn, strict=False)
            # self.policy = self.policy.to(self.device)
                pretrained_cnn = {}
                for old_key, val in self.state_vae["state_dict"].items():
                    if old_key.startswith("encoder."):
                        new_key = "features_extractor." + old_key[len("encoder."):]
                        pretrained_cnn[new_key] = val
                self.policy.load_state_dict(pretrained_cnn, strict=False)
            self.policy = self.policy.to(self.device)
            
        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        
        
        if getattr(self.policy, "use_rnn", True):
            head_input_dim = self.policy.lstm_output_dim + 7
        else:
            head_input_dim = self.policy.features_dim + 7
            
        if self.only_lstm_training:
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                n_seq=self.n_seq,
            )
        else:
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                n_seq=self.n_seq,
                # ppo_input_size=lstm.hidden_size + 7,
                ppo_input_size=head_input_dim,
            )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        iteration: int,
        deterministic: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer, LSTMDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        if self.if_change_maps: # 
            if self.is_forest_env and self.num_timesteps<1.2e6:
                # self.change_maps(env=env, radius=10.0)
                self.change_maps(env=env)
            elif not self.is_forest_env and iteration<3.2e5:
            # elif not self.is_forest_env and self.num_timesteps<1.2e6:
            # elif not self.is_forest_env and self.num_timesteps<1.8e6:
                # self.change_maps(env=env, radius=2)   # 
                self.change_maps(env=env, radius=3.5)
            else:
                self.change_maps(env=env, radius=2)
            env.reset()

        # if self.if_change_maps: # 
        #     self.env.resetRewCoeff()    
        #     self.change_maps(env=env)
        #     env.reset()
            
        while n_steps < n_rollout_steps:


            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # 
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                if getattr(self.policy, "use_rnn", True):
                    latent_pi, latent_vf, lstm_states = self.policy.forward_rnn(obs_tensor, lstm_states, episode_starts)   
                else:
                    latent_pi, latent_vf = self.policy.forward_cnn(obs_tensor)
                actions, values, log_probs = self.policy.forward(latent_pi, latent_vf, deterministic=deterministic)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            # print("render time: ", time.time() - t0)
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx//self.n_seq].get("terminal_observation") is not None
                    and infos[idx//self.n_seq].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx//self.n_seq]["terminal_observation"])[0]
                    # with th.no_grad():
                    #     terminal_lstm_state = (
                    #         lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                    #         lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                    #     )
                    #     # terminal_lstm_state = None
                    #     episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                    #     terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    # rewards[idx] += self.gamma * terminal_value
                    
                    with th.no_grad():
                        if getattr(self.policy, "use_rnn", True):
                            terminal_lstm_state = (
                                lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                                lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                            )
                            episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                            terminal_value = self.policy.predict_values(
                                terminal_obs, terminal_lstm_state, episode_starts
                            )[0]
                        else:
                            # CNN-only path
                            terminal_value = self.policy.predict_values_cnn(terminal_obs)[0]

                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                latent_lstm_pi=latent_pi,
                latent_lstm_vf=latent_vf,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states


        # with th.no_grad():
        #     # Compute value for the last timestep
        #     episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
        #     values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)
        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        with th.no_grad():
            if getattr(self.policy, "use_rnn", True):
                episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)
            else:
                values = self.policy.predict_values_cnn(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def collect_lstm_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        deterministic: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer, LSTMDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        if self.if_change_maps:
            # self.change_maps(env=self.eval_env)
            self.change_maps(env=self.eval_env, radius=3)
            env.reset()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict

                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # save_obs_debug(self._last_obs, n_steps, interval=100, save_dir="./debug")

                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                latent_pi, latent_vf, lstm_states = self.policy.forward_rnn(obs_tensor, lstm_states, episode_starts)
                actions, values, log_probs = self.policy.forward(latent_pi, latent_vf, deterministic=deterministic)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx//self.n_seq].get("terminal_observation") is not None
                    and infos[idx//self.n_seq].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx//self.n_seq]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # print(self.policy.state_dict()['features_extractor.cnn.0.weight'])
        # print(self.policy.state_dict()['pi_features_extractor.cnn.0.weight'])
        # print(self.policy.state_dict()['vf_features_extractor.cnn.0.weight'])
        # print(self.policy.state_dict()['mlp_extractor.policy_net.0.bias'])
        # print(self.policy.state_dict()['mlp_extractor.policy_net.2.bias'])
        # print(self.policy.state_dict()['action_net.0.bias'])
        # print(self.policy.state_dict()['value_net.bias'])
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_ppo_need(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Convert mask from float to bool
                # mask = rollout_data.mask > 1e-8
                # Re-sample the noise matrix because the log_std has changed
                # print(self.policy.log_std)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.latent_lstm_pi,
                    rollout_data.latent_lstm_vf,
                    actions,
                )
                values = values.flatten()
                # print("rollout_data.observations shape: ", rollout_data.observations['image'].shape)
                # print("rollout_data.lstm_states: ", rollout_data.lstm_states[0][0].shape)
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # print("log_prob: ", log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2))

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2))

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # print("loss: ", loss)
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def eval(self, iteration, if_eval=True, max_ep_length=1000) -> None:
        save_path = self.logger.get_dir() + "/TestTraj" # camerl/saved/RecurrentPPO_EVAL_x
        save_vis_path = self._logger.get_dir() + "/TSNE/" + "TSNE_{0:05d}".format(iteration)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_vis_path, exist_ok=True)
        #
        self.policy.eval()
        # rollout trajectory and save the trajectory
        if self.is_forest_env:
            easy_r = 6
            medium_r = 4.0
            hard_r = 3.2
        else:
            # D5-20：easy_r=2.5, medium_r=2; hard_r=1.8
            # D1-5: 2.2, 2, 1.8
            # D10-30: 2.5, 2.2, 1.9
            # D400-500: 9, 8, 7
            easy_r = 2.5
            medium_r = 7
            hard_r = 6
        
        seed_1 = 10
        seed_2 = 20
        seed_3 = 30
        seed_4 = 40
        

        # seed_list = [seed_1]  
        seed_list = [seed_1, seed_2, seed_3, seed_4]  

        levels = [
            ("easy",   easy_r,   "easy"),
            # ("medium", medium_r, "medium"),
            # ("hard",   hard_r,   "hard")
        ]

        results = {}

        for log_suffix, radius, level_tag in levels:
            sum_ave_reward = 0.0
            sum_success_rate = 0.0
            sum_trial_numbers = 0.0
            sum_avg_speed = 0.0
            sum_avg_suc_speed = 0.0
            # sum_avg_mp = 0.0

            for i, seed in enumerate(seed_list):
                print(f"Switching {level_tag} env: obstacle radius={radius}, seed={seed}")
                self.change_maps(env=self.eval_env, seed=seed, radius=radius, if_eval=if_eval)

                # traj_df, ave_reward, success_rate, trial_numbers, avg_speed, avg_mp = traj_rollout(
                traj_df, ave_reward, success_rate, trial_numbers, avg_speed, avg_suc_speed = traj_rollout(
                    self.eval_env, self.policy, max_ep_length=max_ep_length
                )

                traj_df.to_csv(
                    save_path + "/test_traj_{0:05d}_{1}_seed{2}.csv".format(
                        iteration, level_tag, seed
                    )
                )

                success_count = int(round(success_rate * trial_numbers))
                print(
                    f"[{level_tag}] seed={seed}: "
                    f"success={success_count}/{int(trial_numbers)}, "
                    f"avg_speed={avg_speed:.3f}"
                )
                sum_ave_reward += ave_reward
                sum_success_rate += success_rate
                sum_trial_numbers += trial_numbers
                sum_avg_speed += avg_speed
                sum_avg_suc_speed += avg_suc_speed
                # sum_avg_mp += avg_mp

            n_seeds = len(seed_list)
            results[log_suffix] = {
                "ave_reward":     sum_ave_reward / n_seeds,
                "success_rate":   sum_success_rate / n_seeds,
                "trial_numbers":  sum_trial_numbers,
                "avg_speed":      sum_avg_speed / n_seeds,
                "avg_suc_speed": sum_avg_suc_speed / n_seeds,
                # "avg_mp":      sum_avg_mp / n_seeds,
            }
        sr = results["easy"]["success_rate"]

        # easy
        # self.logger.record("test/ave_reward_easy",    results["easy"]["ave_reward"])
        self.logger.record("test/success_rate_easy",  results["easy"]["success_rate"])
        self.logger.record("test/success_rate_e4", f"{sr:.4f}")  
        # self.logger.record("test/trial_numbers_easy", results["easy"]["trial_numbers"])
        self.logger.record("test/avg_speed_easy",     results["easy"]["avg_speed"])
        self.logger.record("test/avg_suc_speed_easy",     results["easy"]["avg_suc_speed"])
        # self.logger.record("test/avg_mp_easy",     results["easy"]["avg_mp"])

        # # medium
        # # self.logger.record("test/ave_reward_medium",    results["medium"]["ave_reward"])
        # self.logger.record("test/success_rate_medium",  results["medium"]["success_rate"])
        # # self.logger.record("test/trial_numbers_medium", results["medium"]["trial_numbers"])
        # self.logger.record("test/avg_speed_medium",     results["medium"]["avg_speed"])
        # self.logger.record("test/avg_mp_medium",     results["medium"]["avg_mp"])

        # # hard
        # # self.logger.record("test/ave_reward_hard",    results["hard"]["ave_reward"])
        # self.logger.record("test/success_rate_hard",  results["hard"]["success_rate"])
        # # self.logger.record("test/trial_numbers_hard", results["hard"]["trial_numbers"])
        # self.logger.record("test/avg_speed_hard",     results["hard"]["avg_speed"])
        # self.logger.record("test/avg_mp_hard",     results["hard"]["avg_mp"])

        self.logger.dump(step=iteration)
        sr = results["easy"]["success_rate"]
        suc_spd = results["easy"]["avg_suc_speed"]
        avg_spd = results["easy"]["avg_speed"]
        print(f"success_rate_easy(avg_suc_speed_easy) = {sr:.4f}({avg_spd:.2f})")


    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 100),
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        new_cfg_dir = self.logger.get_dir() + "/config.yaml"
        with open(new_cfg_dir, "w") as outfile:
            YAML().dump(self.env_cfg, outfile)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, iteration=iteration)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval[0] == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

                for i in range(self.env.rew_dim - 1):
                    self.logger.record(
                        "rewards/{0}".format(self.env.reward_names[i]),
                        safe_mean(
                            [
                                ep_info[self.env.reward_names[i]]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                self.logger.dump(step=self.num_timesteps)

            self.train()

            if log_interval is not None and iteration % log_interval[1] == 0:
                # print(f"iteration:{iteration}, timesteps: {self.num_timesteps}")
                policy_path = self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(policy_path + "/iter_{0:05d}.pth".format(iteration))

                # self.eval(iteration)
        callback.on_training_end()

        return self
    
    def setup_eval(self) -> None:
        self._setup_learn(total_timesteps=0,
                    eval_env=self.eval_env,
                    tb_log_name="RecurrentPPO_EVAL")
    
    def eval_from_outer(self, iteration) -> None:
        self.eval(iteration, if_eval=False, max_ep_length=10000)

    def change_maps(self, env, seed=-1, radius=-1.0, if_eval=False):
        self.finished_save_pc = False
        self.env.spawnObstacles(change_obs=True, seed=seed, radius=radius)
        while not self.env.ifSceneChanged():
            self.env.spawnObstacles(change_obs=False)
            time.sleep(0.02)
        self.env.getPointClouds('', 0, True)
        time.sleep(0.2)
        while(not self.env.getSavingState()):
            time.sleep(0.02)
        if self.is_forest_env:
            time.sleep(12.0)
        else:
            time.sleep(2.0)
        env.readPointClouds(0)
        while(not env.getReadingState()):
            time.sleep(0.02)
        time.sleep(1.0)
        if not if_eval:
            self.finished_save_pc = True

    def change_policy(self, weight):
        self.policy.load_state_dict(weight, strict=False)
    
    def eval_lstm(self, iteration) -> None:
        save_path = self.logger.get_dir() + "/Reconstruction"
        os.makedirs(save_path, exist_ok=True)
        #
        self.policy.eval()
        lstm_rollout(self.eval_env, self.policy, self.device, save_path, iteration)
        # rollout trajectory and save the trajectory
        # traj_df, features, labels = traj_rollout(self.eval_env, self.policy)

    def train_lstm(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        total_loss = 0
        record_loss = 0
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                latent_obs = self.policy.to_latent(rollout_data.observations)
                # recon_current, recon_previous, n_seq, _ = self.policy.predict_lstm(latent_obs, rollout_data.lstm_states.pi, rollout_data.episode_starts)
                # loss = self.lstm_loss_function(rollout_data.observations, recon_current, recon_previous, n_seq)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, rollout_data.lstm_states.pi, rollout_data.episode_starts)
                loss, record = self.lstm_loss_function(rollout_data.observations, recon, n_seq, epoch)
                print("epoch: ", epoch, "  --loss: ", loss.item())
                # print("pre_next_obs: ", pre_next_obs[:, :35])
                # print("next_obs: ", latent_obs[:, :35])
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
                total_loss += loss
                record_loss += record
        return total_loss / self.n_epochs, record_loss / self.n_epochs
    
    def fine_tune_lstm_from_rosbag(self):
        for epoch in range(self.n_epochs):
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            train_loss = 0
            future_loss = 0
            self.dataset_train.load_next_buffer()
            for batch_idx, data in enumerate(self.train_loader):
                obs_th = data.squeeze().unsqueeze(1).to(self.device)
                latent_obs = self.policy.to_latent(obs_th)
                single_hidden_state_shape = self.policy.lstm_hidden_state_shape
                lstm_states = (
                    th.zeros(single_hidden_state_shape,  device=self.device),
                    th.zeros(single_hidden_state_shape,  device=self.device),
                )
                episode_starts = th.zeros((1,), dtype=th.float32, device=self.device)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(obs_th, recon, n_seq, epoch)
                self.policy.optimizer.zero_grad()
                train_loss += loss.item()
                future_loss += record
                loss.backward()
                self.policy.optimizer.step()
                if batch_idx % 5 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss for Future: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(data), record / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}, Future loss: {:.4}'.format(
                epoch, train_loss / len(self.train_loader.dataset), future_loss / len(self.train_loader.dataset)))
            self.logger.record("train/loss", train_loss / len(self.train_loader.dataset))
            self.logger.record("train/future_loss", future_loss / len(self.train_loader.dataset))
            self.logger.dump(step=epoch)
            if epoch % 10 == 0:
                self.test_lstm_from_dataset(epoch)
            if epoch % 20 == 0:
                policy_path =self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(self.logger.get_dir() + "/Policy" + "/iter_{0:05d}.pth".format(epoch))
    
    def train_lstm_from_dataset(self):
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            train_loss = 0
            future_loss = 0
            self.dataset_train.load_next_buffer()
            for batch_idx, data in enumerate(self.train_loader):

                n_seq = data[1][0][0].shape[1]
                observations = {key: obs[0] for (key, obs) in data[0].items()}

                # only train the first n_seq images if your pc don't have enough memory
                # single_hidden_state_shape = self.policy.lstm_hidden_state_shape
                # lstm_states = (
                #     th.zeros(single_hidden_state_shape,  device=self.device),
                #     th.zeros(single_hidden_state_shape,  device=self.device),
                # )
                # episode_starts = th.zeros((1,), dtype=th.float32, device=self.device)
                # img_num = observations['image'].shape[0]
                # observations = {key: obs[0 : int(img_num/n_seq)] for (key, obs) in observations.items()}
                
                lstm_states = (data[1][0][0], data[1][1][0])
                episode_starts = data[2][0]
                latent_obs = self.policy.to_latent(observations)
                # 
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(observations, recon, n_seq, epoch)
                self.policy.optimizer.zero_grad()
                train_loss += loss.item()
                future_loss += record
                loss.backward()
                self.policy.optimizer.step()
                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss for Future: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(data), record / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}, Future loss: {:.4}'.format(
                epoch, train_loss / len(self.train_loader.dataset), future_loss / len(self.train_loader.dataset)))
            self.logger.record("train/loss", train_loss / len(self.train_loader.dataset))
            self.logger.record("train/future_loss", future_loss / len(self.train_loader.dataset))
            self.logger.dump(step=epoch)
            # test the model and save the model each 50 epochs
            if epoch % 10 == 0:
                self.test_lstm_from_dataset(epoch)
            # self.test_lstm_from_dataset(epoch)
            if epoch % 50 == 0:
                policy_path =self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(self.logger.get_dir() + "/Policy" + "/iter_{0:05d}.pth".format(epoch))
            
    def test_lstm_from_dataset(self, epoch):
        self.policy.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        future_loss = 0
        with th.no_grad():
            for data in self.test_loader:
                if not self.fine_tune_from_rosbag:
                    observations = {key: obs[0] for (key, obs) in data[0].items()}

                    # only test the first n_seq images if your pc don't have enough memory
                    # n_seq = data[1][0][0].shape[1]
                    # img_num = observations['image'].shape[0]
                    # observations = {key: obs[0 : int(img_num/n_seq)] for (key, obs) in observations.items()}

                    lstm_states = (data[1][0][0], data[1][1][0])
                    episode_starts = data[2][0]
                else:
                    observations = data.squeeze().unsqueeze(1).to(self.device)
                    single_hidden_state_shape = self.policy.lstm_hidden_state_shape
                    lstm_states = (
                        th.zeros(single_hidden_state_shape,  device=self.device),
                        th.zeros(single_hidden_state_shape,  device=self.device),
                    )
                    episode_starts = th.zeros((1,), dtype=th.float32, device=self.device)
                latent_obs = self.policy.to_latent(observations)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(observations, recon, n_seq, 0)
                test_loss += loss.item()
                future_loss += record
        test_loss /= len(self.test_loader.dataset)
        future_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}, Future loss: {:.4}'.format(test_loss, future_loss))
        self.logger.record("test/loss", test_loss)
        self.logger.record("test/future_loss", future_loss)
        self.logger.dump(step=epoch)
        save_path = self.logger.get_dir() + "/Reconstruction"
        os.makedirs(save_path, exist_ok=True)
        if self.fine_tune_from_rosbag:
            self.plot_depth_image(observations, recon, n_seq, epoch)
        else:
            self.plot_test_image(observations, recon, n_seq, epoch)

    def test_lstm_seperate(self):
        self.policy.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        for data in self.test_loader:
            observations = {key: obs[0] for (key, obs) in data[0].items()}
            lstm_states = (data[1][0][0], data[1][1][0])
            episode_starts = data[2][0]
            latent_obs = self.policy.to_latent(observations)
            recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
            mae, std = self.lstm_test_loss_std(observations, recon, n_seq)
            print('====> Test set loss: {:.4f}, std: {:.4}'.format(mae, std))

    def plot_test_image(self, obs, recon, n_seq, epoch):
        if isinstance(obs, dict):
            obs = obs['image']
        

        shape = obs.shape   # 500, 1, 256, 256
        recon_next_plot = None
        recon_previous_plot = None
        recon_current_plot = None
        obs = obs[0 : int(shape[0]/n_seq), :, :, :].float() / 255.0

        # print(f"[INFO] n_seq: {n_seq}")
        # print(f"[INFO] reconstruction_steps: {self.reconstruction_steps}")


        if recon[0] is not None:
            recon_previous_plot = recon[0][0 : int(shape[0]/n_seq), :, :, :]
        if recon[1] is not None:
            recon_current_plot = recon[1][0 : int(shape[0]/n_seq), :, :, :]

        if recon[2] is not None:
            recon_next_plot = recon[2][0 : int(shape[0]/n_seq), :, :, :]

        saved_images = []
        # save the plot each 20 timesteps
        for i in range(20, int(shape[0]/n_seq), 20):
            plot = []
            if recon_previous_plot is not None:
                plot.append(obs[i-2*self.reconstruction_steps])
                plot.append(recon_previous_plot[i-self.reconstruction_steps])
            if recon_current_plot is not None:
                plot.append(obs[i-self.reconstruction_steps])
                plot.append(recon_current_plot[i-self.reconstruction_steps])
            if recon_next_plot is not None:
                plot.append(obs[i])
                plot.append(recon_next_plot[i-self.reconstruction_steps])
            saved_images.append(th.stack(plot, dim=0))
        save_image(th.cat(saved_images), self.logger.get_dir() + "/Reconstruction" + "/recon_{0:05d}.png".format(epoch))

    def plot_depth_image(self, obs, recon, n_seq, seq_num):
        if isinstance(obs, dict):
            obs = obs['image']
        shape = obs.shape
        recon_next_plot = None
        recon_previous_plot = None
        recon_current_plot = None
        obs = obs[0 : int(shape[0]/n_seq), :, :, :].float() / 255.0
        if recon[0] is not None:
            recon_previous_plot = recon[0][0 : int(shape[0]/n_seq), :, :, :]
        if recon[1] is not None:
            recon_current_plot = recon[1][0 : int(shape[0]/n_seq), :, :, :]
        if recon[2] is not None:
            recon_next_plot = recon[2][0 : int(shape[0]/n_seq), :, :, :]
        save_path = self.logger.get_dir() + "/Reconstruction/Sequence_{0}".format(seq_num)
        save_path3 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_future".format(seq_num)
        save_path2 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_current".format(seq_num)
        save_path1 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_past".format(seq_num)
        save_path0 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/obs".format(seq_num)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path3, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        os.makedirs(save_path1, exist_ok=True)
        os.makedirs(save_path0, exist_ok=True)
        for i in range(10, int(shape[0]/n_seq)-10):
            # save each sequence images to a seperate folder
            save_image(obs[i], save_path0 + "/obs_{0:05d}.png".format(i))
            save_image(recon_current_plot[i], save_path2 + "/recon_current_{0:05d}.png".format(i))
            # save_image(recon_next_plot[i-self.reconstruction_steps], save_path3 + "/recon_future_{0:05d}.png".format(i))
            save_image(recon_previous_plot[i+self.reconstruction_steps], save_path1 + "/recon_past_{0:05d}.png".format(i))

    
    def save_lstm_rollout(self, iteration):
        self.policy.set_training_mode(False)
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            # convert pytorch tensor to numpy array
            observations = {key: obs.cpu().numpy() for (key, obs) in rollout_data.observations.items()}
            lstm_states = (rollout_data.lstm_states.pi[0].cpu().numpy(), rollout_data.lstm_states.pi[1].cpu().numpy())
            episode_starts = rollout_data.episode_starts.cpu().numpy()
            # actions = rollout_data.actions.cpu().numpy()
            # save the rollout data to the file
            save_path = self.logger.get_dir()
            os.makedirs(save_path, exist_ok=True)
            np.savez(save_path + "/rollout_{0:05d}.npz".format(iteration), observations=observations, 
                     lstm_states=lstm_states, episode_starts=episode_starts)


    def lstm_loss_function(self, obs, obs_recon, n_seq, epoch):
        if isinstance(obs, dict):
            obs = obs['image'].float() / 255.0
        else:
            obs = obs.float() / 255.0
        obs_shape = obs.shape
        BCE = 0
        future_loss = 0

        if self.reconstruction_members[0]:
            BCE += F.mse_loss(obs_recon[0], th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, :-self.reconstruction_steps, :], 
                                                       start_dim=0, end_dim=1), reduction='sum')
        if self.reconstruction_members[1]:
            BCE += F.mse_loss(obs_recon[1], obs, reduction='sum')

        if self.reconstruction_members[2]:
            future_loss = F.mse_loss(obs_recon[2], th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, self.reconstruction_steps:, :], 
                                                       start_dim=0, end_dim=1), reduction='sum')
            BCE += future_loss
            future_loss = future_loss.item()
        return BCE, future_loss
    
    def lstm_test_loss_std(self, obs, obs_recon, n_seq):
        if isinstance(obs, dict):
            obs = obs['image'].float() / 255.0
        else:
            obs = obs.float() / 255.0
        obs_shape = obs.shape
        print(obs_recon[0].shape)
        BCE = 0
        if self.reconstruction_members[0]:
            diff_0 = th.abs(obs_recon[0] - th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, :-self.reconstruction_steps, :], 
                                                       start_dim=0, end_dim=1)) * 255.0
            diff_0 = th.sum(diff_0, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_0)
            std = th.std(diff_0)
        if self.reconstruction_members[1]:
            diff_1 = th.abs(obs_recon[1] - obs) * 255.0
            diff_1 = th.sum(diff_1, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_1)
            std = th.std(diff_1)
        if self.reconstruction_members[2]:
            diff_2 = th.abs(obs_recon[2] - th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, self.reconstruction_steps:, :], 
                                                       start_dim=0, end_dim=1)) * 255.0
            diff_2 = th.sum(diff_2, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_2)
            std = th.std(diff_2)
        return BCE.item(), std.item()

    def learn_lstm(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 10),
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        callback.on_training_start(locals(), globals())

        # while self.num_timesteps < total_timesteps:
        while iteration < 2:
        
            continue_training = self.collect_lstm_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, deterministic=True)

            if continue_training is False:
                break
            if not self.save_lstm_dateset:
                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                ave_loss, record_loss = self.train_lstm()
                print("average loss: ", ave_loss)
                self.logger.record("train/future_loss", record_loss)
                self.logger.dump(step=self.num_timesteps)
                if log_interval is not None and iteration % log_interval[1] == 0:
                    policy_path = self.logger.get_dir() + "/Policy"
                    os.makedirs(policy_path, exist_ok=True)
                    self.policy.save(policy_path + "/iter_{0:05d}.pth".format(iteration))
                    self.eval_lstm(iteration)

            else:
                iteration += 1
                self.save_lstm_rollout(iteration)

            callback.on_training_end()

