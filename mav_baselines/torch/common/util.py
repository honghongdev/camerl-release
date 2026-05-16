import numpy as np
import pandas as pd
from torchvision.utils import save_image
import torch as th
from mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import obs_as_tensor

import cv2

vision_columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "goal_z",
    "act11",
    "act12",
    "act13",
    "act14", 
]
columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "goal_z",
    "wx",
    "wy",
    "wz",
    "ax",
    "ay",
    "az",
    "mot1",
    "mot2",
    "mot3",
    "mot4",
    "thrust1",
    "thrust2",
    "thrust3",
    "thrust4",
    "targetx",
    "targety",
    "targetz",
    "targetr",
    "act1",
    "act2",
    "act3",
    "act4",
]

LSIZE, RED_SIZE = 32, 256

def compute_success_avg_speed(traj_df: pd.DataFrame) -> float:
    """
    Success criterion: terminal reward of current episode >= 0.
    Skips the last episode (no next episode to read terminal reward from).
    Returns the mean per-episode average speed across successful episodes.
    """
    if traj_df is None or traj_df.empty:
        return 0.0

    df = traj_df.copy()
    df["episode_id"] = df["episode_id"].astype(int)
    df = df.sort_values(["episode_id", "t"]).reset_index(drop=True)

    speed = np.sqrt(df["vx"].to_numpy()**2 + df["vy"].to_numpy()**2 + df["vz"].to_numpy()**2)
    df["speed"] = speed

    ep_ids = df["episode_id"].unique()
    succ_ep_avg_speeds = []

    for i, ep_id in enumerate(ep_ids[:-1]):
        next_id = ep_ids[i + 1]
        terminal_reward = float(df.loc[df["episode_id"] == next_id, "reward"].iloc[0])
        if terminal_reward >= 0.0:
            succ_ep_avg_speeds.append(float(df.loc[df["episode_id"] == ep_id, "speed"].mean()))

    return float(np.mean(succ_ep_avg_speeds)) if succ_ep_avg_speeds else 0.0

def compute_mission_progress(traj_df: pd.DataFrame, mp_mode: str = "all"):
    """
    Compute Mission Progress (MP) and return:
        - traj_df with per-step mp column
        - avg_mp aggregated by mp_mode

    mp_mode:
        "all"     : all episodes with terminal feedback
        "success" : episodes where terminal reward >= 0
        "failed"  : episodes where terminal reward < 0
    Terminal reward is read from the first step of the next episode.
    The last episode is always skipped.
    """
    if traj_df is None or traj_df.empty:
        return traj_df, 0.0

    df = traj_df.copy()
    df["episode_id"] = df["episode_id"].astype(int)
    df = df.sort_values(["episode_id", "t"]).reset_index(drop=True)
    df["mp"] = 0.0

    episode_ids = df["episode_id"].unique()
    episode_stats = []

    for ep_id in episode_ids:
        g = df[df["episode_id"] == ep_id]

        p0 = g[["px", "py", "pz"]].iloc[0].to_numpy()
        goal = g[["goal_x", "goal_y", "goal_z"]].iloc[0].to_numpy()

        dir_vec = goal - p0
        norm = np.linalg.norm(dir_vec)
        u = dir_vec / norm if norm > 1e-6 else np.zeros(3)

        pos = g[["px", "py", "pz"]].to_numpy()
        mp_vals = (pos - p0) @ u  # (N,3) · (3,) -> (N,)
        
        dist_to_goal = norm  # = np.linalg.norm(goal - p0)
        mp_vals = np.clip(mp_vals, a_min=None, a_max=dist_to_goal)
        

        df.loc[g.index, "mp"] = mp_vals

    for i, ep_id in enumerate(episode_ids[:-1]):
        g = df[df["episode_id"] == ep_id]
        final_mp = float(g["mp"].iloc[-1])

        next_ep_id = episode_ids[i + 1]
        g_next = df[df["episode_id"] == next_ep_id]
        terminal_reward = float(g_next["reward"].iloc[0])
        is_success = terminal_reward >= 0.0

        episode_stats.append((final_mp, is_success))

    if not episode_stats:
        return df, 0.0

    def keep(mp, suc):
        if mp_mode == "all":
            return True
        if mp_mode == "success":
            return suc
        if mp_mode == "failed":
            return not suc
        raise ValueError(f"Unknown mp_mode: {mp_mode}")

    mps = [mp for mp, suc in episode_stats if keep(mp, suc)]
    if not mps:
        avg_mp = 0.0
    else:
        avg_mp = float(np.mean(mps))

    return df, avg_mp

def traj_rollout(env, policy, max_ep_length=1000):
    traj_df = pd.DataFrame(columns=vision_columns)
    max_ep_length = max_ep_length
    # features = np.zeros([max_ep_length, LSIZE], dtype=np.float64)
    # labels = np.zeros([max_ep_length, 1, 28, 28], dtype=np.float64)
    obs = env.reset(random=False)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    ave_reward = 0
    success_trial = 0
    trial = 0
    lstm_states = None
    while trial < 25:
        act, lstm_states = policy.predict(obs, state=lstm_states, deterministic=True)
        act = np.array(act, dtype=np.float64)
        obs, rew, done, info = env.step(act)
        
        # print("image shape:", images.shape, "dtype:", images.dtype)

        # images_f = images.astype(np.float32)

        # valid = np.isfinite(images_f)
        # vmin = float(np.percentile(images_f[valid], 1))
        # vmax = float(np.percentile(images_f[valid], 99))
        # print(f"global vmin={vmin:.4f}, vmax={vmax:.4f}")
     
        # policy.reconstruction_members=[True, True, False]
        # recons = policy.predict_img(lstm_states[0].reshape((1, -1)))
        # if (recons[1] is not None) and (recons[0] is not None):
        #     imgs = np.hstack([(recons[0].reshape([256, 256, 1]) * 255).astype(np.uint8), (recons[1].reshape([256, 256, 1]) * 255).astype(np.uint8)])
        #     cv2.imshow("recon", imgs)
        #     cv2.waitKey(1)
        # elif (recons[1] is not None):
        #     imgs = (recons[1].reshape([256, 256, 1]) * 255).astype(np.uint8)
        #     cv2.imshow("recon", imgs)
        #     cv2.waitKey(1)

        if done[0]==True:
            trial += 1
            lstm_states = None
            if rew[0]>=0:
                success_trial += 1
            # print(f"Current Success: {success_trial} / {trial}")
        ave_reward += rew
        # labels[i, :] = np.expand_dims(env.getLabelImage(), 0)
        episode_id[done] += 1

        state = env.getQuadState()
        action = env.getQuadAct()
        
        # reshape vector
        done = done[:, np.newaxis]
        rew = rew[:, np.newaxis]

        # stack all the data
        data = np.hstack((episode_id, done, rew, state, action))
        data_frame = pd.DataFrame(data=data, columns=vision_columns)

        # append trajectory
        traj_df = pd.concat([traj_df, data_frame], axis=0, ignore_index=True)
    # return traj_df, ave_reward / max_ep_length, success_trial / trial, trial

    if len(traj_df) > 0:
        vx = traj_df["vx"].to_numpy()
        vy = traj_df["vy"].to_numpy()
        vz = traj_df["vz"].to_numpy()
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        avg_speed = float(speed.mean())
    
    avg_speed_success = compute_success_avg_speed(traj_df)
    # avg_speed = avg_speed_success

    
    # print("suc_rate:", success_trial / trial)
    scuc_rate = success_trial / trial

    return (
        traj_df,
        ave_reward / max_ep_length,
        scuc_rate,
        # success_trial / trial,
        trial,
        avg_speed,
        avg_speed_success
    )
    
    # return (
    #     traj_df,
    #     ave_reward / max_ep_length,
    #     success_trial / trial,
    #     trial,
    # )

def lstm_rollout(env, policy, device, logdir, iteration):
    max_ep_length = 200
    obs = env.reset(random=False)
    labels = np.zeros([max_ep_length, 1, 28, 28], dtype=np.float64)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    single_hidden_state_shape = policy.lstm_hidden_state_shape
    _last_episode_starts = np.ones((1,), dtype=bool)
    _last_lstm_states = (
                th.zeros(single_hidden_state_shape,  device=device),
                th.zeros(single_hidden_state_shape,  device=device),
                )
    time_stamp = 0
    saved_images = []

    recon_next_plot = None
    recon_previous_plot = None
    recon_current_plot = None

    for i in range(max_ep_length):
        act, _ = policy.predict(obs, deterministic=True)
        act = np.array(act, dtype=np.float64)
        obs, rew, done, info = env.step(act)
        obs_torch = obs_as_tensor(obs,  device=device)
        latent_obs = policy.to_latent(obs_torch)
        episode_starts = th.tensor(_last_episode_starts, dtype=th.float32, device=device)
        recon, n_seq, _last_lstm_states= policy.predict_lstm(latent_obs, _last_lstm_states, episode_starts, is_eva=True)
        _last_episode_starts = done
        time_stamp += 1
        plot = []
        # state = env.getQuadState()
        if done[0]:
            time_stamp = 0
        if time_stamp % (20-policy.reconstruction_steps) == 0:
            if recon[0] is not None:
                obs_previous = obs_torch['image'][0].clone().detach().float() / 255.0
            
        if time_stamp % 20 == 0:
            if recon[0] is not None:
                recon_previous_plot = recon[0][0]
            if recon[1] is not None:
                recon_current_plot = recon[1][0]
                obs_current = obs_torch['image'][0].clone().detach().float() / 255.0
            if recon[2] is not None:
                recon_next_plot = recon[2][0]

        if time_stamp % (20+policy.reconstruction_steps) == 0:
            obs_next = obs_torch['image'][0].clone().detach().float() / 255.0
            time_stamp = 0
            if recon[0] is not None:
                plot.append(obs_previous)
                plot.append(recon_previous_plot) 
            if recon[1] is not None:
                plot.append(obs_current)
                plot.append(recon_current_plot)
            if recon[2] is not None:
                plot.append(obs_next)
                plot.append(recon_next_plot)

            saved_images.append(th.stack(plot, dim=0))
            # print("timestamp20: ", state)
    save_image(th.cat(saved_images), logdir + "/iter_{0:05d}.png".format(iteration))

def plot3d_traj(ax3d, pos, vel):
    sc = ax3d.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=np.linalg.norm(vel, axis=1),
        cmap="jet",
        s=1,
        alpha=0.5,
    )
    ax3d.view_init(elev=40, azim=50)
    #
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    # ax3d.set_zticks([])

    #
    # ax3d.get_proj = lambda: np.dot(
    # Axes3D.get_proj(ax3d), np.diag([1.0, 1.0, 1.0, 1.0]))
    # zmin, zmax = ax3d.get_zlim()
    # xmin, xmax = ax3d.get_xlim()
    # ymin, ymax = ax3d.get_ylim()
    # x_f = 1
    # y_f = (ymax - ymin) / (xmax - xmin)
    # z_f = (zmax - zmin) / (xmax - xmin)
    # ax3d.set_box_aspect((x_f, y_f * 2, z_f * 2))

def test_vision_policy(env, model):
    max_ep_length = env.max_episode_steps
    num_rollouts = 20
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)
            ep_len += 1

def test_policy(env, model, render=False):
    max_ep_length = env.max_episode_steps
    num_rollouts = 1
    frame_id = 0
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not ((ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)
            #
            # print(obs)
            env.render(ep_len)

            # ======Gray Image=========
            # gray_img = np.reshape(
            #     env.getImage()[0], (env.img_height, env.img_width))
            # cv2.imshow("gray_img", gray_img)
            # cv2.waitKey(100)

            # ======RGB Image=========
            # img =env.getImage(rgb=True) 
            # rgb_img = np.reshape(
            #    img[0], (env.img_height, env.img_width, 3))
            # cv2.imshow("rgb_img", rgb_img)
            # os.makedirs("./images", exist_ok=True)
            # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img)
            # cv2.waitKey(100)

            # # # ======Depth Image=========
            # depth_img = np.reshape(env.getDepthImage()[
            #                        0], (env.img_height, env.img_width))
            # os.makedirs("./depth", exist_ok=True)
            # cv2.imwrite("./depth/img_{0:05d}.png".format(frame_id), depth_img.astype(np.uint16))
            # cv2.imshow("depth", depth_img)
            # cv2.waitKey(100)

            #
            ep_len += 1
            frame_id += 1

    #
    if render:
        env.disconnectUnity()