import os
import collections
import torch
import numpy as np
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
from ruamel.yaml import YAML
from stable_baselines3.common.utils import get_device
from mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy
import mav_baselines
import sys
sys.modules['rpg_baselines_prev'] = mav_baselines

# ── config ────────────────────────────────────────────────────────────────────
# Final trained policy weight (output of train_policy.py --retrain 1)
WEIGHT = "./saved/4_retrain_policy/final_run/Policy/iter_02000.pth"
# ─────────────────────────────────────────────────────────────────────────────


class RobotState:
    def __init__(self, cfg, dim=4) -> None:
        self.acc       = np.zeros(dim-1, dtype=np.float64)
        self.vel       = np.zeros(dim-1, dtype=np.float64)
        self.pos       = np.zeros(dim-1, dtype=np.float64)
        self.quat      = np.zeros(dim,   dtype=np.float64)
        self.vel_world = np.zeros(dim-1, dtype=np.float64)
        self.target    = np.zeros(dim-1, dtype=np.float64)
        act_max = np.array(cfg["simulation"]["act_max"])
        act_min = np.array(cfg["simulation"]["act_min"])
        self.act_mean  = (act_min + act_max) / 2
        self.act_std   = (act_max - act_min) / 2
        self.yaw_rate  = 0
        self.yaw       = 0

    def setState(self, pos, vel, quat):
        self.waypoints = []
        self.vel  = vel
        self.pos  = pos
        self.quat = quat
        self.yaw  = R.from_quat(self.quat).as_euler('zyx')[0]
        if self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        elif self.yaw < -np.pi:
            self.yaw += 2 * np.pi

    def setTarget(self, target):
        self.target = target

    def getState(self):
        return (self.pos.tolist() + self.vel.tolist() + self.target.tolist() + [self.yaw])

    def step(self, input, duration):
        cmd          = input * self.act_std + self.act_mean
        self.acc     = cmd[:3]
        self.yaw_rate = cmd[3]
        acc_world    = self.body2world(self.acc)
        self.vel_world = self.vel + acc_world * duration
        self.vel     = self.vel_world

    def body2world(self, body):
        rot = R.from_quat(self.quat)
        world_flu = np.array(rot.apply(body))
        # FLU to RFU
        world_rfu = np.array([-world_flu[1], world_flu[0], world_flu[2]])
        return world_rfu

    def world2body(self, world):
        rot = R.from_quat(self.quat)
        world_rfu = np.array([world[1], -world[0], world[2]])
        body_rfu  = np.array(rot.inv().apply(world_rfu))
        # RFU to FLU
        return body_rfu

    def get_vel_cmd(self):
        return self.vel_world.tolist() + [self.yaw_rate]


class AvoiderNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.config = YAML().load(open(
            os.environ["AVOIDBENCH_PATH"] + "/../camerl/configs/control/config.yaml", "r"))
        self.robot             = RobotState(self.config)
        self.input_update_freq = self.config["ros"]["input_update_freq"]
        self.use_depth         = self.config["ros"]["use_depth"]
        self.input_height      = self.config["rgb_camera"]["height"]
        self.input_width       = self.config["rgb_camera"]["width"]
        self.velocity_frame    = self.config["ros"]["velocity_frame"]
        self.seq_len           = self.config["ros"]["seq_len"]
        self.goal_obs_dim      = self.config["ros"]["goal_obs_dim"]
        self.trial             = self.config["ros"]["trial"]
        self.iter              = self.config["ros"]["iter"]
        self.pre_steps         = self.config["ros"]["pre_steps"]

        self.device            = get_device("auto")
        self.get_state         = False
        self.env_              = None
        self.target            = None
        self.odometry          = None
        self.next_img          = None
        self.episode_starts    = None
        self.lstm_states       = None
        self.depth             = np.zeros((self.input_height, self.input_width))
        self.net_initialized   = False
        self.depth_queue       = collections.deque([], maxlen=self.input_update_freq)
        self.state_queue       = collections.deque([], maxlen=self.input_update_freq)
        self.time_prediction   = None
        self.reset_queue()
        self._prepare_net_inputs()
        self.create_policy()
        self.net_initialized   = True
        self.final_point_sent  = False
        self.act_np            = None

        # publishers: policy-start flag, velocity command, inference latency
        # subscribers: depth image, odometry, goal path
        self.ctr_activate_flag_pub = rospy.Publisher(
            '/hummingbird/agiros_pilot/ctr_activate_flag', Bool, queue_size=10)
        self.vel_pub  = rospy.Publisher(
            '/hummingbird/agiros_pilot/velocity_command', TwistStamped, queue_size=1)
        self.time_cost_pub = rospy.Publisher(
            "/hummingbird/iter_time", Float32, queue_size=1)
        if self.use_depth:
            self.depth_sub = rospy.Subscriber(
                '/depth', Image, self.depth_callback, queue_size=1)
        self.ground_truth_odom_sub = rospy.Subscriber(
            '/hummingbird/agiros_pilot/odometry', Odometry,
            self.ground_truth_odom_callback, queue_size=1)
        self.target_sub = rospy.Subscriber(
            '/hummingbird/goal_point', Path, self.target_callback, queue_size=1)
        # fixed-rate timers
        self.timer_input   = rospy.Timer(
            rospy.Duration(1.0 / self.input_update_freq), self.update_input_queue)
        self.timer_network = rospy.Timer(
            rospy.Duration(1.0 / self.input_update_freq), self._generate_command)
        self.timer_command = rospy.Timer(
            rospy.Duration(1.0 / 30), self.send_command)

    def depth_callback(self, msg):
        try:
            # depth in mm; convert to meters
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') / 1000.0
            if (np.sum(depth) != 0) and (not np.any(np.isnan(depth))):
                # clip to 12 m and normalize to 0-255
                depth = (np.minimum(depth, 12.0)) / 12.0 * 255.0
                shape = depth.shape
                # crop to square [H, H]
                depth = depth[:, int((shape[1]-shape[0])/2 - 1) : int((shape[1]+shape[0])/2 - 1)]
                dim   = (self.input_height, self.input_width)
                depth = cv2.resize(depth, dim, interpolation=cv2.INTER_AREA)
                self.depth = depth.astype('int')
        except CvBridgeError as e:
            print(e)

    def calRLState(self, state_inputs):
        delta_p      = (np.array(state_inputs[6:9]) - np.array(state_inputs[0:3])).tolist()
        log_distance = np.log(np.sqrt(delta_p[0]**2 + delta_p[1]**2) + 1.0)
        vel_body     = self.robot.world2body(np.array(state_inputs[3:6]))
        horizon_vel  = np.sqrt(vel_body[0]**2 + vel_body[1]**2)
        theta             = np.arctan2(-delta_p[0], delta_p[1])
        horizon_vel_dire  = np.arctan2(vel_body[1], vel_body[0])
        return np.array([log_distance, horizon_vel, theta, horizon_vel_dire,
                         delta_p[2], vel_body[2], self.robot.yaw], dtype=np.float64)

    def reset_queue(self):
        self.depth_queue.clear()
        self.state_queue.clear()
        for _ in range(self.input_update_freq):
            self.depth_queue.append(np.zeros_like(self.depth))
            self.state_queue.append(np.zeros((self.goal_obs_dim,)))

    def select_inputs_in_freq(self, input_list):
        return [input_list[i] for i in self.required_elements]

    def _prepare_net_inputs(self):
        if not self.net_initialized or not self.get_state:
            required_elements = np.arange(
                start=0, stop=self.input_update_freq,
                step=int(np.ceil(self.input_update_freq / self.seq_len)),
                dtype=np.int64)
            required_elements = -1 * (required_elements + 1)
            self.required_elements = [i for i in reversed(required_elements.tolist())]
            self.net_inputs = {
                'image': np.zeros([1, self.seq_len, self.input_height, self.input_width], dtype=np.uint8),
                'state': np.zeros([1, self.seq_len, self.goal_obs_dim], dtype=np.float64),
            }
            return
        depth_inputs  = np.stack(self.select_inputs_in_freq(self.depth_queue), axis=0)
        state_inputs  = np.stack(self.select_inputs_in_freq(self.state_queue), axis=0).squeeze()
        self.robot.setState(np.array(state_inputs[0:3]),
                            np.array(state_inputs[3:6]),
                            state_inputs[9:])
        self.time_prediction = rospy.Time.now()
        self.net_inputs = {
            'image': np.expand_dims(depth_inputs, axis=0),
            'state': np.expand_dims(self.calRLState(state_inputs), axis=0),
        }

    def create_policy(self):
        weight = WEIGHT
        saved_varables = torch.load(weight, map_location=self.device)
        saved_varables["data"]['reconstruction_members'] = [True, True, False]
        self.policy = MultiInputLstmPolicy(features_dim=64, **saved_varables["data"])
        self.policy.action_net = torch.nn.Sequential(self.policy.action_net, torch.nn.Tanh())
        self.policy.load_state_dict(saved_varables["state_dict"], strict=False)
        self.policy.to(self.device)

    def ground_truth_odom_callback(self, msg):
        self.odometry = msg
        self.rot_body = R.from_quat([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def target_callback(self, msg):
        ctr_activate_flag_msg      = Bool()
        ctr_activate_flag_msg.data = True
        self.ctr_activate_flag_pub.publish(ctr_activate_flag_msg)

        self.target = msg.poses[-1].pose.position
        self.robot.setTarget(np.array([self.target.x, self.target.y, self.target.z]))
        self.final_point_sent = False
        self.episode_starts   = None
        print("target: ", self.target)

    def update_input_queue(self, data):
        if self.target is None or self.odometry is None:
            return
        state_inputs = [
            self.odometry.pose.pose.position.x,
            self.odometry.pose.pose.position.y,
            self.odometry.pose.pose.position.z,
            self.odometry.twist.twist.linear.x,
            self.odometry.twist.twist.linear.y,
            self.odometry.twist.twist.linear.z,
            self.target.x, self.target.y, self.target.z,
        ] + self.rot_body.as_quat().tolist()
        self.state_queue.append(state_inputs)
        self.depth_queue.append(self.depth)
        self.get_state = True

    def _generate_command(self, data):
        if not self.get_state or not self.net_initialized:
            return
        start_time = time.time()
        self._prepare_net_inputs()
        if self.episode_starts is None:
            self.episode_starts = np.ones((1,), dtype=bool)
            self.lstm_states    = None
        act, self.lstm_states = self.policy.predict(
            self.net_inputs, state=self.lstm_states, deterministic=True)
        self.act_np = np.array(act, dtype=np.float64)[0]
        end_time = time.time()
        self.time_cost_pub.publish(Float32(end_time - start_time))

        # visualize VAE reconstructions (past and current frame)
        recons = self.policy.predict_img(self.lstm_states[0].reshape((1, -1)))
        if recons[1] is not None and recons[0] is not None:
            imgs = np.hstack([
                (recons[0].reshape([256, 256, 1]) * 255).astype(np.uint8),
                (recons[1].reshape([256, 256, 1]) * 255).astype(np.uint8),
            ])
            cv2.imshow("recon", imgs)
            cv2.waitKey(1)
        elif recons[1] is not None:
            imgs = (recons[1].reshape([256, 256, 1]) * 255).astype(np.uint8)
            cv2.imshow("recon", imgs)
            cv2.waitKey(1)

    def send_command(self, data):
        if self.act_np is None:
            return
        vel_msg = TwistStamped()
        self.robot.step(self.act_np, 1.0 / 30)
        vel_cmd = self.robot.get_vel_cmd()
        vel_msg.header.stamp    = rospy.Time.now()
        vel_msg.header.frame_id = 'world'
        vel_msg.twist.linear.x  = vel_cmd[0]
        vel_msg.twist.linear.y  = vel_cmd[1]
        vel_msg.twist.linear.z  = vel_cmd[2]
        vel_msg.twist.angular.z = vel_cmd[3]
        self.vel_pub.publish(vel_msg)


def main():
    rospy.init_node('avoider_node', anonymous=True)
    avoider = AvoiderNode()
    rospy.spin()

if __name__ == "__main__":
    main()
