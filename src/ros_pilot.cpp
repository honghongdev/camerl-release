#include "ros_pilot.h"

#include <chrono>
#include <functional>

#include "agilib/reference/velocity_reference.hpp"
#include "agilib/types/pose.hpp"
#include "agilib/utils/filesystem.hpp"
#include "agilib/utils/logger.hpp"
#include "agiros_msgs/DebugMsg.h"
#include "agiros_msgs/QuadState.h"
#include "agiros_msgs/Telemetry.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"

namespace agi {

static inline PilotParams loadParams(const ros::NodeHandle& nh) {
  std::string pilot_config;
  const bool got_config = nh.getParam("pilot_config", pilot_config);

  std::string agi_param_dir;
  const bool got_directory = nh.getParam("agi_param_dir", agi_param_dir);

  std::string ros_param_dir;
  nh.getParam("ros_param_dir", ros_param_dir);

  ROS_WARN_STREAM("Pilot Config:        " << pilot_config);
  ROS_WARN_STREAM("Agi Param Directory: " << agi_param_dir);
  ROS_WARN_STREAM("ROS Param Directory: " << ros_param_dir);

  if (!got_config) ROS_FATAL("No parameter directory given!");
  if (!got_directory) ROS_FATAL("No Pilot config file given!");
  if (!got_config || !got_directory) ros::shutdown();
  ROS_INFO("Loading Pilot Params from %s in %s", pilot_config.c_str(),
           ros_param_dir.c_str());

  return PilotParams(fs::path(ros_param_dir) / fs::path(pilot_config),
                     agi_param_dir, fs::path(ros_param_dir) / "quads" / "");
}

RosPilot::RosPilot(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
  : nh_(nh),
    pnh_(pnh),
    params_(loadParams(pnh_)),
    pilot_(params_, RosTime) {
  ROS_INFO_STREAM("Loaded pipeline:\n" << params_.pipeline_cfg_);
  ROS_INFO("Loaded Guard Type:    %s", params_.guard_cfg_.type.c_str());
  if (!params_.guard_cfg_.file.empty())
    ROS_INFO("with parameter file:    %s", params_.guard_cfg_.file.c_str());

  // Pose & Odometry subscribers
  pose_estimate_sub_ =
    pnh_.subscribe("pose_estimate", 1, &RosPilot::poseEstimateCallback, this,
                   ros::TransportHints().tcpNoDelay());
  odometry_estimate_sub_ =
    pnh_.subscribe("odometry_estimate", 1, &RosPilot::odometryEstimateCallback,
                   this, ros::TransportHints().tcpNoDelay());
  imu_sub_ = pnh_.subscribe("imu_in", 1, &RosPilot::imuCallback, this,
                            ros::TransportHints().tcpNoDelay());
  motor_speed_sub_ =
    pnh_.subscribe("motor_speed", 1, &RosPilot::motorSpeedCallback, this,
                   ros::TransportHints().tcpNoDelay());
  // Logic subscribers
  start_sub_ = pnh_.subscribe("start", 1, &RosPilot::startCallback, this);
  force_hover_sub_ =
    pnh_.subscribe("force_hover", 1, &RosPilot::forceHoverCallback, this);
  go_to_pose_sub_ =
    pnh_.subscribe("go_to_pose", 1, &RosPilot::goToPoseCallback, this);
  velocity_sub_ =
    pnh_.subscribe("velocity_command", 1, &RosPilot::velocityCallback, this);
  
  acceleration_sub_ = pnh_.subscribe(
    "accelaration_command", 10, &RosPilot::accelerationCallback, this
    );

  feedthrough_command_sub_ = pnh_.subscribe(
    "feedthrough_command", 1, &RosPilot::feedthroughCommandCallback, this,
    ros::TransportHints().tcpNoDelay());

  task_state_sub_ =
    pnh_.subscribe("task_state", 1, &RosPilot::taskStateCallback, this);

  land_sub_ = pnh_.subscribe("land", 1, &RosPilot::landCallback, this);
  off_sub_ = pnh_.subscribe("off", 1, &RosPilot::offCallback, this);
  enable_sub_ = pnh_.subscribe("enable", 1, &RosPilot::enableCallback, this);

  // Trajectory subscribers
  trajectory_sub_ =
    pnh_.subscribe("trajectory", 1, &RosPilot::trajectoryCallback, this);

  // RL trajectory subscribers
  rl_trajectory_sub_ = pnh_.subscribe(
    "rl_trajectory", 1, &RosPilot::rlTrajectoryCallback, this);

  // Telemetry subscribers
  voltage_sub_ =
    nh_.subscribe("mavros/battery", 1, &RosPilot::voltageCallback, this);

  ctr_activate_sub_ = pnh_.subscribe(
    "ctr_activate_flag", 10, &RosPilot::ctrActivateCallback, this);

  // Publishers
  state_pub_ = pnh_.advertise<agiros_msgs::QuadState>("state", 1);
  state_odometry_pub_ = pnh_.advertise<nav_msgs::Odometry>("odometry", 1);
  telemetry_pub_ = pnh_.advertise<agiros_msgs::Telemetry>("telemetry", 1);
  cmd_pub_ = pnh_.advertise<agiros_msgs::Command>("mpc_command", 1);

  if (params_.pipeline_cfg_.bridge_cfg.type == "RotorS") {
    pilot_.registerExternalBridge(
      std::make_shared<RotorsBridge>(nh_, pnh_, params_.quad_, RosTime));
  } else if (params_.pipeline_cfg_.bridge_cfg.type == "ROS") {
    pilot_.registerExternalBridge(
      std::make_shared<RosBridge>(nh_, pnh_, RosTime));
  }

  if (params_.pipeline_cfg_.bridge_cfg.type == "CTRL") {
    ctrl_feedback_publisher_ =
      std::make_unique<CtrlFeedbackPublisher>(nh_, pnh_);
    pilot_.registerFeedbackCallback(
      std::bind(&CtrlFeedbackPublisher::addFeedback,
                ctrl_feedback_publisher_.get(), std::placeholders::_1));
  }
  pilot_.registerPipelineCallback(std::bind(
    &RosPilot::pipelineCallback, this, std::placeholders::_1,
    std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
    std::placeholders::_5, std::placeholders::_6, std::placeholders::_7));

  run_pipeline_timer_ = nh_.createTimer(
    100, &RosPilot::runPipeline, this);

  reference_publishing_thread_ =
    std::thread(&RosPilot::referencePublisher, this);
}

RosPilot::~RosPilot() { shutdown_ = true; }

void RosPilot::runPipeline(const ros::TimerEvent& event) {
  pilot_.runPipeline(event.current_real.toSec());
}

void RosPilot::poseEstimateCallback(
  const geometry_msgs::PoseStampedConstPtr& msg) {
  Pose pose;
  pose.t = msg->header.stamp.toSec();
  pose.position = fromRosVec3(msg->pose.position);
  pose.attitude = fromRosQuaternion(msg->pose.orientation);

  pilot_.odometryCallback(pose);
}

void RosPilot::odometryEstimateCallback(const nav_msgs::OdometryConstPtr& msg) {
  QuadState state;

  state.setZero();
  state.t = msg->header.stamp.toSec();
  state.p = fromRosVec3(msg->pose.pose.position);
  state.q(Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z));
  state.v = fromRosVec3(msg->twist.twist.linear);
  state.w = fromRosVec3(msg->twist.twist.angular);
  state.mot = motor_speeds_;
  pilot_.odometryCallback(state);
}

void RosPilot::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
  pilot_.imuCallback(ImuSample(msg->header.stamp.toSec(),
                               fromRosVec3(msg->linear_acceleration),
                               fromRosVec3(msg->angular_velocity)));
}

void RosPilot::motorSpeedCallback(const mav_msgs::Actuators& msg) {
  motor_speeds_ =
    (Vector<4>() << msg.angular_velocities[0], msg.angular_velocities[1],
     msg.angular_velocities[2], msg.angular_velocities[3])
      .finished();
  pilot_.motorSpeedCallback(motor_speeds_);
}

void RosPilot::startCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("START command received!");
  pilot_.start();
}

void RosPilot::forceHoverCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("FORCE_HOVER command received!");
  pilot_.forceHover();
}

void RosPilot::goToPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  ROS_INFO("GO_TO_POSE command received!");
  QuadState end_state;
  end_state.setZero();
  end_state.p =
    Vector<3>(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  end_state.q(Quaternion(msg->pose.orientation.w, msg->pose.orientation.x,
                         msg->pose.orientation.y, msg->pose.orientation.z));
  pilot_.goToPose(end_state);
}

void RosPilot::velocityCallback(
  const geometry_msgs::TwistStampedConstPtr& msg) {
  pilot_.setVelocityReference(fromRosVec3(msg->twist.linear),
                              msg->twist.angular.z);
}

void RosPilot::accelerationCallback(
  const agiros_msgs::AccelerationConstPtr& msg) {
  std::vector<Vector<3>> accs;
  std::vector<Scalar> yaw_rates;
  for(int i=0; i<int(msg->accs.size()); i++)
  {
    Vector<3> acc;
    acc << msg->accs[i].twist.linear.x, msg->accs[i].twist.linear.y,
      msg->accs[i].twist.linear.z;
    accs.push_back(acc);
    yaw_rates.push_back(msg->yaw_rate[i]);
  }
  std::cout.precision(6);
  // std::cout << "acc: " << acc.transpose() <<" "<< msg->header.stamp.toSec()<< std::endl;
  pilot_.setAccelerationCommand(accs, yaw_rates, msg->t, msg->header.stamp.toSec());
}

void RosPilot::rlTrajectoryCallback(const agiros_msgs::RLtrajectoryConstPtr& msg)
{
  time_received_prediction_ = msg->header.stamp;
  // get the frame of the trajectory
  std::string frame = msg->header.frame_id;
  // final point, run go to pose command
  if (msg->final_point)
  {
    QuadState end_state;
    agiros_msgs::QuadState state = msg->points.back().state;
    end_state.setZero();
    end_state.p = Eigen::Vector3d(state.pose.position.x,
                                  state.pose.position.y,
                                  state.pose.position.z);
    end_state.q(Quaternion(state.pose.orientation.w,
                            state.pose.orientation.x,
                            state.pose.orientation.y,
                            state.pose.orientation.z));
    pilot_.goToPose(end_state);
  } else {
    SetpointVector sampled_trajectory;
    sampled_trajectory.reserve(msg->points.size());
    for (auto point : msg->points)
    {
      sampled_trajectory.emplace_back(Setpoint());
      sampled_trajectory.back().state.setZero();
      sampled_trajectory.back().state.t = time_received_prediction_.toSec() + point.state.t;
      sampled_trajectory.back().state.p = Eigen::Vector3d(point.state.pose.position.x,
                                                          point.state.pose.position.y,
                                                          point.state.pose.position.z);
      sampled_trajectory.back().state.q(Quaternion(point.state.pose.orientation.w,
                                                  point.state.pose.orientation.x,
                                                  point.state.pose.orientation.y,
                                                  point.state.pose.orientation.z));
      sampled_trajectory.back().state.v = Eigen::Vector3d(point.state.velocity.linear.x,
                                                          point.state.velocity.linear.y,
                                                          point.state.velocity.linear.z);
      sampled_trajectory.back().state.w = Eigen::Vector3d(point.state.velocity.angular.x,
                                                          point.state.velocity.angular.y,
                                                          point.state.velocity.angular.z);
      sampled_trajectory.back().state.a = Eigen::Vector3d(point.state.acceleration.linear.x,
                                                          point.state.acceleration.linear.y,
                                                          point.state.acceleration.linear.z);
    }
    QuadState odom_at_inference;
    odom_at_inference.p = Eigen::Vector3d(msg->ref_pose.position.x,
                                          msg->ref_pose.position.y,
                                          msg->ref_pose.position.z);
    odom_at_inference.q(Quaternion(msg->ref_pose.orientation.w,
                                  msg->ref_pose.orientation.x,
                                  msg->ref_pose.orientation.y,
                                  msg->ref_pose.orientation.z));
    odom_at_inference.v = Eigen::Vector3d(msg->ref_vel.linear.x,
                                          msg->ref_vel.linear.y,
                                          msg->ref_vel.linear.z);
    pilot_.setRLtrajectoryCommand(sampled_trajectory, odom_at_inference, time_received_prediction_.toSec(), frame);    
  }



}

void RosPilot::landCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("LAND command received!");
  pilot_.land();
}

void RosPilot::offCallback(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("OFF command received!");
  pilot_.off();
}

void RosPilot::enableCallback(const std_msgs::BoolConstPtr& msg) {
  ROS_INFO("Computing active: %s!", msg->data ? "true" : "false");
  pilot_.enable(msg->data);
}

void RosPilot::taskStateCallback(const avoid_msgs::TaskStateConstPtr& msg) {
  if (msg->Mission_state > 1 && !initialized_)
  {
    ROS_INFO("Initializing!");
    // pilot_.enable(true);
    // pilot_.forceHover();
    initialized_ = true;
  }
}

void RosPilot::trajectoryCallback(const agiros_msgs::ReferenceConstPtr& msg) {
  ROS_INFO("Received sampled trajectory!");
  SetpointVector sampled_trajectory;
  sampled_trajectory.reserve(msg->points.size());
  const Scalar t_shift = ros::Time::now().toSec() - msg->points[0].state.t;
  for (auto point : msg->points) {
    sampled_trajectory.emplace_back(Setpoint());
    sampled_trajectory.back().state.setZero();
    sampled_trajectory.back().state.t = point.state.t + t_shift;
    sampled_trajectory.back().state.p = fromRosVec3(point.state.pose.position);
    sampled_trajectory.back().state.q(
      fromRosQuaternion(point.state.pose.orientation));
    sampled_trajectory.back().state.v =
      fromRosVec3(point.state.velocity.linear);
    sampled_trajectory.back().state.w =
      fromRosVec3(point.state.velocity.angular);
    sampled_trajectory.back().state.a =
      fromRosVec3(point.state.acceleration.linear);
    sampled_trajectory.back().state.j = fromRosVec3(point.state.jerk);
    sampled_trajectory.back().state.s = fromRosVec3(point.state.snap);

    sampled_trajectory.back().input.t = point.state.t + t_shift;

    if (point.command.is_single_rotor_thrust) {
      if (!fromRosThrusts(point.command.thrusts,
                          &sampled_trajectory.back().input.thrusts)) {
        break;
      }
    } else {
      sampled_trajectory.back().input.collective_thrust =
        point.command.collective_thrust;
      sampled_trajectory.back().input.omega =
        fromRosVec3(point.command.bodyrates);
    }
  }
  pilot_.addSampledTrajectory(sampled_trajectory);
}

void RosPilot::voltageCallback(const sensor_msgs::BatteryStateConstPtr& msg) {
  pilot_.voltageCallback(msg->voltage);
}

void RosPilot::ctrActivateCallback(const std_msgs::BoolConstPtr& msg) {
  std::cout << "ctrActivateCallback" << std::endl;
  pilot_.ctrActivateCallback(msg->data);
}

void RosPilot::feedthroughCommandCallback(
  const agiros_msgs::CommandConstPtr& msg) {
  pilot_.setFeedthroughCommand(fromRosCommand(*msg, pilot_.getTime()));
}

void RosPilot::pipelineCallback(const QuadState& state,
                                const Feedback& feedback,
                                const ReferenceVector& references,
                                const SetpointVector& setpoints,
                                const SetpointVector& outer_setpoints,
                                const SetpointVector& inner_setpoints,
                                const Command& command) {
  agiros_msgs::QuadState msg = toRosQuadState(state);

  nav_msgs::Odometry msg_odo;
  msg_odo.header = msg.header;
  msg_odo.pose.pose = msg.pose;
  msg_odo.twist.twist = msg.velocity;

  cmd_pub_.publish(toRosCommand(command));
  state_odometry_pub_.publish(msg_odo);
  state_pub_.publish(msg);

  agiros_msgs::Telemetry telemetry_msg;
  telemetry_msg.t = state.t;
  telemetry_msg.header = msg.header;
  telemetry_msg.bridge_type.data = pilot_.getActiveBridgeType();
  telemetry_msg.bridge_armed.data = feedback.armed;
  telemetry_msg.guard_triggered.data = pilot_.guardTriggered();

  // get the current reference from the pilot
  if (!setpoints.empty()) {
    telemetry_msg.reference.pose.position =
      toRosPoint(setpoints.front().state.p);
    telemetry_msg.reference.pose.orientation =
      toRosQuaternion(setpoints.front().state.q());
    telemetry_msg.reference.velocity.linear =
      toRosVector(setpoints.front().state.v);
    telemetry_msg.reference.acceleration.linear =
      toRosVector(setpoints.front().state.a);
    telemetry_msg.reference.velocity.angular =
      toRosVector(setpoints.front().state.w);
    telemetry_msg.reference.heading = setpoints.front().state.getYaw();
  }

  if (references != references_) {
    references_ = references;
    reference_publishing_cv_.notify_all();
  }

  const int number_of_references = (int)references.size();
  telemetry_msg.num_references_in_queue = number_of_references;
  if (number_of_references > 0) {
    telemetry_msg.reference_left_duration =
      references.back()->getEndTime() - ros::Time::now().toSec();
  } else {
    telemetry_msg.reference_left_duration = 0.0;
  }
  telemetry_msg.rmse = updateRmse(state, setpoints, references);
  telemetry_msg.voltage = pilot_.getVoltage();

  telemetry_pub_.publish(telemetry_msg);
  if (params_.publish_log_var_) {
    publishLoggerDebugMsg();
  }
}

void RosPilot::referencePublisher() {
  while (!shutdown_ && ros::ok()) {
    std::unique_lock<std::mutex> lk(reference_publishing_mtx_);
    reference_publishing_cv_.wait_for(lk, std::chrono::seconds(1));
    if (shutdown_ || !ros::ok()) break;
    const ReferenceVector references = references_;
    // reference_visualizer_.visualize(references);
  }
}

Scalar RosPilot::updateRmse(const QuadState& state,
                            const SetpointVector& setpoints,
                            const ReferenceVector& references) const {
  static Scalar cumulative_weighted_square_sum = 0.0;
  static Scalar cumulative_time = 0.0;
  static Scalar t_last_sample = NAN;
  static std::shared_ptr<ReferenceBase> last_active_reference;

  // No references given, therefore reset.
  if (references.empty()) {
    last_active_reference.reset();
    t_last_sample = NAN;
    cumulative_weighted_square_sum = 0.0;
    cumulative_time = 0.0;
    return 0.0;
  }

  // Active reference changed, therefore reset.
  if (references.front() != last_active_reference) {
    last_active_reference = references.front();
    t_last_sample = NAN;
    cumulative_weighted_square_sum = 0.0;
    cumulative_time = 0.0;
    return 0.0;
  }

  // Last sample time unknown, therefore reset and set last sample time.
  if (!std::isfinite(t_last_sample)) {
    t_last_sample = state.t;
    cumulative_weighted_square_sum = 0.0;
    cumulative_time = 0.0;
    return 0.0;
  }

  // If no setpoints given, skip.
  if (!setpoints.empty()) {
    const Scalar dt = state.t - t_last_sample;
    t_last_sample = state.t;

    // If time not monotonically increasing, skip.
    if (dt > 0.0) {
      const Vector<3> error = state.p - setpoints.front().state.p;
      cumulative_weighted_square_sum += dt * error.transpose() * error;
      cumulative_time += dt;
    }
  }

  // If samples gathered, return RMSE, otherwise return 0.
  return cumulative_time > 0.0
           ? sqrt(cumulative_weighted_square_sum / cumulative_time)
           : 0.0;
}

Command RosPilot::getCommand() const { return pilot_.getCommand(); }

void RosPilot::advertiseDebugVariable(const std::string& var_name) {
  logger_publishers_[var_name] =
    pnh_.advertise<agiros_msgs::DebugMsg>(var_name, 1);
}

void RosPilot::publishDebugVariable(const std::string& name,
                                    const PublishLogContainer& container) {
  if (logger_publishers_.find(name) == logger_publishers_.end()) {
    // Not found, advertise it and create it
    advertiseDebugVariable(name);
    if (container.advertise) return;
  }

  // Fill message
  agiros_msgs::DebugMsg debug_msg;
  for (unsigned int i = 0; i < container.data.size(); ++i)
    debug_msg.data.push_back(container.data(i));
  // RosTime
  debug_msg.header.stamp = ros::Time::now();
  // Publish
  logger_publishers_[name].publish(debug_msg);
}

void RosPilot::publishLoggerDebugMsg() {
  Logger::for_each_instance(std::bind(&RosPilot::publishDebugVariable, this,
                                      std::placeholders::_1,
                                      std::placeholders::_2));
}

bool RosPilot::getQuadrotor(Quadrotor* const quad) const {
  return pilot_.getQuadrotor(quad);
}

Pilot& RosPilot::getPilot() { return pilot_; }


}  // namespace agi
