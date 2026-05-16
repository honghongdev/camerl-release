#pragma once


#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>

#include <condition_variable>
#include <mutex>

#include "agilib/math/math.hpp"
#include "agilib/pilot/pilot.hpp"
#include "agilib/sampler/time_based/time_sampler.hpp"
#include "bridge/ros_bridge.hpp"
#include "bridge/rotors_bridge.hpp"
#include "ctrl_feedback_publisher.hpp"
#include "time.hpp"
#include "ros_eigen.hpp"
#include "agiros_msgs/Command.h"
#include "agiros_msgs/Reference.h"
#include "agiros_msgs/Acceleration.h"
#include "agiros_msgs/RLtrajectory.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "mav_msgs/Actuators.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/BatteryState.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Float32.h"
#include "std_srvs/SetBool.h"
#include "std_srvs/Trigger.h"
#include "avoid_msgs/TaskState.h"

namespace agi {

class RosPilot {
 public:
  RosPilot(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  RosPilot() : RosPilot(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~RosPilot();
  Command getCommand() const;
  bool getQuadrotor(Quadrotor* const quad) const;
  Pilot& getPilot();

 private:
  void runPipeline(const ros::TimerEvent& event);
  void poseEstimateCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryEstimateCallback(const nav_msgs::OdometryConstPtr& msg);
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);
  void motorSpeedCallback(const mav_msgs::Actuators& msg);
  void startCallback(const std_msgs::EmptyConstPtr& msg);
  void forceHoverCallback(const std_msgs::EmptyConstPtr& msg);
  void goToPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void velocityCallback(const geometry_msgs::TwistStampedConstPtr& msg);
  void accelerationCallback(const agiros_msgs::AccelerationConstPtr& msg);
  void rlTrajectoryCallback(const agiros_msgs::RLtrajectoryConstPtr& msg);
  void landCallback(const std_msgs::EmptyConstPtr& msg);
  void offCallback(const std_msgs::EmptyConstPtr& msg);
  void enableCallback(const std_msgs::BoolConstPtr& msg);
  void trajectoryCallback(const agiros_msgs::ReferenceConstPtr& msg);
  void voltageCallback(const sensor_msgs::BatteryStateConstPtr& msg);
  void feedthroughCommandCallback(const agiros_msgs::CommandConstPtr& msg);
  void taskStateCallback(const avoid_msgs::TaskStateConstPtr& msg);
  void ctrActivateCallback(const std_msgs::BoolConstPtr& msg);

  void advertiseDebugVariable(const std::string& var_name);
  void publishDebugVariable(const std::string& name,
                            const PublishLogContainer& container);
  void publishLoggerDebugMsg();

  void pipelineCallback(const QuadState& state, const Feedback& feedback,
                        const ReferenceVector& references,
                        const SetpointVector& reference_setpoints,
                        const SetpointVector& outer_setpoints,
                        const SetpointVector& inner_setpoints,
                        const Command& command);

  void referencePublisher();
  Scalar updateRmse(const QuadState& state, const SetpointVector& setpoints,
                    const ReferenceVector& references) const;
  // ROS members
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber pose_estimate_sub_;
  ros::Subscriber odometry_estimate_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber motor_speed_sub_;
  ros::Subscriber start_sub_;
  ros::Subscriber force_hover_sub_;
  ros::Subscriber go_to_pose_sub_;
  ros::Subscriber velocity_sub_;
  ros::Subscriber acceleration_sub_;
  ros::Subscriber land_sub_;
  ros::Subscriber off_sub_;
  ros::Subscriber trajectory_sub_;
  ros::Subscriber rl_trajectory_sub_;
  ros::Subscriber enable_sub_;
  ros::Subscriber voltage_sub_;
  ros::Subscriber feedthrough_command_sub_;
  ros::Subscriber task_state_sub_;
  ros::Subscriber ctr_activate_sub_;

  ros::Publisher state_pub_;
  ros::Publisher state_odometry_pub_;
  ros::Publisher telemetry_pub_;
  ros::Publisher cmd_pub_;

  ros::Timer run_pipeline_timer_;

  // Agilib
  PilotParams params_;
  Pilot pilot_;
  Logger logger_{"RosPilot"};

  std::unordered_map<std::string, ros::Publisher> logger_publishers_;

  // Trajectory visualization
  // RosTrajVisualizer reference_visualizer_;
  // RosTrajVisualizer active_reference_visualizer_;
  // RosTrajVisualizer outer_setpoints_visualizer_;
  // RosTrajVisualizer inner_setpoints_visualizer_;

  // CTRL feedback
  std::unique_ptr<CtrlFeedbackPublisher> ctrl_feedback_publisher_;

  // Reference publishing
  bool shutdown_{false};
  bool initialized_{false};
  ros::Time time_received_prediction_;
  std::thread reference_publishing_thread_;
  std::mutex reference_publishing_mtx_;
  std::condition_variable reference_publishing_cv_;
  std::vector<std::shared_ptr<ReferenceBase>> references_;
  Vector<4> motor_speeds_{NAN, NAN, NAN, NAN};
};

}  // namespace agi