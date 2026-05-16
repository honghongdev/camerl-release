#include "bridge/rotors_bridge.hpp"
#include "agilib/math/math.hpp"
#include "mav_msgs/Actuators.h"

namespace agi {

RotorsBridge::RotorsBridge(const ros::NodeHandle& nh,
                           const ros::NodeHandle& pnh, const Quadrotor& quad,
                           const TimeFunction time_function)
  : BridgeBase("RotorSBridge", time_function), nh_(nh), pnh_(pnh), quad_(quad) {
  // rotor_omega_pub_ =
  //   nh_.advertise<mav_msgs::Actuators>("command/motor_speed", 1);
  rates_pub_ = nh_.advertise<rpg_quadrotor_msgs::ControlCommand>(
    "command/rates_command", 1);
}

bool RotorsBridge::sendCommand(const Command& command, const bool active) {
  if (!command.isRatesThrust()) {
    ROS_ERROR("RotorS bridge only allows Rates commands!");
    return false;
  }

  // const Vector<4> motor_omegas =
  //   quad_.clampMotorOmega(quad_.motorThrustToOmega(command.thrusts));
  rpg_quadrotor_msgs::ControlCommand msg;
  msg.header.stamp = ros::Time::now();  //(command.t);
  msg.control_mode = rpg_quadrotor_msgs::ControlCommand::BODY_RATES;
  msg.armed = active;
  msg.expected_execution_time = ros::Time(command.t + 0.01);
  msg.bodyrates.x = command.omega[0];
  msg.bodyrates.y = command.omega[1];
  msg.bodyrates.z = command.omega[2];
  msg.collective_thrust = command.collective_thrust;
  rates_pub_.publish(msg);
  // rotor_omega_pub_.publish(msg);
  return true;
}


}  // namespace agi
