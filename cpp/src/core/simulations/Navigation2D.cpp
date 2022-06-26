#include "core/simulations/Navigation2D.h"

#include "core/Util.h"
#include <boost/functional/hash.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>

namespace simulations {

Navigation2D::Action Navigation2D::Action::Rand() {
  Action action{std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
  return action;
}

uint64_t Navigation2D::Observation::Discretize() const {
  float grid_size = 1.0f;
  list_t<int> data {
    static_cast<int>(floorf(ego_agent_position.x / grid_size)),
    static_cast<int>(floorf(ego_agent_position.y / grid_size))
  };
  return boost::hash_value(data);
}

list_t<list_t<Navigation2D::Action>> Navigation2D::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 8; i++) {
    macro_actions.emplace_back();
    for (size_t j = 0; j < length; j++) {
      macro_actions.back().push_back({static_cast<float>(i) * 2 * PI / 8});
    }
  }
  return macro_actions;
}

list_t<list_t<Navigation2D::Action>> Navigation2D::Action::Deserialize(const list_t<float>& params, size_t macro_length) {
  //TODO: check that StandardMacroActionDeserialization converts 2D Bezier Curve to orientations of vector of the same magnitude
  list_t<list_t<Navigation2D::Action>> macro_actions = StandardMacroActionDeserialization<Navigation2D::Action>(params, macro_length);
  return macro_actions;
}

/* ====== Construction functions ====== */

Navigation2D::Navigation2D() : step(0), _is_terminal(false), _is_failure(false) {

}

Navigation2D Navigation2D::CreateRandom() {
/*
Sets the ego mean starting position and goal position for this simulation
For now we fix the starting position just to test everything first
TODO: create random positions bounded within the given regions
*/
  EGO_START_MEAN = {5.0f, 300.0f};
  GOAL = {650f, 300.0f};

  return SampleBeliefPrior();
}

/* ====== Belief related functions ======*/

Navigation2D Navigation2D::SampleBeliefPrior() {
  Navigation2D sim;
  sim.ego_agent_position.x = std::normal_distribution<float>(
      EGO_START_MEAN.x, EGO_START_STD)(Rng());
  sim.ego_agent_position.y =std::normal_distribution<float>(
      EGO_START_MEAN.y, EGO_START_STD)(Rng());
  return sim;
}

float Navigation2D::Error(const Navigation2D& other) const {
  return (ego_agent_position - other.ego_agent_position).norm();
}

/* ====== Bounds related functions ====== */
// Actually a high probability estimate, assuming actuation noise samples below 3 s.d.
float Navigation2D::BestReward() const {
/*
Reuse the same best reward from LightDark since two simulations share similar properties.
Best reward always comes from the least amount of steps with no collisions.
*/
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (GOAL - ego_agent_position).norm() - EGO_RADIUS);
  float max_distance_per_step = EGO_SPEED * DELTA;
  size_t steps = static_cast<size_t>(round(ceilf(distance / max_distance_per_step)));
  if (steps <= 1) {
    return GOAL_REWARD;
  } else {
    return (1 - powf(GAMMA, static_cast<float>(steps) - 1)) / (1 - static_cast<float>(steps)) * STEP_REWARD +
      powf(GAMMA, static_cast<float>(steps) - 1) * GOAL_REWARD;
  }
}




}