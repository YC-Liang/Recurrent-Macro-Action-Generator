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

/* ====== Stepping Related Functions ====== */
bool Navigation2D::CircleBoxCollision(vector_t pos, float wall_width, float wall_height){
  float wall_centre_x = wall_width / 2.0f;
  float wall_centre_y = wall_width / 2.0f;

  float centre_distance_x = abs(pos.x - wall_centre_x);
  float centre_distance_y = abs(pos.y - wall_centre_y);

  if(centre_distance_x > wall_centre_x + EGO_RADIUS){return false};
  if(centre_distance_y > wall_centre_y + EGO_RADIUS){return false};

  if(centre_distance_x <= wall_centre_x){return true};
  if(centre_distance_y <= wall_centre_y){return true};

  float corner_distance = powf(centre_distance_x - wall_centre_x, 2) + powf(centre_distance_y - wall_centre_y, 2);
  return corner_distance <= powf(EGO_RADIUS, 2);
}

bool Navigation2D::CheckCollision(vector_t pos){
  //check collision with walls
  for(const list_t<float> &wall : CLOSED_WALLS){
    float wall_width = abs(wall[0] - wall[2]);
    float wall_height = abs(wall[1] - wall[3]);
    if(CircleBoxCollision(pos, wall_width, wall_height)){
      return true;
    }
  }
  //check collision with danger zones
  //TODO: add danger zones 
  return false;
}

bool IsInLight(vector_t pos){
  for(const vector_t &light : LIGHT_POSITION){
    if((pos - light).norm() <= LIGHT_RADIUS){
      return true;
    }
  }
  return false;
}

template <bool compute_log_prob>
std::tuple<Navigation2D, float, Navigation2D::Observation, float> Navigation2D::Step(
    const Navigation2D::Action& action, const Navigation2D::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  Navigation2D next_sim = *this;
  float reward;

  /* ====== Step 1: Update state.  ======*/
  next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation); 
  next_sim.step++;
  //update rewards based on the current step
  if(CheckCollision(ego_agent_position)){
    reward = COLLISION_REWARD;
    next_sim._is_failure = true;
    next_sim._is_terminal = true;
  }
  else if((next_sim.ego_agent_position - GOAL).norm() <= EGO_RADIUS){
    reward = GOAL_REWARD;
    next_sim._is_failure = false;
    next_sim._is_terminal = true;
  }
  else if(next_sim.step == MAX_STEPS){
    reward = COLLISION_REWARD;
    next_sim._is_failure = true;
    next_sim._is_terminal = true;
  }
  else{
    reward = STEP_REWARD;
    next_sim._is_failure = false;
    next_sim._is_terminal = false;
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }

  float log_prob = 0;
  bool in_light = IsInLight(next_sim.ego_agent_position);

  if(!observation){
    if(in_light){
      //in lights, the agent accurately localise itself
      new_observation.ego_agent_position = next_sim.ego_agent_position;
    }
    else{
      //outside of lights, the agent recieves noisy localisation readings
      new_observation.ego_agent_position = next_sim.ego_agent_position;
      new_observation.ego_agent_position.x += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
      new_observation.ego_agent_position.y += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
    }
  }
  if constexpr (compute_log_prob){
    //TODO: Figure out what this condition is doing... 
    //compute_log_prob is set to false in DespotPlanner.h, so leave this code here for now should be fine.
    if (in_light) {
      if (std::isnan(new_observation.ego_agent_position.x) && std::isnan(new_observation.ego_agent_position.y)) {
        log_prob += -std::numeric_limits<float>::infinity();
      } else {
        log_prob += NormalLogProb(next_sim.ego_agent_position.x, OBSERVATION_NOISE, new_observation.ego_agent_position.x);
        log_prob += NormalLogProb(next_sim.ego_agent_position.y, OBSERVATION_NOISE, new_observation.ego_agent_position.y);
      }
    } else {
      if (std::isnan(new_observation.ego_agent_position.x) && std::isnan(new_observation.ego_agent_position.y)) {
        log_prob += 0;
      } else {
        log_prob += -std::numeric_limits<float>::infinity();
      }
    }
  }
  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<Navigation2D, float, Navigation2D::Observation, float> Navigation2D::Step<true>(
    const Navigation2D::Action& action, const Navigation2D::Observation* observation) const;
template std::tuple<Navigation2D, float, Navigation2D::Observation, float> Navigation2D::Step<false>(
    const Navigation2D::Action& action, const Navigation2D::Observation* observation) const;

void LightDark::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
}

void LightDark::EncodeContext(list_t<float>& data) {
  GOAL.Encode(data);
  data.emplace_back(LIGHT_POS);
}

//TODO: Add in render functions for visualisation

}