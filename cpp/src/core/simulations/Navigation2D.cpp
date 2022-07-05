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
  //Action action{0};
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
  //initialise starting position, choosing between two
  int rand = std::uniform_int_distribution<>(1, 10)(Rng());
  if(rand <= 5){
    EGO_START_MEAN = RANDOM_START_REGION[0];
  }
  else{
    EGO_START_MEAN = RANDOM_START_REGION[1];
  }

  //initialise goal position, pick a random position in either of the two regions
  rand = std::uniform_int_distribution<>(1, 10)(Rng());
  array_t<int, 4> goal_zone;
  if(rand <= 5){
    goal_zone = GOAL_REGION[0];
  }
  else{
    goal_zone = GOAL_REGION[1];
  }

  int goal_zone_width = abs(goal_zone[0] - goal_zone[2]);
  int goal_zone_height = abs(goal_zone[1] - goal_zone[3]);

  int random_x = std::uniform_int_distribution<>(0, goal_zone_width)(Rng());
  int random_y = std::uniform_int_distribution<>(0, goal_zone_height)(Rng());

  int goal_x = goal_zone[0] + random_x;
  int goal_y = goal_zone[1] - random_y;

  //std::cout << "Picked goal position:" << goal_x << "," << goal_y << std::endl;

  GOAL = {static_cast<float>(goal_x), static_cast<float>(goal_y)};

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
bool Navigation2D::CircleBoxCollision(vector_t pos, float wall_centre_x, float wall_centre_y, float wall_width, float wall_height) const {

  float centre_distance_x = abs(pos.x - wall_centre_x);
  float centre_distance_y = abs(pos.y - wall_centre_y);

  if(centre_distance_x > wall_width/2.0f + EGO_RADIUS){return false;}
  if(centre_distance_y > wall_height/2.0f + EGO_RADIUS){return false;}

  if(centre_distance_x <= wall_width/2.0f){return true;}
  if(centre_distance_y <= wall_height/2.0f){return true;}

  float corner_distance = powf(centre_distance_x - wall_width/2.0f, 2) + powf(centre_distance_y - wall_height/2.0f, 2);
  return corner_distance <= powf(EGO_RADIUS, 2);
}

bool Navigation2D::CheckCollision(vector_t pos) const {
  //check collision with walls
  for(const auto &wall : CLOSED_WALLS){
    float wall_width = abs(wall[0] - wall[2]);
    float wall_height = abs(wall[1] - wall[3]);
    float wall_centre_x = wall[0]+wall_width/2.0f; //centre is left corner + width/height
    float wall_centre_y = wall[1]-wall_height/2.0f;
    if(CircleBoxCollision(pos, wall_centre_x, wall_centre_y, wall_width, wall_height)){
      return true;
    }
  }
  return false;
}

bool Navigation2D::IsInDangerZone(vector_t pos) const {
  for(const auto &zone : DANGER_ZONES){
    float zone_width = abs(zone[0] - zone[2]);
    float zone_height = abs(zone[1] - zone[3]);
    float zone_centre_x = zone[0]+zone_width/2.0f; //centre is left corner + width/height
    float zone_centre_y = zone[1]+zone_height/2.0f;
    if(CircleBoxCollision(pos, zone_centre_x, zone_centre_y, zone_width, zone_height)){return true;}
  }
  return false;
}

bool Navigation2D::IsInLight(vector_t pos) const {
  for(const auto &light : LIGHT_POSITION){
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
  //agent motion contains noises and agent will stop at the current location if bumping into a wall
  vector_t current_position = ego_agent_position;
  next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation); 
  next_sim.ego_agent_position.x += std::normal_distribution<float>(0.0, MOTION_NOISE)(RngDet());
  next_sim.ego_agent_position.y += std::normal_distribution<float>(0.0, MOTION_NOISE)(RngDet());
  if(CheckCollision(next_sim.ego_agent_position)){
    next_sim.ego_agent_position = current_position;
  }
  next_sim.step++;
  //update rewards and other information based on the current step
  if((next_sim.ego_agent_position - GOAL).norm() <= EGO_RADIUS){
    //std::cout << "Reached Goal!" << std::endl;
    reward = GOAL_REWARD;
    next_sim._is_failure = false;
    next_sim._is_terminal = true;
  }
  else if(next_sim.step == MAX_STEPS){
    //std::cout << "Reached maximum allowed steps." << std::endl;
    reward = COLLISION_REWARD;
    next_sim._is_failure = true;
    next_sim._is_terminal = true;
  }
  else if(IsInDangerZone(next_sim.ego_agent_position)){
    //std::cout << "Touched Danger Zone!" << std::endl;
    reward = COLLISION_REWARD;
    next_sim._is_terminal = false;
    next_sim._is_failure = false;
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
      //in lights, the agent localise itself with some sensor error
      new_observation.ego_agent_position = next_sim.ego_agent_position;
      new_observation.ego_agent_position.x += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
      new_observation.ego_agent_position.y += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
    }
    else{
      //outside of lights, the agent receives no observation
      new_observation.ego_agent_position.x = std::numeric_limits<float>::quiet_NaN();
      new_observation.ego_agent_position.y = std::numeric_limits<float>::quiet_NaN();
    }
  }
  if constexpr (compute_log_prob){
    //log_prob is for weights of each particle used in Sampling Importance Resampling
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

void Navigation2D::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
}

void Navigation2D::EncodeContext(list_t<float>& data) {
  //encode goal position
  GOAL.Encode(data);
  //encode light positions
  for(const auto &light : LIGHT_POSITION){
    light.Encode(data);
  }
  //encode closed wall bounding corners
  for(const auto &wall : CLOSED_WALLS){
    for(const auto &point : wall){
      data.emplace_back(point);
    }
  }
  //encode danger zone bounding corners
  for(const auto &zone : DANGER_ZONES){
    for(const auto &point : zone){
      data.emplace_back(point);
    }
  }
}

//TODO: Add in render functions for visualisation
cv::Mat Navigation2D::Render(const list_t<Navigation2D>& belief_sims,
    const list_t<list_t<Action>>& macro_actions, const vector_t& macro_action_start) const {

  constexpr float SCENARIO_MIN = -35.0f; //half width of the environment
  constexpr float SCENARIO_MAX = 30.0f; //half height of the nevironment
  constexpr float RESOLUTION = 0.1f;
  auto to_frame = [&](const vector_t& vector) {
    return cv::Point{
      static_cast<int>((vector.x - SCENARIO_MIN) / RESOLUTION),
      static_cast<int>((SCENARIO_MAX - vector.y) / RESOLUTION)
    };
  };

  auto to_frame_dist = [&](float d) {
    return static_cast<int>(d/RESOLUTION);
  };

  auto x_to_frame = [&](float x) {
    return static_cast<int>((x - SCENARIO_MIN) / RESOLUTION);
  };

  auto y_to_frame = [&](float y){
    return static_cast<int>((SCENARIO_MAX - y) / RESOLUTION);
  };

  //draw the frame
  cv::Mat frame(
      static_cast<int>(abs(SCENARIO_MAX * 2)/RESOLUTION),
      static_cast<int>(abs(SCENARIO_MIN * 2)/RESOLUTION),
      CV_8UC3,
      cv::Scalar(255, 255, 255));

  //draw walls
  for (const auto &wall : CLOSED_WALLS){
    cv::Point p1(x_to_frame(wall[0]), y_to_frame(wall[1]));
    cv::Point p2(x_to_frame(wall[2]), y_to_frame(wall[3]));
    cv::rectangle(frame, p1, p2,
              cv::Scalar(255, 0, 0),
              -1, cv::LINE_8);
  }

  //draw danger zones
  for (const auto &zone : DANGER_ZONES){
    cv::Point p1(x_to_frame(zone[0]), y_to_frame(zone[1]));
    cv::Point p2(x_to_frame(zone[2]), y_to_frame(zone[3]));
    cv::rectangle(frame, p1, p2,
              cv::Scalar(69, 69, 186),
              -1, cv::LINE_8);
  }

  //draw lights
  for(const auto &light : LIGHT_POSITION){
    cv::Point p (x_to_frame(light.x), y_to_frame(light.y));
    cv::circle(frame, p,to_frame_dist(LIGHT_RADIUS), cv::Scalar(104, 43, 159), -1, cv::LINE_AA);
  }

  cv::drawMarker(frame, to_frame(EGO_START_MEAN),
      cv::Scalar(255, 255, 0), cv::MARKER_TILTED_CROSS, 15, 2, cv::LINE_AA);
  cv::drawMarker(frame, to_frame(GOAL),
      cv::Scalar(0, 255, 0), cv::MARKER_TILTED_CROSS, 15, 2, cv::LINE_AA);

  for (const simulations::Navigation2D& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.ego_agent_position), cv::Scalar(0, 255, 255),
        cv::MARKER_CROSS, 8, 2, cv::LINE_4);
  }

  // Draw ego agent.
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  const static list_t<cv::Scalar> colors = {
    {75, 25, 230},
    {49, 130, 245},
    {25, 225, 255},
    {240, 240, 70},
    {75, 180, 60},
    {180, 30, 145},
    {230, 50, 240},
    {216, 99, 67}
  };

  for (size_t i = 0; i < macro_actions.size(); i++) {
    vector_t s = macro_action_start;
    for (const Action& a : macro_actions[i]) {
      vector_t e = s + vector_t(DELTA * EGO_SPEED, 0).rotated(a.orientation);
      
      cv::line(frame, to_frame(s), to_frame(e),
          colors[i], 5, cv::LINE_AA);
      
      s = e;
    }
  }

  if (_is_terminal) {
    if ((ego_agent_position - GOAL).norm() <= EGO_RADIUS) {
      cv::putText(frame,
          "Stop (Success)",
          to_frame(ego_agent_position + vector_t(1.0, - EGO_RADIUS / 2)),
          cv::FONT_HERSHEY_DUPLEX,
          1.0,
          cv::Scalar(0, 255, 0),
          2,
          cv::LINE_AA);
    } else {
      cv::putText(frame,
          "Stop (Failure)",
          to_frame(ego_agent_position + vector_t(1.0, - EGO_RADIUS / 2)),
          cv::FONT_HERSHEY_DUPLEX,
          1.0,
          cv::Scalar(0, 0, 255),
          2,
          cv::LINE_AA);
    }
  }


  return frame;
}


}