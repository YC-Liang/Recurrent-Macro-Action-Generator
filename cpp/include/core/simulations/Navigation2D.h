#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace simulations {

class Navigation2D {

public:

  // Environment parameters.

  // a total of three walls as three vectors. Each vector stores the bounding corner coordinates of the wall.
  static constexpr array_t<array_t<float, 4>, 5> CLOSED_WALLS{
    array_t<float, 4>{-25, 30, 5, 15},
    array_t<float, 4>{-25, 5, 5, -5},
    array_t<float, 4>{-25, -15, 5, -25},
    array_t<float, 4>{10, 13, 35, 8},
    array_t<float, 4>{15, -10, 35, -30}
  };

  static constexpr array_t<array_t<float, 4>, 4> DANGER_ZONES{
    array_t<float, 4>{-25, 15, 5, 13},
    array_t<float, 4>{-25, 7, 5, 5},
    array_t<float, 4>{-25, -28, -5, -30},
    array_t<float, 4>{33, 8, 35, -10}
  };

  //randomisation starting region
  static constexpr array_t<float, 4> STARTING_REGION{-30.0f, 15.0f, -30.0f, -15.0f};

  static constexpr array_t<vector_t, 7> LIGHT_POSITION{
    vector_t{-33, -28},
    vector_t{-25, -25},
    vector_t{5, -5},
    vector_t{5, -25},
    vector_t{10, 8},
    vector_t{13, -15},
    vector_t{7, 28}
  };
  static constexpr float LIGHT_RADIUS = 2.0f;

  static constexpr float EGO_RADIUS = 1.0f;
  static constexpr float EGO_SPEED = 1.0f;

  static constexpr float DELTA = 1.0f;

  static constexpr float EGO_START_STD = 1.0f;
  static constexpr float OBSERVATION_NOISE = 1.0f;
  static constexpr float MOTION_NOISE = 1.0f;

  // Randomization over initialization and context.
  inline static vector_t EGO_START_MEAN;
  inline static vector_t GOAL;

  static constexpr array_t<vector_t, 2> RANDOM_START_REGION{
    vector_t{-30, 10},
    vector_t{-30, -10}
  };

  static constexpr array_t<array_t<int, 4>, 2> GOAL_REGION{
    array_t<int, 4>{15, 28, 30, 10},
    array_t<int, 4>{15, 6, 30, -8}
  };

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 250;
  static constexpr float STEP_REWARD = -0.1f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 250;
  static constexpr size_t PLANNING_TIME = 400;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 50;
  static constexpr float POMCPOW_K_ACTION = 25.0f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.05f;
  static constexpr float POMCPOW_K_OBSERVATION = 10.0f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.1f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 20;

  struct Observation {
    vector_t ego_agent_position;
    uint64_t Discretize() const;
  };

  //change actions
  struct Action {
    float orientation;
    static Action Rand();
    Action() {}
    Action(float orientation) : orientation(orientation) { }
    float Id() const { return orientation; }

    static list_t<list_t<Action>> CreateHandcrafted(size_t length);
    static list_t<list_t<Action>> Deserialize(const list_t<float>& params, size_t macro_length);
  };

  size_t step;
  vector_t ego_agent_position;

  /* ====== Construction functions ====== */
  Navigation2D();
  static Navigation2D CreateRandom();

  /* ====== Belief related functions ====== */
  static Navigation2D SampleBeliefPrior();
  float Error(const Navigation2D& other) const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  bool IsFailure() const { return _is_failure; }
  bool CircleBoxCollision(vector_t pos, float wall_centre_x, float wall_centre_y, float wall_width, float wall_height) const; //circle and rectangle collision
  bool CheckCollision(vector_t pos) const; //check collision with walls and danger zones during stepping
  bool IsInLight(vector_t pos) const;
  bool IsInDangerZone(vector_t pos) const;
  template <bool compute_log_prob>
  std::tuple<Navigation2D, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  static void EncodeContext(list_t<float>& data);
  cv::Mat Render(const list_t<Navigation2D>& belief_sims,
      const list_t<list_t<Action>>& macro_actions={},
      const vector_t& macro_action_start={}) const;

private:

  bool _is_terminal;
  bool _is_failure;

};

}

