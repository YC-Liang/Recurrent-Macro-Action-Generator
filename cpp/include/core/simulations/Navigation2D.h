#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace simulations {

class Navigation2D {

public:

  // Environment parameters.
  static constexpr float MAP_HALF_HEIGHT = 30.0f;
  static constexpr float MAP_HALF_WIDTH = 30.0f;

  // a total of three walls as three vectors. Each vector stores the bounding corner coordinates of the wall.
  static constexpr array_t<array_t<float, 4>, 5> CLOSED_WALLS{
    array_t<float, 4>{-20, 30, 5, 15},
    array_t<float, 4>{-20, 5, 5, -6},
    array_t<float, 4>{-20, -14, 5, -25},
    array_t<float, 4>{10, 13, 30, 8},
    array_t<float, 4>{15, -10, 30, -30}
  };

  static constexpr array_t<array_t<float, 4>, 4> DANGER_ZONES{
    array_t<float, 4>{-20, 15, 5, 13}, //center is -10, 14
    array_t<float, 4>{-20, 7, 5, 5},
    array_t<float, 4>{-20, -28, -5, -30},
    array_t<float, 4>{28, 8, 30, -10}
  };

  //randomisation starting region
  //static constexpr array_t<float, 4> STARTING_REGION{-25.0f, 15.0f, -25.0f, -15.0f};

  static constexpr array_t<vector_t, 20> ALL_LIGHT_POSITION{
    vector_t{-22, 28},
    vector_t{7, 28},
    vector_t{-22, 15},
    vector_t{7, 15},
    vector_t{-22, 5},
    vector_t{-22, -6},
    vector_t{7, 5},
    vector_t{7, -6},
    vector_t{-22, -14},
    vector_t{-22, -25},
    vector_t{7, -14},
    vector_t{7, -25},
    vector_t{20, 15},
    vector_t{20, 6},
    vector_t{15, -10},
    vector_t{15, -25},
    vector_t{-28, 22},
    vector_t{-28, 0},
    vector_t{-28, -22},
    vector_t{-7, -10}
  };

  static constexpr float LIGHT_RADIUS = 2.0f;

  static constexpr float EGO_RADIUS = 1.0f;
  static constexpr float EGO_SPEED = 1.0f;

  static constexpr float DELTA = 1.0f;

  static constexpr float EGO_START_STD = 1.0f;
  static constexpr float OBSERVATION_NOISE = 0.1f;
  static constexpr float MOTION_NOISE = 0.3f;

  // Randomization over initialization and context.
  inline static vector_t EGO_START_MEAN;
  inline static vector_t GOAL;

  static constexpr array_t<vector_t, 2> RANDOM_START_REGION{
    vector_t{-25, 10},
    vector_t{-25, -10}
  };

  
  static constexpr array_t<array_t<float, 4>, 3> GOAL_REGION{
    array_t<float, 4>{6, 28, 9, -28},
    array_t<float, 4>{10, 28, 28, 15},
    array_t<float, 4>{10, 6, 28, -8}
  };

  /*
  static array_t<vector_t, 7> LIGHT_POSITION{
    vector_t{-28, -26},
    vector_t{-10, -13},
    vector_t{6, -6},
    vector_t{13, -20},
    vector_t{10, 8},
    vector_t{15, 15},
    vector_t{-28, 28}
  };*/

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 150;
  static constexpr float STEP_REWARD = -0.2f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.99f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 150;
  static constexpr size_t PLANNING_TIME = 300;

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
    bool trigger = false;
    static Action Rand();
    Action() {}
    Action(float orientation) : orientation(orientation) { }
    float Id() const { return orientation; }

    static list_t<list_t<Action>> CreateHandcrafted(size_t length);
    static list_t<list_t<Action>> Deserialize(const list_t<float>& params, size_t macro_length);
  };

  size_t step;
  vector_t ego_agent_position;
  inline static thread_local list_t<vector_t> LIGHT_POSITION;
  /* ====== Construction functions ====== */
  Navigation2D();
  static Navigation2D CreateRandom();
  static Navigation2D SampleInitial(); //samples the start position of the actual agent

  /* ====== Belief related functions ====== */
  static Navigation2D SampleBeliefPrior(); //samples the belief prior which is sampled from two locations since the agent doesn't know its position initially
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

