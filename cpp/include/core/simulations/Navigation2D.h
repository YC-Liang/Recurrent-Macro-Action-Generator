#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace simulations {

class Navigation2D {

public:

  // Environment parameters.
  // map sizes
  static constexpr float MAP_HEIGHT = 600.0f;
  static constexpr float MAP_WIDTH = 700.0f;

  // a total of three walls as three vectors. Each vector stores the bounding corner coordinates of the wall.
  static constexpr array_t<list_t<float>, 3> CLOSED_WALLS{
    list_t<float>{100, 0, 400, 150},
    list_t<float>{100, 250, 400, 350},
    list_t<float>{100, 450, 400, 600}
  };

  static constexpr array_t<list_t<float>, 2> DANGER_ZONES{
    list_t<float>{100, 150, 400, 170},
    list_t<float>{100, 230, 400, 250}
  };

  //randomisation starting region
  static constexpr array_t<float, 4> STARTING_REGION{0.0f, 150.0f, 0.0f, 450.0f};
  static constexpr array_t<float, 4> GOAL_REGION{550.0f, 150.0f, 700.0f, 450.0f};

  static constexpr array_t<vector_t, 2> LIGHT_POSITION{
    vector_t{50, 550},
    vector_t{400, 350}
  };
  static constexpr float LIGHT_RADIUS = 20.0f;

  static constexpr float EGO_RADIUS = 5.0f;
  static constexpr float EGO_SPEED = 10.0f;

  static constexpr float DELTA = 1.0f;

  static constexpr float EGO_START_STD = 1.0f;
  static constexpr float OBSERVATION_NOISE = 0.3f;

  // Randomization over initialization and context.
  inline static vector_t EGO_START_MEAN;
  inline static vector_t GOAL;
  /*
  static constexpr array_t<vector_t, 4> RANDOMIZATION_REGION{ // ego start mean, goal, light pos are bounded within this.
      vector_t{-4, 4},
      vector_t{4, 4},
      vector_t{4, -4},
      vector_t{-4, -4}
  };*/

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 240;
  static constexpr float STEP_REWARD = -0.1f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 240;
  static constexpr size_t PLANNING_TIME = 500;

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
    Action(float orientation) : trigger(false), orientation(orientation) { }
    float Id() const { return trigger ? std::numeric_limits<float>::quiet_NaN() : orientation; }

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
  bool CircleBoxCollision(vector_t pos, float wall_width, float wall_height) const; //circle and rectangle collision
  bool CheckCollision(vector_t pos) const; //check collision with walls and danger zones during stepping
  bool IsInLight(vector_t pos) const;
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

