#pragma once

// Macro-action parameters
#define BELIEF_SAMPLES 100

// Define simulation.
#if defined SIM_LightDark
  #include "core/simulations/LightDark.h"
  typedef simulations::LightDark ExpSimulation;
#elif defined SIM_LightDarkCirc
  #include "core/simulations/LightDarkCirc.h"
  typedef simulations::LightDarkCirc ExpSimulation;
#elif defined SIM_PuckPush
  #include "core/simulations/PuckPush.h"
  typedef simulations::PuckPush ExpSimulation;
#elif defined SIM_PuckPushHard
  #include "core/simulations/PuckPushHard.h"
  typedef simulations::PuckPushHard ExpSimulation;
#elif defined SIM_VdpTag
  #include "core/simulations/VdpTag.h"
  typedef simulations::VdpTag ExpSimulation;
#elif defined SIM_DriveHard
  #include "core/simulations/DriveHard.h"
  typedef simulations::DriveHard ExpSimulation;
#elif defined SIM_Navigation2D
  #include "core/simulations/Navigation2D.h"
  typedef simulations::Navigation2D ExpSimulation;
#endif
