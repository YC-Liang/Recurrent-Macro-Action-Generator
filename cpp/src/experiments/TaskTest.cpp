#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <opencv2/highgui.hpp>

typedef Belief<ExpSimulation> ExpBelief;

int main(int argc, char** argv) {
  // Initialize simulation.
  ExpSimulation sim = ExpSimulation::CreateRandom();

  // Initialize belief.
  ExpBelief belief = ExpBelief::FromInitialState();

  // Test action deserilisation
  /*
  size_t macro_length = 24;
  list_t<float> params = {3.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f,
 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f, 359.f,   0.f, 128.f, 197.f, 197.f,
 197.f, 197.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f, 128.f,
 128.f, 128.f, 339.f, 339.f, 339.f, 339.f, 275.f, 197.f, 128.f, 128.f, 128.f, 128.f, 197.f, 197.f,
 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 128.f, 128.f,
 128.f, 128.f, 359.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 339.f, 197.f, 197.f, 197.f, 197.f,
 197.f, 197.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f, 197.f,  40.f,   0.f,
   0.f, 219.f, 219.f, 219.f, 219.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f, 197.f, 197.f,
 197.f, 197.f, 197.f, 197.f, 197.f, 197.f, 197.f,   0.f, 197.f,  40.f, 219.f,   0.f,   0.f,   0.f,
   0.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 21conda list | grep cv9.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f,
 219.f, 306.f, 306.f, 306.f, 128.f,   3.f,  40.f,  40.f,  40.f,  40.f,  40.f,  40.f,  40.f,   3.f,
   3.f,   3.f,   3.f, 217.f,   3.f,   3.f,   3.f,   3.f, 217.f, 217.f, 217.f, 217.f, 217.f, 217.f,
 339.f, 306.f, 306.f, 306.f, 306.f, 306.f, 306.f, 306.f,   3.f,  40.f,  40.f,  40.f, 217.f,   3.f,
  40.f,  40.f,  40.f, 217.f,   3.f, 306.f, 306.f, 219.f, 219.f, 219.f};

  list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(params, macro_length);
  
  for(auto macro : macro_actions){
    for(auto action : macro){
      std::cout << action.orientation << ", ";
    }
    std::cout << std::endl;
  }

  std:: cout << "Finished printing macro actions" << std::endl;*/

  /*
  size_t macro_length = 8;
  list_t<float> params = {-0.1121f,  0.1713f,  0.1805f, -0.2834f,  0.9073f,  0.1487f,  0.0244f,  0.4556f,  0.0323f,
  0.1687f,  0.7341f,  0.4728f,  0.205f,  -0.003f,  -0.049f,   0.0464f,  0.9761f, -0.0263f,
  0.2906f, -0.0252f,  0.218f,   0.0376f,  0.9201f,  0.1391f,  0.0953f, -0.1763f,  0.1492f,
 -0.1984f,  0.9305f, -0.1799f,  0.134f,   0.2727f,  0.2408f,  0.2921f,  0.7923f,  0.3696f,
  0.1764f, -0.5525f,  0.5156f,  0.0336f,  0.5242f,  0.3491f,  0.1984f,  0.0566f,-0.0136f,
  0.0799f,  0.9734f,  0.0578f};

  list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(params, macro_length);
  for(auto macro : macro_actions){
    for(auto action: macro){
      std::cout << action.orientation << ", ";
    }
    std::cout << std::endl;
  } */


  while (!sim.IsTerminal()) {
    list_t<ExpSimulation> samples;
    for (size_t i = 0; i < 1000; i++) {
      samples.emplace_back(belief.Sample());
    }
    cv::Mat frame = sim.Render(samples);
    cv::imshow("Frame", frame);
    cv::waitKey(1000 * ExpSimulation::DELTA);

    auto action = ExpSimulation::Action::Rand();
    //action.trigger = true;
    auto result = sim.Step<false>(action);
    belief.Update(action, std::get<2>(result));
    float reward = std::get<1>(result);
    /*
    if (std::abs(reward) - 0.2f >= 0.1f){
      std::cout << "True Step reward:" << reward << std::endl;
    }*/
    //std::cout << "Number of step: " << sim.step << std::endl;
    std::cout << "True step reward: " << reward << std::endl;
    sim = std::get<0>(result);
  }
}
