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
   0.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f, 219.f,
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
