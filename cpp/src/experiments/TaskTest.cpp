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

  while (!sim.IsTerminal()) {
    list_t<ExpSimulation> samples;
    for (size_t i = 0; i < 1000; i++) {
      samples.emplace_back(belief.Sample());
    }
    cv::Mat frame = sim.Render(samples);
    cv::imshow("Frame", frame);
    cv::waitKey(1000 * ExpSimulation::DELTA);

    auto action = ExpSimulation::Action::Rand();
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
