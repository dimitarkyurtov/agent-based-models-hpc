#include <memory>
#include <utility>

#include "GameOfLifeSimulation.h"

GameOfLifeSimulation::GameOfLifeSimulation(
    int& argc, char**& argv, std::unique_ptr<Space<Cell>> space,
    std::shared_ptr<ModelCUDA<Cell>> model,
    ParallelABM::Environment& environment)
    : ParallelABM::SimulationCUDA<Cell>(argc, argv, std::move(space), model,
                                        environment) {}
