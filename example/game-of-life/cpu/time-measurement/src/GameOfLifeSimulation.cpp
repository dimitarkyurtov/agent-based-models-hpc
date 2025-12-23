#include "GameOfLifeSimulation.h"

GameOfLifeSimulation::GameOfLifeSimulation(
    int& argc, char**& argv, std::unique_ptr<Space<Cell>> space,
    std::shared_ptr<ModelCPU<Cell>> model,
    ParallelABM::Environment& environment, bool sync_regions_every_timestep)
    : SimulationCPU<Cell>(argc, argv, std::move(space), model, environment,
                          sync_regions_every_timestep) {}
