#ifndef PARALLELABM_PARALLELABM_H
#define PARALLELABM_PARALLELABM_H

/**
 * @file ParallelABM.h
 * @brief Umbrella header for the ParallelABM library
 *
 * Include this single header to access all ParallelABM functionality.
 */

// Core components
#include "ParallelABM/Agent.h"
#include "ParallelABM/Model.h"
#include "ParallelABM/Space.h"

// Model implementations
#include "ParallelABM/ModelCPU.h"

#ifdef PARALLELABM_CUDA_ENABLED
#include "ParallelABM/ModelCUDA.h"
#include "ParallelABM/SimulationCUDA.h"
#endif

// MPI components
#include "ParallelABM/MPICoordinator.h"
#include "ParallelABM/MPINode.h"
#include "ParallelABM/MPIWorker.h"

// Simulation framework
#include "ParallelABM/Simulation.h"

#endif  // PARALLELABM_PARALLELABM_H
