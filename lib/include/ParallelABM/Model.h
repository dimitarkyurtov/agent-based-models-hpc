#ifndef PARALLELABM_MODEL_H
#define PARALLELABM_MODEL_H

// Forward declaration
class Agent;

/**
 * @class Model
 * @brief Abstract base class defining the computational model for agent
 * interactions.
 *
 * The Model class represents the computational logic that defines how agents in
 * the simulation interact with one another. This is an abstract interface that
 * serves as the foundation for device-specific implementations (CPU, CUDA,
 * etc.).
 *
 * PURPOSE:
 * ========
 * The Model class exists to:
 * 1. Define the interaction rules between agents in the simulation
 * 2. Provide a uniform interface for the library to execute these interactions
 * 3. Enable device-specific optimizations through polymorphism
 *
 * The actual implementation of the interaction logic is provided by derived
 * classes (e.g., ModelCPU, ModelCUDA) which contain device-specific interaction
 * rules.
 *
 * USAGE IN THE LIBRARY:
 * =====================
 * The library uses Model objects to:
 * 1. Process local regions of agents after workload distribution
 * 2. Apply interaction rules that define the simulation behavior
 * 3. Update agent states based on their interactions with neighbors
 *
 * INTERACTION RULES:
 * ==================
 * An interaction rule defines what happens when agents interact:
 * - Agent state updates (position, velocity, properties, etc.)
 * - Force calculations (physics simulations)
 * - Behavioral rules (agent-based modeling)
 * - Communication between agents
 * - Environmental effects
 *
 * The specific implementation depends on the simulation requirements and the
 * computational device being used (CPU threads, GPU blocks, etc.).
 *
 * DEVICE-SPECIFIC IMPLEMENTATIONS:
 * =================================
 * Derived classes implement this interface for specific computational devices:
 * - ModelCPU: Uses standard C++ functions for CPU execution
 * - ModelCUDA: Uses CUDA device functions for GPU execution
 * - Future: ModelOpenCL, ModelVulkan, etc.
 *
 * Each implementation stores the appropriate function pointer and provides
 * the execution logic for that device.
 *
 */
class Model {
 public:
  /**
   * @brief Default constructor.
   */
  Model() = default;

  /**
   * @brief Copy constructor.
   */
  Model(const Model&) = default;

  /**
   * @brief Copy assignment operator.
   */
  Model& operator=(const Model&) = default;

  /**
   * @brief Move constructor.
   */
  Model(Model&&) = default;

  /**
   * @brief Move assignment operator.
   */
  Model& operator=(Model&&) = default;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~Model() = default;
};

#endif  // PARALLELABM_MODEL_H
