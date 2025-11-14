#ifndef PARALLELABM_AGENT_H
#define PARALLELABM_AGENT_H

/**
 * @class Agent
 * @brief User-defined class representing individual agents in the simulation.
 *
 * The Agent class is a minimal interface that serves as a marker for agent
 * objects in the simulation. The library does not require any specific methods
 * or member variables from this class - it is entirely user-defined based on
 * the specific requirements of the simulation.
 *
 * IMPORTANT: This class exists only as a type identifier within the library's
 * containers (such as the agents vector in the Space class). The actual
 * implementation and all its properties, methods, and state are defined by the
 * user according to their simulation needs.
 *
 * DEVICE COMPATIBILITY REQUIREMENTS:
 * ===================================
 * Since Agent objects are passed directly as pointers to computational devices
 * (CPU/GPU) for calculations, the user's concrete implementation MUST consider
 * device-specific constraints:
 *
 * For CUDA GPU Compatibility:
 * ---------------------------
 * 1. Data Members: Only use types that are compatible with device code:
 *    - Built-in types (int, float, double) are safe
 *    - Avoid STL containers (std::vector, std::string, etc.) as they use host
 * memory
 *    - Use fixed-size arrays or pointers to device-allocated memory instead
 *
 * 2. Member Functions:
 *    - Virtual functions have limited support and overhead on GPU
 *    - Prefer simple, inlineable functions marked with __host__ __device__
 *    - Avoid dynamic polymorphism if possible
 *
 * 3. Memory Layout:
 *    - Keep data members tightly packed for coalesced memory access
 *    - Avoid padding and alignment issues
 *    - Consider using struct-of-arrays (SoA) instead of array-of-structs (AoS)
 * for better GPU memory access patterns
 *
 * 4. Construction/Destruction:
 *    - Constructors and destructors should be device-compatible
 *    - Avoid complex initialization that relies on host-only features
 *
 * 5. Memory Management:
 *    - Do not allocate dynamic memory (new/malloc) within device code
 *    - All necessary memory should be pre-allocated
 *
 * For CPU Compatibility:
 * ----------------------
 * When running on CPU, there are fewer restrictions:
 * - Full C++ feature set is available
 * - STL containers can be used freely
 * - Virtual functions and inheritance work normally
 * - Dynamic memory allocation is permitted
 *
 * IMPLEMENTATION STRATEGIES:
 * ==========================
 *
 * Strategy 1 - Minimal CPU-Only Agent:
 * -------------------------------------
 * class Agent {
 * public:
 *     double x, y;           // Position
 *     double vx, vy;         // Velocity
 *     int id;                // Unique identifier
 *
 *     void update(double dt) {
 *         x += vx * dt;
 *         y += vy * dt;
 *     }
 * };
 *
 * Strategy 2 - CUDA-Compatible Agent:
 * ------------------------------------
 * class Agent {
 * public:
 *     float x, y, z;         // Position (float for better GPU performance)
 *     float vx, vy, vz;      // Velocity
 *     int id;                // Unique identifier
 *
 *     __host__ __device__
 *     void update(float dt) {
 *         x += vx * dt;
 *         y += vy * dt;
 *         z += vz * dt;
 *     }
 * };
 *
 * Strategy 3 - Hybrid Design with Compile-Time Selection:
 * --------------------------------------------------------
 * #ifdef __CUDA_ARCH__
 *     // GPU-specific implementation
 *     class Agent {
 *         float position[3];
 *         float velocity[3];
 *         // ...
 *     };
 * #else
 *     // CPU-specific implementation
 *     class Agent {
 *         std::vector<double> position;
 *         std::vector<double> velocity;
 *         // ...
 *     };
 * #endif
 *
 * USAGE IN THE LIBRARY:
 * =====================
 * The library uses Agent objects through:
 * 1. Storage in containers (std::vector<Agent> in Space class)
 * 2. Passing pointers to computational devices for parallel processing
 * 3. Communication between MPI nodes (requires Agent to be serializable)
 *
 * For MPI communication, consider:
 * - Simple Agent classes can use MPI_BYTE datatype
 * - Complex Agents may need custom MPI datatypes or serialization
 * - Avoid pointers as members if possible (complicate serialization)
 *
 * RECOMMENDATIONS:
 * ================
 * - Start simple: use only built-in types (int, float, double)
 * - Test thoroughly: ensure device code compiles and runs correctly
 * - Profile performance: memory layout significantly affects GPU performance
 * - Document assumptions: clearly state which devices your Agent supports
 * - Consider templates: parameterize Agent by backend type if needed
 *
 * @note The library does not enforce any structure on this class. It is the
 * user's responsibility to ensure their Agent implementation is compatible with
 * their chosen computational backend (CPU/CUDA/etc.).
 */
class Agent {
 public:
  /**
   * @brief Default constructor.
   *
   * Users should implement this according to their needs.
   * For device compatibility, ensure the constructor can execute on the target
   * device.
   */
  Agent() = default;

  /**
   * @brief Copy constructor.
   */
  Agent(const Agent&) = default;

  /**
   * @brief Copy assignment operator.
   */
  Agent& operator=(const Agent&) = default;

  /**
   * @brief Move constructor.
   */
  Agent(Agent&&) = default;

  /**
   * @brief Move assignment operator.
   */
  Agent& operator=(Agent&&) = default;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   *
   * Note: Virtual functions add overhead on GPU. If GPU performance is
   * critical, consider using a non-virtual destructor or avoiding inheritance
   * altogether.
   */
  virtual ~Agent() = default;

  // Users can add any members and methods here according to their simulation
  // requirements and device compatibility constraints.
};

#endif  // PARALLELABM_AGENT_H
