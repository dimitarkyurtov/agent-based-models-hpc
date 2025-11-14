#ifndef PARALLELABM_MPINODE_H
#define PARALLELABM_MPINODE_H

/**
 * @class MPINode
 * @brief Abstract base class for MPI communication abstraction.
 *
 * Represents a node in the MPI communication network. This class serves as the
 * base for both coordinator and worker nodes in a distributed simulation.
 */
class MPINode {
 protected:
  int rank;  // MPI rank of this node

 public:
  /**
   * @brief Constructs an MPINode with the given rank.
   * @param rank The MPI rank of this node
   */
  explicit MPINode(int rank);

  /**
   * @brief Copy constructor.
   */
  MPINode(const MPINode&) = default;

  /**
   * @brief Copy assignment operator.
   */
  MPINode& operator=(const MPINode&) = default;

  /**
   * @brief Move constructor.
   */
  MPINode(MPINode&&) = default;

  /**
   * @brief Move assignment operator.
   */
  MPINode& operator=(MPINode&&) = default;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~MPINode() = default;

  /**
   * @brief Gets the MPI rank of this node.
   * @return The MPI rank
   */
  [[nodiscard]] int GetRank() const;

  /**
   * @brief Sets the MPI rank of this node.
   * @param rank The new MPI rank
   */
  void SetRank(int rank);
};

#endif  // PARALLELABM_MPINODE_H
