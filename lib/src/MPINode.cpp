#include "ParallelABM/MPINode.h"

MPINode::MPINode(int rank) : rank(rank) {}

int MPINode::GetRank() const { return rank; }

void MPINode::SetRank(int rank) { this->rank = rank; }
