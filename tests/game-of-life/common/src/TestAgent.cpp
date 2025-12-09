#include "TestAgent.h"

TestAgent::TestAgent() : alive(false), next_alive(false), x(0), y(0) {}

TestAgent::TestAgent(int x, int y, bool alive)
    : alive(alive), next_alive(false), x(x), y(y) {}

void TestAgent::ApplyNextState() { alive = next_alive; }
