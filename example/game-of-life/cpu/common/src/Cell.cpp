#include "Cell.h"

Cell::Cell() : alive(false), next_alive(false), x(0), y(0) {}

Cell::Cell(int x, int y, bool alive)
    : alive(alive), next_alive(false), x(x), y(y) {}

void Cell::ApplyNextState() { alive = next_alive; }
