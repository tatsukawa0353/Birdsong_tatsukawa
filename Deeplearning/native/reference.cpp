// Single fixed condition through the original, unmodified implementation.
#include "BirdsongModel_2.h"
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    BirdsongModel model(1e-7, 4.0/3.5e4, 0.35, 3.5e7, 3e6, argv[1]);
    model.saveData();
    const int steps=std::atoi(argv[2]);
    for(int i=0;i<steps;++i) model.step();
    return 0;
}
