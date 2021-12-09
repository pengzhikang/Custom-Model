#include "pzk.hpp"
#include <iostream>
int main(int argc, char **argv) {
    if(argc == 3 && argv[1] == std::string("--json"))
    {
        PzkM a(argv[2]);
        a.meta.printinfo();
    }
    return 0;
}