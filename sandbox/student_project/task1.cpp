#include <iostream>
#include "tasks.hpp"

using namespace std;

int main() {
    for (auto i = 0; i < 100; ++i)
        cout << random_number() << endl;
    return EXIT_SUCCESS;
}

