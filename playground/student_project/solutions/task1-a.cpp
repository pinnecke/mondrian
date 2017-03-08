#include <iostream>
#include <limits>
#include <random>

using namespace std;

default_random_engine generator;

int random_number()
{
    static uniform_int_distribution<int> distribution(numeric_limits<int>::min(), numeric_limits<int>::max());
    return distribution(generator);
};

int main() {
    for (auto i = 0; i < 100; ++i)
        cout << random_number() << endl;
    return EXIT_SUCCESS;
}

