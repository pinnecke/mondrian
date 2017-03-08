#include <iostream>
#include <limits>
#include <random>

using namespace std;

default_random_engine generator;

int random_number(int lower_bound = numeric_limits<int>::min(), int upper_bound = numeric_limits<int>::max())
{
    static uniform_int_distribution<int> distribution(lower_bound, upper_bound);
    return distribution(generator);
};

int main() {
    for (auto i = 0; i < 100; ++i)
        cout << random_number() << endl;
    return EXIT_SUCCESS;
}

