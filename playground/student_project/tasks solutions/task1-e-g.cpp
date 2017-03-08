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

int *create_column(unsigned long num_of_elements, bool fill_with_random = true, bool fill = true)
{
    auto result = (int *) malloc (num_of_elements * sizeof(int));
    if (fill) {
        for (auto i = 0; i < num_of_elements; ++i)
            result[i] = fill_with_random ? random_number() : i;
    }
    return result;
}

void delete_column(int *column)
{
    free (column);
}

int main() {
    auto column = create_column(2e6);
    for (auto i = 0; i < 2e6; ++i) {
        cout << column[i] << endl;
    }
    delete_column(column);

    return EXIT_SUCCESS;
}

