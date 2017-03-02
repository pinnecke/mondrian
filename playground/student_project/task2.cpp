#include <iostream>
#include "tasks.hpp"

using namespace std;

int main()
{
    auto column = create_column(2e6);
    for (auto i = 0; i < 2e6; ++i) {
        cout << column[i] << endl;
    }
    delete_column(column);

    return EXIT_SUCCESS;
}

