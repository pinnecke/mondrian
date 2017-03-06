//
// Created by gabriel on 06.03.17.
//

#include<utils/kernel.h>
#include <modules/version/main.h>

int version_main(int argc, char **argv) {
    printf("Mondrian version %s (compiled %s)\n", MND_VERSION, __DATE__);
    return EXIT_SUCCESS;
}
