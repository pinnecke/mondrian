//
// Created by gabriel on 06.03.17.
//
#include <utils/kernel.h>
#include <modules/show/c/main.h>

int c_main(int argc, char **argv)
{
    int c;
    FILE *file;
    file = fopen("license/gpl-2.0c.txt", "r");
    if (file) {
        while ((c = getc(file)) != EOF)
            putchar(c);
        fclose(file);
    }

    return 0;
}
