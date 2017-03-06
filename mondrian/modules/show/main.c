//
// Created by gabriel on 06.03.17.
//

#include <modules/manifest.h>

int show_main(int argc, char *argv[])
{
    begin_module_context("show <topic>",
                         "You must specify a topic to show. See the topic list below.",
                         "These are common topics that you can display:");

    begin_group("Information on the license")
    install_module(w, "w", "Displays the warranties made by this software.");
    install_module(c, "c", "Displays the license terms about redistribution.");
    end_group()

    end_module_context();
}