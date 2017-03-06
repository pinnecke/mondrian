//
// Created by gabriel on 06.03.17.
//

#include <utils/kernel.h>
#include <modules/modules.h>
#include <utils/alg.h>
#include <utils/log.h>

struct man_page_ref
{
    const char *help_pattern;
    const char *man_page_name;
};

struct man_page_ref man_page_references[MAX_MODULES_COUNT];
unsigned next_man_page_references_id = 0;

void show_help(struct modules_context *context);
void show_unknown_command(struct modules_context *context, const char *command);

enum MND_RESULT mod_init_modules(struct modules_context *context, const char *error_msg, const char *usage_args,
                                  const char *command_list_desc)
{
    if ((error_msg == NULL) || (usage_args == NULL) || (command_list_desc == NULL))
        return_failure(MND_ILLEGAL_ARGUMENT);

    struct modules_context new_context;
    new_context.next_modules_id = new_context.next_group_id = 0;
    new_context.error_msg = strdup(error_msg);
    new_context.usage_args = strdup(usage_args);
    new_context.command_list_desc = strdup(command_list_desc);
    *context = new_context;

    return_ok();
}

enum MND_RESULT mod_install_group(unsigned *group_id, struct modules_context *context, const char *desc)
{
    if ((desc == NULL) || (context == NULL))
        return MND_ILLEGAL_ARGUMENT;
    if (context->next_group_id == MAX_GROUP_COUNT)
        return MND_REJECTED_NO_SPACE;

    *group_id = context->next_group_id;

    struct module_group group = {
            .desc = strdup(desc),
            .num_modules = 0,
            .modules = mnd_malloc(sizeof(struct mod_module) * INIT_MODULE_COUNT_PER_GROUP),
            .cap_modules = INIT_MODULE_COUNT_PER_GROUP
    };

    debugf("Group installed (context=%p, id=%u): '%s'", context, context->next_group_id, desc);

    context->groups[context->next_group_id++] = group;

    return_ok();
}

enum MND_RESULT mod_install_module(struct modules_context *context, unsigned group_id,
                                    const char *command, const char *desc,
                                    int (*entry_point)(int, char *[]))
{
    enum MND_RESULT command_already_registered(struct modules_context *context, const char *command);

    if ((command == NULL) || (desc == NULL) || (context == NULL) || (group_id >= context->next_group_id))
        return MND_ILLEGAL_ARGUMENT;
    if (context->next_modules_id == MAX_MODULES_COUNT)
        return MND_REJECTED_NO_SPACE;

    /* check for duplicate command name */
    if (command_already_registered(context, command) == MND_YES)
        return MND_COMMAND_ALREADY_REGISTERED;

    /* resize group link register if required */
    struct module_group *group = &context->groups[group_id];
    if (group->num_modules == group->cap_modules) {
        struct mod_module *resized_groups = NULL;
        unsigned resized_capacity = group->cap_modules * MODULE_COUNT_PER_GROUP_RESIZE_FACTOR;

        if ((resized_groups = realloc(group->modules, sizeof(struct mod_module) * resized_capacity)) == NULL)
            return MND_MALLOC_FAILED;
        else {
            group->cap_modules = resized_capacity;
            group->modules = resized_groups;
        }
    }

    /* register module */
    struct mod_module module =
            {
                    .desc = strdup(desc),
                    .command = strdup(command),
                    .num_sub_modules = 0,
                    .sub_modules = NULL,
                    .entry_point = entry_point
            };

    context->modules[context->next_modules_id++] = module;

    /* link module in group */
    group->modules[group->num_modules++] = module;

    return_ok();
}

enum MND_RESULT mod_install_man_page(const char *help_pattern, const char *man_page_name)
{
    if ((help_pattern == NULL) || (man_page_name == NULL))
        return MND_ILLEGAL_ARGUMENT;

    if (next_man_page_references_id == MAX_MODULES_COUNT)
        return MND_REJECTED_NO_SPACE;
    else {
        for (unsigned i = 0; i < next_man_page_references_id; i++) {
            if (strcmp(man_page_references[i].help_pattern, help_pattern) == 0) {
                return MND_HELP_PATTERN_ALREADY_REGISTERED;
            }
        }

        struct man_page_ref page_ref = {
                .help_pattern = strdup(help_pattern),
                .man_page_name = strdup(man_page_name)
        };
        man_page_references[next_man_page_references_id++] = page_ref;
    }

    return_ok();
}

enum MND_RESULT mod_start_module(int *return_code, struct modules_context *context, int argc, char *argv[])
{
    if (context == NULL)
        return MND_ILLEGAL_ARGUMENT;

    if (argc == 1) {
        show_help(context);
        *return_code = EXIT_SUCCESS;
        return MND_NO_PROGRAM_ARGS;
    } else {
        const char *command = argv[1];

        if (strcmp(command, "help") == 0)
        {
            if (argc == 2) {
                show_help(context);
                *return_code = EXIT_SUCCESS;
                return MND_NO_PROGRAM_ARGS;
            }

            const char *system_command = "man ";
            const char *MND_HOME = "/Users/marcus/git/racoondb";    // TODO: Home auslesen
            const char *man_dir = "/man/";

            char help_for_command_chain[COMMAND_CHAIN_MAX_LEN] = "\0";
            unsigned help_for_command_chain_len = 0;

            for (unsigned i = 2; i < argc; i++)
            {
                char *sub_command = argv[i];
                help_for_command_chain_len += strlen(sub_command) + 1;
                if (help_for_command_chain_len >= COMMAND_CHAIN_MAX_LEN) {
                    perror("ERROR: Command chain for request is too long.\n");
                    return MND_BUFFER_OVERFLOW;
                } else {
                    strcat(help_for_command_chain, sub_command);
                    if (i + 1 < argc) {
                        strcat(help_for_command_chain, " ");
                    }
                }
            }

            for (unsigned i = 0; i < next_man_page_references_id; i++) {
                if (strcmp(man_page_references[i].help_pattern, help_for_command_chain) == 0) {
                    const char *man_page_file = man_page_references[i].man_page_name;
                    char system_man_call_str[SYSTEM_MAN_MAX_LEN] = "\0";
                    unsigned system_man_call_len = strlen(system_command) + strlen(MND_HOME) + strlen(man_dir) + strlen(man_page_file) + 1;
                    if (system_man_call_len >= SYSTEM_MAN_MAX_LEN) {
                        perror("ERROR: Path length is too long. Consider to move ${MND_HOME}, or recompile with larger 'SYSTEM_MAN_MAX_LEN' buffer size.\n");
                        return MND_BUFFER_OVERFLOW;
                    } else {
                        strcat(system_man_call_str, system_command);
                        strcat(system_man_call_str, MND_HOME);
                        strcat(system_man_call_str, man_dir);
                        strcat(system_man_call_str, man_page_file);
                        system(system_man_call_str);
                    }
                    return MND_OK;
                }
            }

            printf("The command '%s' is unknown in this context.\n", help_for_command_chain);
            show_unknown_command(context, help_for_command_chain);
            return MND_NO_SUCH_ELEMENT;
        }
        else {
            for (unsigned i = 0; i < context->next_modules_id; i++) {
                struct mod_module *module = &context->modules[i];
                if (strcmp(module->command, command) == 0) {
                    *return_code = module->entry_point(argc - 1, argv + 1);
                    return MND_OK;
                }
            }
        }

        show_unknown_command(context, command);
        return MND_NO_SUCH_ELEMENT;
    }
}

enum MND_RESULT mod_clean_up(struct modules_context *context)
{
    debug("Running modules cleanup")
    for (unsigned i = 0; i < context->next_group_id; i++)
        free (context->groups[i].modules);
    free(context->command_list_desc);
    free(context->error_msg);
    free(context->usage_args);
    debug("Cleanup finished")
    return MND_OK;
}

void show_help(struct modules_context *context)
{
    assert (context != NULL);

    printf("%s\n\n"
                   "usage: mnd %s\n\n%s\n\n",
           context->error_msg,
           context->usage_args,
           context->command_list_desc);

    for (unsigned group_id = 0; group_id < context->next_group_id; group_id++) {
        struct module_group *group = &context->groups[group_id];
        if (group->num_modules > 0) {
            printf("%-*.*s\n", 80, 80, group->desc);

            for (unsigned module_idx = 0; module_idx < group->num_modules; module_idx++) {
                struct mod_module *module = &group->modules[module_idx];
                printf("   %-15s   %-*.*s\n", module->command, 59, 59, module->desc);
            }
            printf("\n");
        }
    }

    printf("\nSee 'mnd help <command>' to read the man page about a specific command.\n");
}

void show_unknown_command(struct modules_context *context, const char *command)
{
    printf("Mondrian: '%s' is not a mondrian command. See 'mnd help'.\n\n"
                   "Did you mean this?\n", command);

    for (int i = 0; i < context->next_modules_id; i++) {
        printf("   %s (%d)\n", context->modules[i].command, levenshtein_distance(context->modules[i].command, command));
    }

    printf("\n");
}

enum MND_RESULT command_already_registered(struct modules_context *context, const char *command)
{
    for (unsigned i = 0; i < context->next_modules_id; i++) {
        if (strcmp(context->modules[i].command, command) == 0)
            return MND_YES;
    }
    return MND_NO;
}
