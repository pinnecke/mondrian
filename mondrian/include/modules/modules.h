//
// Created by gabriel on 06.03.17.
//

#ifndef MND_MODULES_H
#define MND_MODULES_H

#include <utils/result.h>
#include <stdlib.h>

#define MAX_GROUP_COUNT                       100
#define MAX_MODULES_COUNT                     100
#define INIT_MODULE_COUNT_PER_GROUP            50
#define MODULE_COUNT_PER_GROUP_RESIZE_FACTOR  1.4
#define COMMAND_CHAIN_MAX_LEN                2048
#define SYSTEM_MAN_MAX_LEN                   2048

struct module_group
{
    const char *desc;
    struct mod_module *modules;
    unsigned num_modules, cap_modules;
};

struct mod_module
{
    const char *desc, *command;
    struct mod_module *sub_modules;
    unsigned num_sub_modules;
    int (*entry_point)(int, char *[]);
};

struct modules_context
{
    char *error_msg, *usage_args, *command_list_desc;

    struct mod_module modules[MAX_MODULES_COUNT];
    unsigned          next_modules_id;

    struct module_group groups[MAX_GROUP_COUNT];
    unsigned            next_group_id;
};

#define begin_modules() \
    int main(int argc, char *argv[]) {

#define end_modules()        \
    }

#define define_module(module_command_name, main_function) \
    static int __mod_##module_command_name##_init(int argc, char *argv[]) {  return main_function(argc, argv);  }

#define begin_module_context(usage_help, error_msg, command_list_desc) \
    struct modules_context context; \
    mod_init_modules(&context, error_msg, usage_help, command_list_desc); \
    unsigned current_group; \
    mod_install_group(&current_group, &context, "Default"); \


#define end_module_context() \
    int return_code;\
    mod_start_module(&return_code, &context, argc, argv); \
    mod_clean_up(&context); \
    return return_code;

#define install_module(module_name, module_command_name, module_description) \
    mod_install_module(&context, current_group, module_command_name, module_description, __mod_##module_name##_init); \
    mod_install_man_page(module_command_name, "mnd_" #module_name);

#define install_man_page(help_pattern, man_page_name) \
    mod_install_man_page(help_pattern, man_page_name);

#define begin_group(description) \
    mod_install_group(&current_group, &context, description); \

#define end_group()


enum MND_RESULT mod_init_modules(struct modules_context *context, const char *error_msg, const char *usage_args,
                                  const char *command_list_desc);

enum MND_RESULT mod_install_group(unsigned *group_id, struct modules_context *context, const char *desc);

enum MND_RESULT mod_install_module(struct modules_context *context, unsigned group_id,
                                    const char *command, const char *desc, int (*entry_point)(int, char *[]));

enum MND_RESULT mod_install_man_page(const char *help_pattern, const char *man_page_name);

enum MND_RESULT mod_start_module(int *return_code, struct modules_context *context, int argc, char *argv[]);

enum MND_RESULT mod_clean_up(struct modules_context *context);

#endif
