#pragma once
#include "host_options.h"

// These choices are used only during create. Each plan retains its own settings.
inline thread_local int awe_additional_operation = 1;
struct awe_additional_scope {
    int previous;
    explicit awe_additional_scope(int operation) : previous(awe_additional_operation) {
        awe_additional_operation = operation;
    }
    ~awe_additional_scope() {
        awe_additional_operation = previous;
    }
};
// Imported engines use compiled defaults and explicit operation selection.
inline const char *awe_additional_env(const char *name) {
    if (!std::strcmp(name, "L8_SYRK")) return awe_additional_operation == 2 ? "1" : nullptr;
    if (!std::strcmp(name, "L8_TRMM")) return awe_additional_operation == 3 ? "1" : nullptr;
    if (!std::strcmp(name, "ZL8_HERK")) return awe_additional_operation == 4 ? "1" : nullptr;
    return nullptr;
}
