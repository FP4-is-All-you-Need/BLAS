#include "engine_loader.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <utility>

namespace awe {
EngineModule::EngineModule(std::string path) : path_(std::move(path)) {
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
        const char *message = dlerror();
        throw std::runtime_error("Cannot load engine: " + path_ + ": " +
                                 (message ? message : "unknown loader error"));
    }
}
EngineModule::~EngineModule() {
    if (handle_) {
        dlclose(handle_);
    }
}
std::shared_ptr<EngineModule> EngineModule::open(const std::string &path) {
    // Retain successful modules until process exit; CUDA registrations and CRT
    // owner state must not outlive the module while another plan still uses it.
    struct Registry {
        std::mutex mutex;
        std::map<std::string, std::shared_ptr<EngineModule>> modules;
    };
    static Registry *registry = new Registry;
    const std::string canonical = std::filesystem::canonical(path).string();
    std::lock_guard<std::mutex> lock(registry->mutex);
    auto found = registry->modules.find(canonical);
    if (found != registry->modules.end()) {
        return found->second;
    }
    auto module = std::shared_ptr<EngineModule>(new EngineModule(canonical));
    registry->modules.emplace(canonical, module);
    return module;
}
void *EngineModule::symbol(const char *name) const {
    if (!name || !*name) {
        throw std::invalid_argument("Empty engine symbol name");
    }
    dlerror();
    void *address = dlsym(handle_, name);
    const char *error = dlerror();
    if (error || !address) {
        throw std::runtime_error("Missing engine symbol " + std::string(name) + " in " + path_);
    }
    return address;
}
}
