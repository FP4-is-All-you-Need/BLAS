#pragma once
#include <memory>
#include <mutex>
#include <string>

namespace awe {
// The caller holds mutex() across every operation on a module's plans.
class EngineModule {
public:
    static std::shared_ptr<EngineModule> open(const std::string &path);
    ~EngineModule();
    EngineModule(const EngineModule &) = delete;
    EngineModule &operator=(const EngineModule &) = delete;
    void *symbol(const char *name) const;
    std::mutex &mutex() { return mutex_; }
    const std::string &path() const { return path_; }
private:
    explicit EngineModule(std::string path);
    std::string path_;
    void *handle_ = nullptr;
    std::mutex mutex_;
};
}
