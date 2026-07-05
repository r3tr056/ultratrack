#pragma once
#include <ultratrack/core/types.hpp>

namespace ultratrack {

struct LifecycleConfig {
    int confirmation_threshold = 3;
    int max_age = 30;
    int max_tentative_age = 5;
};

class TrackLifecycle {
public:
    explicit TrackLifecycle(const LifecycleConfig& cfg);

    void onMatched(Track& track);
    void onMissed(Track& track);
    bool shouldRemove(const Track& track) const;
    bool shouldConfirm(const Track& track) const;

private:
    LifecycleConfig cfg_;
};

} // namespace ultratrack
