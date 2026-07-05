#include <ultratrack/benchmark/harness.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: benchmark_cli <image_dir> <gt.txt> <model.onnx>\n";
        return 1;
    }
    ultratrack::BenchmarkHarness harness;
    auto seq = harness.loadSequenceWithGT(argv[1], argv[2]);
    if (!seq.has_value()) {
        std::cerr << "Failed to load sequence: " << seq.error().message() << "\n";
        return 1;
    }

    ultratrack::TrackerSettings settings;
    settings.detector_model_path = argv[3];
    settings.mode = ultratrack::TrackingMode::BALANCED;

    auto result = harness.runVariant("ultratrack_balanced", settings, seq.value());
    harness.report({result});
    return 0;
}
