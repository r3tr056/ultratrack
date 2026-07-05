# Task 1 Report: Modernize CMake Build System

## What I Implemented

Replaced the monolithic, hardcoded-OpenCV `CMakeLists.txt` with a modern CMake super-build:

- **Top-level `CMakeLists.txt`** – project definition, C++17 standard, build options (`BUILD_TESTS`, `ULTRATRACK_WITH_ONNXRUNTIME`), vcpkg-based dependency discovery (`OpenCV`, `Threads`, `spdlog`, `nlohmann_json`, optional `OnnxRuntime`), install/export rules, and package-config generation.
- **`src/CMakeLists.txt`** – builds the `ultratrack` static library from the existing source files and preserves the original SIMD and Windows compiler flags. Also keeps the original CLI as `ultratrack_cli` with output name `ultratrack`.
- **`tests/CMakeLists.txt`** – builds a single `ultratrack_tests` executable with Catch2 v3 and auto-discovers tests via `catch_discover_tests`.
- **`cmake/ultratrack-config.cmake.in`** – install-time package config that finds all required dependencies for consumers.
- **`CMakePresets.json`** – default preset using the local vcpkg toolchain and the Visual Studio 18 2026 generator installed on this machine.

The existing test files were converted from standalone `assert`-based programs to Catch2 `TEST_CASE`s so they can be linked into the unified `ultratrack_tests` target. All original test logic was preserved.

## Files Changed

- `CMakeLists.txt` (rewritten)
- `src/CMakeLists.txt` (created)
- `tests/CMakeLists.txt` (created)
- `cmake/ultratrack-config.cmake.in` (created)
- `CMakePresets.json` (created)
- `tests/test_displacement.cpp` (converted to Catch2)
- `tests/test_scale.cpp` (converted to Catch2)
- `tests/test_features.cpp` (converted to Catch2)
- `tests/test_tracker.cpp` (converted to Catch2)

## What I Tested

### 1. Configure and build library

```bash
cmake -B build -S . -DBUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --target ultratrack
```

Result: `ultratrack.lib` produced successfully.

### 2. Build and run tests

```bash
cmake --build build --config Release --target ultratrack_tests
ctest --test-dir build --output-on-failure -C Release
```

Result:

```text
100% tests passed, 0 tests failed out of 28
Total Test time (real) = 0.41 sec
```

### 3. Preset build

```bash
cmake --preset=default
cmake --build build --config Release --target ultratrack_tests
ctest --test-dir build --output-on-failure -C Release
```

Result: same 28 tests passed.

### 4. Install / consumer smoke test

```bash
cmake --install build --config Release --prefix _install
```

Installed headers, static library, and cmake config files correctly.

A temporary downstream consumer using `find_package(ultratrack CONFIG REQUIRED)` and `target_link_libraries(consumer PRIVATE ultratrack::ultratrack)` configured and built successfully when the vcpkg toolchain was also supplied.

## TDD Evidence

The existing tests were kept and adapted, not removed. They now exercise the same behavior through Catch2:

- Displacement predictor: 6 tests
- Scale estimator: 6 tests
- Feature extraction: 11 tests
- Tracker integration: 5 tests

Total: 28 tests, all green.

## Self-Review Findings

- **Positive:** The hardcoded OpenCV path is gone; dependencies come from vcpkg. The library target is usable from tests and a downstream consumer. SIMD/Windows flags are preserved. Install exports are complete and versioned.
- **Deviation from brief:** The brief listed future source files (`core/types.cpp`, `detector/opencv_dnn_backend.cpp`, etc.) that do not yet exist in the worktree. I used only the files that currently exist. The brief also assumed an `include/ultratrack/` layout; the current repository has headers directly under `include/`, so the install rule reflects that.
- **Static library:** I made `ultratrack` `STATIC` because the existing headers have no `__declspec(dllexport)` annotations. This avoids Windows shared-library symbol-export issues while still producing a clean install/export.
- **Generator note:** The installed vcpkg packages (including Catch2) were built with Visual Studio 18 2026 / MSVC 14.50. The `CMakePresets.json` default preset therefore uses that generator; using an older Visual Studio generator causes ABI link errors with Catch2.
- **OnnxRuntime:** The option exists but was left `OFF` because OnnxRuntime is not installed in this vcpkg tree. The config file correctly guards the `find_dependency(OnnxRuntime)` call.

## Issues / Concerns

- `cmake --build build --target test` does not work on Visual Studio generators; CMake reserves the `test` target name but only creates it for Makefile/Ninja generators. On Visual Studio the equivalent targets are `RUN_TESTS` or direct `ctest`. The task's explicitly requested commands (`cmake --build build --target ultratrack_tests` and `ctest --test-dir build --output-on-failure`) both pass.
- The `ultratrack_cli` executable preserves the original CLI entry point but is not installed. Later tasks may decide whether to ship it.
