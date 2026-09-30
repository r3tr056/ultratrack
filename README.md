# UltraTrack

UltraTrack is a C++17 object tracking project with a legacy video CLI and a
Phase 1 SDK. The SDK combines an OpenCV DNN detector, KCF correlation tracking,
Kalman prediction, a Hungarian-style assignment heuristic with IoU gating, and track lifecycle
management. It is a development foundation with several unfinished production
features.

## Development workflow

Use `development` for all development work and commits. `main` preserves the
existing published baseline at `460a01d`; promotion to `main` is a separate,
explicit decision. The SDK branch and the previous local `master` work have
been consolidated into `development`, including pending tracker fixes and
regression tests. See `AGENTS.md` for repository instructions.

## Code layout

| Location | Purpose |
| --- | --- |
| `include/ultratrack/` | Public SDK, session, result, detector and tracking interfaces |
| `src/detector/` | OpenCV DNN implementation and backend selection |
| `src/tracker/` | KCF, Kalman, displacement, scale and HOG/gray/color features |
| `src/tracking_engine/` | Data association, track lifecycle and track management |
| `src/tracking_session.cpp` | Detection, tracking, annotation and stage timing |
| `src/telemetry/` | Local stage latency collection |
| `src/benchmark/` and `tools/` | MOT input loading and benchmark CLI scaffolding |
| `src/main.cpp`, `src/ultratrack.cpp`, `include/ultratrack.hpp` | Legacy tracker and video CLI |
| `examples/simple_tracker.cpp` | SDK example for a model and camera/video input |
| `tests/` | Catch2 unit tests and synthetic MOT fixtures |
| `scripts/` | Python setup/gallery helper and unfinished Tkinter GUI |
| `src/gpu/` and `src/gst/` | GPU/GStreamer source retained from earlier work; absent from current build targets |

## Build on Windows

Requirements: CMake, a C++17 compiler, OpenCV 4.5 or newer, spdlog,
nlohmann_json, and Catch2 3 for tests. `vcpkg.json` lists the dependencies.
Installed dependencies are local and ignored by Git; they must match your
compiler. The current checkout retains them under `vcpkg/installed/x64-windows`.

The installed tool inventory on 1 October 2026 is:

| Component | Existing installation |
| --- | --- |
| Git and Git Credential Manager | Git 2.55.0; GitHub browser login available |
| CMake and CTest | 4.4.3 |
| C++ compiler | Visual Studio Build Tools 2022, MSVC 19.44.35229 |
| Windows SDK | 10.0.26100.0 |
| C++ dependencies | OpenCV 4.12.0, spdlog 1.17.0, nlohmann-json 3.12.0, Catch2 3.15.1, x64-windows |
| Other tools on PATH | Ninja, Python 3.12.10, Node.js 22.23.2, npm, Go, Docker, ODW |
| Tools absent from PATH | GitHub CLI, dotnet, mypy, pytest, OCR review |

Use CMake with the existing MSVC toolchain and one vcpkg dependency prefix.
Ninja and the other language runtimes are already available, but are not needed
for this verified build path. No additional compiler or package manager is
needed. Inspect the installed tools and dependencies before changing setup.

The active system Python has PyYAML and Ruff, but lacks OpenCV, NumPy, Pillow,
requests, tqdm, mypy and pytest. Codex also has an existing bundled runtime with
NumPy and Pillow for artifact work. Neither Python environment currently covers
the entire helper requirements file; no Python packages were installed for this
C++ consolidation.

From PowerShell with Visual Studio Build Tools 2022 installed:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$PWD/vcpkg/installed/x64-windows"
$env:PATH = "$PWD/vcpkg/installed/x64-windows/bin;$env:PATH"
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure -C Release
build/src/Release/ultratrack.exe --help
```

`build.bat` configures, builds and tests without pausing. It accepts additional
CMake arguments, such as `-G "Visual Studio 17 2022" -A x64` and
`-DCMAKE_PREFIX_PATH="C:/path/to/installed/x64-windows"`.

For a fresh clone, install the manifest dependencies with vcpkg and pass its
`scripts/buildsystems/vcpkg.cmake` as `CMAKE_TOOLCHAIN_FILE`. The `default` and
`linux-ninja` configure presets expect a vcpkg checkout in the repository root.
On Linux, use an appropriate generator and dependency prefix, then run
`cmake --build build` and `ctest --test-dir build --output-on-failure`.

The build produces `ultratrack` (static SDK library), `ultratrack_cli` (executable
named `ultratrack`), `simple_tracker`, `benchmark_cli`, and `ultratrack_tests`.
The SDK supports CMake installation/export as `ultratrack::ultratrack`.

## Current capabilities and gaps

- OpenCV DNN is the implemented detector backend. ONNX Runtime and TensorRT
  backend selections return `NOT_IMPLEMENTED`.
- Sessions accept BGR and RGB CPU frames. Other session frame formats are not
  implemented.
- Tracking modes select HOG (FAST), HOG plus grayscale (BALANCED), or HOG plus
  grayscale and Color Names (ACCURATE). Feature maps are resized to the HOG
  grid before concatenation.
- Licensing is a Phase 1 stub that accepts any nonempty key. Activation and
  remote telemetry fields do not provide production services.
- Benchmark latency is measured, but MOTA, MOTP, HOTA and IDF1 remain zero
  placeholders. These are not tracking accuracy results.
- The legacy header and SDK headers define different `Track`, `Detection` and
  `ErrorCode` types in the same namespace. Their API/type separation needs work
  before combining both interfaces in one client or shipping the SDK.
- Assignment uses a greedy row/column reduction heuristic; it does not implement
  a globally optimal Hungarian solver.
- The Python GUI contains unfinished download/training functions. It is not
  the primary SDK interface.
- Model files are not bundled. Real inference, hardware acceleration and
  performance targets require separate model/hardware validation.

## Plan for further work

1. Keep using `development`, the existing MSVC/CMake tools and the single
   installed vcpkg prefix. Inventory versions before any tool or SDK setup.
2. Resolve the legacy/SDK type and error-code collisions before exposing a
   combined public API. Add a client compilation check for the chosen boundary.
3. Supply a compatible detector model fixture and run the eight currently
   skipped inference/session checks before making runtime or performance claims.
4. Choose the next production feature explicitly: alternate detector backends,
   real licensing, accuracy metrics, or the Python interface. Implement only the
   chosen scope and reuse available dependencies.

## Verification and local cleanup

The consolidation was verified on 1 October 2026 with MSVC 19.44 and OpenCV
4.12.0. The Release build succeeded. CTest discovered 138 tests: 130 passed,
8 skipped because the detector model was unavailable, and none failed.
`codex-verify` printed `no recognised stack`; CMake and CTest supplied the actual
C++ checks.

Stale CMake output from the previous machine was moved outside the repository
into system TEMP. Installed vcpkg dependencies were preserved; Catch2 3.15.1
was rebuilt with the current compiler to replace incompatible copied binaries.
The obsolete SDK checkout and the verified build output were also moved to the
recovery folder in system TEMP, leaving one active checkout and no build cache
in the repository. Build output and common tool caches are ignored. Recovery
snapshots are kept outside the repository.
