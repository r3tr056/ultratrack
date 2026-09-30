# UltraTrack repository workflow

## Branches and commits

- Keep `main` and `development` as the repository's two branches.
- Perform development and commit all development changes on `development`.
- `main` is the baseline; promote changes to it only when the user requests it.
- Reuse the primary checkout. Create another branch or worktree only if the user
  explicitly requests an exception to this workflow.
- Preserve existing work and history when consolidating branches.

## Build and verification

This is a C++17 CMake project. Read `README.md` for dependencies and setup.
On Windows with the repository's installed vcpkg dependencies, use:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$PWD/vcpkg/installed/x64-windows"
$env:PATH = "$PWD/vcpkg/installed/x64-windows/bin;$env:PATH"
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure -C Release
```

For another toolchain, configure the appropriate generator and dependency path.
Report skipped model-dependent tests separately from passed tests. `codex-verify`
does not recognize this C++ stack, so it cannot replace CMake and CTest checks.

## Cache and scope

Keep build output and tool caches ignored. Do not delete model files, lookup
tables, installed dependencies, tracking state, or Git history as cache cleanup.
Keep recovery snapshots and one-off verification output in system TEMP.
Limit changes to the requested task and read files before editing them.
