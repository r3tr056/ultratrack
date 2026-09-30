@echo off
setlocal
echo Building UltraTracker...
cd /d "%~dp0"
if exist "%~dp0vcpkg\installed\x64-windows\bin" set "PATH=%~dp0vcpkg\installed\x64-windows\bin;%PATH%"

REM Pass CMake configuration arguments through to support local toolchains.
cmake -S . -B build -DBUILD_TESTS=ON %*
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

cmake --build build --config Release
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

ctest --test-dir build --output-on-failure -C Release
if %errorlevel% neq 0 (
    echo Tests failed!
    exit /b 1
)

echo.
echo Build and tests completed successfully!
echo Executable: build\src\Release\ultratrack.exe
echo.
echo To run the tracker:
echo   build\src\Release\ultratrack.exe --help
echo.
