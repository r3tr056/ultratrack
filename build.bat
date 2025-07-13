@echo off
echo Building UltraTracker...
cd /d "%~dp0"

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Configure and build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable: build\Release\Release\ultratrack.exe
echo.
echo To run the tracker:
echo   cd build\Release\Release
echo   ultratrack.exe --help
echo.
pause
