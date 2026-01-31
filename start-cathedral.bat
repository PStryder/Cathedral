@echo off
REM Cathedral Startup Script for Windows
REM Requires Docker Desktop to be running

echo.
echo  ========================================
echo   Cathedral - Memory-Augmented Chat
echo  ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if .env exists
if not exist .env (
    echo [INFO] No .env file found. Creating from template...
    copy .env.docker .env
    echo.
    echo [IMPORTANT] Please edit .env and add your API keys:
    echo   - OPENROUTER_API_KEY
    echo   - OPENAI_API_KEY
    echo.
    echo Then run this script again.
    notepad .env
    pause
    exit /b 0
)

REM Check if API keys are set
findstr /C:"your-key-here" .env >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] It looks like you haven't set your API keys in .env
    echo Please edit .env and add your actual API keys.
    echo.
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "%CONTINUE%"=="y" (
        notepad .env
        exit /b 0
    )
)

echo [INFO] Starting Cathedral...
echo.

REM Pull latest images and start
docker-compose pull
docker-compose up -d

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Cathedral.
    echo Run 'docker-compose logs' for details.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Cathedral is starting up!
echo.
echo   Web Interface: http://localhost:8000
echo   Health Check:  http://localhost:8000/api/health
echo.
echo   View logs:     docker-compose logs -f
echo   Stop:          docker-compose down
echo.

REM Wait for health check
echo [INFO] Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Open browser
echo [INFO] Opening browser...
start http://localhost:8000

echo.
echo Press any key to view logs (Ctrl+C to exit)...
pause >nul
docker-compose logs -f
