@echo off
REM Cathedral Stop Script for Windows

echo.
echo [INFO] Stopping Cathedral...
docker-compose down
echo.
echo [SUCCESS] Cathedral stopped.
echo.
echo To remove all data (fresh start), run:
echo   docker-compose down -v
echo.
pause
