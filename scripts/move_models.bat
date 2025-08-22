@echo off
title Move Models to Correct Location
echo ========================================
echo WAN2.2 Model Setup Helper
echo ========================================
echo.

echo Required models should be moved from:
echo   local_installation\models\
echo.
echo To:
echo   models\
echo.

echo Models to move:
echo   - WAN2.2-T2V-A14B
echo   - WAN2.2-I2V-A14B  
echo   - WAN2.2-TI2V-5B
echo.

echo You can:
echo 1. Copy the folders manually using Windows Explorer
echo 2. Or run these commands:
echo.
echo   xcopy "local_installation\models\WAN2.2-T2V-A14B" "models\WAN2.2-T2V-A14B" /E /I
echo   xcopy "local_installation\models\WAN2.2-I2V-A14B" "models\WAN2.2-I2V-A14B" /E /I
echo   xcopy "local_installation\models\WAN2.2-TI2V-5B" "models\WAN2.2-TI2V-5B" /E /I
echo.

pause