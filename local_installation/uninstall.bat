@echo off
echo WAN2.2 Uninstaller
echo ==================
echo.
echo This will remove WAN2.2 shortcuts and start menu entries.
echo The installation files will remain for manual deletion.
echo.
set /p confirm="Are you sure you want to uninstall? (y/n): "
if /i not "%confirm%"=="y" goto :cancel

echo.
echo Removing WAN2.2 shortcuts...

REM Remove desktop shortcuts (.lnk and .bat files)
del "%USERPROFILE%\Desktop\WAN2.2 Video Generator.lnk" 2>nul
del "%USERPROFILE%\Desktop\WAN2.2 Video Generator.bat" 2>nul
del "%USERPROFILE%\Desktop\WAN2.2 UI.lnk" 2>nul
del "%USERPROFILE%\Desktop\WAN2.2 UI.bat" 2>nul

REM Remove start menu entries
rmdir /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\WAN2.2" 2>nul

echo.
echo WAN2.2 shortcuts have been removed.
echo.
echo To completely remove WAN2.2, you can manually delete this folder:
echo %~dp0
echo.
echo Thank you for using WAN2.2!
echo.
pause
goto :end

:cancel
echo Uninstall cancelled.
pause

:end