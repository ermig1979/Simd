@echo off

echo Start Simd Library building:

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

if not errorlevel 1 ( call :BUILD Debug Win32 )

if not errorlevel 1 ( call :BUILD Debug x64 )

if not errorlevel 1 ( call :BUILD Release Win32 )

if not errorlevel 1 ( call :BUILD Release x64 )

echo. & echo All configurations were built.

pause
goto :eof
::------------------------------------------------------------------------------------------------
:BUILD

set CONFIGURATION=%1
set PLATFORM=%2

echo. & echo Start build "%CONFIGURATION%|%PLATFORM%":

devenv ..\vs2022\Simd.sln /Build "%CONFIGURATION%|%PLATFORM%"

if errorlevel 1 ( exit /b 1 ) 

goto :eof
::------------------------------------------------------------------------------------------------

