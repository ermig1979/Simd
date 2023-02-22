@echo off

echo Start Simd Library building:

call GetThreadCount.cmd

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

msbuild /m /p:Configuration=%CONFIGURATION% /p:Platform=%PLATFORM% -maxCpuCount:%ThreadCount% ..\vs2022\Simd.sln

if errorlevel 1 ( exit /b 1 ) 

goto :eof
::------------------------------------------------------------------------------------------------

