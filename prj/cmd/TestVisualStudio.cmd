@echo off

set FILTER_INCLUDE=
if "%FILTER_INCLUDE%" == "" (
	set PREFIX=all
) else (
	set PREFIX=%FILTER_INCLUDE%
)
set YEARS=%date:~6,4%
set MONTHS=%date:~3,2%
set DAYS=%date:~0,2%
set HOURS=%time:~0,2%
if "%HOURS:~0,1%" == " " set HOURS=0%HOURS:~1,1%
SET MINUTES=%time:~3,2%
set ROOT=..\..\

if not errorlevel 1 ( call :TEST 2013 v120 Win32 )
if not errorlevel 1 ( call :TEST 2013 v120 x64 )
if not errorlevel 1 ( call :TEST 2015 v140 Win32 )
if not errorlevel 1 ( call :TEST 2015 v140 x64 )
if not errorlevel 1 ( call :TEST 2017 v141 Win32 )
if not errorlevel 1 ( call :TEST 2017 v141 x64 )
if not errorlevel 1 ( call :TEST 2019 v142 Win32 )
if not errorlevel 1 ( call :TEST 2019 v142 x64 )

pause
goto :eof
::------------------------------------------------------------------------------------------------
:TEST
set VERSION=%1
set TOOLSET=%2
set PLATFORM=%3

set BIN=%ROOT%\bin\%TOOLSET%\%PLATFORM%\Release\Test.exe
set LOG=%ROOT%test\%YEARS%_%MONTHS%_%DAYS%\%PREFIX%_vs%VERSION%_%PLATFORM%_%YEARS%_%MONTHS%_%DAYS%__%HOURS%_%MINUTES%.txt

if NOT EXIST "%BIN%" (
	echo File '%BIN%' is not exist! You have to compile it with using of Visual Studio %VERSION% {%PLATFORM%/Release}!
) else (
	echo Start test of Visual Studio %VERSION% {%TOOLSET%\%PLATFORM%\Release}:
	%BIN% -m=a -tt=1 -wt=1 -r=%ROOT% -fi=%FILTER_INCLUDE% -ot=%LOG% 
	if errorlevel 1 ( exit /b 1 ) 
)

goto :eof
::------------------------------------------------------------------------------------------------
