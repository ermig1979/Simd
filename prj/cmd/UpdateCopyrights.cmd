@echo off

set START_DATE=1/1/2018
set VALID_COPYRIGHT=2011-2018

call :FIND_ALL 2011-2017

pause

goto :eof
::-----------------------------------------------------------------------------
:FIND_ALL

set OLD_COPYRIGHT=%1

call :FIND_IN ..\..\src\Simd %OLD_COPYRIGHT%
call :FIND_IN ..\..\src\Test %OLD_COPYRIGHT%
call :FIND_IN ..\..\src\Use  %OLD_COPYRIGHT%

goto :eof
::-----------------------------------------------------------------------------
:FIND_IN

set DIR=%1
set OLD_COPYRIGHT=%2

set LIST=list.txt

xcopy %DIR% /D:%START_DATE% /L /S  > %LIST%

For /f "tokens=* delims=" %%f in (%LIST%) do (
    if exist %%f (
		find /c "%VALID_COPYRIGHT%" "%%f" | find ": 0" 1>nul && call :REPLACE %%f %OLD_COPYRIGHT%
	)
)

erase %LIST%

goto :eof
::-----------------------------------------------------------------------------
:REPLACE

set NAME=%1
set OLD_COPYRIGHT=%2

echo Update copyrights in "%NAME%"

find /c "%OLD_COPYRIGHT%" "%NAME%" | find ": 1" 1>nul && powershell -Command "(Get-Content "%NAME%") -replace '%OLD_COPYRIGHT%', '%VALID_COPYRIGHT%' | Set-Content "%NAME%"

goto :eof
::-----------------------------------------------------------------------------