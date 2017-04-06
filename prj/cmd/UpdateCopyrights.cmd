@echo off

set START_DATE=1/18/2017
set VALID_COPYRIGHT=2011-2017
set COPYRIGHT_2016=2011-2016

call :FIND_IN ..\..\src\Simd
call :FIND_IN ..\..\src\Test

pause

goto :eof
::-----------------------------------------------------------------------------
:FIND_IN

set DIR=%1

set LIST=list.txt

xcopy %DIR% /D:%START_DATE% /L /S  > %LIST%

For /f "tokens=* delims=" %%f in (%LIST%) do (
    if exist %%f (
		find /c "%VALID_COPYRIGHT%" "%%f" | find ": 0" 1>nul && call :REPLACE %%f
	)
)

erase %LIST%

goto :eof
::-----------------------------------------------------------------------------
:REPLACE

set NAME=%1

echo Update copyrights in "%NAME%"

find /c "%COPYRIGHT_2016%" "%NAME%" | find ": 1" 1>nul && powershell -Command "(Get-Content "%NAME%") -replace '%COPYRIGHT_2016%', '%VALID_COPYRIGHT%' | Set-Content "%NAME%"

goto :eof
::-----------------------------------------------------------------------------