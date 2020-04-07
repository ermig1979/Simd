@echo off

set FI=
set ROOT=..\..\
set BIN=%ROOT%\bin\v120\x64\Release\Test.exe
set LOG=%ROOT%test\all_vs2013_%date:~6,4%_%date:~3,2%_%date:~0,2%.txt

if NOT EXIST "%BIN%" (
	echo File '%BIN%' is not exist!
	echo You have to compile it with using of Visual Studio 2013 {x64/Release}!
	goto end
) 

%BIN% -m=a -tt=1 -wt=1 -r=%ROOT% -fi=%FI% -ot=%LOG% 

:end
pause