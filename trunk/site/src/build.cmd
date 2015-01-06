@echo off

call ..\..\prj\cmd\GetVersion.cmd

set DOXYGEN=doxygen.exe
set OUT_DIR=%TRUNK_DIR%\site\out
set CONFIG_TXT=%TRUNK_DIR%\site\src\config.txt
set TMP_TXT=%TRUNK_DIR%\site\src\tmp.txt

for %%X in (%DOXYGEN%) do (set DOXYGEN_FOUND=%%~$PATH:X)
if not defined DOXYGEN_FOUND (
echo Execution file "%DOXYGEN%" is not found!
pause
exit 0
)

if not exist %OUT_DIR% mkdir %OUT_DIR%

if exist %OUT_DIR% (
echo Delete old help files:
erase %OUT_DIR%\* /q /s /f
rmdir %OUT_DIR% /q /s
)

call ..\..\prj\cmd\FindAndReplace.cmd @VERSION@ %FULL_VERSION% %CONFIG_TXT%>%TMP_TXT%

%DOXYGEN% %TMP_TXT%

erase %TMP_TXT% /q /s /f

xcopy .\*.html %OUT_DIR%\* /y /i /s
xcopy .\*.png %OUT_DIR%\* /y /i /s
::xcopy .\*.js %OUT_DIR%\* /y /i /s
::xcopy .\*.css %OUT_DIR%\* /y /i /s

call %TRUNK_DIR%\site\out\index.html



