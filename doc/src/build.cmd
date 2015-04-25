@echo off

echo Try to generate help with using doxygen:

set DOXYGEN=doxygen.exe
set HELP_DIR=..\..\doc\help

for %%X in (%DOXYGEN%) do (set DOXYGEN_FOUND=%%~$PATH:X)
if not defined DOXYGEN_FOUND (
echo Execution file "%DOXYGEN%" is not found!
pause
exit 0
)

if exist %HELP_DIR% (
echo Delete old help files:
erase %HELP_DIR%\* /q /s /f
rmdir %HELP_DIR% /q /s
)

mkdir %HELP_DIR%

%DOXYGEN% ..\..\doc\src\config.txt

::call %HELP_DIR%\modules.html

::pause
