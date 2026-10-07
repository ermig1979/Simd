@echo off

echo Try to generate help with using doxygen:

set DOXYGEN=doxygen.exe
set DOCS_DIR=..\..\docs
set HELP_DIR=%DOCS_DIR%\help

for %%X in (%DOXYGEN%) do (set DOXYGEN_FOUND=%%~$PATH:X)
if not defined DOXYGEN_FOUND (
 echo Execution file "%DOXYGEN%" is not found!
 pause
 exit 0
)

if exist %HELP_DIR% (
 echo Delete old help files:
 erase %HELP_DIR%\* /q /s /f
)

if not exist %HELP_DIR% (
 mkdir %HELP_DIR%
)

%DOXYGEN% ..\txt\DoxygenConfig.txt

erase %DOCS_DIR%\*.tmp /q /s /f

::call %HELP_DIR%\modules.html

::pause
