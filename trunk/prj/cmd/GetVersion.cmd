@echo off

echo Try to estimate SVN revision:

set SUBWCREV_EXE=SubWCRev.exe
set TRUNK_DIR=..\..
set SIMD_VERSION_TEMPLATE=%TRUNK_DIR%\prj\cmd\SimdVersion.h.tmpl
set SIMD_VERSION_FILE=%TRUNK_DIR%\src\Simd\SimdVersion.h
set VERSION_FILE=%TRUNK_DIR%\prj\cmd\Version.cmd

for %%X in (%SUBWCREV_EXE%) do (set SUBWCREV_EXE_FOUND=%%~$PATH:X)
if not defined SUBWCREV_EXE_FOUND (
echo Execution file "%SUBWCREV_EXE%" is not found!
exit 0
)

%SUBWCREV_EXE% %TRUNK_DIR%
if ERRORLEVEL 1 exit 0

if exist %VERSION_FILE% (
    call %VERSION_FILE%
) else (
	set VERSION=0
)
set LAST_VERSION=%VERSION%

echo set VERSION=$WCREV$>%VERSION_FILE%
%SUBWCREV_EXE% %TRUNK_DIR% %VERSION_FILE% %VERSION_FILE%
call %VERSION_FILE%

if not %LAST_VERSION% == %VERSION% set NEED_TO_UPDATE=1
if not exist %SIMD_VERSION_FILE% set NEED_TO_UPDATE=1

if defined NEED_TO_UPDATE (
echo Create or update file "%SIMD_VERSION_FILE%".
%SUBWCREV_EXE% %TRUNK_DIR% %SIMD_VERSION_TEMPLATE% %SIMD_VERSION_FILE%
) else (
echo Skip updating of file "%SIMD_VERSION_FILE%" because there are not any changes.
)
