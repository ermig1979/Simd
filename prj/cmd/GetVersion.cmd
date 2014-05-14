@echo off

echo Try to estimate SVN revision:

set SUBWCREV_EXE=SubWCRev.exe
set TRUNK_DIR=..\..
set USER_VERSION_TXT=%TRUNK_DIR%\prj\txt\UserVersion.txt
set FULL_VERSION_TXT=%TRUNK_DIR%\prj\txt\FullVersion.txt
set SIMD_VERSION_H_TXT=%TRUNK_DIR%\prj\txt\SimdVersion.h.txt
set SIMD_VERSION_H=%TRUNK_DIR%\src\Simd\SimdVersion.h
set FIND_AND_REPLACE_CMD=%TRUNK_DIR%\prj\cmd\FindAndReplace.cmd

for %%X in (%SUBWCREV_EXE%) do (set SUBWCREV_EXE_FOUND=%%~$PATH:X)
if not defined SUBWCREV_EXE_FOUND (
echo Execution file "%SUBWCREV_EXE%" is not found!
exit 0
)

%SUBWCREV_EXE% %TRUNK_DIR%
if ERRORLEVEL 1 exit 0

set /p USER_VERSION=<%USER_VERSION_TXT%

if exist %FULL_VERSION_TXT% (
set /p FULL_VERSION=<%FULL_VERSION_TXT%
) else (
set FULL_VERSION=0.0.0.0
)
set LAST_VERSION=%FULL_VERSION%

echo %USER_VERSION%.$WCREV$>%FULL_VERSION_TXT%
%SUBWCREV_EXE% %TRUNK_DIR% %FULL_VERSION_TXT% %FULL_VERSION_TXT%
set /p FULL_VERSION=<%FULL_VERSION_TXT%

if not %LAST_VERSION% == %FULL_VERSION% set NEED_TO_UPDATE=1
if not exist %SIMD_VERSION_H% set NEED_TO_UPDATE=1

if not defined NEED_TO_UPDATE (
echo Skip updating of file "%SIMD_VERSION_H%" because there are not any changes.
) else (
echo Create or update file "%SIMD_VERSION_H%".
call %FIND_AND_REPLACE_CMD% @VERSION@ %FULL_VERSION% %SIMD_VERSION_H_TXT%>%SIMD_VERSION_H%
)
