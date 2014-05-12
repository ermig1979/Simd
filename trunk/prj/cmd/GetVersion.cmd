@echo off

echo Try to estimate SVN revision:

set SUBWCREV_EXE=SubWCRev.exe
set TRUNK_DIR=..\..
set USER_VERSION_TXT=%TRUNK_DIR%\prj\txt\Version.txt
set SIMD_VERSION_H_TXT=%TRUNK_DIR%\prj\txt\SimdVersion.h.txt
set SIMD_VERSION_H=%TRUNK_DIR%\src\Simd\SimdVersion.h
set VERSION_CMD=%TRUNK_DIR%\prj\cmd\Version.cmd
set FIND_AND_REPLACE_CMD=%TRUNK_DIR%\prj\cmd\FindAndReplace.cmd

set USER_VERSION=
for /f "delims=" %%i in ('type %USER_VERSION_TXT%') do set USER_VERSION=%%i

for %%X in (%SUBWCREV_EXE%) do (set SUBWCREV_EXE_FOUND=%%~$PATH:X)
if not defined SUBWCREV_EXE_FOUND (
echo Execution file "%SUBWCREV_EXE%" is not found!
exit 0
)

%SUBWCREV_EXE% %TRUNK_DIR%
if ERRORLEVEL 1 exit 0

if exist %VERSION_CMD% (
    call %VERSION_CMD%
) else (
	set VERSION=0
)
set LAST_VERSION=%VERSION%

echo set VERSION=%USER_VERSION%.$WCREV$>%VERSION_CMD%
%SUBWCREV_EXE% %TRUNK_DIR% %VERSION_CMD% %VERSION_CMD%
call %VERSION_CMD%

if not %LAST_VERSION% == %VERSION% set NEED_TO_UPDATE=1
if not exist %SIMD_VERSION_H% set NEED_TO_UPDATE=1

if not defined NEED_TO_UPDATE (
echo Skip updating of file "%SIMD_VERSION_H%" because there are not any changes.
exit 0
)

echo Create or update file "%SIMD_VERSION_H%".

call %FIND_AND_REPLACE_CMD% @VERSION@ %VERSION% %SIMD_VERSION_H_TXT%>%SIMD_VERSION_H%
