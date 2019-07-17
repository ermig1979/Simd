@echo on

echo Extract project version:

set TRUNK_DIR=%1
set USER_VERSION_TXT=%TRUNK_DIR%\prj\txt\UserVersion.txt
set FULL_VERSION_TXT=%TRUNK_DIR%\prj\txt\FullVersion.txt
set SIMD_VERSION_H_TXT=%TRUNK_DIR%\prj\txt\SimdVersion.h.txt
set SIMD_VERSION_H=%TRUNK_DIR%\src\Simd\SimdVersion.h
set FIND_AND_REPLACE_CMD=%TRUNK_DIR%\prj\cmd\FindAndReplace.cmd

set /p USER_VERSION=<%USER_VERSION_TXT%

if exist %FULL_VERSION_TXT% (
set /p FULL_VERSION=<%FULL_VERSION_TXT%
) else (
set FULL_VERSION=UNKNOWN
)

set LAST_VERSION=%FULL_VERSION%

echo %USER_VERSION%>%FULL_VERSION_TXT%

set /p FULL_VERSION=<%FULL_VERSION_TXT%

echo Current version: %FULL_VERSION%
echo Last version: %LAST_VERSION%

if not %LAST_VERSION% == %FULL_VERSION% set NEED_TO_UPDATE=1
if not exist %SIMD_VERSION_H% set NEED_TO_UPDATE=1

if not defined NEED_TO_UPDATE (
echo Skip updating of file "%SIMD_VERSION_H%" because there are not any changes.
) else (
echo Create or update file "%SIMD_VERSION_H%".
call %FIND_AND_REPLACE_CMD% @VERSION@ %FULL_VERSION% %SIMD_VERSION_H_TXT%>%SIMD_VERSION_H%
)

