@echo off

set TRUNK_DIR=%1
set PRINT_INFO=%2
set USER_VERSION_TXT=%TRUNK_DIR%\prj\txt\UserVersion.txt
set FULL_VERSION_TXT=%TRUNK_DIR%\prj\txt\FullVersion.txt
set SIMD_VERSION_H_TXT=%TRUNK_DIR%\prj\txt\SimdVersion.h.txt
set SIMD_VERSION_H=%TRUNK_DIR%\src\Simd\SimdVersion.h
set FIND_AND_REPLACE_CMD=%TRUNK_DIR%\prj\cmd\FindAndReplace.cmd

if not "%PRINT_INFO%" == "0" ( echo Extract project version: )

set /p USER_VERSION=<%USER_VERSION_TXT%

if exist %FULL_VERSION_TXT% (
	set /p FULL_VERSION=<%FULL_VERSION_TXT%
) else (
	set FULL_VERSION=UNKNOWN
)

set LAST_VERSION=%FULL_VERSION%

echo %USER_VERSION%>%FULL_VERSION_TXT%
where /Q git > nul
if not errorlevel 1 (
	git -C %TRUNK_DIR% rev-parse 2>nul
	if not errorlevel 1 (
		git -C %TRUNK_DIR% rev-parse --short HEAD>%FULL_VERSION_TXT%
		set /p GIT_REVISION=<%FULL_VERSION_TXT%
		git -C %TRUNK_DIR% rev-parse --abbrev-ref HEAD>%FULL_VERSION_TXT%
		set /p GIT_BRANCH=<%FULL_VERSION_TXT%
		echo %USER_VERSION%.%GIT_BRANCH%-%GIT_REVISION%>%FULL_VERSION_TXT%
	)
)
set /p FULL_VERSION=<%FULL_VERSION_TXT%

set NEED_TO_UPDATE=0
if not %LAST_VERSION% == %FULL_VERSION% (
	if not "%PRINT_INFO%" == "0" ( echo Last project version '%LAST_VERSION%' is not equal to current version '%FULL_VERSION%'. )
	set NEED_TO_UPDATE=1
) else (
	if not "%PRINT_INFO%" == "0" ( echo Last project version '%LAST_VERSION%' is equal to current version '%FULL_VERSION%'. )
)
if not exist %SIMD_VERSION_H% set NEED_TO_UPDATE=1

if %NEED_TO_UPDATE% == 1 (
	if not "%PRINT_INFO%" == "0" ( echo Create or update file '%SIMD_VERSION_H%'. )
	call %FIND_AND_REPLACE_CMD% @VERSION@ %FULL_VERSION% %SIMD_VERSION_H_TXT%>%SIMD_VERSION_H%
) else (
	if not "%PRINT_INFO%" == "0" ( echo Skip updating of file '%SIMD_VERSION_H%' because there are not any changes. )
)


