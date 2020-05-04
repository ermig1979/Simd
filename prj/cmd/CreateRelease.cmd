@echo off

set ARCHIVER="C:\Program Files\7-Zip\7z.exe"
if not exist %ARCHIVER% (
echo Execution file "%ARCHIVER%" is not exists!
exit 1
)

call .\GetVersion.cmd ..\..

call .\GenerateHelp.cmd

set OUT_DIR=%TRUNK_DIR%\zip\
set TMP_DIR=%TRUNK_DIR%\zip\simd

if not exist %OUT_DIR% mkdir %OUT_DIR%

if exist %TMP_DIR% (
echo Delete old files:
erase %TMP_DIR%\* /q /s /f
rmdir %TMP_DIR% /q /s
)

echo Copy new files:
mkdir %TMP_DIR%
mkdir %TMP_DIR%\src
mkdir %TMP_DIR%\prj
mkdir %TMP_DIR%\docs
mkdir %TMP_DIR%\data

xcopy %TRUNK_DIR%\src\* %TMP_DIR%\src\* /y /i /s
xcopy %TRUNK_DIR%\prj\* %TMP_DIR%\prj\* /y /i /s
xcopy %TRUNK_DIR%\docs\* %TMP_DIR%\docs\* /y /i /s
xcopy %TRUNK_DIR%\data\* %TMP_DIR%\data\* /y /i /s

echo Erase temporary files:
erase %TMP_DIR%\prj\*.user /q /s /f
erase %TMP_DIR%\prj\*.suo /q /s /f
erase %TMP_DIR%\prj\*.ncb /q /s /f
erase %TMP_DIR%\prj\*.depend /q /s /f
erase %TMP_DIR%\prj\*.layout /q /s /f
erase %TMP_DIR%\prj\*.cbTemp /q /s /f
erase %TMP_DIR%\prj\*.pdb /q /s /f
erase %TMP_DIR%\prj\*.pgm /q /s /f
erase %TMP_DIR%\prj\*.ppm /q /s /f
erase %TMP_DIR%\prj\*.jpg /q /s /f
erase %TMP_DIR%\prj\Ocv.props /q /s /f
erase %TMP_DIR%\prj\cmd\UpdateCopyrights.cmd /q /s /f
erase %TMP_DIR%\prj\cmd\CreateRelease.cmd /q /s /f
erase %TMP_DIR%\prj\cmd\GenerateHelp.cmd /q /s /f
erase %TMP_DIR%\prj\vs*\*.txt /q /s /f
erase %TMP_DIR%\prj\vs2019\.vs\* /q /s /f
rmdir %TMP_DIR%\prj\vs2019\.vs /q /s
erase %TMP_DIR%\docs\*.tmp /q /s /f

echo Create ZIP archive:
%ARCHIVER% a -tzip -r %OUT_DIR%\simd.%FULL_VERSION%.zip %TMP_DIR%

