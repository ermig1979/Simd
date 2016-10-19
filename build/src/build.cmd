@echo off

set RAR_EXE="C:\Program Files\WinRAR\WinRAR.exe"
if not exist %RAR_EXE% (
echo Execution file "%RAR_EXE%" is not exists!
exit 1
)

call ..\..\prj\cmd\GetVersion.cmd

call ..\..\doc\src\build.cmd

set OUT_DIR=%TRUNK_DIR%\build\out
set TMP_DIR=%TRUNK_DIR%\build\out\simd

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
mkdir %TMP_DIR%\doc
mkdir %TMP_DIR%\data

xcopy %TRUNK_DIR%\src\* %TMP_DIR%\src\* /y /i /s
xcopy %TRUNK_DIR%\prj\* %TMP_DIR%\prj\* /y /i /s
xcopy %TRUNK_DIR%\doc\* %TMP_DIR%\doc\* /y /i /s
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
erase %TMP_DIR%\doc\src\*.lnk /q /s /f
erase %TMP_DIR%\doc\*.tmp /q /s /f

echo Create ZIP archive:
%RAR_EXE% a -afzip -ep1 -r %OUT_DIR%\simd.%FULL_VERSION%.zip %TMP_DIR%



