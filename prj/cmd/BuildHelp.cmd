@echo off

call GetVersion.cmd

set DOXYGEN=doxygen.exe
set DOXYGEN_OUT=%TRUNK_DIR%\doc
set DOXYGEN_HTML=help
set OUT_DIR=%DOXYGEN_OUT%\%DOXYGEN_HTML%
set CONFIG=".\Config.txt"

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

echo Create or update file %CONFIG%.
echo PROJECT_NAME="Simd Library">%CONFIG%
echo PROJECT_NUMBER=%VERSION%>>%CONFIG%
echo OUTPUT_DIRECTORY=%DOXYGEN_OUT%>>%CONFIG%
echo INPUT=..\..\src\Simd>>%CONFIG%
echo EXTRACT_ALL=NO>>%CONFIG%
echo SHOW_INCLUDE_FILES=NO>>%CONFIG%
echo SHOW_USED_FILES=NO>>%CONFIG%
echo INLINE_SOURCES=NO>>%CONFIG%
echo SOURCE_BROWSER=NO>>%CONFIG%
::echo FILE_PATTERNS=SimdLib.h SimdTypes.h SimdUtils.h SimdPoint.h SimdRectangle.h SimdView.h>>%CONFIG%
echo FILE_PATTERNS=SimdLib.h SimdTypes.h>>%CONFIG%
echo GENERATE_HTML=YES>>%CONFIG%
echo HTML_OUTPUT=%DOXYGEN_HTML%>>%CONFIG%
echo SEARCHENGINE=NO>>%CONFIG%
echo GENERATE_LATEX=NO>>%CONFIG%
echo OPTIMIZE_OUTPUT_FOR_C=YES>>%CONFIG%
echo EXTRACT_STATIC=YES>>%CONFIG%
echo ALPHABETICAL_INDEX=NO>>%CONFIG%
echo HIDE_UNDOC_MEMBERS=YES>>%CONFIG% 
echo HIDE_UNDOC_CLASSES=YES>>%CONFIG%

%DOXYGEN% %CONFIG%

pause



