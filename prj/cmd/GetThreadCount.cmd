@echo off
for /f "tokens=*" %%f in ('wmic cpu get ThreadCount /value ^| find "="') do set %%f