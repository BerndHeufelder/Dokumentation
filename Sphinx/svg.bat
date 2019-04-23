@ECHO OFF

REM change directory to ...\Sphinx\source
set dir_source=%~dp0source
pushd %dir_source%
set dir_pdf2svg="..\..\SW-Tools\pdf2svg\pdf2svg.exe"
setlocal enabledelayedexpansion

for /r %%i in (*.pdf) do (
	REM %%~pi path of current file in loop, %%~ni name of current file in loop, %%~xi file extension (.pdf)
	
	echo %%~ni
	if "%%~ni" == "PMSM_Modell_in_dq" set list=i ii iii iv
	if "%%~ni" == "Digitale_Regler_V1.2" set list=1 2 3
	
	echo !list!
	for %%u in (!list!) do (
		%dir_pdf2svg% "%%~pi%%~ni.pdf" "%%~pi%%~ni-%%u.svg" %%u
	)
)
endlocal

cd ..
