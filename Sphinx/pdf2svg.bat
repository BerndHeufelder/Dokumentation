@ECHO OFF

REM change directory to ...\Sphinx\source
set dir_source=%~dp0source
pushd %dir_source%
set dir_pdf2svg="..\..\SW-Tools\pdf2svg\dist-32bits\pdf2svg.exe"
setlocal enabledelayedexpansion

for /r %%i in (*.pdf) do (
	REM %%~pi path of current file in loop, %%~ni name of current file in loop, %%~xi file extension (.pdf)
	
	set list=
	if "%%~ni" == "PMSM_Modell_in_dq" set list=7 8 9
	if "%%~ni" == "Digitale_Regler_V1.2" set list=1 2 3
REM    if "%%~ni" == "federn_lin_rot" set list=1 2 3
REM    if "%%~ni" == "Continuous_Acceleration_and_Duty_Time" set list=1 2 3
	if "%%~ni" == "input_shaping" set list=1
REM    if "%%~ni" == "IntroductionToRobotics" set list=164 165

	echo "%%~ni / pages to convert: !list!"
	for %%u in (!list!) do (
		echo "       %%u: %%~ni-%%u.svg"
		%dir_pdf2svg% "%%~pi%%~ni.pdf" "%%~pi%%~ni-%%u.svg" %%u
	)
	echo "--------------------------------------------------"
)
endlocal

cd ..
