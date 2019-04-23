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

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=BesiProjekteundzugehoerigesWissenswertes

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
