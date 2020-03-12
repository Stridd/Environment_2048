SET anaconda_path=E:/Programs/Anaconda3/

call %anaconda_path%/Scripts/activate.bat %anaconda_path%
echo Activating Cython Environment
call conda activate cython

call echo Deleting older files
call del /f  Environment_2048.cpp
call del /f  Environment_2048.cp37-win_amd64.pyd
call rmdir /Q /S build

call echo Building project
rem call python setup.py build_ext --inplace

call echo Done building project