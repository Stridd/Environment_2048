SET anaconda_path=E:/Programs/Anaconda3/

call %anaconda_path%/Scripts/activate.bat %anaconda_path%
echo Activating Cython Environment
call conda activate torchRL

call jupyter-notebook Tests.ipynb