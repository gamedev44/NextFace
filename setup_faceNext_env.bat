@echo off
echo [*] Nuking old env if it exists...
conda deactivate
conda remove -n faceNext --all -y

echo [*] Creating fresh env with Python 3.9...
conda create -n faceNext python=3.9 -y

echo [*] Activating env...
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" faceNext

echo [*] Installing core packages...
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge opencv -y
conda install -c 1adrianb face_alignment=1.2.0 -y
conda install -c anaconda h5py -y
pip install redner-gpu

echo [*] Installing mediapipe globally using py launcher...
py -m pip install --user mediapipe

echo [*] All done.

REM Ask the user to browse for the project directory to open
echo [*] Please choose the project directory to open:
for /f "delims=" %%a in ('powershell -command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; if($f.ShowDialog() -eq 'OK') { $f.SelectedPath }"') do set "projectPath=%%a"

echo [*] Opening folder: %projectPath%
cd /d "%projectPath%"
start .
cmd /k