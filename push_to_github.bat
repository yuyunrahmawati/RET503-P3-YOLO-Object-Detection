@echo off
setlocal
echo ========================================================
echo   UPLOAD / PUSH REPOSITORI KE GITHUB - YOLO OBJECT DETECTION
echo ========================================================
echo.
set "GIT_EXE=C:\Users\LENOVO\AppData\Local\Programs\Git\cmd\git.exe"

set /p REPO_URL="Masukkan URL Repository GitHub Anda (contoh: https://github.com/yuyunrahmawati/RET503-P3-YOLO.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] URL tidak boleh kosong!
    pause
    exit /b 1
)

echo.
echo [1/3] Menyiapkan remote origin...
"%GIT_EXE%" remote remove origin 2>nul
"%GIT_EXE%" remote add origin %REPO_URL%

echo.
echo [2/3] Memastikan branch main...
"%GIT_EXE%" branch -M main

echo.
echo [3/3] Mengunggah (push) ke GitHub...
"%GIT_EXE%" push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   [BERHASIL] Repositori berhasil diunggah ke GitHub!
    echo ========================================================
) else (
    echo.
    echo [ERROR] Gagal melakukan push. Pastikan repository sudah dibuat di GitHub dan koneksi internet aktif.
)
pause
