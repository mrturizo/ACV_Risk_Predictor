# Script para recrear el repositorio desde cero
Write-Host "=== Recreando repositorio Git ===" -ForegroundColor Cyan

# 1. Eliminar .git si existe
if (Test-Path .git) {
    Write-Host "Eliminando .git existente..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .git
}

# 2. Inicializar nuevo repositorio
Write-Host "Inicializando nuevo repositorio..." -ForegroundColor Yellow
git init

# 3. Configurar usuario
Write-Host "Configurando usuario como mrturizo..." -ForegroundColor Yellow
git config user.name "mrturizo"
git config user.email "martin.romero@uao.edu.co"

# 4. Verificar configuración
Write-Host "`n=== Configuración ===" -ForegroundColor Cyan
Write-Host "Usuario: $(git config user.name)"
Write-Host "Email: $(git config user.email)"

# 5. Agregar todos los archivos
Write-Host "`nAgregando archivos..." -ForegroundColor Yellow
git add .

# 6. Hacer commit inicial
Write-Host "Creando commit inicial..." -ForegroundColor Yellow
git commit -m "Initial commit: ACV Risk Predictor - Web and Desktop app"

# 7. Verificar commit
Write-Host "`n=== Commit creado ===" -ForegroundColor Cyan
git log --pretty=format:"%h - %an <%ae> - %s" -1

# 8. Renombrar rama a main
Write-Host "`nRenombrando rama a main..." -ForegroundColor Yellow
git branch -M main

# 9. Conectar con GitHub
Write-Host "`nConectando con GitHub..." -ForegroundColor Yellow
git remote add origin https://github.com/mrturizo/ACV_Risk_Predictor.git

# 10. Verificar remote
Write-Host "`n=== Remote configurado ===" -ForegroundColor Cyan
git remote -v

Write-Host "`n=== Listo para hacer push ===" -ForegroundColor Green
Write-Host "Ejecuta: git push -f -u origin main" -ForegroundColor Yellow

