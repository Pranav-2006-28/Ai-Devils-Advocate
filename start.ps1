# AI Devil's Advocate - Start All Servers
# Run this script from the project root: ai-devils-advocate\

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== AI Devil's Advocate Startup ===" -ForegroundColor Cyan

# Kill any existing processes on port 8000
Write-Host "Freeing port 8000..." -ForegroundColor Yellow
$portPid = (netstat -ano | findstr ":8000 " | Select-String "LISTENING" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1)
if ($portPid) {
    Stop-Process -Id $portPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Start Backend
Write-Host "Starting FastAPI backend on port 8000..." -ForegroundColor Green
$backendPath = Join-Path $root "backend"
$uvicorn = Join-Path $root "venv\Scripts\uvicorn.exe"
Start-Process -FilePath $uvicorn -ArgumentList "app.main:app --host 0.0.0.0 --port 8000" -WorkingDirectory $backendPath -WindowStyle Normal

Write-Host "Backend starting... (model load takes ~30s)" -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Frontend
Write-Host "Starting Vite frontend on port 5173..." -ForegroundColor Green
$frontendPath = Join-Path $root "frontend"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k npm run dev" -WorkingDirectory $frontendPath -WindowStyle Normal

Write-Host ""
Write-Host "=== Both servers launched! ===" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000/health" -ForegroundColor White
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Wait ~30 seconds for the AI model to finish loading before uploading a PDF." -ForegroundColor Yellow
