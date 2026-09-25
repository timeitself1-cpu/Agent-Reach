<#
.SYNOPSIS
    One-shot setup + launcher for Agent Reach.

.DESCRIPTION
    - Finds the project (this folder, or extracts agent_reach.zip from here / Downloads)
    - Verifies Python 3.10+, creates .venv, installs requirements (skips if unchanged)
    - Ensures Ollama is installed, running, and has the model pulled
    - Runs the pipeline with any options you pass

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_agent_reach.ps1
    .\run_agent_reach.ps1 -Loop -Interval 30
    .\run_agent_reach.ps1 -NoLLM -Top 10 -Sources hackernews,github,arxiv
    .\run_agent_reach.ps1 -JsonOut reports\latest.json
#>
[CmdletBinding()]
param(
    [switch]$Loop,
    [double]$Interval = 30,
    [switch]$NoLLM,
    [int]$Top = 0,
    [string[]]$Sources,
    [string]$JsonOut,
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR")]
    [string]$LogLevel = "INFO",
    [string]$Model = "llama3.1:8b",
    [string]$EmbedModel = "nomic-embed-text",
    [string]$OllamaHost = "http://localhost:11434",
    [switch]$SkipOllamaInstall,
    [switch]$NoPause          # never wait for Enter on errors (for scheduled/automated runs)
)

# "Continue", not "Stop": Windows PowerShell 5.1 can turn native-command stderr (pip, ollama)
# into terminating errors. Every native call checks $LASTEXITCODE; key cmdlets use -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg){ Write-Host "    $msg" -ForegroundColor Yellow }
function Wait-IfInteractive {
    # keep a double-clicked / "Run with PowerShell" window open so the error can be read
    if (-not $NoPause -and [Environment]::UserInteractive -and $Host.Name -eq "ConsoleHost") {
        Write-Host ""
        Read-Host "Press Enter to close" | Out-Null
    }
}
function Fail([string]$msg)       { Write-Host "ERROR: $msg" -ForegroundColor Red; Wait-IfInteractive; exit 1 }

# ------------------------------------------------------------------ 1. locate project
Write-Step "Locating Agent Reach project"
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

function Find-ProjectRoot([string]$start) {
    $candidates = @($start, (Join-Path $start "agent_reach"))
    foreach ($c in $candidates) {
        if ((Test-Path (Join-Path $c "requirements.txt")) -and (Test-Path (Join-Path $c "agent_reach\main.py"))) {
            return (Resolve-Path $c).Path
        }
    }
    return $null
}

$root = Find-ProjectRoot $scriptDir
if (-not $root) {
    $zipCandidates = @(
        (Join-Path $scriptDir "agent_reach.zip"),
        (Join-Path $env:USERPROFILE "Downloads\agent_reach.zip")
    )
    $zip = $zipCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $zip) {
        Fail "Project not found. Put this script inside the extracted agent_reach folder, or next to agent_reach.zip."
    }
    $dest = Split-Path $zip -Parent
    Write-Ok "Extracting $zip -> $dest"
    Expand-Archive -Path $zip -DestinationPath $dest -Force -ErrorAction Stop
    $root = Find-ProjectRoot $dest
    if (-not $root) { Fail "Extracted archive but could not find requirements.txt + agent_reach\main.py" }
}
Set-Location -LiteralPath $root -ErrorAction Stop
Write-Ok "Project: $root"

# ------------------------------------------------------------------ 2. python
Write-Step "Checking Python (3.10+ required)"
$pyCmd = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = "py"; $pyArgs = @("-3")
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and
          ((Get-Command python).Source -notmatch "WindowsApps")) {
    $pyCmd = "python"
} else {
    Fail "Python not found. Install it with:  winget install -e --id Python.Python.3.12   (then reopen PowerShell)"
}
$ver = & $pyCmd @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0 -or -not $ver) { Fail "Could not run Python." }
$parts = $ver.Trim().Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Fail "Python $ver found; 3.10+ required."
}
Write-Ok "Python $ver"

# ------------------------------------------------------------------ 3. venv + deps
Write-Step "Preparing virtual environment"
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    & $pyCmd @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
    Write-Ok "Created .venv"
}
$reqHash = (Get-FileHash (Join-Path $root "requirements.txt") -Algorithm SHA256).Hash
$stamp = Join-Path $root ".venv\.requirements.sha256"
$installed = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
if ($installed -ne $reqHash) {
    Write-Ok "Installing requirements (log: .venv\pip-install.log)..."
    & $venvPy -m pip install --upgrade pip --disable-pip-version-check 2>&1 | ForEach-Object { "$_" } | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "pip self-upgrade failed (continuing with the current pip)" }
    $pipLog = Join-Path $root ".venv\pip-install.log"
    & $venvPy -m pip install -r requirements.txt --disable-pip-version-check 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $pipLog | Where-Object { $_ -match "^(Collecting|Successfully|ERROR|error:)" } |
        ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "---- last lines of $pipLog ----" -ForegroundColor Yellow
        Get-Content $pipLog -Tail 25 | ForEach-Object { Write-Host "    $_" }
        Fail "pip install failed (full log: $pipLog)"
    }
    Set-Content -Path $stamp -Value $reqHash -NoNewline -ErrorAction Stop
    Write-Ok "Dependencies installed"
} else {
    Write-Ok "Dependencies up to date"
}

if (-not (Test-Path (Join-Path $root ".env")) -and (Test-Path (Join-Path $root ".env.example"))) {
    Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env") -ErrorAction Stop
    Write-Ok "Created .env from .env.example (edit AGENT_REACH_CONTACT_EMAIL when convenient)"
}

# ------------------------------------------------------------------ 4. ollama
function Test-Ollama {
    try { return Invoke-RestMethod -Uri "$OllamaHost/api/tags" -TimeoutSec 3 } catch { return $null }
}

if (-not $NoLLM) {
    Write-Step "Checking Ollama at $OllamaHost"
    $tags = Test-Ollama
    if (-not $tags) {
        $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaExe) {
            $fallback = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
            if (Test-Path $fallback) { $ollamaExe = Get-Item $fallback }
        }
        if (-not $ollamaExe -and -not $SkipOllamaInstall) {
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                Write-Warn2 "Ollama not found - installing via winget..."
                winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements | Out-Host
                $fallback = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
                if (Test-Path $fallback) { $ollamaExe = Get-Item $fallback }
            }
        }
        if ($ollamaExe) {
            $exePath = if ($ollamaExe.Source) { $ollamaExe.Source } else { $ollamaExe.FullName }
            Write-Warn2 "Starting 'ollama serve' in the background..."
            Start-Process -FilePath $exePath -ArgumentList "serve" -WindowStyle Hidden
            for ($i = 0; $i -lt 30 -and -not $tags; $i++) { Start-Sleep -Seconds 1; $tags = Test-Ollama }
        }
    }

    if (-not $tags) {
        Write-Warn2 "Ollama is not reachable - continuing with heuristic clustering (--no-llm)."
        $NoLLM = $true
    } else {
        Write-Ok "Ollama is running"
        $names = @($tags.models | ForEach-Object { $_.name })
        $base = $Model.Split(":")[0]
        $hasModel = ($names -contains $Model) -or ($names -contains "$Model`:latest") -or
                    (($Model -notmatch ":") -and ($names | Where-Object { $_.Split(":")[0] -eq $base }))
        if (-not $hasModel) {
            Write-Warn2 "Pulling $Model (one-time, ~4.7 GB)..."
            $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
            $ollamaPath = if ($ollamaCmd) { $ollamaCmd.Source } else { Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
            & $ollamaPath pull $Model
            if ($LASTEXITCODE -ne 0) {
                Write-Warn2 "Model pull failed - continuing with heuristic clustering."
                $NoLLM = $true
            }
        } else {
            Write-Ok "Model $Model available"
        }
        # embedding model for density clustering (small, ~274 MB)
        if (-not $NoLLM) {
            $hasEmbed = ($names -contains $EmbedModel) -or ($names -contains "$EmbedModel`:latest") -or
                        ($names | Where-Object { $_.Split(":")[0] -eq $EmbedModel.Split(":")[0] })
            if (-not $hasEmbed) {
                Write-Warn2 "Pulling $EmbedModel (one-time, ~274 MB)..."
                $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
                $ollamaPath = if ($ollamaCmd) { $ollamaCmd.Source } else { Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
                & $ollamaPath pull $EmbedModel
                if ($LASTEXITCODE -ne 0) { Write-Warn2 "Embedding model pull failed - clustering falls back to lexical grouping." }
            } else {
                Write-Ok "Embedding model $EmbedModel available"
            }
        }
    }
}

# ------------------------------------------------------------------ 5. run
$env:AGENT_REACH_OLLAMA_HOST = $OllamaHost
$env:AGENT_REACH_OLLAMA_MODEL = $Model
$env:AGENT_REACH_EMBED_MODEL = $EmbedModel
$env:PYTHONIOENCODING = "utf-8"

$runArgs = @("-m", "agent_reach", "--log-level", $LogLevel)
if ($Loop)    { $runArgs += @("--loop", "--interval", "$Interval") }
if ($NoLLM)   { $runArgs += "--no-llm" }
if ($Top -gt 0) { $runArgs += @("--top", "$Top") }
if ($Sources) { $runArgs += "--sources"; $runArgs += ($Sources | ForEach-Object { $_ -split "," } | Where-Object { $_ }) }
if ($JsonOut) { $runArgs += @("--json-out", $JsonOut) }

Write-Step ("Running: python " + ($runArgs -join " "))
Write-Host ""
& $venvPy @runArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host ""
    Write-Host "Agent Reach exited with code $code - see the messages above." -ForegroundColor Red
    Wait-IfInteractive
} elseif (-not $Loop) {
    Write-Host ""
    Write-Host "Run complete." -ForegroundColor Green
    Wait-IfInteractive
}
exit $code
