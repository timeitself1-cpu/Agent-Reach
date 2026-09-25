<#
.SYNOPSIS
    Set up and test the Paperclip research node on Windows (PowerShell 5.1+).

.DESCRIPTION
    Modes:
      Check     Cross-process msvcrt GPU lock self-test + Ollama settings (default)
      Wrapper   Run research_agent.cli once on a sample request and validate result.json
      Apply     Create or update the Paperclip agent from paperclip.yaml
      LiveTest  Assign a smoke-test ticket to the agent and wait for done/blocked
      All       Check, Wrapper, Apply, LiveTest in order, stopping at the first failure
      Bridge    Launch python -m paperclip_bridge (needs the PAPERCLIP_* variables that
                a Paperclip heartbeat injects; stdout is left to the bridge)

    OLLAMA_MAX_LOADED_MODELS / OLLAMA_NUM_PARALLEL are read by the Ollama *server*
    when it starts. Setting them here only affects a server started from this
    session: use -RestartOllama to restart it with them, or -PersistOllamaEnv to
    store them for your user so the tray app picks them up on its next start.

.EXAMPLE
    .\run_research_node.ps1 -Mode Check -RestartOllama
    .\run_research_node.ps1 -Mode Wrapper -ResearchRoot C:\dev\research-toolkit
    .\run_research_node.ps1 -Mode All -ResearchRoot C:\dev\research-toolkit -Config .\paperclip.yaml
#>
param(
    [ValidateSet("Check", "Wrapper", "Apply", "LiveTest", "All", "Bridge")]
    [string]$Mode = "Check",
    [string]$BridgeRoot = "",
    [string]$Python = "",
    [string]$ResearchRoot = "",
    [string]$ResearchPython = "",
    [string]$Graph = "research_agent.graph:graph",
    [string]$Config = "",
    [string]$LockPath = "C:\ProgramData\paperclip_bridge\gpu.lock",
    [switch]$PersistOllamaEnv,
    [switch]$RestartOllama,
    [int]$SmokeTimeoutSec = 1800
)

# PowerShell 5.1 turns native stderr into terminating errors under "Stop".
$ErrorActionPreference = "Continue"
$OllamaUrl = "http://127.0.0.1:11434"

function Say([string]$Text, [string]$Color = "Gray") {
    if ($Mode -eq "Bridge") { [Console]::Error.WriteLine($Text) } else { Write-Host $Text -ForegroundColor $Color }
}
function Pass([string]$Text) { Say "  [PASS] $Text" "Green" }
function Fail([string]$Text) { Say "  [FAIL] $Text" "Red" }

# ------------------------------------------------------------------ paths
if (-not $BridgeRoot) {
    foreach ($candidate in @($PSScriptRoot, (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))) {
        if ($candidate -and (Test-Path (Join-Path $candidate "paperclip_bridge\__init__.py"))) {
            $BridgeRoot = $candidate
            break
        }
    }
}
if (-not $BridgeRoot -or -not (Test-Path (Join-Path $BridgeRoot "paperclip_bridge\__init__.py"))) {
    Say "Cannot find the paperclip_bridge package. Pass -BridgeRoot <Agent-Reach checkout>." "Red"
    exit 2
}
if (-not $Python) {
    $venvPython = Join-Path $BridgeRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { $Python = $venvPython } else { $Python = "python" }
}
if (-not $ResearchPython) {
    $ResearchPython = $Python
    if ($ResearchRoot) {
        $toolkitPython = Join-Path $ResearchRoot ".venv\Scripts\python.exe"
        if (Test-Path $toolkitPython) { $ResearchPython = $toolkitPython }
    }
}
if (-not $Config) {
    $Config = Join-Path $PSScriptRoot "paperclip.yaml"
}
$env:PYTHONIOENCODING = "utf-8"
$env:RESEARCH_BRIDGE_GPU_LOCK_PATH = $LockPath

# ------------------------------------------------------------------ Ollama
function Get-OllamaVersion {
    try {
        return (Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 3).version
    } catch {
        return $null
    }
}

function Start-OllamaServer {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Fail "ollama.exe is not on PATH"
        return $false
    }
    Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Get-OllamaVersion) { return $true }
    }
    Fail "ollama serve did not answer on $OllamaUrl within 30 s"
    return $false
}

function Initialize-Ollama {
    Say "Ollama" "Cyan"
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL = "1"
    Pass "session: OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_NUM_PARALLEL=1"
    if ($PersistOllamaEnv) {
        [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
        [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
        Pass "stored for your user account (the tray app uses them after its next start)"
    }

    $version = Get-OllamaVersion
    if ($version -and $RestartOllama) {
        Say "  restarting Ollama with this session's settings..."
        Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
        $version = $null
    }
    if (-not $version) {
        if (-not (Start-OllamaServer)) { return $false }
        Pass "ollama serve $(Get-OllamaVersion) started from this session (settings applied)"
    } elseif (-not $RestartOllama) {
        Say "  [WARN] Ollama $version was already running; it uses the settings it started with." "Yellow"
        Say "         Re-run with -RestartOllama (or -PersistOllamaEnv, then restart the tray app)." "Yellow"
    }

    try {
        $loaded = (Invoke-RestMethod -Uri "$OllamaUrl/api/ps" -TimeoutSec 3).models
        $names = @($loaded | ForEach-Object { $_.name }) -join ", "
        if (-not $names) { $names = "none" }
        Say "  loaded models: $names"
    } catch {
        Say "  (could not read /api/ps)"
    }
    $lmStudio = Get-Process -Name "LM Studio", "lms" -ErrorAction SilentlyContinue
    if ($lmStudio) {
        Say "  [WARN] LM Studio is running. Unload its model: Ollama cannot see LM Studio's VRAM use." "Yellow"
    }
    return $true
}

# ------------------------------------------------------------------ GPU lock
function Test-GpuLock {
    Say "GPU lock ($LockPath)" "Cyan"
    $lockDir = Split-Path -Parent $LockPath
    if (-not (Test-Path $lockDir)) { New-Item -ItemType Directory -Path $lockDir -Force | Out-Null }

    Push-Location $BridgeRoot
    $raw = & $Python -m paperclip_bridge lock-selftest $LockPath
    $code = $LASTEXITCODE
    Pop-Location

    $report = $null
    try { $report = ($raw | Select-Object -Last 1) | ConvertFrom-Json } catch { $report = $null }
    if (-not $report) {
        Fail "self-test produced no report (exit $code): $raw"
        return $false
    }
    if ($report.backend -ne "msvcrt") {
        Fail "expected the msvcrt backend on Windows, got '$($report.backend)'"
        return $false
    }
    if ($code -ne 0 -or -not $report.ok) {
        foreach ($f in $report.failures) { Fail $f }
        return $false
    }
    Pass "msvcrt: a second process is blocked while the lock is held"
    Pass "msvcrt: a waiting process gets the lock after release"
    Pass "msvcrt: the OS frees the lock when the holding process is killed"
    return $true
}

# ------------------------------------------------------------------ wrapper
function Test-Wrapper {
    Say "research_agent.cli" "Cyan"
    if (-not $ResearchRoot -or -not (Test-Path (Join-Path $ResearchRoot "research_agent\cli.py"))) {
        Fail "pass -ResearchRoot <folder that contains research_agent\cli.py>"
        return $false
    }
    Say "  runs the real graph once, outside the GPU lock: stop Paperclip heartbeats first." "Yellow"
    $work = Join-Path $env:TEMP ("research_node_" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    $request = Join-Path $work "request.json"
    $result = Join-Path $work "result.json"
    $payload = @{
        schema_version = "1.0"
        request_id     = "wrapper-test"
        title          = "Wrapper test: one recent AI paper on agent tool use"
        instructions   = "Summarise one paper in three sentences."
        created_at     = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json
    # UTF-8 without BOM: Windows PowerShell's Set-Content -Encoding UTF8 writes a BOM.
    [IO.File]::WriteAllText($request, $payload, (New-Object Text.UTF8Encoding($false)))

    $env:RESEARCH_AGENT_GRAPH = $Graph
    Push-Location $ResearchRoot
    & $ResearchPython -m research_agent.cli --in $request --out $result | Out-Host
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Fail "research_agent.cli exited with $code (files in $work)"
        return $false
    }

    Push-Location $BridgeRoot
    $check = & $Python -m paperclip_bridge validate $result
    $valid = $LASTEXITCODE
    Pop-Location
    if ($valid -ne 0) {
        Fail "result.json does not match the bridge contract (files in $work)"
        return $false
    }
    $summary = $check | ConvertFrom-Json
    if ($summary.status -eq "error") {
        $err = (Get-Content -Raw -Path $result | ConvertFrom-Json).error
        Fail "graph reported an error: $err"
        return $false
    }
    Pass "result.json is valid: status=$($summary.status), papers=$($summary.papers), handoff=$($summary.handoff)"
    Say "  files: $work"
    return $true
}

# ------------------------------------------------------------------ Paperclip
function Invoke-Setup([string]$Command) {
    Say "Paperclip: $Command ($Config)" "Cyan"
    if (-not (Test-Path $Config)) {
        Fail "config not found: $Config"
        return $false
    }
    Push-Location $BridgeRoot
    if ($Command -eq "smoke") {
        & $Python -m paperclip_bridge.setup_agent smoke --config $Config --timeout $SmokeTimeoutSec | Out-Host
    } else {
        & $Python -m paperclip_bridge.setup_agent $Command --config $Config | Out-Host
    }
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Fail "setup_agent $Command exited with $code"
        return $false
    }
    return $true
}

# ------------------------------------------------------------------ main
if ($Mode -eq "Bridge") {
    # Top level, not a function: the bridge's stdout must reach Paperclip unchanged.
    if (-not $env:PAPERCLIP_RUN_ID -or -not $env:PAPERCLIP_API_KEY) {
        Say "PAPERCLIP_RUN_ID / PAPERCLIP_API_KEY are not set. The bridge is started by a Paperclip" "Yellow"
        Say "heartbeat, which injects them. For a live test use -Mode LiveTest instead." "Yellow"
    }
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL = "1"
    Set-Location $BridgeRoot
    & $Python -m paperclip_bridge
    exit $LASTEXITCODE
}

Say "Research node: $Mode  (bridge: $BridgeRoot, python: $Python)" "White"
$steps = @()
switch ($Mode) {
    "Check"    { $steps = @("lock", "ollama") }
    "Wrapper"  { $steps = @("wrapper") }
    "Apply"    { $steps = @("apply") }
    "LiveTest" { $steps = @("smoke") }
    "All"      { $steps = @("lock", "ollama", "wrapper", "apply", "smoke") }
}
$failed = @()
foreach ($step in $steps) {
    $ok = $false
    switch ($step) {
        "ollama"  { $ok = Initialize-Ollama }
        "lock"    { $ok = Test-GpuLock }
        "wrapper" { $ok = Test-Wrapper }
        "apply"   { $ok = Invoke-Setup "apply" }
        "smoke"   { $ok = Invoke-Setup "smoke" }
    }
    if (-not $ok) {
        $failed += $step
        # Check reports every problem; the other modes stop at the first failure.
        if ($Mode -ne "Check") {
            Say "Stopped at step '$step'." "Red"
            exit 1
        }
    }
}
if ($failed.Count -gt 0) {
    Say ("Failed: " + ($failed -join ", ")) "Red"
    exit 1
}
Say "All steps passed." "Green"
exit 0
