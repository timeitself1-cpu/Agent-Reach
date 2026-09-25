<#
.SYNOPSIS
    Agent Reach cloud handoff: sync this folder to GitHub, dispatch the cloud-runner
    workflow, confirm the run is live, and stream its logs.

.DESCRIPTION
    1. Verifies git + GitHub CLI and your GitHub login (you sign in yourself if needed).
    2. Initialises / reuses the git repo, sets origin, bases on origin/<branch> if it exists.
    3. Enforces .gitignore, refuses to commit secret-looking files or content.
    4. Commits with the handoff message and pushes; verifies the remote SHA matches.
    5. Dispatches .github/workflows/cloud-runner.yml with the runtime payload.
    6. Polls until the new run is in_progress (or finished), prints its URL, streams logs.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_cloud_handoff.ps1
    .\run_cloud_handoff.ps1 -NoLLM -ExtraArgs "--top 10"
    .\run_cloud_handoff.ps1 -Model llama3.1:8b   # slower, larger labelling model
    .\run_cloud_handoff.ps1 -NoWatch
    .\run_cloud_handoff.ps1 -Mirror          # also delete remote files that don't exist locally
#>
[CmdletBinding()]
param(
    [string]$RepoUrl = "https://github.com/timeitself1-cpu/Agent-Reach.git",
    [string]$Branch = "main",
    [string]$Workflow = "cloud-runner.yml",
    [string]$CommitMessage = "feat(cloud-handoff): transfer state to cloud runner [ci skip/trigger]",
    [string]$SessionStateId = ("handoff-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$Entrypoint = "agent_reach",
    [string]$ExtraArgs = "",
    [ValidateSet("llama3.2:3b", "llama3.1:8b")]
    [string]$Model = "llama3.2:3b",   # cloud labelling model; the CPU runner generates ~2-3x faster with 3b
    [switch]$NoLLM,
    [switch]$Mirror,
    [switch]$NoWatch,
    [switch]$PushOnly,       # commit + push + verify, but do not dispatch a cloud run
    [switch]$KeepWorkflow,   # kept for compatibility: an existing cloud-runner.yml is never overwritten now
    [switch]$ResetWorkflow,  # overwrite .github/workflows/cloud-runner.yml with the embedded version
    [int]$StartTimeoutSec = 180
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns any native-command stderr output (e.g. gh
# "not logged in", git fetch/push progress) into a terminating error when combined with redirection.
# Every native call below checks $LASTEXITCODE explicitly; key cmdlets use -ErrorAction Stop.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

function Step([string]$m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "    $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "    $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

function Invoke-Git {
    # plain $args (no param block) so flags like -v / -m are passed to git, not bound as PowerShell common parameters
    $out = & git @args 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { Fail ("git " + ($args -join " ") + " failed:`n" + ($out | Out-String)) }
    return $out
}

# Embedded copy of .github/workflows/cloud-runner.yml, written if the file is missing.
$CloudRunnerYaml = @'
name: cloud-runner

# Cloud takeover for Agent Reach: runs the full pipeline on a GitHub-hosted runner
# (Ollama on CPU, llama3.2:3b by default) and carries the SQLite velocity history between
# runs through the Actions cache. Standard runners have no GPU: llama3.1:8b generates at
# ~5 tok/s there, llama3.2:3b is roughly 2-3x faster.

on:
  workflow_dispatch:
    inputs:
      session_state_id:
        description: "Handoff session id (echoed into logs and the run summary)"
        required: false
        default: "manual"
      entrypoint:
        description: "Python module to run"
        required: false
        default: "agent_reach"
      extra_args:
        description: "Extra CLI args, e.g. --top 10 --sources hackernews github"
        required: false
        default: ""
      no_llm:
        description: "Skip Ollama and use deterministic clustering"
        type: boolean
        required: false
        default: false
      model:
        description: "Ollama labelling model (3b is ~2-3x faster on the CPU runner)"
        type: choice
        required: false
        default: "llama3.2:3b"
        options:
          - "llama3.2:3b"
          - "llama3.1:8b"
  repository_dispatch:
    types: [cloud-handoff]

concurrency:
  group: agent-reach-cloud-runner   # one run at a time so the SQLite history never forks
  cancel-in-progress: false

permissions:
  contents: read

env:
  MODEL: ${{ github.event.inputs.model || github.event.client_payload.model || 'llama3.2:3b' }}
  AGENT_REACH_OLLAMA_MODEL: ${{ github.event.inputs.model || github.event.client_payload.model || 'llama3.2:3b' }}
  EMBED_MODEL: nomic-embed-text
  SESSION_STATE_ID: ${{ github.event.inputs.session_state_id || github.event.client_payload.session_state_id || 'dispatch' }}
  ENTRYPOINT: ${{ github.event.inputs.entrypoint || github.event.client_payload.entrypoint || 'agent_reach' }}
  EXTRA_ARGS: ${{ github.event.inputs.extra_args || github.event.client_payload.extra_args || '' }}
  NO_LLM: ${{ github.event.inputs.no_llm || github.event.client_payload.no_llm || 'false' }}
  AGENT_REACH_DB_PATH: state/agent_reach.db
  AGENT_REACH_CONTACT_EMAIL: ${{ vars.AGENT_REACH_CONTACT_EMAIL || 'agent-reach@example.invalid' }}
  # CPU-only runner: give the model room per call and bound the LLM workload
  AGENT_REACH_OLLAMA_TIMEOUT_S: "900"
  AGENT_REACH_MAX_ITEMS_FOR_LLM: "150"   # grouping is embedding-based; LLM only labels real clusters
  AGENT_REACH_LLM_MAX_RETRIES: "1"
  PYTHONIOENCODING: utf-8
  PYTHONUNBUFFERED: "1"

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - name: Session banner
        run: |
          echo "Agent Reach cloud handoff"
          echo "  session_state_id : $SESSION_STATE_ID"
          echo "  entrypoint       : $ENTRYPOINT"
          echo "  extra_args       : $EXTRA_ARGS"
          echo "  no_llm           : $NO_LLM"
          echo "  model            : $MODEL"
          echo "  commit           : $GITHUB_SHA"

      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements.txt

      - name: Install Python dependencies
        run: python -m pip install --disable-pip-version-check -r requirements.txt

      - name: Restore velocity history (SQLite)
        uses: actions/cache/restore@v5
        with:
          path: state
          key: agent-reach-db-${{ github.run_id }}
          restore-keys: agent-reach-db-

      - name: Restore Ollama model cache
        if: env.NO_LLM != 'true'
        id: model-cache
        uses: actions/cache/restore@v5
        with:
          path: ~/.ollama/models
          key: ollama-models-${{ env.MODEL }}-${{ env.EMBED_MODEL }}

      - name: Install and start Ollama
        if: env.NO_LLM != 'true'
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          sudo systemctl stop ollama 2>/dev/null || true
          mkdir -p "$HOME/.ollama/models"
          OLLAMA_MODELS="$HOME/.ollama/models" nohup ollama serve > "$RUNNER_TEMP/ollama.log" 2>&1 &
          for i in $(seq 1 60); do
            curl -sf http://127.0.0.1:11434/api/tags > /dev/null && break
            sleep 1
          done
          curl -sf http://127.0.0.1:11434/api/tags > /dev/null || { cat "$RUNNER_TEMP/ollama.log"; exit 1; }
          OLLAMA_MODELS="$HOME/.ollama/models" ollama pull "$MODEL"
          OLLAMA_MODELS="$HOME/.ollama/models" ollama pull "$EMBED_MODEL"
          ollama list

      - name: Save Ollama model cache
        if: env.NO_LLM != 'true' && steps.model-cache.outputs.cache-hit != 'true'
        uses: actions/cache/save@v5
        with:
          path: ~/.ollama/models
          key: ollama-models-${{ env.MODEL }}-${{ env.EMBED_MODEL }}

      - name: Run Agent Reach
        run: |
          mkdir -p state reports
          ARGS=(--json-out "reports/report-${GITHUB_RUN_ID}.json" --log-level INFO)
          if [ "$NO_LLM" = "true" ]; then ARGS+=(--no-llm); fi
          # shellcheck disable=SC2206
          EXTRA=($EXTRA_ARGS)
          set -o pipefail
          # stderr (all progress logs) streams live to the job log AND is saved to pipeline.log;
          # stdout (the final executive report) goes to the log and report.txt
          python -u -m "$ENTRYPOINT" "${ARGS[@]}" "${EXTRA[@]}" \
            2> >(tee reports/pipeline.log >&2) | tee reports/report.txt

      - name: Publish run summary
        if: always()
        run: |
          {
            echo "## Agent Reach cloud run"
            echo ""
            echo "- Session: \`$SESSION_STATE_ID\`"
            echo "- Commit: \`$GITHUB_SHA\`"
            echo "- Model: \`$MODEL\` (no_llm: $NO_LLM)"
            echo ""
            echo '```text'
            if [ -f reports/report.txt ]; then cat reports/report.txt; else echo "no report produced"; fi
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"

      - name: Save velocity history (SQLite)
        if: always()
        uses: actions/cache/save@v5
        with:
          path: state
          key: agent-reach-db-${{ github.run_id }}

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v6
        with:
          name: agent-reach-report-${{ github.run_id }}
          path: |
            reports/
            ${{ runner.temp }}/ollama.log
          if-no-files-found: warn
          retention-days: 14
'@

$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -LiteralPath $root -ErrorAction Stop
$ownerRepo = ($RepoUrl -replace '^https://github.com/', '' -replace '\.git$', '').Trim('/')

# ------------------------------------------------------------------ 0. tools + auth
Step "Checking tools"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git not found. Install it:  winget install -e --id Git.Git   (then reopen PowerShell)"
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn "GitHub CLI not found - installing via winget..."
        winget install -e --id GitHub.cli --accept-package-agreements --accept-source-agreements | Out-Host
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Fail "GitHub CLI (gh) missing. Install: winget install -e --id GitHub.cli" }
}
Ok ("git " + ((git --version) -replace 'git version ', '') + " | " + ((gh --version | Select-Object -First 1)))

& gh auth status --hostname github.com *> $null
if ($LASTEXITCODE -ne 0) {
    Warn "Not signed in to GitHub CLI - a browser sign-in will open. Complete it yourself, then the script continues."
    & gh auth login --hostname github.com --git-protocol https --web --scopes "repo,workflow"
    if ($LASTEXITCODE -ne 0) { Fail "GitHub sign-in did not complete." }
}
& gh auth setup-git *> $null   # let git push use the gh credential
$scopes = (& gh auth status --hostname github.com 2>&1 | ForEach-Object { "$_" } | Out-String)
if ($scopes -match "Token scopes:" -and $scopes -notmatch "workflow") {
    Warn "Your token lacks the 'workflow' scope (needed to push .github/workflows). Refreshing..."
    & gh auth refresh --hostname github.com --scopes "repo,workflow"
}

& gh repo view $ownerRepo --json name *> $null
if ($LASTEXITCODE -ne 0) { Fail "Cannot access $ownerRepo with your GitHub login. Check the URL and that the repo exists." }
Ok "Access to $ownerRepo confirmed"

# ------------------------------------------------------------------ 1. repo + remote
Step "Preparing git repository"
if (-not (Test-Path (Join-Path $root ".git"))) {
    Invoke-Git init | Out-Null
    Ok "Initialised new repository"
}
$remotes = (& git remote) -split "`n"
if ($remotes -contains "origin") {
    $current = (& git remote get-url origin).Trim()
    if ($current -ne $RepoUrl) { Invoke-Git remote set-url origin $RepoUrl | Out-Null; Ok "origin: $current -> $RepoUrl" }
} else {
    Invoke-Git remote add origin $RepoUrl | Out-Null
}
Invoke-Git remote -v | ForEach-Object { Write-Host "    $_" }

& git config user.name *> $null
if ($LASTEXITCODE -ne 0) {
    $login = (& gh api user --jq .login).Trim()
    Invoke-Git config user.name $login | Out-Null
    Invoke-Git config user.email "$login@users.noreply.github.com" | Out-Null
    Ok "Commit identity set to $login (repo-local)"
}

& git fetch origin 2>&1 | Out-Null
& git rev-parse --verify --quiet "refs/remotes/origin/$Branch" *> $null
$remoteHasBranch = ($LASTEXITCODE -eq 0)
& git rev-parse --verify --quiet HEAD *> $null
$hasLocalCommits = ($LASTEXITCODE -eq 0)

if ($remoteHasBranch -and -not $hasLocalCommits) {
    # Adopt remote history without touching the working tree, then overlay local files.
    Invoke-Git symbolic-ref HEAD "refs/heads/$Branch" | Out-Null
    Invoke-Git reset --mixed "origin/$Branch" | Out-Null
    if (-not $Mirror) {
        $deleted = & git ls-files --deleted
        foreach ($f in $deleted) { if ($f) { Invoke-Git checkout -- $f | Out-Null } }
        if ($deleted) { Ok ("Kept " + @($deleted).Count + " remote-only file(s) (use -Mirror to delete them)") }
    }
    Ok "Based on origin/$Branch"
} elseif ($remoteHasBranch -and $hasLocalCommits) {
    Invoke-Git checkout -B $Branch | Out-Null
    & git merge-base --is-ancestor "origin/$Branch" HEAD
    if ($LASTEXITCODE -ne 0) {
        Warn "Local branch is behind/diverged from origin/$Branch - rebasing local commits on top"
        Invoke-Git pull --rebase origin $Branch | Out-Null
    }
} else {
    Invoke-Git checkout -B $Branch | Out-Null
    Ok "Remote has no '$Branch' yet - it will be created"
}

# ------------------------------------------------------------------ 2. secret hygiene
Step "Staging files (secrets excluded)"
if (-not (Test-Path ".gitignore")) { Fail ".gitignore missing - refusing to stage without it." }
foreach ($required in @(".env", "*.db", ".venv/")) {
    if (-not (Select-String -Path ".gitignore" -SimpleMatch $required -Quiet)) { Add-Content -LiteralPath ".gitignore" -Value $required -ErrorAction Stop; Warn "Added '$required' to .gitignore" }
}
# untrack anything ignored that was committed earlier
$trackedIgnored = & git ls-files -ci --exclude-standard
foreach ($f in $trackedIgnored) { if ($f) { Invoke-Git rm --cached --quiet -- $f | Out-Null; Warn "Untracked ignored file: $f" } }

Invoke-Git add -A | Out-Null
$staged = @(& git diff --cached --name-only) | Where-Object { $_ }

$badNames = $staged | Where-Object { $_ -match '(^|/)\.env($|\.)' -and $_ -notmatch '\.env\.example$' -or $_ -match '\.(pem|key|p12|pfx|db|db-wal|db-shm)$' }
if ($badNames) { Fail ("Refusing to commit secret/state files:`n  " + ($badNames -join "`n  ")) }

$secretPattern = '(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,}|sk-[A-Za-z0-9]{32,}|sk-ant-[A-Za-z0-9\-_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)'
$leaks = @()
foreach ($f in $staged) {
    if ((Test-Path -LiteralPath $f) -and ((Get-Item -LiteralPath $f -Force).Length -lt 2MB)) {
        $hit = Select-String -LiteralPath $f -Pattern $secretPattern -List -ErrorAction SilentlyContinue
        if ($hit) { $leaks += "$f (line $($hit.LineNumber))" }
    }
}
if ($leaks) { Fail ("Possible secrets in staged files - fix before pushing:`n  " + ($leaks -join "`n  ")) }
Ok ("Staged " + $staged.Count + " changed path(s); secret scan clean")
$wfPath = Join-Path $root ".github/workflows/$Workflow"
$wfExpected = $CloudRunnerYaml.Replace("`r`n", "`n") + "`n"
if ($Workflow -eq "cloud-runner.yml") {
    $wfCurrent = if (Test-Path -LiteralPath $wfPath) { [System.IO.File]::ReadAllText($wfPath).Replace("`r`n", "`n") } else { $null }
    # the repo's workflow is the source of truth; the embedded copy only bootstraps a missing file
    if ($wfCurrent -and $wfCurrent -ne $wfExpected -and -not $ResetWorkflow) {
        Warn "Keeping existing .github/workflows/$Workflow (differs from the embedded copy; -ResetWorkflow overwrites it)"
    } elseif ($wfCurrent -ne $wfExpected) {
        New-Item -ItemType Directory -Force -Path (Split-Path $wfPath -Parent) -ErrorAction Stop | Out-Null
        try {
            [System.IO.File]::WriteAllText($wfPath, $wfExpected, (New-Object System.Text.UTF8Encoding($false)))
        } catch {
            Fail "Could not write ${wfPath}: $($_.Exception.Message)"
        }
        Invoke-Git add -- ".github/workflows/$Workflow" | Out-Null
        $staged = @(& git diff --cached --name-only) | Where-Object { $_ }
        if ($wfCurrent) { Ok "Updated .github/workflows/$Workflow to the version embedded in this script" }
        else { Ok "Wrote .github/workflows/$Workflow from the embedded template" }
    }
} elseif (-not (Test-Path -LiteralPath $wfPath)) {
    Fail ".github/workflows/$Workflow not found in $root"
}

# ------------------------------------------------------------------ 3. commit + push
Step "Committing and pushing"
if ($staged.Count -gt 0) {
    Invoke-Git commit -m $CommitMessage | Out-Null
    Ok ("Committed " + (& git rev-parse --short HEAD))
} else {
    Ok "Nothing new to commit - pushing current HEAD"
}
$pushOut = & git push -u origin $Branch 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { Fail ("git push failed:`n" + ($pushOut | Out-String)) }
$localSha = (& git rev-parse HEAD).Trim()
$remoteSha = ((& git ls-remote origin "refs/heads/$Branch") -split "\s+")[0]
if ($localSha -ne $remoteSha) { Fail "Remote $Branch is $remoteSha but local HEAD is $localSha" }
Ok "Remote verified: $Branch @ $($localSha.Substring(0,7)) on $ownerRepo"

# ------------------------------------------------------------------ 4. dispatch
if ($PushOnly) {
    Write-Host ""
    Write-Host "================ PUSH SUMMARY ================" -ForegroundColor Cyan
    Write-Host ("  Repository : https://github.com/$ownerRepo")
    Write-Host ("  Branch     : $Branch")
    Write-Host ("  Commit     : $localSha")
    Write-Host ("  Message    : " + ((& git log -1 --format=%s) -join ""))
    Write-Host ("  Start a run: gh workflow run $Workflow --repo $ownerRepo   (or the Run workflow button)")
    Write-Host "==============================================" -ForegroundColor Cyan
    exit 0
}
Step "Dispatching $Workflow"
$defaultBranch = (& gh repo view $ownerRepo --json defaultBranchRef --jq .defaultBranchRef.name).Trim()
if ($defaultBranch -and $defaultBranch -ne $Branch) {
    Warn "Repo default branch is '$defaultBranch'. GitHub only lists workflows that exist on the default branch; dispatching with --ref $Branch."
}
$dispatchedAt = (Get-Date).ToUniversalTime().AddSeconds(-5)
$noLlmValue = if ($NoLLM) { "true" } else { "false" }

$wfArgs = @("workflow", "run", $Workflow, "--repo", $ownerRepo, "--ref", $Branch,
            "-f", "session_state_id=$SessionStateId", "-f", "entrypoint=$Entrypoint",
            "-f", "extra_args=$ExtraArgs", "-f", "no_llm=$noLlmValue", "-f", "model=$Model")
$ok = $false
for ($i = 1; $i -le 6 -and -not $ok; $i++) {   # a just-pushed workflow can take a few seconds to register
    $out = & gh @wfArgs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -eq 0) { $ok = $true } else { Warn "Dispatch not accepted yet ($(($out | Out-String).Trim())); retry $i/6"; Start-Sleep -Seconds 5 }
}
if (-not $ok) { Fail "Could not dispatch $Workflow. Check that Actions is enabled: https://github.com/$ownerRepo/settings/actions" }
Ok "Dispatched (session_state_id=$SessionStateId, entrypoint=$Entrypoint, no_llm=$noLlmValue, model=$Model)"

# ------------------------------------------------------------------ 5. confirm running
Step "Waiting for the run to start"
$run = $null
$deadline = (Get-Date).AddSeconds($StartTimeoutSec)
while ((Get-Date) -lt $deadline) {
    $json = & gh run list --repo $ownerRepo --workflow $Workflow --event workflow_dispatch --branch $Branch --limit 5 `
        --json databaseId,status,conclusion,url,createdAt,headSha 2>$null
    if ($LASTEXITCODE -eq 0 -and $json) {
        $candidate = ($json | ConvertFrom-Json) |
            Where-Object { ([datetime]$_.createdAt).ToUniversalTime() -ge $dispatchedAt -and $_.headSha -eq $localSha } |
            Select-Object -First 1
        if ($candidate) {
            $run = $candidate
            if ($run.status -in @("in_progress", "completed")) { break }
            Write-Host ("    status: " + $run.status) -ForegroundColor DarkGray
        }
    }
    Start-Sleep -Seconds 5
}
if (-not $run) { Fail "No run appeared within $StartTimeoutSec s. Open https://github.com/$ownerRepo/actions" }

Write-Host ""
Write-Host "================ CLOUD HANDOFF SUMMARY ================" -ForegroundColor Cyan
Write-Host ("  Repository : https://github.com/$ownerRepo")
Write-Host ("  Commit     : $localSha")
Write-Host ("  Session    : $SessionStateId")
Write-Host ("  Run ID     : " + $run.databaseId)
Write-Host ("  Status     : " + $run.status + $(if ($run.conclusion) { " / " + $run.conclusion } else { "" }))
Write-Host ("  Run URL    : " + $run.url)
Write-Host ("  Logs       : gh run view " + $run.databaseId + " --repo $ownerRepo --log")
Write-Host "=======================================================" -ForegroundColor Cyan

if ($run.status -ne "in_progress" -and $run.status -ne "completed") {
    Warn "Run is still '$($run.status)' (queued runners can take a while). Watch it at the URL above."
}

# ------------------------------------------------------------------ 6. monitor
if (-not $NoWatch) {
    Step "Streaming run progress (Ctrl+C stops watching; the cloud run keeps going)"
    & gh run watch $run.databaseId --repo $ownerRepo --interval 15 --exit-status
    $watchExit = $LASTEXITCODE
    Step "Final job log tail"
    & gh run view $run.databaseId --repo $ownerRepo --log 2>$null | Select-Object -Last 60
    Write-Host ""
    Write-Host ("Report artifact: gh run download " + $run.databaseId + " --repo $ownerRepo") -ForegroundColor Cyan
    exit $watchExit
}
exit 0
