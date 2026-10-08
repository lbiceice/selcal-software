# SelCal Windows check: locked environment, full test suite, end-to-end acceptance.
#
# From the repository root, in Windows PowerShell 5.1 or PowerShell 7:
#   powershell -ExecutionPolicy Bypass -File scripts\windows_check.ps1 -Python "py -3.12"
# Optional: -Index <PyPI mirror URL>, -Records <records made elsewhere by the same version>,
# -LegacyRecords <records made by an earlier version>, -Wheelhouse <offline wheels folder>.
# -TestWorkRoot <short physical folder>, or SELCAL_WINDOWS_TEST_ROOT, selects the test scratch
# root. By default it is SelCal-Test on the repository's volume (never the system TEMP folder).
#
# Each run writes a new folder and a results ZIP next to the repository. Its short, owned test
# environment remains at the path recorded in runner-summary.json; test-work-evidence.zip keeps
# the test artifacts but excludes that environment. Failures still produce a
# summary and an archive. Exit 1 means at least one check failed; exit 0 means all steps passed.

param(
    [string]$Python = "py -3.12",
    [string]$Index = "",
    [string]$Records = "",
    [string]$LegacyRecords = "",
    [string]$Wheelhouse = "",
    [string]$TestWorkRoot = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # progress bars make archive cmdlets very slow
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = (Get-Date -Format "yyyyMMdd-HHmmss-fff") + "-" + [guid]::NewGuid().ToString("N").Substring(0, 6)
$out = Join-Path (Split-Path $repo -Parent) "windows-check-$stamp"
New-Item -ItemType Directory -Path $out | Out-Null
$log = Join-Path $out "steps.log"
$script:failureCount = 0
$script:failureNames = ""
$script:steps = New-Object System.Collections.ArrayList
# R14 Windows return (W14-01): the first failed environment step blocks every step that needs
# that environment, so one root cause is not reported as a chain of unrelated failures.
$script:setupFailure = $null
$script:sourceManifestStatus = "not supplied"
$script:sourceManifestCount = 0
$script:sourceManifestSHA256 = $null
$startedUTC = [DateTime]::UtcNow.ToString("o")
$previousPath = $env:PATH
$previousUvOffline = $env:UV_OFFLINE
$previousPipNoIndex = $env:PIP_NO_INDEX
$previousPipFindLinks = $env:PIP_FIND_LINKS
$previousUvNoIndex = $env:UV_NO_INDEX
$previousUvFindLinks = $env:UV_FIND_LINKS
$previousUvPythonDownloads = $env:UV_PYTHON_DOWNLOADS
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousPipCacheDir = $env:PIP_CACHE_DIR
$previousUvCacheDir = $env:UV_CACHE_DIR
$previousTestRoot = $env:SELCAL_WINDOWS_TEST_ROOT
$archive = "$out.zip"
$venv = $null
$pytestBase = $null
$testWork = $null
$testWorkArchive = $null
$pytestExit = $null
$nativeGateExit = $null
$script:nativeIndex = 0
$script:pythonEncodings = @{}
# Native output is captured as raw bytes (R17 Windows item 13); see that file for the reasons.
. (Join-Path $PSScriptRoot "windows_native_capture.ps1")

# Write to the console and append to the log as UTF-8 (PowerShell 5.1 would otherwise use UTF-16).
function Write-Log {
    param([string]$Text)
    Write-Host $Text
    Add-Content -Path $log -Value $Text -Encoding UTF8
}

function Write-JsonFile {
    param([string]$FilePath, [object]$Value)
    $json = ConvertTo-Json -InputObject $Value -Depth 8
    [System.IO.File]::WriteAllText($FilePath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

# BEGIN short-test-work
# CPython's Windows venv redirector has a physical executable-path limit independent of long
# file support. Keep the executable and fixture base short; keep repository/long-path fixtures
# unchanged. Never use a junction, reuse another run, delete evidence, or silently fall back.
function New-SelCalWindowsTestWork {
    param([string]$Repository, [string]$RequestedRoot = "", [string]$EnvironmentRoot = "")
    $source = "repository_volume_default"
    if ($RequestedRoot) { $candidate = $RequestedRoot; $source = "parameter" }
    elseif ($EnvironmentRoot) { $candidate = $EnvironmentRoot; $source = "SELCAL_WINDOWS_TEST_ROOT" }
    else { $candidate = Join-Path ([IO.Path]::GetPathRoot($Repository)) "SelCal-Test" }
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        throw "TEST_WORK_ROOT_INVALID: use an absolute short physical directory: $candidate"
    }
    $root = [IO.Path]::GetFullPath($candidate).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ($root -notmatch '^[A-Za-z]:\\' -or ($root + '\') -eq [IO.Path]::GetPathRoot($root)) {
        throw "TEST_WORK_ROOT_INVALID: a local non-root directory is required: $root"
    }
    $runId = [guid]::NewGuid().ToString("N")
    $run = Join-Path $root ("r-" + $runId.Substring(0, 16))
    $environment = Join-Path $run "venv"
    $pythonPath = Join-Path $environment "Scripts\python.exe"
    $pytestPath = Join-Path $run "pytest"
    if ($pythonPath.Length -gt 100 -or $pytestPath.Length -gt 64) {
        throw "TEST_WORK_ROOT_TOO_LONG: python length $($pythonPath.Length), pytest base length $($pytestPath.Length). Choose a shorter -TestWorkRoot; no fallback was used."
    }
    # Inspect every existing ancestor before creating anything, including a linked root.
    $ancestorPath = $root
    while ($ancestorPath) {
        if (Test-Path -LiteralPath $ancestorPath) {
            $ancestor = Get-Item -LiteralPath $ancestorPath -Force -ErrorAction Stop
            if (-not $ancestor.PSIsContainer -or ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "TEST_WORK_ROOT_NOT_PHYSICAL: $ancestorPath"
            }
        }
        $nextAncestor = Split-Path $ancestorPath -Parent
        if ($nextAncestor -eq $ancestorPath) { break }
        $ancestorPath = $nextAncestor
    }
    $null = New-Item -ItemType Directory -Path $root -Force -ErrorAction Stop
    $null = New-Item -ItemType Directory -Path $run -ErrorAction Stop
    # Recheck after creation, before any venv or test executes; never accept a redirected path.
    $ancestor = Get-Item -LiteralPath $run -Force -ErrorAction Stop
    while ($null -ne $ancestor) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "TEST_WORK_ROOT_NOT_PHYSICAL: $($ancestor.FullName)"
        }
        $ancestor = $ancestor.Parent
    }
    $temporary = Join-Path $run "tmp"
    $cache = Join-Path $run "cache"
    $null = New-Item -ItemType Directory -Path $temporary -ErrorAction Stop
    $null = New-Item -ItemType Directory -Path $cache -ErrorAction Stop
    $layout = [ordered]@{
        schema = "selcal.windows-test-work.v1"
        run_id = $runId
        repository = $Repository
        root_source = $source
        root = $root
        run = $run
        venv = $environment
        python = $pythonPath
        python_length = $pythonPath.Length
        pytest_base = $pytestPath
        pytest_base_length = $pytestPath.Length
        temporary = $temporary
        cache = $cache
        created_utc = [DateTime]::UtcNow.ToString("o")
        retained = $true
    }
    [IO.File]::WriteAllText((Join-Path $run "test-work-layout.json"),
        (ConvertTo-Json -InputObject $layout -Depth 5), (New-Object Text.UTF8Encoding($false)))
    return [pscustomobject]$layout
}
# END short-test-work

function Add-Failure {
    param([string]$StepName)
    $script:failureCount += 1
    $script:failureNames += "$StepName; "
    Write-Log "FAILED: $StepName"
}

# Run one step. $StepName is only a label; the command and its arguments are passed explicitly,
# so no variable of the caller is shadowed. A command that cannot be started, or a nonzero exit
# code, counts as a failure. -Setup marks an environment step; -NeedsSetup steps are recorded as
# BLOCKED (not run, not a separate failure) once an environment step has failed.
function Invoke-Step {
    param([string]$StepName, [string]$Exe, [string[]]$Arguments, [string]$CapturePath = "",
          [switch]$Setup, [switch]$NeedsSetup, [string]$Output = "python")
    Write-Log "=== $StepName"
    if (($Setup -or $NeedsSetup) -and $script:setupFailure) {
        Write-Log "--- BLOCKED by: $script:setupFailure"
        [void]$script:steps.Add([ordered]@{name = $StepName; exit_code = $null; status = "BLOCKED"; blocked_by = $script:setupFailure})
        return
    }
    Write-Log "argv: $Exe $($Arguments -join ' ')"
    Write-Log "process paths: executable length $($Exe.Length); cwd $((Get-Location).ProviderPath) (length $(((Get-Location).ProviderPath).Length))"
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) {
        Write-Log "--- cannot start: $Exe"
        Add-Failure "$StepName (cannot start)"
        [void]$script:steps.Add([ordered]@{name = $StepName; exit_code = $null; status = "cannot start"})
        if ($Setup) { $script:setupFailure = $StepName }
        return
    }
    $script:nativeIndex += 1
    $raw = Join-Path $out ("native-{0:D2}" -f $script:nativeIndex)
    $program = (Get-Command $Exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1)
    try {
        if (-not $program) { throw "not an executable program: $Exe" }
        # The encoding of the actual producer: uv writes UTF-8; Python reports its own encodings
        # for the same executable and start-up flags (R18 Windows item 2: no guessing).
        if ($Output -eq "utf-8") { $encodings = @("utf-8", "utf-8"); $basis = "uv writes UTF-8" }
        else {
            $flags = @(foreach ($argument in $Arguments) { if ($argument -match '^(-I|-E|-3(\.\d+)?)$') { $argument } else { break } })
            $key = $program.Source + "|" + ($flags -join " ")
            if (-not $script:pythonEncodings.ContainsKey($key)) { $script:pythonEncodings[$key] = Get-PythonOutputEncoding $program.Source $flags "$raw.encoding" }
            $encodings = $script:pythonEncodings[$key]; $basis = "reported by $($program.Source) $($flags -join ' ')"
        }
        Write-Log "output encoding: stdout $($encodings[0]), stderr $($encodings[1]) ($basis)"
        $result = Invoke-NativeCapture -Exe $program.Source -Arguments $Arguments -RawPath $raw -Encoding $encodings[0] -ErrorEncoding $encodings[1] -OnLine {
            param($line)
            Write-Log $line
            if ($CapturePath) { Add-Content -LiteralPath $CapturePath -Value $line -Encoding UTF8 }
        }
        $code = $result.ExitCode
        $global:LASTEXITCODE = $code
        Write-Log "raw output: $(Split-Path $raw -Leaf).stdout / .stderr"
    } catch {
        Write-Log "--- invocation error: $_"
        $code = -1
    }
    [void]$script:steps.Add([ordered]@{name = $StepName; exit_code = $code; status = $(if ($code -eq 0) {"PASS"} else {"FAIL"})})
    Write-Log "--- exit $code"
    if ($code -ne 0) {
        Add-Failure "$StepName (exit $code)"
        if ($Setup) { $script:setupFailure = $StepName }
    }
}

function Test-SourceManifest {
    $manifest = Join-Path $repo "SOURCE_MANIFEST.tsv"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        Write-Log "source manifest: not supplied (repository-mode run)"
        return
    }
    $script:sourceManifestStatus = "FAIL"
    $script:sourceManifestSHA256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $rootPrefix = $repo.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($line in [System.IO.File]::ReadAllLines($manifest)) {
        if (-not $line.Trim() -or $line -eq "path`tbytes`tsha256") { continue }
        $fields = $line -split "`t"
        if ($fields.Count -ne 3 -or $fields[1] -notmatch '^\d+$' -or $fields[2] -notmatch '^[a-fA-F0-9]{64}$') {
            throw "SOURCE_MANIFEST_MISMATCH: invalid row"
        }
        if ([System.IO.Path]::IsPathRooted($fields[0]) -or -not $seen.Add($fields[0])) {
            throw "SOURCE_MANIFEST_MISMATCH: absolute or repeated path $($fields[0])"
        }
        $target = [System.IO.Path]::GetFullPath((Join-Path $repo $fields[0]))
        if (-not $target.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "SOURCE_MANIFEST_MISMATCH: path outside source tree"
        }
        $item = Get-Item -LiteralPath $target -Force -ErrorAction Stop
        if ($item.PSIsContainer -or $item.Length -ne [long]$fields[1]) {
            throw "SOURCE_MANIFEST_MISMATCH: size or file type $($fields[0])"
        }
        $ancestor = $item
        while ($ancestor -and $ancestor.FullName -ne $repo) {
            if ($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "SOURCE_MANIFEST_MISMATCH: linked path $($fields[0])"
            }
            $ancestor = Get-Item -LiteralPath (Split-Path $ancestor.FullName -Parent) -Force -ErrorAction Stop
        }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $fields[2]) {
            throw "SOURCE_MANIFEST_MISMATCH: SHA256 $($fields[0])"
        }
        $script:sourceManifestCount += 1
    }
    if ($script:sourceManifestCount -eq 0) { throw "SOURCE_MANIFEST_MISMATCH: empty manifest" }
    $script:sourceManifestStatus = "PASS"
    Write-Log "source manifest: PASS ($script:sourceManifestCount files)"
}

# "-Python" is either a path to python.exe (spaces allowed) or a launcher command such as
# "py -3.12".
if (Test-Path -LiteralPath $Python) { $pyParts = @($Python) } else { $pyParts = $Python.Split(" ") }
$py = $null
$locationPushed = $false
# Read-only: the developer test suite creates paths up to about 215 characters below its pytest
# folder, so with this package layout Windows must allow long paths. Never changed here.
$longPaths = $null
try {
    $longPaths = (Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
} catch { $longPaths = $null }
try {
    Write-Log "repository: $repo (path length $($repo.Length))"
    $testWork = New-SelCalWindowsTestWork -Repository $repo -RequestedRoot $TestWorkRoot -EnvironmentRoot $env:SELCAL_WINDOWS_TEST_ROOT
    # Child probes must use the same resolved choice, including a -TestWorkRoot override.
    $env:SELCAL_WINDOWS_TEST_ROOT = $testWork.root
    $venv = $testWork.venv
    $py = $testWork.python
    $pytestBase = $testWork.pytest_base
    $env:TEMP = $testWork.temporary
    $env:TMP = $testWork.temporary
    $env:PIP_CACHE_DIR = Join-Path $testWork.cache "pip"
    $env:UV_CACHE_DIR = Join-Path $testWork.cache "uv"
    Write-JsonFile (Join-Path $out "test-work-layout.json") $testWork
    Write-Log "test work root: $($testWork.root) ($($testWork.root_source)); owned run $($testWork.run)"
    Write-Log "test executable: $py (length $($py.Length)); pytest base $pytestBase (length $($pytestBase.Length))"
    Write-Log "TEMP/TMP: $($testWork.temporary); caches: $($testWork.cache). Original environment values are restored on exit."
    if ($longPaths -eq 0) {
        Write-Log "NOTE: Windows long paths are disabled (LongPathsEnabled=0). Test paths below $pytestBase reach about $($pytestBase.Length + 215) characters. This does not by itself explain a failure: if a step fails, see the long-path note at the end. This check does not change the setting."
    }
    Write-JsonFile (Join-Path $out "environment.json") ([ordered]@{
        started_utc = $startedUTC
        powershell_version = $PSVersionTable.PSVersion.ToString()
        os_version = [Environment]::OSVersion.VersionString
        is_64bit_os = [Environment]::Is64BitOperatingSystem
        is_64bit_process = [Environment]::Is64BitProcess
        python_requested = $Python
        long_paths_enabled = $longPaths
        pytest_base_length = $pytestBase.Length
        test_work = $testWork
        runner_cwd = $repo
        runner_cwd_length = $repo.Length
        offline_requested = [bool]$Wheelhouse
    })
    Test-SourceManifest
    $installSourceArgs = @()
    if ($Wheelhouse) {
        $wheelhousePath = (Resolve-Path -LiteralPath $Wheelhouse -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $wheelhousePath -PathType Container)) { throw "Wheelhouse must be a directory" }
        $installSourceArgs = @("--no-index", "--find-links", $wheelhousePath)
        $env:UV_OFFLINE = "1"
        # Nested build/uv commands launched by the tests must inherit the same local source.
        # An escaped file URI also preserves paths containing spaces in environment options.
        $wheelhouseLink = ([System.Uri]$wheelhousePath).AbsoluteUri
        $env:PIP_NO_INDEX = "1"
        $env:PIP_FIND_LINKS = $wheelhouseLink
        $env:UV_NO_INDEX = "1"
        $env:UV_FIND_LINKS = $wheelhouseLink
        $env:UV_PYTHON_DOWNLOADS = "never"
        Write-Log "dependency mode: OFFLINE ($wheelhousePath)"
    } elseif ($Index) {
        $installSourceArgs = @("-i", $Index)
    }
    $workflowFile = [System.IO.File]::ReadAllBytes((Join-Path $repo "src\selcal\workflow.py"))
    if ($workflowFile -contains 13) {
        Add-Failure "line endings: CRLF DETECTED (extract the supplied ZIP without newline conversion)"
    } else {
        Write-Log "line endings: LF OK"
    }

    Invoke-Step -Setup -StepName "create venv" -Exe $pyParts[0] -Arguments (@($pyParts | Select-Object -Skip 1) + @("-m", "venv", $venv))
    Invoke-Step -Setup -StepName "Python environment" -Exe $py -Arguments @("-c", "import json, platform, sys; print(json.dumps({'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(), 'machine':platform.machine()}, indent=2))") -CapturePath (Join-Path $out "python-environment.json")
    # W14-01: CPython 3.12.0 ships pip 23.2.1, which decodes requirement files with the locale
    # code page (cp936) and failed on a UTF-8 path comment. Pin the tested pip from the same
    # offline source first; the lock export below is also written without the path header.
    Invoke-Step -Setup -StepName "bootstrap pip" -Exe $py -Arguments (@("-m", "pip", "install", "-q", "--disable-pip-version-check") + $installSourceArgs + @("pip==26.2.1"))
    Invoke-Step -Setup -StepName "install uv" -Exe $py -Arguments (@("-m", "pip", "install", "-q", "--disable-pip-version-check") + $installSourceArgs + @("uv"))
    Invoke-Step -Setup -StepName "install build tools" -Exe $py -Arguments (@("-m", "pip", "install", "-q", "--disable-pip-version-check") + $installSourceArgs + @("setuptools>=77", "wheel", "packaging"))
    $requirements = Join-Path $out "requirements-locked.txt"
    Push-Location $repo
    $locationPushed = $true
    Invoke-Step -Setup -StepName "export lock" -Output "utf-8" -Exe $py -Arguments @("-m", "uv", "export", "--locked", "--all-extras", "--no-hashes", "--no-emit-project", "--no-header", "-o", $requirements)
    Invoke-Step -Setup -StepName "install locked requirements" -Exe $py -Arguments (@("-m", "pip", "install", "-q", "--disable-pip-version-check") + $installSourceArgs + @("-r", $requirements))
    Invoke-Step -Setup -StepName "install selcal" -Exe $py -Arguments (@("-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "--no-build-isolation") + $installSourceArgs + @("-e", $repo))
    Invoke-Step -NeedsSetup -StepName "installed package versions" -Exe $py -Arguments @("-m", "pip", "freeze", "--all") -CapturePath (Join-Path $out "pip-freeze.txt")

    $env:PATH = (Join-Path $venv "Scripts") + ";" + $env:PATH
    if (Test-Path -LiteralPath $pytestBase) { throw "Refuse existing pytest evidence directory: $pytestBase" }
    Invoke-Step -NeedsSetup -StepName "full test suite" -Exe $py -Arguments @("-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEs", "--basetemp=$pytestBase", "--junitxml=$(Join-Path $out 'pytest.xml')")
    $pytestExit = $script:steps[-1].exit_code
    Invoke-Step -NeedsSetup -StepName "native Windows evidence" -Exe $py -Arguments @("-I", "tests\_windows_acceptance.py", (Join-Path $out "pytest.xml")) -CapturePath (Join-Path $out "native-gate.log")
    $nativeGateExit = $script:steps[-1].exit_code
    $acceptance = @("scripts\acceptance_check.py", "--out", $out, "--precision")
    if ($Records) { $acceptance += @("--records", $Records) }
    if ($LegacyRecords) { $acceptance += @("--legacy-records", $LegacyRecords) }
    Invoke-Step -NeedsSetup -StepName "acceptance check" -Exe $py -Arguments $acceptance
} catch {
    Add-Failure "runner error: $_"
} finally {
    if ($locationPushed) { Pop-Location }
    $env:PATH = $previousPath
    $env:UV_OFFLINE = $previousUvOffline
    $env:PIP_NO_INDEX = $previousPipNoIndex
    $env:PIP_FIND_LINKS = $previousPipFindLinks
    $env:UV_NO_INDEX = $previousUvNoIndex
    $env:UV_FIND_LINKS = $previousUvFindLinks
    $env:UV_PYTHON_DOWNLOADS = $previousUvPythonDownloads
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:PIP_CACHE_DIR = $previousPipCacheDir
    $env:UV_CACHE_DIR = $previousUvCacheDir
    $env:SELCAL_WINDOWS_TEST_ROOT = $previousTestRoot
    $summaryPath = Join-Path $out "runner-summary.json"
    $summary = [ordered]@{
        schema = "selcal.windows-check.v1"
        started_utc = $startedUTC
        finished_utc = [DateTime]::UtcNow.ToString("o")
        status = $(if ($script:failureCount -gt 0) { "FAIL" } else { "PASS" })
        "failure_count" = $script:failureCount
        failures = $script:failureNames
        first_environment_failure = $script:setupFailure
        blocked_steps = @($script:steps | Where-Object { $_.status -eq "BLOCKED" } | ForEach-Object { $_.name })
        source_manifest_status = $script:sourceManifestStatus
        source_manifest_files = $script:sourceManifestCount
        source_manifest_sha256 = $script:sourceManifestSHA256
        pytest_base = $pytestBase
        test_work = $testWork
        test_work_archive = $null
        pytest_exit_code = $pytestExit
        native_gate_exit_code = $nativeGateExit
        "steps" = @($script:steps.ToArray())
        results_archive = [System.IO.Path]::GetFileName($archive)
    }
    Write-JsonFile $summaryPath $summary
    if ($script:failureCount -gt 0) {
        Write-Log "FAILED steps ($script:failureCount): $script:failureNames"
        # AUD-06: the long-path setting is one possible cause, never assumed. Diagnose in order.
        Write-Log "Diagnosis order: (1) first_environment_failure in runner-summary.json is the step that failed first; later failures may follow from it. (2) For a missing-path or path error, compare the failing path's length with 260 and read its error code (for example 3 = path not found, 206 = file name or extension too long). (3) long_paths_enabled ($longPaths) is diagnostic context, not proof that every Windows API accepts long paths: process creation can still fail with WinError 206 for a long executable path when it is 1. Inspect the executable path, working directory and accessed file path separately, using the short physical test-work paths recorded in runner-summary.json. (4) A process's working directory is limited to about 258 characters even when long paths are enabled (WinError 267 when starting a process); SelCal starts its own child processes in the helper's working directory, so report the starting folder and its length if this appears."
    } else {
        Write-Log "ALL STEPS PASSED."
    }
    Write-Log "Results folder: $out"
    Write-Log "Return this ZIP: $archive"
    try {
        . (Join-Path $PSScriptRoot "windows_evidence_archive.ps1")
        if ($testWork) {
            $testWorkArchive = Join-Path $out "test-work-evidence.zip"
            New-SelCalEvidenceArchive -EvidenceDirectory $testWork.run -DestinationPath $testWorkArchive -ExcludedDirectories @($venv)
            $summary.test_work_archive = [IO.Path]::GetFileName($testWorkArchive)
            Write-JsonFile $summaryPath $summary
        }
        New-SelCalEvidenceArchive -EvidenceDirectory $out -DestinationPath $archive
    } catch {
        Add-Failure "results archive: $_"
        $summary.status = "FAIL"
        $summary.failure_count = $script:failureCount
        $summary.failures = $script:failureNames
        $summary.results_archive = $null
        Write-JsonFile $summaryPath $summary
        Write-Log "Archive could not be created. Return the results folder and the retained test-work folder recorded in runner-summary.json, excluding its venv."
    }
}
if ($script:failureCount -gt 0) { exit 1 }
exit 0
