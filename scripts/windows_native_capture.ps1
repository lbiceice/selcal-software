# BEGIN native-capture (canonical copy: scripts/windows_native_capture.ps1; tests keep copies equal)
# R17 Windows item 13 / R18 item 2: native output is read as raw bytes and decoded with the encoding
# its producer actually uses, never with the console code page and never by guessing. The caller
# names that encoding: for Python, Get-PythonOutputEncoding asks the same interpreter with the same
# start-up flags (the locale code page, e.g. cp936, unless UTF-8 mode or PYTHONIOENCODING apply);
# the lock exporter (a Rust tool) writes UTF-8; a child PowerShell uses the console output code page. A line that is not
# valid in the stated encoding is marked as such, not silently decoded otherwise (cp936 bytes can
# also be valid UTF-8: C3 A6 is one Chinese character in cp936 and "ae" in UTF-8). The exact bytes
# stay in <RawPath>.stdout / <RawPath>.stderr. The console code page and the child's environment
# are not changed. Lines appear while the program runs; each stream keeps its own order.
function Join-NativeArguments([string[]]$Arguments) {
    $quoted = foreach ($argument in $Arguments) {
        if ($argument -ne "" -and $argument -notmatch '[\s"]') { $argument; continue }
        $text = '"'; $slashes = 0
        foreach ($character in $argument.ToCharArray()) {
            if ($character -eq '\') { $slashes += 1; continue }
            if ($character -eq '"') { $text += ('\' * (2 * $slashes + 1)) + '"' }
            else { $text += ('\' * $slashes) + $character }
            $slashes = 0
        }
        $text + ('\' * (2 * $slashes)) + '"'
    }
    return ($quoted -join ' ')
}

# A strict decoder for a .NET encoding or a Python codec name; throws for an unknown name.
function Get-StrictEncoding($Encoding) {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        [System.Text.Encoding]::RegisterProvider([System.Text.CodePagesEncodingProvider]::Instance)
    }
    if ($Encoding -is [System.Text.Encoding]) { $page = $Encoding.CodePage }
    else {
        $name = ([string]$Encoding).Trim().ToLowerInvariant().Replace('_', '-')
        if ($name -in @("utf-8", "utf8")) { $page = 65001 }
        elseif ($name -in @("ascii", "us-ascii")) { $page = 20127 }
        elseif ($name -match '^cp(\d+)$') { $page = [int]$Matches[1] }
        elseif ($name -in @("gbk", "gb2312")) { $page = 936 }
        elseif ($name -eq "mbcs") {
            $page = [int](Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Nls\CodePage' -Name ACP -ErrorAction Stop).ACP
        }
        else { $page = [System.Text.Encoding]::GetEncoding($name).CodePage }
    }
    if ($page -eq 65001) { return New-Object System.Text.UTF8Encoding($false, $true) }
    return [System.Text.Encoding]::GetEncoding($page, [System.Text.EncoderFallback]::ExceptionFallback, [System.Text.DecoderFallback]::ExceptionFallback)
}

function ConvertFrom-NativeLine([byte[]]$Bytes, [System.Text.Encoding]$Encoding) {
    $length = $Bytes.Length
    if ($length -gt 0 -and $Bytes[$length - 1] -eq 13) { $length -= 1 }
    try { return $Encoding.GetString($Bytes, 0, $length) }
    catch {
        $lenient = [System.Text.Encoding]::GetEncoding($Encoding.CodePage)
        return "[not valid $($Encoding.WebName); exact bytes in the raw file] " + $lenient.GetString($Bytes, 0, $length)
    }
}

function Read-NativeLines([string]$Path, [ref]$Offset, [bool]$Final, [System.Text.Encoding]$Encoding) {
    $stream = New-Object System.IO.FileStream($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        $available = $stream.Length - $Offset.Value
        if ($available -le 0) { return @() }
        $buffer = New-Object byte[] $available
        [void]$stream.Seek($Offset.Value, [System.IO.SeekOrigin]::Begin)
        $read = 0
        while ($read -lt $available) {
            $count = $stream.Read($buffer, $read, $available - $read)
            if ($count -le 0) { break }
            $read += $count
        }
    } finally { $stream.Dispose() }
    $lines = New-Object System.Collections.ArrayList
    $start = 0
    for ($i = 0; $i -lt $read; $i++) {
        if ($buffer[$i] -ne 10) { continue }
        $line = New-Object byte[] ($i - $start)
        [Array]::Copy($buffer, $start, $line, 0, $i - $start)
        [void]$lines.Add((ConvertFrom-NativeLine $line $Encoding))
        $start = $i + 1
    }
    if ($Final -and $start -lt $read) {
        $line = New-Object byte[] ($read - $start)
        [Array]::Copy($buffer, $start, $line, 0, $read - $start)
        [void]$lines.Add((ConvertFrom-NativeLine $line $Encoding))
        $start = $read
    }
    $Offset.Value += $start
    return $lines.ToArray()
}

# Runs Exe with Arguments in the current folder. Encoding (stdout) and ErrorEncoding (stderr,
# default: Encoding) state what the program writes. Returns @{ExitCode; Lines; Encoding;
# ErrorEncoding; InvalidLines}; throws if the program cannot be started. OnLine (optional) receives
# every decoded line as it arrives.
function Invoke-NativeCapture {
    param([string]$Exe, [string[]]$Arguments, [string]$RawPath, [Parameter(Mandatory = $true)]$Encoding,
          $ErrorEncoding = $null, [scriptblock]$OnLine = $null)
    if ($null -eq $ErrorEncoding) { $ErrorEncoding = $Encoding }
    $decoders = @((Get-StrictEncoding $Encoding), (Get-StrictEncoding $ErrorEncoding))
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Exe
    $info.Arguments = Join-NativeArguments $Arguments
    $info.WorkingDirectory = (Get-Location -PSProvider FileSystem).ProviderPath
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $paths = @("$RawPath.stdout", "$RawPath.stderr")
    $sinks = @(foreach ($path in $paths) {
        New-Object System.IO.FileStream($path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    })
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    $all = New-Object System.Collections.ArrayList
    $invalid = 0
    try {
        [void]$process.Start()
        $copies = @(
            $process.StandardOutput.BaseStream.CopyToAsync($sinks[0]),
            $process.StandardError.BaseStream.CopyToAsync($sinks[1])
        )
        $offsets = @([long]0, [long]0)
        $finished = $false
        while (-not $finished) {
            $finished = $process.WaitForExit(500)
            if ($finished) { $process.WaitForExit(); [System.Threading.Tasks.Task]::WaitAll($copies) }
            foreach ($index in 0, 1) {
                $sinks[$index].Flush()
                $offset = $offsets[$index]
                foreach ($line in (Read-NativeLines $paths[$index] ([ref]$offset) $finished $decoders[$index])) {
                    if ($line.StartsWith("[not valid ")) { $invalid += 1 }
                    [void]$all.Add($line)
                    if ($OnLine) { & $OnLine $line }
                }
                $offsets[$index] = $offset
            }
        }
        return @{ ExitCode = $process.ExitCode; Lines = $all.ToArray(); Encoding = $decoders[0].WebName; ErrorEncoding = $decoders[1].WebName; InvalidLines = $invalid }
    } finally {
        foreach ($sink in $sinks) { $sink.Dispose() }
        $process.Dispose()
    }
}

# The encodings this Python writes to redirected stdout and stderr, reported by the same executable
# with the same start-up flags (for example -I, or the py launcher's -3.12). Returns @(stdout,
# stderr) as Python codec names; throws if it cannot tell. The answer stays in RawPath.stdout.
function Get-PythonOutputEncoding([string]$Exe, [string[]]$Flags, [string]$RawPath) {
    $result = Invoke-NativeCapture -Exe $Exe -Arguments ($Flags + @("-c", "import sys; print(sys.stdout.encoding); print(sys.stderr.encoding)")) -RawPath $RawPath -Encoding "ascii"
    $names = @($result.Lines | Where-Object { $_ -match '^[A-Za-z0-9_.-]+$' })
    if ($result.ExitCode -ne 0 -or $names.Count -ne 2) { throw "Python did not report its output encoding (exit $($result.ExitCode))" }
    return $names
}
# END native-capture
