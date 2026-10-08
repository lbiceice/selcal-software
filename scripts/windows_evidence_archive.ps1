# Shared Windows evidence archive; regular files only, never follow reparse points.
function New-SelCalEvidenceArchive {
    param(
        [Parameter(Mandatory=$true)][string]$EvidenceDirectory,
        [Parameter(Mandatory=$true)][string]$DestinationPath,
        [string[]]$ExcludedDirectories = @()
    )
    $evidenceRoot = [IO.Path]::GetFullPath($EvidenceDirectory)
    if ($evidenceRoot -eq [IO.Path]::GetPathRoot($evidenceRoot)) { throw "Filesystem root cannot be evidence." }
    $evidenceRoot = $evidenceRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
    $destination = [IO.Path]::GetFullPath($DestinationPath)
    $rootItem = Get-Item -LiteralPath $evidenceRoot -Force -ErrorAction Stop
    if (-not $rootItem.PSIsContainer -or -not $rootItem.Parent) { throw "Evidence root must be a non-root directory." }
    $destinationParent = Get-Item -LiteralPath (Split-Path $destination -Parent) -Force -ErrorAction Stop
    if (-not $destinationParent.PSIsContainer) { throw "Archive parent must be an existing directory." }
    foreach ($start in @($rootItem, $destinationParent)) {
        $ancestor = $start
        while ($null -ne $ancestor) {
            if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked evidence/archive ancestor is not allowed." }
            $ancestor = $ancestor.Parent
        }
    }
    $prefix = $evidenceRoot + [IO.Path]::DirectorySeparatorChar
    if ($destination -eq $evidenceRoot -or $destination.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Archive destination must be outside the evidence root."
    }
    if (Test-Path -LiteralPath $destination) { throw "Refuse existing evidence archive: $destination" }
    $manifestPath = Join-Path $evidenceRoot "archive-omissions.json"
    if (Test-Path -LiteralPath $manifestPath) { throw "Refuse existing archive omission manifest; retain previous evidence." }
    $excluded = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $ExcludedDirectories) {
        $full = [IO.Path]::GetFullPath($path).TrimEnd([IO.Path]::DirectorySeparatorChar)
        if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $full -Leaf) -ne "venv") {
            throw "Only explicit venv directories within the evidence root may be excluded."
        }
        [void]$excluded.Add($full)
    }
    Add-Type -AssemblyName System.IO.Compression -ErrorAction Stop
    $archiveStream = [IO.File]::Open($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $zip = $null
    try {
        $zip = [IO.Compression.ZipArchive]::new($archiveStream, [IO.Compression.ZipArchiveMode]::Create, $false)
        $pending = New-Object 'System.Collections.Generic.Stack[System.IO.DirectoryInfo]'
        $pending.Push($rootItem)
        $omitted = New-Object System.Collections.ArrayList
        while ($pending.Count -gt 0) {
            $directory = $pending.Pop()
            foreach ($item in $directory.EnumerateFileSystemInfos()) {
                $relative = $item.FullName.Substring($prefix.Length).Replace('\', '/')
                if ($excluded.Contains($item.FullName)) { continue }
                if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    [void]$omitted.Add([ordered]@{ path = $relative; reason = "ReparsePoint" })
                    continue
                }
                if ($item.Attributes -band [IO.FileAttributes]::Directory) {
                    [void]$zip.CreateEntry($relative + "/")
                    $pending.Push([IO.DirectoryInfo]$item)
                    continue
                }
                $fileStream = [IO.File]::Open($item.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
                try {
                    $entryStream = $zip.CreateEntry($relative, [IO.Compression.CompressionLevel]::Optimal).Open()
                    try { $fileStream.CopyTo($entryStream) } finally { $entryStream.Dispose() }
                } finally { $fileStream.Dispose() }
            }
        }
        $manifest = [ordered]@{
            schema = "selcal.windows-evidence-omissions.v1"
            omitted = @($omitted.ToArray())
            excluded_venvs = @($excluded | ForEach-Object { $_.Substring($prefix.Length).Replace('\', '/') })
        }
        $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes((ConvertTo-Json -InputObject $manifest -Depth 5))
        $manifestStream = [IO.File]::Open($manifestPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $manifestStream.Write($bytes, 0, $bytes.Length) } finally { $manifestStream.Dispose() }
        $entryStream = $zip.CreateEntry("archive-omissions.json").Open()
        try { $entryStream.Write($bytes, 0, $bytes.Length) } finally { $entryStream.Dispose() }
    } finally {
        if ($null -ne $zip) { $zip.Dispose() } else { $archiveStream.Dispose() }
    }
}
