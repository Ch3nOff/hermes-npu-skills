# Sampler detail: atribusi utilisasi GPU per instance engine (pid + engtype).
param(
    [int]$DurationSec = 24,
    [string]$OutFile = "$env:TEMP\npu_gpu_detail.csv"
)

$deadline = (Get-Date).AddSeconds($DurationSec)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("timestamp,instance,util_pct")

while ((Get-Date) -lt $deadline) {
    try {
        $smp = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples |
            Where-Object { $_.CookedValue -gt 0.5 } |
            Sort-Object CookedValue -Descending |
            Select-Object -First 6
        $ts = (Get-Date).ToString("HH:mm:ss.fff")
        foreach ($s in $smp) {
            $lines.Add("$ts,$($s.InstanceName),$([math]::Round($s.CookedValue,2))")
        }
        if ($smp.Count -eq 0) { $lines.Add("$ts,(idle),0") }
    } catch { $lines.Add("$(Get-Date -Format HH:mm:ss.fff),(counter-error),0") }
    Start-Sleep -Milliseconds 700
}

$lines | Out-File -FilePath $OutFile -Encoding utf8
Write-Output "DONE. FILE=$OutFile"
