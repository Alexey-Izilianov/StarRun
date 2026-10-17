# Фразы десантника голосом Windows «Microsoft Pavel» (WinRT SpeechSynthesizer: голоса OneCore старому
# System.Speech не видны). Вход — файл «имя<TAB>текст» в UTF-8, выход — <OutDir>\<имя>.wav.
# Запуск (Windows PowerShell 5.1): powershell -NoProfile -File voice.ps1 -Lines lines.tsv -OutDir out
param([string]$Lines, [string]$OutDir, [string]$Voice = "Pavel", [double]$Rate = 1.08)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await($op, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $null = $task.Wait()
    $task.Result
}

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$synth.Voice = ([Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like "*$Voice*" })[0]
$synth.Options.SpeakingRate = $Rate
foreach ($line in [IO.File]::ReadAllLines($Lines, [Text.Encoding]::UTF8)) {
    $name, $text = $line -split "`t", 2
    $stream = Await ($synth.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    $size = [uint32]$stream.Size
    $null = Await ($reader.LoadAsync($size)) ([uint32])
    $bytes = New-Object byte[] $size
    $reader.ReadBytes($bytes)
    [IO.File]::WriteAllBytes((Join-Path $OutDir "$name.wav"), $bytes)
}
