param(
  [string]$Command = "preflight",
  [string]$Config = "config/config.json",
  [string]$Output = ""
)
$env:PYTHONPATH = (Resolve-Path "src").Path
$argsList = @("-m", "cd4perturb.cli", $Command, "--config", $Config)
if ($Output -ne "") { $argsList += @("--output", $Output) }
python @argsList
