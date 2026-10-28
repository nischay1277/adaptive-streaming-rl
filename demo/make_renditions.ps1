# Builds a six-level HLS ladder from one source video with ffmpeg.
# Usage:  .\make_renditions.ps1 -Source input.mp4 -Out renditions
# IMPORTANT: set $Ladder to the SAME bitrate levels (kbps) and $SegSeconds to the
# SAME chunk duration that env.py uses. The values below are placeholders
# (the common Pensieve ladder); check env.py and edit them.
param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$Out = "renditions",
  [int]$SegSeconds = 4
)

$Ladder = @(300, 750, 1200, 1850, 2850, 4300)   # kbps, EDIT to match env.py
$Heights = @(240, 360, 480, 720, 720, 1080)     # one height per level

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  throw "ffmpeg not found on PATH. Install it first (for example: winget install Gyan.FFmpeg)."
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

for ($i = 0; $i -lt $Ladder.Count; $i++) {
  $br = $Ladder[$i]; $h = $Heights[$i]
  $dir = Join-Path $Out "q$i"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  # Constant bitrate, keyframe every segment so every level has aligned boundaries.
  $gop = $SegSeconds * 30
  ffmpeg -y -i $Source -an -vf "scale=-2:$h" -r 30 `
    -c:v libx264 -preset veryfast -b:v "${br}k" -maxrate "${br}k" -bufsize "$($br*2)k" `
    -g $gop -keyint_min $gop -sc_threshold 0 `
    -f hls -hls_time $SegSeconds -hls_playlist_type vod `
    -hls_segment_filename "$dir/seg%04d.ts" "$dir/index.m3u8"
}

# Master playlist so a player sees all levels.
$master = "#EXTM3U`n"
for ($i = 0; $i -lt $Ladder.Count; $i++) {
  $master += "#EXT-X-STREAM-INF:BANDWIDTH=$($Ladder[$i]*1000),RESOLUTION=0x$($Heights[$i])`nq$i/index.m3u8`n"
}
Set-Content -Path (Join-Path $Out "master.m3u8") -Value $master -NoNewline
Write-Host "Done. Verify with: ffprobe -v error -show_entries format=bit_rate,duration $Out\q0\index.m3u8"
