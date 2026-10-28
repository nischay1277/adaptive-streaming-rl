param(
  [int]$Port = 8110,
  [string]$Checkpoint = "artifacts_adaptive_ppo\ppo_adaptive.pt"
)

python policy_server.py --port $Port --checkpoint $Checkpoint
