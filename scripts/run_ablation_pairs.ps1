Get-ChildItem artifacts/judge_prompts/ablation_pairs -Directory | ForEach-Object {
  $pair = $_.Name

  # Claude
  python scripts/run_llm_judge.py `
    --input_dir "artifacts/judge_prompts/ablation_pairs/$pair" `
    --group "ablation_pairs/$pair" `
    --judge_name "claude-haiku-4-5-20251001" `
    --replicas 3 `
    --provider openai `
    --api_key_env COMETAPI_API_KEY `
    --base_url https://api.cometapi.com/v1 `
    --api_style chat_completions `
    --temperature 0.3 `
    --overwrite

  # Gemini
  python scripts/run_llm_judge.py `
    --input_dir "artifacts/judge_prompts/ablation_pairs/$pair" `
    --group "ablation_pairs/$pair" `
    --judge_name "gemini-2.5-flash-lite" `
    --replicas 3 `
    --provider openai `
    --api_key_env COMETAPI_API_KEY `
    --base_url https://api.cometapi.com/v1 `
    --api_style chat_completions `
    --temperature 0.3 `
    --overwrite

  # OpenAI
  python scripts/run_llm_judge.py `
    --input_dir "artifacts/judge_prompts/ablation_pairs/$pair" `
    --group "ablation_pairs/$pair" `
    --judge_name "gpt-4o-mini" `
    --replicas 3 `
    --provider openai `
    --api_key_env OPENAI_API_KEY `
    --base_url https://api.openai.com/v1 `
    --api_style responses `
    --temperature 0.3 `
    --overwrite
}
