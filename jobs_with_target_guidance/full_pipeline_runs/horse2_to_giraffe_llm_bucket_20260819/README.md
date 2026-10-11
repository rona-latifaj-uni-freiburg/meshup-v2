# Horse2 To Giraffe LLM Bucket Full Pipeline

This folder is reserved for one full pilot run:

1. choose a fixed PartField bucket count with `partfield_bucket_llm.py`;
2. run PartField co-segmentation for `horse2` and `giraffe` into `partfield_segments/`;
3. deform horse to giraffe for 6000 epochs into `outputs/`;
4. render a bucket-colored start/result turntable into `videos/`.

Run with:

```bash
sbatch jobs_with_target_guidance/full_pipeline_runs/horse2_to_giraffe_llm_bucket_20260819/run_full_pipeline.sh
```

Set `PARTFIELD_BUCKET_SELECTOR_PROVIDER=openai` and `OPENAI_API_KEY` for a live OpenAI selector call. Without an API key, `provider=auto` uses the calibrated offline fallback and records that provider in the decision JSON.
