<!--
Thanks for contributing to GTR! Delete any section that doesn't apply — a one-line
docs fix doesn't need a Performance section, and a pure refactor doesn't need one either.
-->

## Summary

<!--
What breaks/is missing today, and what does this change do about it?
For a bug fix: include a minimal repro or the exact error/traceback.
For a feature: what it enables and why it belongs here.
-->

## Changes

<!-- Bullet list of what changed, if it's not already obvious from the summary. -->

## Verification

<!--
Commands you ran and their result, e.g.:
  python train.py -c configs/det/coco_finetune/gtr_s.yml --test-only -r weights/det/gtr_s_coco.pth  # AP 53.6, matches README
  python engine/gtr/backbone/csrc/test_gla.py  # ALL PASS
If the change touches model, loss, data or evaluation code, evaluate at least one affected
released checkpoint and confirm it still matches the README numbers.
For TensorRT changes, include the tensorrt_plugin/verify_trt.py result.
Note anything you could NOT verify (no GPU, no TensorRT, dataset not available, etc.).
-->

## Performance

<!--
Only for perf-sensitive changes. Include: hardware, measurement method (what you ran,
how many repetitions), and a before/after table or numbers. README latencies are median
single-image FP16 forward times on one RTX 4090, e.g.:
  python tools/benchmark/torch_speed.py --configs configs/det/coco_finetune/gtr_s.yml --dtype fp16  # report p50
Delete this section otherwise.
-->

## Related issues

<!-- Fixes #, Closes #, Part of #, or a link to the discussion that prompted this. -->
