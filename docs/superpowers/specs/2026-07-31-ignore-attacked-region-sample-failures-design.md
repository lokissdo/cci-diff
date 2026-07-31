# Ignore Attacked-Region Sample Failures

## Problem

`scripts/run_attacked_region_300.sh` invokes the clean CCI pilot without its
existing `--continue_on_error` option. A source-specific failure, such as an
undetectable FaceNet source face, therefore stops the entire generation run.

## Design

Pass `--continue_on_error` from the attacked-region launcher's shared
`run_generation` command. The pilot will record failed candidates in
`failures.jsonl`, omit incomplete samples from its result rows, and continue
with later variants and samples using its existing tested behavior.

This change does not alter face detection, sample eligibility, replacement
selection, exact-count validation, or downstream metric requirements.

## Testing

Extend the attacked-region scheduler test to require
`--continue_on_error` in the pilot invocation. Run that focused test and the
existing clean-pilot continuation regression test.
