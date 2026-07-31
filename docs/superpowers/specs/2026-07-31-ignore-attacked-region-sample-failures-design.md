# Exclude Attacked-Region Image 10260

## Problem

Image `10260` passes the pilot's initial eligibility check but FaceNet cannot
detect a face after the source passes through the runtime image path. Its
failure stops the attacked-region generation run.

## Design

Add a targeted exclusion JSON containing `{"smile": [10260]}` and pass it to
the attacked-region launcher's shared `run_generation` command through the
pilot's existing `--exclude_ids_json` option. The deterministic selector will
skip only image `10260` and select the next eligible image to retain the
requested 300-image cohort.

This change does not alter face detection, general error handling,
exact-count validation, or downstream metric requirements.

## Testing

Extend the attacked-region scheduler test to require the targeted exclusion
file and `--exclude_ids_json` in the pilot invocation. Run that focused test
and the existing clean-pilot exclusion tests.
