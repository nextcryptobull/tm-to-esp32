# Examples

Reference integration snippets for dropping the converted PPE models into
an ESP-IDF / xiaozhi-esp32 firmware build.

## Files

- `ppe_inference.h` — C API for running all binary PPE classifiers on one
  frame and getting a structured "what's missing" result + a natural
  language string suitable for injection into the xiaozhi LLM prompt.

## Integration steps

1. Convert each binary TM model with the script:
   ```bash
   python ../../tm_to_esp32.py -i ./hat_model   -o ./out_hat   -q int8 -r ../../samples/hat/present   --c-header
   python ../../tm_to_esp32.py -i ./vest_model  -o ./out_vest  -q int8 -r ../../samples/vest/present  --c-header
   # ...etc
   ```
2. Copy each `model_data.h`, `model_data.cc`, `labels.h` into this folder
   with a unique prefix (e.g., `hat_model_data.h`, `vest_model_data.h`).
3. Include `ppe_inference.h` from your firmware's main app.
4. Implement `ppe_init()` and `ppe_infer()` against TFLite-Micro's
   `MicroInterpreter` API (the header file declares the contract; the
   `.cc` implementation is left to your firmware build to avoid
   pulling in TFLite-Micro as a submodule of this repo).
5. In your xiaozhi prompt-building code:
   ```cpp
   ppe_result_t r;
   ppe_infer(frame_int8, &r);
   char summary[128];
   ppe_describe(&r, summary, sizeof(summary));
   // Inject `summary` into the LLM system prompt or user turn.
   ```

## Why the .cc implementation isn't included here

TFLite-Micro's `MicroInterpreter` API and the ESP-IDF component structure
vary across firmware versions. To avoid locking this repo to a specific
TFLite-Micro version, the .cc file is intentionally left as an exercise
for your firmware build. The header declares the contract; the
implementation is mechanical and follows the standard TFLite-Micro
"arena + resolver + op resolver + interpreter + invoke" pattern.

A reference .cc implementation is on the roadmap — open an issue if you
need it sooner.
