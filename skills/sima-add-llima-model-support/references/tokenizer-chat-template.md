# Add Tokenizer or Chat-Template Compatibility

Use this route when model computation is supported but prompt rendering,
special tokens, tokenizer assets, or tool-call formatting is not.

## Procedure

1. Inventory `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`,
   `chat_template.jinja`, `chat_template.json`, processor files, and GGUF
   tokenizer metadata.
2. Verify BOS, EOS, PAD, stop, image, and other control-token IDs.
3. Preserve template precedence:
   - explicit runtime override;
   - compiled configuration;
   - `chat_template.jinja`;
   - `chat_template.json`;
   - `tokenizer_config.json`;
   - embedded GGUF template.
4. Render the same structured messages with upstream Hugging Face/Jinja and
   Minja, then compare exact text and token IDs.
5. Prefer a standards-compatible Minja fix over a model-specific template
   rewrite.
6. Add a model-specific runtime branch only for a genuine protocol difference,
   such as image-token placement or a distinct tool-call wire format.

## Example: Gemma Template with Adjacent Literals

Request:

> A supported Gemma checkpoint fails because its upstream template contains
> adjacent quoted string literals that Jinja accepts but Minja rejects.

Investigation:

- Render representative system, user, assistant, and tool messages with the
  upstream Jinja environment.
- Reduce the failure to the smallest adjacent-literal expression.
- Confirm the construct is general template syntax rather than a
  Gemma-specific semantic rule.

Implementation:

- Extend Minja to concatenate adjacent literals according to the upstream
  semantics.
- Keep the original checkpoint template unchanged.
- Avoid embedding a replacement Gemma template in C++.

Validation:

- Add a focused Minja regression for adjacent literals.
- Compare full prompt rendering and token IDs for single-turn, multi-turn, and
  tool-use messages.
- Package the tokenizer/template assets and run CLI or HTTP generation on
  Modalix.

