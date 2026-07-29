# Add Tokenizer or Prompt Compatibility

Use when computation works but tokenization, templates, special tokens,
multimodal messages, or tool calls do not.

## Contract

LLiMa always renders LLM/VLM templates with Minja on Modalix. Do not add
another renderer or hardcode a checkpoint-specific replacement.

Use the source implementation only as the oracle:

- Hugging Face: Transformers/Jinja
- GGUF: embedded template plus a compatible Jinja renderer

For identical structured messages, Minja must match reference prompt text,
token IDs, generation suffix, and image-token placement.

## Procedure

1. Inventory tokenizer, template, processor, and GGUF tokenizer metadata.
2. Apply format-specific selection:
   - Hugging Face: runtime override, compiled config,
     `chat_template.jinja`, `chat_template.json`, then
     `tokenizer_config.json`.
   - GGUF: embedded `tokenizer.chat_template`. HF overrides do not apply;
     adding them is a separate runtime contract change.
3. Verify BOS/EOS/PAD/stop/image/control strings and IDs.
4. Define accepted system, user, assistant, image, tool-definition, tool-call,
   and tool-result messages.
5. Compare reference and Minja rendering exactly.
6. Fix the smallest boundary:
   - asset selection: ingestion/packaging;
   - standard Jinja syntax: Minja;
   - different message protocol: model-specific runtime handling.

## Minja Ownership

Implement and test Minja fixes in the repository configured by
`third_party/minja` in `.gitmodules`; then update the LLiMa submodule pointer.
Never leave a dirty vendored checkout or unexplained local patch.

Prefer standards-compatible syntax support over template rewrites. Run the
focused Minja regression, confirm
`git -C third_party/minja status --short` is clean, and record the upstream
commit.

## Example: Tool-Aware Template

For a supported architecture whose template adds tools:

- compare ordinary, multi-turn, tool-definition, tool-call, and tool-result
  messages with the upstream oracle;
- distinguish syntax, asset-selection, and protocol differences;
- keep the upstream template when Minja can render it;
- add model-specific runtime code only for a genuinely different protocol; and
- test exact prompt/token IDs, malformed-tool recovery, packaged assets, and
  CLI or HTTP generation on Modalix.
