# Documentation localization

English source pages remain in `docs/`. Localized mirrors use the same relative
paths below `docs/i18n/<locale>/`.

Supported locales are Korean (`ko`), Japanese (`ja`), Traditional Chinese for
Taiwan (`zh-Hant` / `zh-Hant-TW`), and Ukrainian for Ukraine (`uk` / `uk-UA`).
Product and model names, commands, options, API symbols, paths, links, code,
and literal output remain unchanged. Generated translations are drafts until
reviewed by a native technical reviewer.

Run `sima-i18n check --require-complete` from the repository root to validate
coverage, protected Markdown structure, and source freshness. Use
`sima-i18n translate --locale <locale> --all --write` to add missing pages.
`translation-sources.json` is maintained by the CLI and must not be edited by
hand.
