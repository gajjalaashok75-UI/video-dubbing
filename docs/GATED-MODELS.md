# Gated models: getting the voice model

Multiva's voice model, [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5),
is a **gated** repository on the Hugging Face Hub. Gated means the author
requires you to accept a licence before the weights can be downloaded, and the
download must be authenticated with a token. An anonymous fetch is refused with
`401 Client Error ... Cannot access gated repo`, and writes only the README.

Everything else Multiva downloads is open and needs no account. This one model
does, and it is the model that speaks the dub, so without it no video can be
dubbed into any Indian language. Transcription, translation, subtitles and lip
sync all work without it.

## Get the weights (once)

1. **Sign in to Hugging Face** and open the model page:
   <https://huggingface.co/ai4bharat/IndicF5>

2. **Accept the licence.** Click the prompt on the model page and confirm. The
   page will say you have been granted access. This is per-account, and it is
   instant.

3. **Create a read token** at <https://huggingface.co/settings/tokens>. A
   `read` token is enough; nothing here needs write access.

4. **Put the token in `.env`**, beside `run.sh`:

   ```bash
   cp .env.example .env
   # then edit .env and set:
   # HF_TOKEN=hf_...
   ```

   Exporting it in the shell works too, for one run:

   ```bash
   export HF_TOKEN=hf_...
   ```

   A real environment variable wins over the `.env` file. Both are read the
   same way by `run.sh`, the studio's setup page, and
   `scripts/download_models.py`.

5. **Download the models.** Either press *Download everything missing* on the
   studio's setup page, or run:

   ```bash
   ./venv/bin/python scripts/download_models.py
   ```

   `--check` reports state without downloading:

   ```bash
   ./venv/bin/python scripts/download_models.py --check
   ```

   A present gated model prints `[present]`; a missing one prints
   `[MISSING] ... (gated - needs HF_TOKEN)`.

## What happens without a token

Nothing breaks. A gated model that cannot be fetched is **skipped silently**
and the rest of the download continues:

- `scripts/download_models.py` prints `skipped:` with the reason, finishes the
  other models, and exits `0`.
- The studio's setup page lists the model as missing and shows the reason.
- `/api/boot` carries a note naming this document.
- `/languages` marks the Indian languages `available: false` with the same
  hint, so the studio refuses a dub into one up front instead of failing
  several minutes in.
- Starting a dub into an Indian language returns `400` with the steps above.

## Why the weights are checked by file, not by folder

`models.is_cached()` treats a model as present only when the files it actually
loads are on disk (`config.json`, `model.safetensors`, and so on). A folder
containing nothing but a `README.md` is not a model. This matters precisely for
gated repos: the failed fetch is what leaves the README behind, and counting it
as "downloaded" is how a gated model used to look ready while every render
failed.

## If you have a token and it still fails

- **Access not granted yet.** The token only works after step 2. Re-open the
  model page; it should show your access as granted.
- **Wrong token type.** A `read` token is required. Fine-grained tokens need
  *Read access to contents of all public gated repos you can access* enabled.
- **The token is not reaching the process.** Check it is visible:
  `./venv/bin/python -c "import sys; sys.path.insert(0,'engine'); import models; print(bool(models.hf_token()))"`.
  It prints `True` when the token is seen.
- **A different gated model.** Add it to `MODELS` in `engine/models.py` with
  `gated=True` and the files it loads, and it is handled the same way.

## Do not use an unofficial mirror

Unofficial mirrors of IndicF5 exist on the Hub with byte-identical weights.
They are not a safe substitute: the model is loaded with
`AutoModel.from_pretrained(..., trust_remote_code=True)`, which **executes the
repository's own `model.py`**. The official `model.py` lives only in the gated
repository, so there is nothing to verify a mirror's copy against. Point the
loader at the official repo and use a token.
