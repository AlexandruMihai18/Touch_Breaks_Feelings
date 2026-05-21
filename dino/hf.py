from __future__ import annotations


def auth_kwargs(token: str | None) -> dict:
    return {"token": token} if token else {}


def explain_hf_load_error(exc: OSError, model_id: str) -> OSError:
    message = str(exc).lower()
    if "gated repo" not in message and "restricted" not in message and "401" not in message:
        return exc
    return OSError(
        f"Cannot access Hugging Face model {model_id!r}. This checkpoint is likely gated. "
        "Request access on Hugging Face, then run `huggingface-cli login`, or pass "
        "`--hf-token YOUR_TOKEN`."
    )

