import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from retrieval.retrieve_docs import retrieve_relevant_docs

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
SYSTEM_MSG = "You are a helpful DevOps planning assistant."

TEMPERATURE = 0.2
TOP_K_DOCS = 2

MAX_TOKENS = 2600  # reasonable on free tier

MAX_DOC_CHARS = 8000  # prevents RAG prompt from getting huge

REQUESTS_FILE = Path("data/requests/requests.txt")
PROMPTS_DIR = Path("prompts")
OUTPUTS_DIR = Path("outputs")


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def parse_numbered_requests(text: str) -> List[Tuple[str, str]]:
    """
    Parses requests formatted like:
      1. ...
      2. ...
    Returns list of (request_id, request_text)
    """
    pattern = r"(?m)^\s*(\d+)\.\s*(.+?)(?=^\s*\d+\.|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    out = []
    for num, body in matches:
        cleaned = " ".join(body.strip().split())
        out.append((f"req{num}", cleaned))
    return out


def render_prompt(template: str, request: str, documentation: str = "") -> str:
    return template.replace("{REQUEST}", request).replace("{DOCUMENTATION}", documentation)


def get_hf_token() -> str:
    """
    Accept either:
      - HF_TOKEN (what your script uses)
      - HUGGINGFACEHUB_API_TOKEN (common HF env var)
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return token or ""


def call_llm(client: OpenAI, prompt: str) -> str:
    """
    HF router can rate-limit or throw transient 5xx.
    Retry with exponential backoff + jitter.
    """
    last_err = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            return (content or "").strip()

        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[warn] LLM call failed (attempt {attempt+1}/6): {e}")
            print(f"[warn] Retrying in {sleep_s:.1f}s...\n")
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def sanity_check(client: OpenAI) -> None:
    """
    Cheap test call so you know the token/base_url/model combo works
    before running the full 15-call experiment.
    """
    print("Running sanity check call...")
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0,
        max_tokens=10,
    )
    out = (resp.choices[0].message.content or "").strip()
    print(f"Sanity check output: {out!r}")
    if out != "OK":
        print("[warn] Sanity check did not return exactly 'OK' (not fatal), but token/model are responding.")


def main():
    print("=== HF Router Experiment Runner starting ===")

    token = get_hf_token()
    token_present = bool(token)
    print(f"HF token present? {token_present}")

    if not token_present:
        raise RuntimeError(
            "No Hugging Face token found in env.\n"
            "Set one of these in THIS terminal session:\n"
            "  export HF_TOKEN='hf_...'\n"
            "  export HUGGINGFACEHUB_API_TOKEN='hf_...'\n"
            "Then re-run: python3 -m experiments.run_experiments_hf"
        )

    client = OpenAI(base_url=HF_BASE_URL, api_key=token)

    # quick validation before spending calls
    sanity_check(client)

    # Load prompts
    prompt_naive = load_prompt("naive.txt")
    prompt_structured = load_prompt("structured.txt")
    prompt_structured_rag = load_prompt("structured_rag.txt")

    # Load + parse requests
    req_text = REQUESTS_FILE.read_text(encoding="utf-8")
    requests = parse_numbered_requests(req_text)
    if not requests:
        raise RuntimeError("No requests parsed. Check data/requests/requests.txt format.")

    # Create run folder
    run_id = datetime.now().strftime("hf_run_%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / run_id
    (run_dir / "naive").mkdir(parents=True, exist_ok=True)
    (run_dir / "structured").mkdir(parents=True, exist_ok=True)
    (run_dir / "structured_rag").mkdir(parents=True, exist_ok=True)

    manifest: Dict = {
        "run_id": run_id,
        "backend": "hf_router_openai_compat",
        "base_url": HF_BASE_URL,
        "model": HF_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_k_docs": TOP_K_DOCS,
        "max_doc_chars": MAX_DOC_CHARS,
        "requests": [],
    }

    for req_id, req in requests:
        print(f"\nRunning {req_id}: {req}")

        # Variant A: naive
        p_a = render_prompt(prompt_naive, request=req)
        out_a = call_llm(client, p_a)
        (run_dir / "naive" / f"{req_id}.md").write_text(out_a, encoding="utf-8")

        # Variant B: structured
        p_b = render_prompt(prompt_structured, request=req)
        out_b = call_llm(client, p_b)
        (run_dir / "structured" / f"{req_id}.md").write_text(out_b, encoding="utf-8")

        # Variant C: structured + grounded
        docs = retrieve_relevant_docs(req, top_k=TOP_K_DOCS)
        docs_blob = "\n\n---\n\n".join(docs)

        # Guardrail: cap documentation injected into prompt
        docs_blob = docs_blob[:MAX_DOC_CHARS]

        p_c = render_prompt(prompt_structured_rag, request=req, documentation=docs_blob)
        out_c = call_llm(client, p_c)
        (run_dir / "structured_rag" / f"{req_id}.md").write_text(out_c, encoding="utf-8")

        manifest["requests"].append(
            {
                "request_id": req_id,
                "request_text": req,
                "saved_outputs": {
                    "naive": str(run_dir / "naive" / f"{req_id}.md"),
                    "structured": str(run_dir / "structured" / f"{req_id}.md"),
                    "structured_rag": str(run_dir / "structured_rag" / f"{req_id}.md"),
                },
            }
        )

    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone! Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
