import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from retrieval.retrieve_docs import retrieve_relevant_docs

# ---- Config ----
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
TOP_K_DOCS = 2

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
    return (
        template.replace("{REQUEST}", request)
        .replace("{DOCUMENTATION}", documentation)
    )


def call_llm(client: OpenAI, prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=TEMPERATURE,
    )
    return resp.output_text


def main():
    print("=== Week 4 Experiment Runner starting ===")

    key_present = bool(os.getenv("OPENAI_API_KEY"))
    print(f"OPENAI_API_KEY present? {key_present}")

    if not key_present:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this terminal session.\n"
            "Run: export OPENAI_API_KEY='sk-...'\n"
            "Then re-run: python3 experiments/run_experiments.py"
        )

    # Load prompts
    prompt_naive = load_prompt("naive.txt")
    prompt_structured = load_prompt("structured.txt")
    prompt_structured_rag = load_prompt("structured_rag.txt")

    # Load + parse requests
    req_text = REQUESTS_FILE.read_text(encoding="utf-8")
    requests = parse_numbered_requests(req_text)
    if not requests:
        raise RuntimeError("No requests parsed. Check data/requests/requests.txt format.")

    # Prepare output directory
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / run_id
    (run_dir / "naive").mkdir(parents=True, exist_ok=True)
    (run_dir / "structured").mkdir(parents=True, exist_ok=True)
    (run_dir / "structured_rag").mkdir(parents=True, exist_ok=True)

    manifest: Dict = {
        "run_id": run_id,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_k_docs": TOP_K_DOCS,
        "requests": [],
    }

    client = OpenAI()

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
