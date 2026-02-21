import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

from openai import OpenAI

from retrieval.retrieve_docs import retrieve_relevant_docs

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
SYSTEM_MSG = "You are a helpful DevOps planning assistant."

TEMPERATURE = 0.2

# ReAcTree-lite constraints (deterministic)
MAX_SUBTASKS = 7
TOP_K_DOCS_PER_NODE = 1

# Token budgets (bounded + reproducible)
DECOMPOSE_MAX_TOKENS = 900
EXPAND_MAX_TOKENS = 2200

# Guardrail for injected docs
MAX_DOC_CHARS = 8000

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


def render_template(template: str, mapping: Dict[str, str]) -> str:
    """
    Supports both styles so we don't get tripped up:
      {REQUEST} / {DOCUMENTATION}
      {{REQUEST}} / {{SUBTASK_JSON}} / {{DOC_SNIPPET}}
    """
    out = template
    for k, v in mapping.items():
        out = out.replace(f"{{{k}}}", v)          # {KEY}
        out = out.replace(f"{{{{{k}}}}}", v)      # {{KEY}}
    return out


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return token or ""


def call_llm(client: OpenAI, prompt: str, max_tokens: int) -> str:
    """
    Retry with exponential backoff + jitter (same pattern as Week 5).
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
                max_tokens=max_tokens,
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


def extract_json_object(text: str) -> Any:
    """
    Robustly extract JSON from model output:
    - Strips ```json fences if present
    - Attempts full json.loads
    - If that fails, finds the first {...} block and parses it
    """
    cleaned = text.strip()

    # Remove code fences if present
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(m.group(0))


def render_tree_markdown(request_text: str, subtasks: List[Dict[str, Any]], expansions: Dict[str, str]) -> str:
    lines = []
    lines.append("# ReAcTree-lite Plan\n")
    lines.append("## Original Request")
    lines.append(request_text.strip() + "\n")

    lines.append("## Subtasks")
    for st in subtasks:
        sid = st.get("id", "")
        title = st.get("title", "")
        lines.append(f"- **{sid}**: {title}")
    lines.append("\n---\n")

    for st in subtasks:
        sid = st.get("id", "")
        title = st.get("title", "")
        lines.append(f"## {sid}: {title}\n")
        lines.append(expansions.get(sid, "_(no expansion output)_"))
        lines.append("\n---\n")

    return "\n".join(lines)


def main():
    print("=== ReAcTree-lite Runner starting ===")

    token = get_hf_token()
    print(f"HF token present? {bool(token)}")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found in env.\n"
            "Set one of these in THIS terminal session:\n"
            "  export HF_TOKEN='hf_...'\n"
            "  export HUGGINGFACEHUB_API_TOKEN='hf_...'\n"
            "Then re-run: python3 -m experiments.run_experiments_tree"
        )

    client = OpenAI(base_url=HF_BASE_URL, api_key=token)
    sanity_check(client)

    # Load prompts you created
    prompt_decompose = load_prompt("decompose_tree.txt")
    prompt_expand = load_prompt("expand_node.txt")

    # Load + parse requests
    req_text = REQUESTS_FILE.read_text(encoding="utf-8")
    requests = parse_numbered_requests(req_text)
    if not requests:
        raise RuntimeError("No requests parsed. Check data/requests/requests.txt format.")

    # Create run folder
    run_id = datetime.now().strftime("hf_tree_%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / run_id
    tree_dir = run_dir / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict = {
        "run_id": run_id,
        "backend": "hf_router_openai_compat",
        "base_url": HF_BASE_URL,
        "model": HF_MODEL,
        "temperature": TEMPERATURE,
        "max_subtasks": MAX_SUBTASKS,
        "top_k_docs_per_node": TOP_K_DOCS_PER_NODE,
        "max_doc_chars": MAX_DOC_CHARS,
        "decompose_max_tokens": DECOMPOSE_MAX_TOKENS,
        "expand_max_tokens": EXPAND_MAX_TOKENS,
        "requests": [],
    }

    for req_id, req in requests:
        print(f"\n[DECOMPOSE] {req_id}: {req}")

        decompose_prompt = render_template(
            prompt_decompose,
            {"REQUEST": req}
        )
        decompose_out = call_llm(client, decompose_prompt, max_tokens=DECOMPOSE_MAX_TOKENS)

        # Parse JSON tree
        try:
            tree_obj = extract_json_object(decompose_out)
        except Exception as e:
            # Save raw for debugging
            raw_path = tree_dir / f"{req_id}_decompose_raw.txt"
            raw_path.write_text(decompose_out, encoding="utf-8")
            raise RuntimeError(f"Decompose output not parseable JSON for {req_id}. Saved raw to {raw_path}. Error: {e}")

        subtasks = (tree_obj.get("subtasks") or [])[:MAX_SUBTASKS]

        tree_json_path = tree_dir / f"{req_id}_tree.json"
        tree_json_path.write_text(
            json.dumps({"request": req, "subtasks": subtasks}, indent=2),
            encoding="utf-8",
        )

        expansions: Dict[str, str] = {}

        for st in subtasks:
            sid = st.get("id", "")
            title = st.get("title", "")
            goal = st.get("goal", "")
            print(f"[EXPAND] {req_id} {sid}: {title}")

            subtask_json = json.dumps(st, indent=2)

            # Per-node retrieval (top_k=1)
            docs = retrieve_relevant_docs(
                query=f"{req}\nSubtask: {title}\nGoal: {goal}",
                top_k=TOP_K_DOCS_PER_NODE
            )
            docs_blob = "\n\n---\n\n".join(docs)[:MAX_DOC_CHARS]

            expand_prompt = render_template(
                prompt_expand,
                {
                    "REQUEST": req,
                    "SUBTASK_JSON": subtask_json,
                    "DOC_SNIPPET": docs_blob,
                }
            )

            expansions[sid or title or goal or f"subtask_{len(expansions)+1}"] = call_llm(
                client,
                expand_prompt,
                max_tokens=EXPAND_MAX_TOKENS
            )

            time.sleep(0.2)

        tree_md = render_tree_markdown(req, subtasks, expansions)
        tree_md_path = tree_dir / f"{req_id}_tree.md"
        tree_md_path.write_text(tree_md, encoding="utf-8")

        manifest["requests"].append(
            {
                "request_id": req_id,
                "request_text": req,
                "saved_outputs": {
                    "tree_json": str(tree_json_path),
                    "tree_md": str(tree_md_path),
                },
            }
        )

    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅ Done! Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()