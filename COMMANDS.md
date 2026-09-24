# VeriRAG: commands

Run everything from the project folder (`~/verirag`) with the virtual environment active.

## 1. One-time setup

```bash
mkdir verirag && cd verirag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install fonttools          # optional: silences PDF font warnings
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`

## 2. Hugging Face token

1. Go to https://huggingface.co/settings/tokens
2. Create a **Fine-grained** token and tick **Make calls to Inference Providers**
3. Set it in every new terminal:

```bash
export HF_TOKEN=hf_your_token
echo $HF_TOKEN                 # check it is set
```

Windows PowerShell: `$env:HF_TOKEN="hf_your_token"`

Do not set `HF_HUB_OFFLINE=1`. It blocks the hosted LLM call. If it is set, run `unset HF_HUB_OFFLINE`.

## 3. Mode A: any arXiv paper (main app)

```bash
python app_paper.py
```

Open http://127.0.0.1:7860, enter a paper title, arXiv ID or URL, then ask a question. Stop with `Ctrl+C`.

The first question on a new paper is slower (download, chunk, index). Later questions on the same paper are fast.

Quick test without the UI:

```bash
python -c "
from paperqa import PaperQA
qa = PaperQA()
r = qa.ask('Attention Is All You Need', 'What optimizer was used?')
print(r['answer'])
print('grounded:', r['grounded'], 'first draft:', r['initial'], 'avg entailment:', r['avg_entail'], 'loops:', r['loops'])
"
```

Test the paper loader alone:

```bash
python -c "from paperloader import load_paper; m,c=load_paper('Attention Is All You Need'); print(m['title'], len(c), 'chunks')"
```

Suggested tests:
1. `Attention Is All You Need`: "What optimizer and learning rate schedule were used?"
2. `1706.03762` (arXiv ID path, uses the cached PDF)
3. Off-topic question on the same paper (should abstain or show low scores)
4. A different field, e.g. `Denoising Diffusion Probabilistic Models`
5. Untick **Verifier loop** and ask again: the line should read `Verifier off`

## 4. Mode B: fixed quantum error correction corpus

```bash
python ingest.py               # downloads ~30 PDFs, writes chunks.jsonl (a few minutes)
wc -l chunks.jsonl
head -c 400 chunks.jsonl
```

Smoke test (the first run builds the FAISS index):

```bash
python -c "from pipeline import VeriRAG; r=VeriRAG().ask('What is the surface code?'); print(r['answer']); print('grounded:', r['grounded'], 'loops:', r['loops'])"
```

Launch the corpus UI:

```bash
python app.py
```

If you change the corpus or re-run `ingest.py`, delete the old index first:

```bash
rm -rf index
```

## 5. Evaluation (baseline vs hybrid vs full VeriRAG)

Add more questions to `data/eval_questions.json` first (aim for about 20).

```bash
python evaluate.py
cat results.json
```

It prints faithfulness, abstain rate and average claims per configuration, scored by an independent NLI judge. It makes many LLM calls, so it can use up free credits.

## 6. Change settings

```bash
grep LLM_MODEL config.py                                                                 # current LLM
sed -i 's|^LLM_MODEL = .*|LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"|' config.py    # switch model
grep ENTAIL_THRESHOLD config.py                                                          # strictness (default 0.5)
sed -i 's|^ENTAIL_THRESHOLD = .*|ENTAIL_THRESHOLD = 0.8|' config.py                      # stricter check
```

Thresholds above about 0.9 start rejecting correct paraphrases.

List chat models your token can call:

```bash
python -c "
import os, requests
r = requests.get('https://router.huggingface.co/v1/models', headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN']}, timeout=30)
print('\n'.join(sorted(m['id'] for m in r.json()['data'])))
"
```

Avoid reasoning models (Qwen3, DeepSeek-R1). They emit `<think>` text that breaks claim splitting and the planner's JSON.

## 7. Network: fix flaky DNS (Linux, NetworkManager)

```bash
nmcli -t -f NAME,TYPE connection show --active       # find your Wi-Fi connection name
nmcli connection modify "<wifi name>" ipv4.dns "8.8.8.8 1.1.1.1" ipv4.ignore-auto-dns yes && nmcli connection up "<wifi name>"
resolvectl status wlo1 | grep "DNS Servers"          # should show 8.8.8.8 1.1.1.1
getent hosts huggingface.co export.arxiv.org         # should print IPs
```

Use your real connection name in place of `<wifi name>` (keep the quotes).

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` from router.huggingface.co | Token lacks Inference Providers permission | Create a fine-grained token with that permission |
| `model_not_supported` | Model not served by your enabled providers | Pick a model from the list command in section 6 |
| `ImportError: cannot import name 'VeriRAG'` | `pipeline.py` is empty or missing | Check `wc -l *.py` for 0-line files |
| `Errno -3 Temporary failure in name resolution` | Flaky DNS | See section 7 |
| `OfflineModeIsEnabled` | `HF_HUB_OFFLINE=1` is set | `unset HF_HUB_OFFLINE` |
| `total chunks: 0` after ingest | Old `ingest.py` using `download_pdf` | Use the `requests`-based version |
| `Insufficient evidence...` on a good question | Threshold or retrieval too strict | Lower `ENTAIL_THRESHOLD`, or raise `per_q` in `pipeline.py` |
| `No arXiv paper found` | Paper not on arXiv, or title too vague | Use the arXiv ID |
| `fontTools is required...` spam | PDF fonts, harmless | `pip install fonttools` |
| Claim table looks cut off | Table is wider than the panel | Scroll it sideways to see all four columns |
| An empty `.py` file after `Ctrl+C` on `cat >` | `cat > file` truncates the file immediately | Re-create the file with the full heredoc |

## 9. Git (optional)

```bash
git init && printf ".venv/\n__pycache__/\npapers/\nindex/\nchunks.jsonl\nresults.json\n" > .gitignore
git add . && git commit -m "add VeriRAG self-auditing arXiv QA agent"
```
