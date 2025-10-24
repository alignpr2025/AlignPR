
AlignPaper2Repo

AlignPaper2Repo provides the reproducibility package and Visual Studio Code extension for our research work Align2Repo — enabling alignment between research papers and their corresponding source code repositories.
Installation
Option 1: Install from Source

Clone this repository and install the VS Code extension manually:

cd src
vsce package
code --install-extension alignpr.AlignPaper2Repo-*.vsix

Option 2: Install from Marketplace

You can also install directly from the Visual Studio Marketplace:
AlignPaper2Repo on VS Code Marketplace
Model Setup

To reproduce experiments, you can use proprietary or open-source models.
Proprietary Models
Model	API Endpoint	Embedding Model	Notes
GPT-5 Pro	https://api.openai.com/v1	text-embedding-3-small	Requires OpenAI API key
Gemini-2.5 Pro	https://generativelanguage.googleapis.com/v1beta/openai/	gemini-embedding-001	Requires Google API key
Claude-4.5 Sonnet	https://openrouter.ai/api/v1	text-embedding-3-small	Requires OpenRouter API key
Open-Source Models
Model	Endpoint	API Key	Notes
Llama-3 8B / Llama-3.3 70B	http://localhost:11434/v1	ollama	Pull and run locally using Ollama
Qwen-3 8B / Qwen-3 30B	http://localhost:11434/v1	ollama	No API key required
Tip: You can adjust model endpoints and API keys from
VS Code Settings → Extensions → AlignPaper2Repo Configuration (Cmd + , on macOS).
Running Experiments
RQ1 – Code–Paper Matching

To reproduce RQ1 results:

Upload your CSV dataset via the AlignPaper2Repo VS Code extension.

Record latency and cost (required for RQ4).

Run:

python src/evaluate.py --gt ./results_117/dataset.csv --models_folder ./results_117/


Results are stored in the compare_out folder.
RQ2 & RQ3 – Alignment Confidence and Misattribution

The folder rq_2_3_data contains paper contents and corresponding implementation files.
1. Run one of the following scripts:
python gemini.py
python gpt.py
python claude.py
python open_source.py
2.The results are saved under results_for_rq2_rq3/.
Generate plots using:
python plot.py --folder ./results_for_rq2_rq3
VS Code Extension Usage
Open VS Code and install AlignPaper2Repo.

Press Cmd + Shift + P, search for AlignPaper2Repo: Configure API Keys, and enter keys for your chosen models.

Press Cmd + Shift + P again → choose Open AlignPaper2Repo Panel.

Upload a paper and its implementation repository.

Click Build Index and wait for the completion notification.

You’ll see:

A confidence indicator for each prediction

Five suggested prompts based on paper context

A result view showing the relevant code span and a concise explanation

The extension highlights how confident the LLM is in locating the correct code segment based on your query.

Reproducibility Notes

RQ1 results can be reproduced from data under results_117/.

RQ2 and RQ3 rely on manually validated paper–repository pairs.

RQ4 measures latency and cost across all models under consistent hardware and API conditions.

Citation

If you use AlignPaper2Repo, please cite our paper (to appear in ICSE/FSE 2025).

AlignPaper2Repo (2025). Visual Studio Code Extension.
https://marketplace.visualstudio.com/items?itemName=alignpr.AlignPaper2Repo
