# AlignPaper2Repo

**AlignPaper2Repo**  provides the reproducibility package and Visual Studio Code extension for our research work Align2Repo — enabling alignment between research papers and their corresponding source code repositories.
##dataset
Please download the dataset from this link and put it in the root folder.
## Installation

## Option 1: Install from Source

Clone this repository and install the VS Code extension manually:

```bash
cd src
vsce package
code --install-extension alignpr.AlignPaper2Repo-*.vsix

```
## Option 2: Install from Marketplace
You can also install directly from the Visual Studio Marketplace:
[Align2PaperRepo on vs code market place](https://marketplace.visualstudio.com/items?itemName=alignpr.AlignPaper2Repo)
## Model Setup

To reproduce experiments, you can use proprietary or open-source models.

### Proprietary models
#### For GPT 5
```python
https://api.openai.com/v1
text-embedding-3-small
```
#### For Gemini
```python
https://generativelanguage.googleapis.com/v1beta/openai/text-embedding-3-small
gemini-embedding-001
```
#### For Claude
```python
https://openrouter.ai/api/v1
text-embedding-3-small
```
#### For llama and qwen open source models


```python
http://localhost:11434/v1
mxbai-embed-large
```
### TIPS:
You can adjust model endpoints and API keys from
VS Code Settings → Extensions → AlignPaper2Repo Configuration (Cmd + , on macOS).
## Running Experiments
### RQ1 – Code–Paper Matching
To reproduce RQ1 results:

1. Upload your CSV dataset via the AlignPaper2Repo VS Code extension.

2. Record latency and cost (required for RQ4).
```python
python src/evaluate.py --gt ./results_117/dataset.csv --models_folder ./results_117/
```
Results are stored in the **compare_out** folder.
### RQ2 & RQ3 – Alignment Confidence and Misattribution

The folder rq_2_3_data contains paper contents and corresponding implementation files.
1.Run one of the following scripts:
```python
python gemini.py
python gpt.py
python claude.py
python open_source.py
```
2.The results are saved under results_for_rq2_rq3/.

3.Generate plots using:
```python
python plot.py --folder ./results_for_rq2_rq3
```
## VS Code Extension Usage

1. Open VS Code and install AlignPaper2Repo.

2. Press Cmd + Shift + P, search for AlignPaper2Repo: Configure API Keys, and enter keys for your chosen models.

Press Cmd + Shift + P again → choose Open AlignPaper2Repo Panel.

3. Upload a paper and its implementation repository.

Click Build Index and wait for the completion notification.

You’ll see:

-> A confidence indicator for each prediction

-> Five suggested prompts based on paper context

-> A result view showing the relevant code span and a concise explanation

-> The extension highlights how confident the LLM is in locating the correct code segment based on your query.
Run:
## Reproducibility Notes

RQ1 results can be reproduced from data under results_117/.

RQ2 and RQ3 rely on manually validated paper–repository pairs.

RQ4 measures latency and cost across all models under consistent hardware and API conditions.

## Extension
 
[Visual Studio Code Extension](https://marketplace.visualstudio.com/items?itemName=alignpr.AlignPaper2Repo)


## License

[MIT](https://choosealicense.com/licenses/mit/)