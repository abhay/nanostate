---
name: read-github-code
description: Use this skill when you want to look at a reference implementation from GitHub for inspiration (e.g. Mamba, S4, RWKV)
---

**IMPORTANT: This skill is for interactive sessions only. Do NOT use during autonomous experiment loops (program.md). If running autonomously, read existing summaries in knowledge/ instead.**

You will be given a GitHub repository URL or a specific file path, for example:

https://github.com/state-spaces/mamba

### Step 1: Identify the key files

Use the gh CLI to list the repo contents and find the core implementation files. For ML repos this is usually in `src/`, `model/`, or the root. Focus on the model
definition and training loop. Skip tests, configs, and boilerplate.

```bash
gh api repos/{owner}/{repo}/contents/{path} --jq '.[].name'
```

### Step 2: Read the implementation

Fetch the raw content of the key files:

```bash
gh api repos/{owner}/{repo}/contents/{file_path} --jq '.content' | base64 -d
```

Read the core model code. For SSM repos, focus on:
- State space parameterization (how A, B, C, D are defined)
- Discretization method
- Scan/convolution implementation
- Gating and activation functions
- Initialization
- Block structure and residual connections

### Step 3: Connect to nanostate

Read our train.py and explicitly compare:
- What does their implementation do that ours doesn't?
- What initialization do they use vs our random init?
- What gating/activation patterns do they use?
- How is their block structure different from our SSMBlock?

### Step 4: Report

Write a summary to `knowledge/reference_{tag}.md` where tag describes the repo (e.g. `mamba_impl`, `s4_original`, `rwkv_v6`). Include specific code snippets from both their
implementation and ours, with concrete suggestions for what to adapt.
