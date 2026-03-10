---
name: read-arxiv-paper
description: Use this skill when asked to read an arxiv paper given an arxiv URL
---

**IMPORTANT: This skill is for interactive sessions only. Do NOT use during autonomous experiment loops (program.md). If running autonomously, read existing summaries in knowledge/ instead.**

You will be given a URL of an arxiv paper, for example:

https://www.arxiv.org/abs/2601.07372

### Part 1: Normalize the URL

The goal is to fetch the TeX Source of the paper (not the PDF!), the URL always looks like this:

https://www.arxiv.org/src/2601.07372

Notice the /src/ in the url. Once you have the URL:

### Part 2: Download the paper source

Fetch the url to a local .tar.gz file. A good location is `~/.cache/nanostate/knowledge/{arxiv_id}.tar.gz`.

(If the file already exists, there is no need to re-download it).

### Part 3: Unpack the file in that folder

Unpack the contents into `~/.cache/nanostate/knowledge/{arxiv_id}` directory.

### Part 4: Locate the entrypoint

Every latex source usually has an entrypoint, such as `main.tex` or something like that.

### Part 5: Read the paper

Once you've found the entrypoint, Read the contents and then recurse through all other relevant source files to read the paper.

### Part 6: Report

Once you've read the paper, produce a summary into a markdown file at `./knowledge/summary_{tag}.md`. Use the local knowledge directory (easier to open and reference),
not `~/.cache`. Generate a reasonable tag like `hippo_initialization` or `selective_state_spaces`. Make sure the tag doesn't exist yet.

The summary should be written in the context of the nanostate project: how does this paper relate to our S4D implementation in train.py? What ideas could we try? Read
the relevant parts of train.py and explicitly connect the paper's insights to our code.

Adapted from [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) read-arxiv-paper skill, modified for nanostate's SSM focus.
