---
name: initiate-boglodite
description: "Use when initializing the boglodite repository structure and cloning required tool repositories."
---

# Initiate Boglodite

In the repository root, create the required directories:

```bash
mkdir data
mkdir sandbox
mkdir tools
```

Inside the `data` directory, download the preloaded F3 seismic dataset (1.2 GB):

```bash
gdown --folder "https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs?resourcekey=0-0ZhV_OJ3TKN1ShFAGcrOzQ" -O ./data/
```

Inside the `tools` directory, clone the required repositories:

```bash
git clone https://github.com/bolgebrygg/MalenoV
git clone https://github.com/crild/facies_net
```

