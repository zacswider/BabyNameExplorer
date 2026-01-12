## Overview

This is a 99% vibe-coded explorer of the [Kaggle baby names dataset](https://www.kaggle.com/datasets/thedevastator/us-baby-names-by-year-of-birth). It allows you to view historical trends for names ranked by recent popularity with some very basic filter and search functionality.

### Run with UV

By far the simplest way to get this code running on your machine is to use UV to automatically install the dependencies and start the main entry point. Visit the [UV website](https://docs.astral.sh/uv/getting-started/installation/) and follow the instructions to install the tool. As of Jan 2026, you can do this with:

MacOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once you have installed UV, you can run the Baby Names Explorer with:

```bash
uv run "https://raw.githubusercontent.com/zacswider/BabyNameExplorer/main/explore.py"
```
