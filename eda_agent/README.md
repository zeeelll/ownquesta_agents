# EDA Agent

This package provides a small LangChain-based exploratory data analysis (EDA) agent.

Quick start

1. Install dependencies (see ../requirements.txt)

2. Run the CLI:

```bash
python -m eda_agent.config path/to/data.csv
```

Or import the agent in Python:

```py
from eda_agent import create_agent
agent = create_agent()
result = agent.run("Perform complete exploratory data analysis on file data.csv")
```
