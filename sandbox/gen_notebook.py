import json

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Redrob-v4: Team Antigravity - Sandbox Environment\n",
        "\n",
        "This notebook serves as the reproducible sandbox environment. It runs the full ranking pipeline end-to-end (Stages 1 through 5, including feature extraction, BM25 indexing, and embedding generation) on a small sample of candidates (<= 100).\n",
        "\n",
        "It completes well within the 5 minute compute budget on a standard CPU."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 1. Clone the repository\n",
        "!git clone https://github.com/achyutadixit/redrob-v4.git\n",
        "%cd redrob-v4\n",
        "\n",
        "# 2. Install dependencies\n",
        "!pip install sentence-transformers rank_bm25 numpy pandas"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Provide Sample Data\n",
        "You can upload your own `candidates.jsonl` via the Colab file browser (on the left panel) into the `redrob-v4` directory, or use the cell below to generate a dummy sample of 10 candidates."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "\n",
        "dummy_candidates = [\n",
        "    {\n",
        "        \"candidate_id\": f\"CAND_99990{i}\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 6.0 + i,\n",
        "            \"current_title\": \"Machine Learning Engineer\",\n",
        "            \"location\": \"Pune, Maharashtra\",\n",
        "            \"summary\": \"Experienced in building retrieval systems using FAISS, embeddings, PyTorch and Pinecone. Active contributor on github.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"Tech Startup Inc\",\n",
        "                \"title\": \"Machine Learning Engineer\",\n",
        "                \"duration_months\": 36,\n",
        "                \"description\": \"Built production vector search systems using FAISS, Elasticsearch and Qdrant. Evaluated using NDCG and A/B testing.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"Python\", \"proficiency\": \"Expert\", \"duration_months\": 60},\n",
        "            {\"name\": \"Machine Learning\", \"proficiency\": \"Expert\", \"duration_months\": 60},\n",
        "            {\"name\": \"FAISS\", \"proficiency\": \"Intermediate\", \"duration_months\": 24}\n",
        "        ],\n",
        "        \"redrob_signals\": {\n",
        "            \"notice_period_days\": 30,\n",
        "            \"recruiter_response_rate\": 0.9\n",
        "        }\n",
        "    } for i in range(10)\n",
        "]\n",
        "\n",
        "with open(\"sample_candidates.jsonl\", \"w\") as f:\n",
        "    for c in dummy_candidates:\n",
        "        f.write(json.dumps(c) + \"\\n\")\n",
        "\n",
        "print(\"Created sample_candidates.jsonl with 10 records.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Run the Full Pipeline\n",
        "By passing `--candidates`, the script automatically reruns all precompute stages (including embedding generation) before ranking."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Run the end-to-end pipeline\n",
        "!python rank.py --candidates sample_candidates.jsonl --out outputs/sandbox_submission.csv"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### View Results"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('outputs/sandbox_submission.csv')\n",
        "display(df)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("sandbox/sandbox.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
