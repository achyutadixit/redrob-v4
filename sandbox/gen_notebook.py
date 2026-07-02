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
        "You can upload your own `candidates.jsonl` via the Colab file browser (on the left panel) into the `redrob-v4` directory, or use the cell below to generate a dummy sample of 10 diverse candidates."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "import random\n",
        "\n",
        "dummy_candidates = [\n",
        "    # 1. The Perfect Fit\n",
        "    {\n",
        "        \"candidate_id\": \"CAND_999901\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 7.5,\n",
        "            \"current_title\": \"Machine Learning Engineer\",\n",
        "            \"location\": \"Pune, Maharashtra\",\n",
        "            \"summary\": \"Experienced in building retrieval systems using embeddings, PyTorch and Pinecone. Active contributor on github.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"Tech Startup Inc\",\n",
        "                \"industry\": \"technology\",\n",
        "                \"title\": \"Machine Learning Engineer\",\n",
        "                \"duration_months\": 48,\n",
        "                \"description\": \"Built production vector search systems using FAISS, Elasticsearch and Qdrant. Evaluated using NDCG and A/B testing for real users.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"Python\", \"proficiency\": \"Expert\", \"duration_months\": 60},\n",
        "            {\"name\": \"Machine Learning\", \"proficiency\": \"Expert\", \"duration_months\": 60},\n",
        "            {\"name\": \"FAISS\", \"proficiency\": \"Intermediate\", \"duration_months\": 24}\n",
        "        ],\n",
        "        \"redrob_signals\": {\"notice_period_days\": 30, \"recruiter_response_rate\": 0.9}\n",
        "    },\n",
        "    # 2. Good but high notice period\n",
        "    {\n",
        "        \"candidate_id\": \"CAND_999902\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 6.0,\n",
        "            \"current_title\": \"Applied Scientist\",\n",
        "            \"location\": \"Bangalore\",\n",
        "            \"summary\": \"NLP engineer specializing in search and ranking systems using transformers and embeddings.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"E-commerce Giant\",\n",
        "                \"title\": \"Data Scientist\",\n",
        "                \"duration_months\": 36,\n",
        "                \"description\": \"Developed dense retrieval pipelines. Monitored embedding drift and recall@k.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"NLP\", \"proficiency\": \"Expert\", \"duration_months\": 48},\n",
        "            {\"name\": \"PyTorch\", \"proficiency\": \"Expert\", \"duration_months\": 48}\n",
        "        ],\n",
        "        \"redrob_signals\": {\"notice_period_days\": 90, \"recruiter_response_rate\": 0.8}\n",
        "    },\n",
        "    # 3. Wrong Domain (Computer Vision)\n",
        "    {\n",
        "        \"candidate_id\": \"CAND_999903\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 8.0,\n",
        "            \"current_title\": \"Computer Vision Engineer\",\n",
        "            \"location\": \"Noida\",\n",
        "            \"summary\": \"Expert in object detection, segmentation and YOLO models using PyTorch.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"Auto Drive\",\n",
        "                \"title\": \"CV Engineer\",\n",
        "                \"duration_months\": 60,\n",
        "                \"description\": \"Trained CNNs and deployed object detection models for autonomous vehicles.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"Computer Vision\", \"proficiency\": \"Expert\", \"duration_months\": 72},\n",
        "            {\"name\": \"PyTorch\", \"proficiency\": \"Expert\", \"duration_months\": 72}\n",
        "        ],\n",
        "        \"redrob_signals\": {\"notice_period_days\": 15, \"recruiter_response_rate\": 0.9}\n",
        "    },\n",
        "    # 4. Consulting Heavy, No vector search\n",
        "    {\n",
        "        \"candidate_id\": \"CAND_999904\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 10.0,\n",
        "            \"current_title\": \"Data Engineer\",\n",
        "            \"location\": \"Chennai\",\n",
        "            \"summary\": \"Data engineering and analytics pipelines using PyTorch and traditional ML.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"TCS\",\n",
        "                \"title\": \"Data Engineer\",\n",
        "                \"duration_months\": 120,\n",
        "                \"description\": \"Built ETL pipelines and deployed scikit-learn models for clients.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"Machine Learning\", \"proficiency\": \"Intermediate\", \"duration_months\": 60},\n",
        "            {\"name\": \"SQL\", \"proficiency\": \"Expert\", \"duration_months\": 120}\n",
        "        ],\n",
        "        \"redrob_signals\": {\"notice_period_days\": 90, \"recruiter_response_rate\": 0.5}\n",
        "    },\n",
        "    # 5. LLM Wrapper / Junior\n",
        "    {\n",
        "        \"candidate_id\": \"CAND_999905\",\n",
        "        \"profile\": {\n",
        "            \"years_of_experience\": 2.0,\n",
        "            \"current_title\": \"AI Developer\",\n",
        "            \"location\": \"Pune\",\n",
        "            \"summary\": \"Building chatbots with Langchain, OpenAI, and Streamlit.\"\n",
        "        },\n",
        "        \"career_history\": [\n",
        "            {\n",
        "                \"company\": \"AI Agency\",\n",
        "                \"title\": \"AI Dev\",\n",
        "                \"duration_months\": 24,\n",
        "                \"description\": \"Created RAG wrappers using OpenAI API and Langchain.\"\n",
        "            }\n",
        "        ],\n",
        "        \"skills\": [\n",
        "            {\"name\": \"Langchain\", \"proficiency\": \"Expert\", \"duration_months\": 12},\n",
        "            {\"name\": \"OpenAI\", \"proficiency\": \"Expert\", \"duration_months\": 12}\n",
        "        ],\n",
        "        \"redrob_signals\": {\"notice_period_days\": 0, \"recruiter_response_rate\": 0.9}\n",
        "    }\n",
        "]\n",
        "\n",
        "with open(\"sample_candidates.jsonl\", \"w\") as f:\n",
        "    for c in dummy_candidates:\n",
        "        f.write(json.dumps(c) + \"\\n\")\n",
        "\n",
        "print(f\"Created sample_candidates.jsonl with {len(dummy_candidates)} diverse records.\")"
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
