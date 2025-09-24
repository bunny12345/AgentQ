import os
import re
import json
import time
import tarfile
import tempfile
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError

from langchain import hub
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor


# ==================== CONFIG ====================
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

S3_BUCKET  = os.getenv("EMBED_S3_BUCKET", "monitoring-agent-embeddings-repo")
S3_PREFIX  = os.getenv("EMBED_S3_PREFIX", "code-index")

AMAZON_Q_APP_ID  = os.getenv("AMAZON_Q_APP_ID", "")
AMAZON_Q_USER_ID = os.getenv("AMAZON_Q_USER_ID", "")

LLM_MODEL_ID   = os.getenv("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "600"))

TOP_K = int(os.getenv("TOP_K", "8"))

s3 = boto3.client("s3", region_name=AWS_REGION)
qbiz = boto3.client("qbusiness", region_name=AWS_REGION) if AMAZON_Q_APP_ID else None

_bedrock_embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=AWS_REGION)

# ==================== CACHING HELPERS ====================
def _faiss_local_path(repo_key: str) -> str:
    return f"/tmp/{repo_key}_faiss"

def _faiss_s3_key(owner: str, repo: str, branch: str = "main") -> str:
    return f"{S3_PREFIX}/{owner}/{repo}/{branch}/faiss.tar.gz"

def _load_faiss_from_s3(owner: str, repo: str, branch: str = "main") -> EnsembleRetriever:
    repo_key = f"{owner}_{repo}_{branch}"
    local_path = _faiss_local_path(repo_key)

    # If already cached in /tmp, reuse it
    if os.path.exists(local_path):
        return FAISS.load_local(local_path, _bedrock_embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": TOP_K})

    # Otherwise download from S3
    key = _faiss_s3_key(owner, repo, branch)
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "faiss.tar.gz")
        s3.download_file(S3_BUCKET, key, tar_path)
        with tarfile.open(tar_path, "r:gz") as tarf:
            tarf.extractall(td)
        faiss_store = FAISS.load_local(os.path.join(td, "faiss"), _bedrock_embeddings, allow_dangerous_deserialization=True)

        # Cache in /tmp for reuse
        faiss_store.save_local(local_path)
        return faiss_store.as_retriever(search_kwargs={"k": TOP_K})


# ==================== AMAZON Q FALLBACK ====================
INFRA_HINTS = re.compile(
    r"\b(aws|amazon|bedrock|s3|iam|vpc|subnet|sg|security[-_\s]?group|rds|dynamodb|lambda|apigateway|"
    r"ecs|ecr|eks|cloudwatch|cloudtrail|alb|nlb|route53|terraform|tfvars?|k8s|kubernetes|helm|"
    r"ansible|packer|prometheus|grafana|otel|observability|cidr|nat|igw|asg|ec2)\b", re.I
)

def looks_infra(text: str) -> bool:
    return bool(INFRA_HINTS.search(text or ""))

def amazon_q_fallback(question: str, tries: int = 2) -> Dict[str, Any]:
    if not qbiz or not AMAZON_Q_APP_ID or not AMAZON_Q_USER_ID:
        return {"error": "amazon_q_not_configured"}
    last_err = None
    for i in range(tries):
        try:
            resp = qbiz.chat(
                applicationId=AMAZON_Q_APP_ID,
                userId=AMAZON_Q_USER_ID,
                messages=[{"role": "user", "content": [{"text": question}]}],
            )
            content = ""
            output = resp.get("output") or resp
            msgs = (output.get("messages") or []) if isinstance(output, dict) else []
            if msgs and "content" in msgs[0]:
                parts = msgs[0]["content"]
                if parts and "text" in parts[0]:
                    content = parts[0]["text"]
            if content.strip():
                return {"answer": content.strip()}
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (2 ** i))
    return {"error": "amazon_q_failed", "detail": str(last_err) if last_err else "unknown"}


# ==================== LLM ====================
def _make_llm():
    return ChatBedrock(
        model_id=LLM_MODEL_ID,
        region_name=AWS_REGION,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        streaming=False,
    )


# ==================== MAIN QUERY ====================
def run_query(query: str, alert: str, repos: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        retrievers = []
        for r in repos:
            if not isinstance(r, dict):
                raise TypeError(f"Repo entry is not a dict: {r}")
            try:
                retrievers.append(_load_faiss_from_s3(r["owner"], r["repo"], r.get("branch", "main")))
            except Exception as e:
                return {"error": f"Failed to load repo {r}", "detail": str(e)}

        if not retrievers:
            return {"error": "no_index", "hint": "No FAISS indexes loaded."}

        if len(retrievers) == 1:
            RETRIEVER = retrievers[0]
        else:
            RETRIEVER = EnsembleRetriever(retrievers=retrievers, weights=[1.0]*len(retrievers))

        docs = RETRIEVER.invoke((query or "") + "\n" + (alert or ""))[:TOP_K]

        if docs:
            context = "\n\n".join([
                f"{d.metadata.get('repo')}:{d.metadata.get('path')}:{d.metadata.get('line_start')}-{d.metadata.get('line_end')}\n{d.page_content}"
                for d in docs
            ])
            prompt = (
                f"You are an assistant. Use ONLY these snippets to answer.\n\n"
                f"USER QUESTION:\n{query or '(no question)'}\n\n"
                f"ALERT:\n{alert or '(none)'}\n\n"
                f"CODE CONTEXT:\n{context}\n\nAnswer clearly and concisely."
            )
            resp = _make_llm().invoke(prompt)
            return {"answer": resp.content, "from": "repo"}
        else:
            if looks_infra(query + " " + alert):
                qresp = amazon_q_fallback(query or alert or "infra question", tries=2)
                if "answer" in qresp:
                    return {"answer": qresp["answer"], "from": "amazon_q"}
                else:
                    return {"error": "amazon_q_unavailable", "detail": qresp, "from": "fallback"}
            return {"answer": "No relevant snippets found.", "from": "none"}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ==================== LAMBDA HANDLER ====================
# ==================== LAMBDA HANDLER ====================

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("EVENT: %s", json.dumps(event))

def lambda_handler(event, context):
    """
    AWS Lambda entrypoint.
    Handles both direct Lambda invoke events and API Gateway proxy events.
    """

    try:
        # Normalize API Gateway body
        body = {}
        if "body" in event:
            raw_body = event["body"]

            # Decode base64 if API Gateway says so
            if event.get("isBase64Encoded"):
                import base64
                raw_body = base64.b64decode(raw_body).decode("utf-8")

            if isinstance(raw_body, str):
                body = json.loads(raw_body or "{}")
            elif isinstance(raw_body, dict):
                body = raw_body
        else:
            # Direct Lambda invoke
            body = event if isinstance(event, dict) else {}

        query = (body.get("query") or "").strip()
        alert = (body.get("alert") or "").strip()

        # Define repos to search
        repos = [
            {"owner": "bunny12345", "repo": "langchain", "branch": "main"},
            {"owner": "bunny12345", "repo": "AgentQ", "branch": "main"},
            {"owner": "bunny12345", "repo": "chatvista_ai", "branch": "main"},
            # add more repos here
        ]

        # 🔥 wrap run_query in debug try/except
        try:
            result = run_query(query, alert, repos)
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result)
            }
        except Exception as e:
            import traceback
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
            }

    except Exception as outer:
        import traceback
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": str(outer),
                "traceback": traceback.format_exc()
            })
        }


# ==================== CLI ====================
if __name__ == "__main__":
    user_query = input("Query (optional): ").strip()
    user_alert = input("Alert (optional): ").strip()

    repos = [
        {"owner": "bunny12345", "repo": "langchain", "branch": "main"},
        {"owner": "bunny12345", "repo": "AgentQ", "branch": "main"},
        {"owner": "bunny12345", "repo": "chatvista_ai", "branch": "main"}
        # add more repos here
    ]

    result = run_query(user_query, user_alert, repos)
    print(json.dumps(result, indent=2))
