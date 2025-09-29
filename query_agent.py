import os, re, json, time, tarfile, tempfile, urllib.request
from urllib.parse import quote
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET  = os.getenv("EMBED_S3_BUCKET", "monitoring-agent-embeddings-repo")
S3_PREFIX  = os.getenv("EMBED_S3_PREFIX", "code-index")
GH_TOKEN   = os.getenv("GITHUB_TOKEN", "")

LLM_MODEL_ID   = os.getenv("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "600"))
TOP_K = int(os.getenv("TOP_K", "8"))

s3 = boto3.client("s3", region_name=AWS_REGION)
_bedrock_embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=AWS_REGION)

# ---------------- GitHub helpers ----------------
def _gh_headers():
    h = {"Accept": "application/vnd.github+json", "User-Agent": "agent-query"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _http_json(url: str):
    req = urllib.request.Request(url, headers=_gh_headers())
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read().decode("utf-8"))

def get_recent_commit_diffs(owner: str, repo: str, branch: str = "main", n: int = 3):
    commits = _http_json(f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/commits?sha={quote(branch)}&per_page={n}")
    diffs = []
    for c in commits:
        sha = c["sha"]
        commit_data = _http_json(f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/commits/{sha}")
        for f in commit_data.get("files", []):
            if "patch" in f:
                diffs.append(f"Commit {sha} file {f['filename']} ({f.get('status')}):\n{f['patch']}")
    return diffs

# ---------------- FAISS loading ----------------
def _faiss_local_path(repo_key: str) -> str:
    return f"/tmp/{repo_key}_faiss"

def _faiss_s3_key(owner: str, repo: str, branch: str="main") -> str:
    stable_key = f"{S3_PREFIX}/{owner}/{repo}/{branch}/faiss.tar.gz"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=stable_key)
        return stable_key
    except ClientError:
        pass
    prefix = f"{S3_PREFIX}/{owner}/{repo}/{branch}/faiss-"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    if "Contents" in resp:
        latest = max(resp["Contents"], key=lambda x: x["LastModified"])
        return latest["Key"]
    raise FileNotFoundError(f"No FAISS index for {owner}/{repo}@{branch}")

def _load_faiss_from_s3(owner: str, repo: str, branch: str="main"):
    repo_key = f"{owner}_{repo}_{branch}"
    local_path = _faiss_local_path(repo_key)
    if os.path.exists(local_path):
        return FAISS.load_local(local_path, _bedrock_embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": TOP_K})
    key = _faiss_s3_key(owner, repo, branch)
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "faiss.tar.gz")
        s3.download_file(S3_BUCKET, key, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(td)
        faiss_store = FAISS.load_local(os.path.join(td, "faiss"), _bedrock_embeddings, allow_dangerous_deserialization=True)
        faiss_store.save_local(local_path)
        return faiss_store.as_retriever(search_kwargs={"k": TOP_K})

# ---------------- Query execution ----------------
def _make_llm():
    return ChatBedrock(model_id=LLM_MODEL_ID, region_name=AWS_REGION, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)

def run_query(query: str, alert: str, repos: List[Dict[str,str]]) -> Dict[str, Any]:
    retrievers = []
    commit_contexts = []
    for r in repos:
        retrievers.append(_load_faiss_from_s3(r["owner"], r["repo"], r.get("branch","main")))
        commit_contexts += get_recent_commit_diffs(r["owner"], r["repo"], r.get("branch","main"), n=3)

    RETRIEVER = retrievers[0] if len(retrievers) == 1 else EnsembleRetriever(retrievers=retrievers, weights=[1.0]*len(retrievers))
    docs = RETRIEVER.invoke((query or "") + "\n" + (alert or ""))[:TOP_K]

    context_parts = []
    if docs:
        context_parts.append("\n\n".join([f"{d.metadata.get('repo')}:{d.metadata.get('path')}\n{d.page_content}" for d in docs]))
    if commit_contexts:
        context_parts.append("\n\n".join(commit_contexts))

    if not context_parts:
        return {"answer": "No relevant snippets or commits found.", "from": "none"}

    context_str = "\n\n".join(context_parts)
    prompt = (
       f"You are an assistant. Use ONLY these snippets and commit diffs.\n\n"
       f"QUESTION: {query}\n"
       f"ALERT: {alert}\n\n"
       f"CONTEXT:\n{context_str}"
    )

    resp = _make_llm().invoke(prompt)
    return {"answer": resp.content, "from": "repo+commits"}


# ==================== LAMBDA HANDLER ====================
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda entrypoint.
    Handles both direct Lambda invoke events and API Gateway proxy events.
    """

    logger.info("EVENT: %s", json.dumps(event))
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
# ---------------- CLI ----------------
if __name__ == "__main__":
    q = input("Query: ").strip()
    a = input("Alert: ").strip()
    repos = [
        {"owner":"bunny12345","repo":"langchain","branch":"main"},
        {"owner":"bunny12345","repo":"AgentQ","branch":"main"},
        {"owner":"bunny12345","repo":"chatvista_ai","branch":"main"}
    ]
    print(json.dumps(run_query(q,a,repos), indent=2))
