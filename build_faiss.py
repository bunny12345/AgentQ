# build_faiss.py
import os, re, json, tarfile, tempfile, urllib.request
from urllib.parse import quote
import boto3
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET  = os.getenv("EMBED_S3_BUCKET", "monitoring-agent-embeddings-repo")
S3_PREFIX  = os.getenv("EMBED_S3_PREFIX", "code-index")
GH_TOKEN   = os.getenv("GITHUB_TOKEN", "")

s3 = boto3.client("s3", region_name=AWS_REGION)
emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=AWS_REGION)

CODE_EXTS = (".py",".js",".ts",".json",".yaml",".yml",".tf",".java",".go",".sh",".md")
EXCLUDE_DIRS = ("node_modules","dist","build",".git",".cache","venv","__pycache__")

def _gh_headers():
    h = {"Accept": "application/vnd.github+json", "User-Agent": "agent-indexer"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _http_json(url: str):
    req = urllib.request.Request(url, headers=_gh_headers())
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read().decode("utf-8"))

def _gh_latest_sha(owner: str, repo: str, branch: str):
    url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/commits?sha={quote(branch)}&per_page=1"
    data = _http_json(url)
    return data[0]["sha"] if data else branch

def _gh_tree(owner: str, repo: str, ref: str):
    url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/git/trees/{quote(ref)}?recursive=1"
    return _http_json(url)

def _gh_file(owner: str, repo: str, path: str, ref: str) -> str:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    req = urllib.request.Request(raw_url, headers=_gh_headers())
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return r.read().decode("utf-8", errors="replace")
    except:
        return ""

def _should_index(path: str) -> bool:
    if any(d in path for d in EXCLUDE_DIRS):
        return False
    return path.endswith(CODE_EXTS) or any(path.endswith(ext) for ext in CODE_EXTS)

def index_repo(owner: str, repo: str, branch: str="main"):
    sha = _gh_latest_sha(owner, repo, branch)
    sha_key     = f"{S3_PREFIX}/{owner}/{repo}/{branch}/faiss-{sha}.tar.gz"
    stable_key  = f"{S3_PREFIX}/{owner}/{repo}/{branch}/faiss.tar.gz"

    try:
        s3.head_object(Bucket=S3_BUCKET, Key=sha_key)
        print(f"[INFO] {owner}/{repo}@{branch} already indexed at {sha}, syncing stable alias...")
        s3.copy_object(Bucket=S3_BUCKET, CopySource={"Bucket": S3_BUCKET, "Key": sha_key}, Key=stable_key)
        print(f"[OK] Synced {stable_key} -> {sha_key}")
        return
    except ClientError:
        pass

    tree = _gh_tree(owner, repo, sha)
    docs = []
    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if not _should_index(path):
            continue
        content = _gh_file(owner, repo, path, sha)
        if not content.strip():
            continue
        docs.append(Document(page_content=content, metadata={"path": path, "repo": repo, "sha": sha}))

    if not docs:
        print(f"[WARN] No docs for {owner}/{repo}")
        return

    faiss = FAISS.from_documents(docs, emb)
    with tempfile.TemporaryDirectory() as td:
        faiss.save_local(td)
        tarpath = os.path.join(td, "faiss.tar.gz")
        with tarfile.open(tarpath, "w:gz") as tf:
            tf.add(td, arcname="faiss")
        with open(tarpath, "rb") as f:
            data = f.read()
            s3.put_object(Bucket=S3_BUCKET, Key=sha_key, Body=data)
            s3.put_object(Bucket=S3_BUCKET, Key=stable_key, Body=data)
    print(f"[OK] Indexed {owner}/{repo}@{sha}, uploaded to {sha_key} and {stable_key}")

if __name__ == "__main__":
    repos = json.loads(open("repos.json").read())
    for r in repos:
        index_repo(r["owner"], r["repo"], r.get("branch", "main"))
