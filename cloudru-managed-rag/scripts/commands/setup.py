"""Cloud.ru Managed RAG Infrastructure Setup pipeline.

Full automation for creating Managed RAG infrastructure:
  1. extract-info   -- decode JWT from browser token
  2. ensure-sa      -- create/find Service Account
  3. ensure-role    -- assign managed_rag.admin
  4. create-access-key -- create access key (secret shown only once!)
  5. get-tenant-id  -- get tenant_id for S3
  6. ensure-bucket  -- create S3 bucket
  7. upload-docs    -- upload documents to bucket
  8. create-kb      -- create Knowledge Base
  9. wait-active    -- poll until KNOWLEDGEBASE_ACTIVE
 10. save-env       -- save .env with credentials
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IAM_HOST = "iam.api.cloud.ru"
CONSOLE_HOST = "console.cloud.ru"
RAG_API_HOST = "managed-rag.api.cloud.ru"
S3_ENDPOINT = "https://s3.cloud.ru"
S3_REGION = "ru-central-1"

NO_PROXY_HOSTS = (
    "managed-rag.api.cloud.ru",
    "iam.api.cloud.ru",
    "s3.cloud.ru",
    "console.cloud.ru",
    "managed-rag.inference.cloud.ru",
)

DEFAULT_SA_NAME = "managed-rag-sa"
DEFAULT_FILE_EXTENSIONS = "txt,pdf"
DEFAULT_KB_POLL_INTERVAL = 15  # seconds
DEFAULT_KB_POLL_TIMEOUT = 600  # 10 minutes
DEFAULT_ACCESS_KEY_TTL = 365  # days (BFF expects uint32 in range [0, 10000])
DEFAULT_ENV_PATH = os.path.expanduser(
    "~/.openclaw/workspace/skills/managed-rag-skill/.env"
)

ALL_STEPS = [
    "extract-info",
    "ensure-sa",
    "ensure-role",
    "create-access-key",
    "get-tenant-id",
    "ensure-bucket",
    "upload-docs",
    "create-kb",
    "wait-active",
    "save-env",
]


# ---------------------------------------------------------------------------
# Helpers: proxy bypass
# ---------------------------------------------------------------------------


def _setup_no_proxy() -> None:
    """Append Cloud.ru hosts to no_proxy to bypass corporate proxies."""
    current = os.environ.get("no_proxy", "")
    hosts_to_add = [h for h in NO_PROXY_HOSTS if h not in current]
    if hosts_to_add:
        separator = "," if current else ""
        os.environ["no_proxy"] = current + separator + ",".join(hosts_to_add)
        os.environ["NO_PROXY"] = os.environ["no_proxy"]
    # Remove proxy env vars that would override no_proxy for http.client / urllib
    for pvar in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(pvar, None)


# Force bypass on module import
_no_proxy = ",".join(NO_PROXY_HOSTS)
os.environ["no_proxy"] = _no_proxy + "," + os.environ.get("no_proxy", "")
for _pvar in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_pvar, None)


# ---------------------------------------------------------------------------
# Helpers: output
# ---------------------------------------------------------------------------


def emit(data: Dict[str, Any]) -> None:
    """Print a JSON object to stdout (one line)."""
    print(json.dumps(data, ensure_ascii=False))


def make_error(step: str, message: str, code: Optional[int] = None) -> Dict[str, Any]:
    """Build an error result dict (does NOT emit -- caller uses ctx.record)."""
    result: Dict[str, Any] = {"step": step, "error": message}
    if code is not None:
        result["code"] = code
    return result


# ---------------------------------------------------------------------------
# Helpers: httpx-based HTTPS requests for BFF/Console API
# ---------------------------------------------------------------------------


def _bff_request(
    method: str,
    path: str,
    token: str,
    body: Optional[Any] = None,
    timeout: float = 30.0,
) -> Tuple[int, Dict[str, Any]]:
    """Perform BFF (console) HTTPS request using httpx. Returns (status, data)."""
    url = f"https://{CONSOLE_HOST}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Origin": "https://console.cloud.ru",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(verify=True, timeout=timeout, proxy=None) as client:
            resp = client.request(method, url, headers=headers, json=body)
            status = resp.status_code
            raw = resp.text
    except Exception as exc:
        return 0, {"_error": str(exc)}

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {"_raw": raw}
    return status, data


def _api_request(
    host: str,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: float = 30.0,
) -> Tuple[int, Dict[str, Any]]:
    """Generic HTTPS request via httpx for public API / IAM calls."""
    url = f"https://{host}{path}"
    hdrs = dict(headers or {})
    hdrs.setdefault("Content-Type", "application/json")
    try:
        with httpx.Client(verify=True, timeout=timeout, proxy=None) as client:
            resp = client.request(method, url, headers=hdrs, json=body)
            status = resp.status_code
            raw = resp.text
    except Exception as exc:
        return 0, {"_error": str(exc)}
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {"_raw": raw}
    return status, data


# ---------------------------------------------------------------------------
# Helpers: urllib-based HTTPS requests (kept for S3 operations)
# ---------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _https_request(
    host: str,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: int = 30,
) -> Tuple[int, Dict[str, Any]]:
    """urllib-based HTTPS request. Kept for S3 AWS Signature V4 operations."""
    hdrs = dict(headers or {})
    hdrs.setdefault("Content-Type", "application/json")

    url = f"https://{host}{path}"
    encoded_body = json.dumps(body).encode("utf-8") if body is not None else None

    req = urllib.request.Request(url, data=encoded_body, headers=hdrs, method=method)

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_ssl_context()),
    )

    try:
        resp = opener.open(req, timeout=timeout)
        raw = resp.read().decode("utf-8", errors="replace")
        status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        status = e.code
    except Exception as exc:
        return 0, {"_error": str(exc)}

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {"_raw": raw}
    return status, data


# ---------------------------------------------------------------------------
# Helpers: headers
# ---------------------------------------------------------------------------


def _auth_headers(token: str) -> Dict[str, str]:
    """Standard Authorization header for public API."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Helpers: JWT decoding (no third-party libs)
# ---------------------------------------------------------------------------


def _b64_decode_segment(segment: str) -> bytes:
    """Decode a base64url segment with padding fix."""
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded)


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decode the payload of a JWT without signature verification."""
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("Invalid JWT: expected at least 2 dot-separated segments")
    payload_bytes = _b64_decode_segment(parts[1])
    return json.loads(payload_bytes)


# ---------------------------------------------------------------------------
# Helpers: IAM token exchange
# ---------------------------------------------------------------------------


def get_iam_token(key_id: str, secret: str) -> str:
    """Exchange an access key pair for an IAM bearer token."""
    status, data = _api_request(
        IAM_HOST,
        "POST",
        "/api/v1/auth/token",
        headers={"Content-Type": "application/json"},
        body={"key_id": key_id, "secret": secret},
    )
    if status != 200:
        raise RuntimeError(
            f"IAM token exchange failed (HTTP {status}): {json.dumps(data)}"
        )
    token = data.get("access_token") or data.get("token")
    if not token:
        raise RuntimeError(f"No token in IAM response: {json.dumps(data)}")
    return token


# ---------------------------------------------------------------------------
# Pipeline context -- carries state between steps
# ---------------------------------------------------------------------------


class PipelineContext:
    """Mutable bag of state threaded through pipeline steps."""

    def __init__(
        self,
        token: str,
        project_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        sa_name: str = DEFAULT_SA_NAME,
        bucket_name: Optional[str] = None,
        docs_path: Optional[str] = None,
        kb_name: Optional[str] = None,
        file_extensions: str = DEFAULT_FILE_EXTENSIONS,
        output_env: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.token: str = token
        self.project_id: Optional[str] = project_id
        self.customer_id: Optional[str] = customer_id
        self.sa_name: str = sa_name
        self.bucket_name: Optional[str] = bucket_name
        self.docs_path: Optional[str] = docs_path
        self.kb_name: Optional[str] = kb_name
        self.file_extensions: str = file_extensions
        self.output_env: Optional[str] = output_env
        self.dry_run: bool = dry_run

        self.sa_id: Optional[str] = None
        self.key_id: Optional[str] = None
        self.key_secret: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.kb_id: Optional[str] = None
        self.search_url: Optional[str] = None
        self.log_group_id: Optional[str] = None
        self.iam_token: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

    def record(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self.results.append(result)
        emit(result)
        return result

    def ensure_iam_token(self) -> str:
        """Get or refresh an IAM token from the access key pair."""
        if self.iam_token:
            return self.iam_token
        if not self.key_id or not self.key_secret:
            raise RuntimeError(
                "Access key not available yet -- run create-access-key first"
            )
        self.iam_token = get_iam_token(self.key_id, self.key_secret)
        return self.iam_token


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step_extract_info(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 1: Decode JWT to extract project_id, customer_id, user sub."""
    step = "extract-info"
    try:
        payload = decode_jwt_payload(ctx.token)
    except Exception as exc:
        return ctx.record(make_error(step, f"JWT decode failed: {exc}"))

    project_id = (
        ctx.project_id
        or payload.get("project_id")
        or payload.get("proj")
        or payload.get("prj")
    )
    customer_id = (
        ctx.customer_id
        or payload.get("customer_id")
        or payload.get("cid")
    )
    user_sub = payload.get("sub", "")

    if not project_id:
        return ctx.record(make_error(
            step,
            "project_id not found in token. Browser OIDC tokens don't contain project_id. "
            "Pass --project-id explicitly. You can find it in the console URL: "
            "console.cloud.ru/spa/svp?projectId=<THIS_VALUE>"
        ))

    if not customer_id:
        return ctx.record(make_error(
            step,
            "customer_id not found in token. Browser OIDC tokens don't contain customer_id. "
            "Pass --customer-id explicitly. You can find it in the console URL: "
            "console.cloud.ru/spa/svp?customerId=<THIS_VALUE>"
        ))

    ctx.project_id = project_id
    ctx.customer_id = customer_id

    result = {
        "step": step,
        "project_id": project_id,
        "customer_id": customer_id,
        "user_sub": user_sub,
    }
    return ctx.record(result)


def step_ensure_sa(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 2: Find or create a Service Account via BFF (httpx)."""
    step = "ensure-sa"
    sa_name = ctx.sa_name

    if not ctx.project_id:
        return ctx.record(
            make_error(step, "project_id is required -- run extract-info first")
        )
    if not ctx.customer_id:
        return ctx.record(
            make_error(step, "customer_id is required -- run extract-info first")
        )

    if ctx.dry_run:
        ctx.sa_id = ctx.sa_id or "dry-run-sa-id"
        return ctx.record({"step": step, "sa_name": sa_name, "dry_run": True})

    # --- helper: search SA by name across project's accounts ---
    def _find_sa_by_name() -> Optional[str]:
        list_path = f"/u-api/bff-console/v1/projects/{ctx.project_id}/service-accounts"
        list_status, list_data = _bff_request("GET", list_path, ctx.token)
        if list_status != 200:
            return None
        accounts = list_data.get(
            "accounts",
            list_data.get("service_accounts", list_data.get("items", [])),
        )
        for sa in accounts:
            if sa.get("name") == sa_name:
                return sa.get("id")
        return None

    # Try to find existing SA first
    existing_id = _find_sa_by_name()
    if existing_id:
        ctx.sa_id = existing_id
        return ctx.record(
            {"step": step, "sa_id": ctx.sa_id, "sa_name": sa_name, "created": False}
        )

    # Create new SA via BFF v2
    create_body = {
        "name": sa_name,
        "description": "Service account for Managed RAG",
        "customerId": ctx.customer_id,
        "serviceRoles": [],
        "projectId": ctx.project_id,
        "projectRole": "PROJECT_ROLE_PROJECT_ADMIN",
        "artifactRoles": [],
        "artifactRegistries": [],
        "s3eRoles": [],
        "s3eBuckets": [],
    }
    status, data = _bff_request(
        "POST",
        "/u-api/bff-console/v2/service-accounts/add",
        ctx.token,
        body=create_body,
    )

    if status == 409:
        emit({"step": step, "info": f"SA '{sa_name}' already exists (409), looking up by name..."})
        found_id = _find_sa_by_name()
        if found_id:
            ctx.sa_id = found_id
            return ctx.record(
                {"step": step, "sa_id": ctx.sa_id, "sa_name": sa_name, "created": False}
            )
        # Name taken by another project — try with suffix
        for suffix in range(2, 6):
            alt_name = f"{sa_name}-{suffix}"
            emit({"step": step, "info": f"Trying alternative name '{alt_name}'..."})
            create_body["name"] = alt_name
            alt_status, alt_data = _bff_request(
                "POST",
                f"/u-api/bff-console/v2/service-accounts",
                ctx.token,
                body=create_body,
            )
            if alt_status in (200, 201):
                ctx.sa_id = alt_data.get("id") or alt_data.get("service_account", {}).get("id")
                ctx.sa_name = alt_name
                return ctx.record(
                    {"step": step, "sa_id": ctx.sa_id, "sa_name": alt_name, "created": True}
                )
            if alt_status != 409:
                break
        return ctx.record(
            make_error(
                step,
                f"SA '{sa_name}' and variants already exist in customer {ctx.customer_id}. "
                f"Use --sa-name with a unique name.",
                409,
            )
        )

    if status not in (200, 201):
        return ctx.record(
            make_error(step, f"Failed to create SA via BFF: {json.dumps(data)}", status)
        )

    ctx.sa_id = data.get("id") or data.get("service_account", {}).get("id")
    if not ctx.sa_id:
        return ctx.record(
            make_error(step, f"SA created but id not found in response: {json.dumps(data)}")
        )
    return ctx.record(
        {"step": step, "sa_id": ctx.sa_id, "sa_name": sa_name, "created": True}
    )


def step_ensure_role(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 3: Assign managed_rag.admin service role to the SA via BFF (httpx).

    Non-fatal -- if the endpoint fails, a warning is logged and the pipeline
    continues. The SA may already have sufficient permissions via PROJECT_ADMIN.
    """
    step = "ensure-role"
    role = "managed_rag.admin"

    if not ctx.sa_id or not ctx.project_id:
        return ctx.record(
            make_error(step, "sa_id and project_id required -- run previous steps")
        )

    if ctx.dry_run:
        return ctx.record(
            {"step": step, "role": role, "sa_id": ctx.sa_id, "dry_run": True}
        )

    # Try multiple known endpoints for role assignment
    endpoints = [
        (
            f"/u-api/bff-console/v2/service-accounts/{ctx.sa_id}/service-roles",
            {"projectId": ctx.project_id, "roles": [role]},
        ),
        (
            f"/u-api/bff-console/v1/service-accounts/{ctx.sa_id}/service-roles",
            {"projectId": ctx.project_id, "roles": [role]},
        ),
        (
            f"/u-api/iam-sa/v1/{ctx.customer_id}/service-accounts/{ctx.sa_id}/roles",
            {"projectId": ctx.project_id, "serviceRoles": [role]},
        ),
    ]

    status, data = 404, {}
    for path, body in endpoints:
        status, data = _bff_request("POST", path, ctx.token, body=body)
        if status in (200, 201, 409):
            return ctx.record({"step": step, "role": role, "sa_id": ctx.sa_id})
        if status != 404:
            break

    # Non-fatal: log a warning and continue pipeline.
    emit({
        "step": step,
        "warning": (
            f"Failed to assign service role '{role}' (HTTP {status}): {json.dumps(data)}. "
            f"This is non-fatal -- SA has PROJECT_ROLE_PROJECT_ADMIN from creation. "
            f"If Managed RAG operations fail later, assign '{role}' manually in the console."
        ),
    })
    return ctx.record({
        "step": step,
        "role": role,
        "sa_id": ctx.sa_id,
        "warning": f"Service role assignment failed (HTTP {status}), continuing with PROJECT_ADMIN",
    })


def step_create_access_key(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 4: Create an access key for the SA via BFF (httpx). Secret shown only once!"""
    step = "create-access-key"

    if not ctx.sa_id:
        return ctx.record(
            make_error(step, "sa_id required -- run ensure-sa first")
        )

    if ctx.dry_run:
        ctx.key_id = ctx.key_id or "dry-run-key-id"
        ctx.key_secret = ctx.key_secret or "dry-run-key-secret"
        return ctx.record(
            {"step": step, "sa_id": ctx.sa_id, "ttl": DEFAULT_ACCESS_KEY_TTL, "dry_run": True}
        )

    body = {
        "description": "Access key for Managed RAG S3 and API (auto-created)",
        "ttl": DEFAULT_ACCESS_KEY_TTL,
    }
    path = f"/u-api/bff-console/v1/service-accounts/{ctx.sa_id}/access_keys"
    status, data = _bff_request("POST", path, ctx.token, body=body)

    if status not in (200, 201):
        return ctx.record(
            make_error(step, f"Failed to create access key via BFF: {json.dumps(data)}", status)
        )

    ctx.key_id = data.get("key_id") or data.get("access_key", {}).get("key_id") or data.get("id")
    ctx.key_secret = data.get("secret") or data.get("access_key", {}).get("secret")

    if not ctx.key_id or not ctx.key_secret:
        return ctx.record(
            make_error(step, f"Unexpected response format: {json.dumps(data)}")
        )

    return ctx.record(
        {"step": step, "key_id": ctx.key_id, "secret": ctx.key_secret}
    )


def step_get_tenant_id(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 5: Get tenant_id from the S3 controller BFF (httpx)."""
    step = "get-tenant-id"

    if not ctx.project_id:
        return ctx.record(
            make_error(step, "project_id required -- run extract-info first")
        )

    if ctx.dry_run:
        ctx.tenant_id = ctx.tenant_id or "dry-run-tenant-id"
        return ctx.record(
            {"step": step, "project_id": ctx.project_id, "dry_run": True}
        )

    path = f"/u-api/s3e-controller/v2/projects/{ctx.project_id}"
    status, data = _bff_request("GET", path, ctx.token)

    if status != 200:
        return ctx.record(
            make_error(step, f"Failed to get tenant_id: {json.dumps(data)}", status)
        )

    ctx.tenant_id = (
        data.get("tenant_id")
        or data.get("tenantId")
        or data.get("tenant", {}).get("id")
        or data.get("data", {}).get("tenant_id")
    )
    if not ctx.tenant_id:
        return ctx.record(
            make_error(step, f"tenant_id not found in response: {json.dumps(data)}")
        )

    return ctx.record({"step": step, "tenant_id": ctx.tenant_id})


def step_ensure_bucket(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 6: Create an S3 bucket via BFF (s3e-controller).

    Uses BFF endpoint so the bucket is registered in Cloud.ru platform
    and accessible by Managed RAG for indexing.
    """
    step = "ensure-bucket"
    bucket_name = ctx.bucket_name

    if not bucket_name:
        return ctx.record(
            make_error(step, "--bucket-name is required")
        )
    if not ctx.tenant_id:
        return ctx.record(
            make_error(step, "tenant_id required -- run get-tenant-id first")
        )

    if ctx.dry_run:
        return ctx.record(
            {"step": step, "bucket_name": bucket_name, "dry_run": True}
        )

    # Create bucket via BFF s3e-controller (registers in platform)
    # Match UI payload exactly — no global_name/domain_name, quota=0
    bucket_body = {
        "name": bucket_name,
        "storage_class": "STANDARD",
        "quotas": [{"type": "BUCKET_SIZE", "value": 0, "unit": "GB"}],
    }
    status, data = _bff_request(
        "POST",
        f"/u-api/s3e-controller/v1/tenants/{ctx.tenant_id}/buckets",
        ctx.token,
        body=bucket_body,
    )

    created = False
    if status in (200, 201):
        created = True
        # Save log_group_id from response for telemetry_configuration in KB
        ctx.log_group_id = data.get("log_group_id") if isinstance(data, dict) else None
    elif status == 409 or (isinstance(data, dict) and "already exists" in str(data).lower()):
        # Bucket already exists — ok
        created = False
    else:
        return ctx.record(
            make_error(step, f"Failed to create bucket via BFF: {json.dumps(data)}", status)
        )

    return ctx.record({"step": step, "bucket_name": bucket_name, "created": created})


# ---------------------------------------------------------------------------
# S3 upload via BFF request-signature (project-owned files)
# ---------------------------------------------------------------------------


def _upload_file_via_bff(
    token: str, bucket_name: str, s3_key: str, local_path: pathlib.Path,
    tenant_id: str = "",
) -> None:
    """Upload a file to S3 using BFF request-signature.

    This ensures files are owned by the project (not an SA), which is required
    for Managed RAG to read them via Search API.

    Flow:
      1. Construct AWS4 canonical request for PUT
      2. POST string_to_sign to BFF → get signature
      3. PUT file to S3 with signed Authorization header
    """
    import hashlib

    file_bytes = local_path.read_bytes()
    content_type = "application/octet-stream"
    if local_path.suffix == ".txt":
        content_type = "text/plain"
    elif local_path.suffix == ".pdf":
        content_type = "application/pdf"

    content_length = str(len(file_bytes))

    # Timestamp
    now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    date_short = now[:8]
    scope = f"{date_short}/{S3_REGION}/s3/aws4_request"

    # S3 URL
    s3_host = "s3.cloud.ru"
    s3_path = f"/{bucket_name}/{s3_key}"
    query = "isFileManager=true&x-id=PutObject"

    # Use UNSIGNED-PAYLOAD (matching UI behavior)
    content_sha = "UNSIGNED-PAYLOAD"

    # Canonical request — match UI signed headers exactly
    canonical_headers = (
        f"content-length:{content_length}\n"
        f"content-type:{content_type}\n"
        f"host:{s3_host}\n"
        f"x-amz-content-sha256:{content_sha}\n"
        f"x-amz-date:{now}\n"
        f"x-amz-storage-class:STANDARD\n"
    )
    signed_headers = "content-length;content-type;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class"

    canonical_request = (
        f"PUT\n"
        f"{s3_path}\n"
        f"{query}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{content_sha}"
    )
    canonical_hash = hashlib.sha256(canonical_request.encode()).hexdigest()

    string_to_sign = f"AWS4-HMAC-SHA256\n{now}\n{scope}\n{canonical_hash}"

    # Get signature from BFF
    status, sig_data = _bff_request(
        "POST",
        "/u-api/s3e-controller/v2/access/request-signature",
        token,
        body={"string_to_sign": string_to_sign},
    )
    if status not in (200, 201) or "signature" not in str(sig_data):
        raise RuntimeError(f"request-signature failed: HTTP {status} {json.dumps(sig_data)}")

    signature = sig_data.get("signature", "")
    key_id = sig_data.get("key_id", "")

    # Credential = tenant_id:key_id (matching UI Authorization header)
    credential_id = f"{tenant_id}:{key_id}" if tenant_id else key_id
    auth_header = (
        f"AWS4-HMAC-SHA256 "
        f"Credential={credential_id}/{scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    # PUT to S3
    s3_url = f"https://{s3_host}{s3_path}?{query}"
    resp = httpx.put(
        s3_url,
        content=file_bytes,
        headers={
            "Authorization": auth_header,
            "Content-Type": content_type,
            "Content-Length": content_length,
            "Host": s3_host,
            "x-amz-content-sha256": content_sha,
            "x-amz-date": now,
            "x-amz-storage-class": "STANDARD",
        },
        timeout=120,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"S3 PUT failed: HTTP {resp.status_code} {resp.text[:200]}")


def step_upload_docs(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 7: Upload documents from a local folder to the S3 bucket.

    Uses boto3 with AWS Signature V4 -- NOT rewritten to httpx.
    """
    step = "upload-docs"
    docs_path = ctx.docs_path
    bucket_name = ctx.bucket_name

    if not docs_path:
        return ctx.record(
            make_error(step, "--docs-path is required")
        )
    if not bucket_name:
        return ctx.record(
            make_error(step, "--bucket-name is required")
        )

    docs_dir = pathlib.Path(docs_path).expanduser().resolve()
    if not docs_dir.is_dir():
        return ctx.record(
            make_error(step, f"docs-path is not a directory: {docs_dir}")
        )

    # Collect files matching extensions
    extensions = {
        ext.strip().lstrip(".")
        for ext in (ctx.file_extensions or DEFAULT_FILE_EXTENSIONS).split(",")
    }
    files_to_upload: List[pathlib.Path] = []
    for ext in extensions:
        files_to_upload.extend(docs_dir.rglob(f"*.{ext}"))
    for ext in list(extensions):
        files_to_upload.extend(docs_dir.rglob(f"*.{ext.upper()}"))
    files_to_upload = sorted(set(files_to_upload))

    if not files_to_upload:
        return ctx.record(
            make_error(step, f"No files with extensions {extensions} found in {docs_dir}")
        )

    if ctx.dry_run:
        return ctx.record(
            {
                "step": step,
                "files_found": len(files_to_upload),
                "extensions": sorted(extensions),
                "dry_run": True,
            }
        )

    if not ctx.tenant_id or not ctx.key_id or not ctx.key_secret:
        return ctx.record(
            make_error(step, "S3 credentials required -- run previous steps")
        )

    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        return ctx.record(
            make_error(step, "boto3 is not installed -- pip install boto3")
        )

    s3_access_key = f"{ctx.tenant_id}:{ctx.key_id}"
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=ctx.key_secret,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )

    total_size = 0
    uploaded = 0
    for fpath in files_to_upload:
        relative = fpath.relative_to(docs_dir)
        s3_key = str(relative)
        file_size = fpath.stat().st_size
        try:
            # ACL=bucket-owner-full-control is CRITICAL:
            # Without it, Managed RAG Search API cannot read the files
            s3.upload_file(
                str(fpath), bucket_name, s3_key,
                ExtraArgs={
                    "StorageClass": "STANDARD",
                    "ACL": "bucket-owner-full-control",
                },
            )
            uploaded += 1
            total_size += file_size
        except Exception as exc:
            emit({"step": step, "warning": f"Failed to upload {relative}: {exc}"})

    return ctx.record(
        {"step": step, "files_uploaded": uploaded, "total_size_bytes": total_size}
    )


def _resolve_log_group_id(ctx: PipelineContext) -> str:
    """Get logaas_log_group_id for telemetry configuration.

    CRITICAL: Empty string breaks Search API deployment. Must provide a real ID.
    Sources: bucket creation response, or BFF log-groups endpoint.
    """
    if ctx.log_group_id:
        return ctx.log_group_id

    # Try to get from BFF log-groups (bucket-level)
    if ctx.tenant_id and ctx.token:
        status, data = _bff_request(
            "GET",
            f"/u-api/s3e-controller/v1/tenants/{ctx.tenant_id}/log-groups",
            ctx.token,
        )
        if status == 200:
            groups = data if isinstance(data, list) else data.get("items", data.get("log_groups", []))
            if groups and isinstance(groups, list) and len(groups) > 0:
                gid = groups[0].get("id") or groups[0].get("log_group_id") or ""
                if gid:
                    ctx.log_group_id = gid
                    return gid

    # Try to get from existing KB versions on the project (via SA IAM token)
    if ctx.project_id:
        try:
            iam_token = ctx.ensure_iam_token()
            import httpx
            with httpx.Client(transport=httpx.HTTPTransport(proxy=None), timeout=15) as c:
                r = c.get(
                    f"https://{RAG_API_HOST}/v1/knowledge-bases?project_id={ctx.project_id}&page_size=10",
                    headers={"Authorization": f"Bearer {iam_token}"},
                )
                if r.status_code == 200:
                    for kb in r.json().get("data", []):
                        kb_id = kb.get("knowledgebase_id", "")
                        if not kb_id:
                            continue
                        rv = c.get(
                            f"https://{RAG_API_HOST}/v1/knowledge-bases/versions?knowledgebase_id={kb_id}&project_id={ctx.project_id}&page_size=1",
                            headers={"Authorization": f"Bearer {iam_token}"},
                        )
                        if rv.status_code == 200:
                            for v in rv.json().get("data", []):
                                opts = v.get("knowledge_base_version_settings", {}).get("knowledge_base_settings", {}).get("options", {})
                                lgid = opts.get("logaas_log_group_id", "")
                                if lgid:
                                    ctx.log_group_id = lgid
                                    return lgid
        except Exception:
            pass

    # Fallback: return empty (may break Search API)
    return ""


def _build_kb_payload(ctx: PipelineContext) -> Dict[str, Any]:
    """Build the Knowledge Base creation payload."""
    extensions = [
        ext.strip().lstrip(".")
        for ext in (ctx.file_extensions or DEFAULT_FILE_EXTENSIONS).split(",")
    ]

    extractors = []
    if "txt" in extensions:
        extractors.append(
            {
                "file_patterns": ["txt"],
                "text_extractor": {
                    "recursive_char_splitter": {
                        "separators": "\n\n",
                        "is_separator_regex": False,
                        "keep_separator": "KeepSeparator_None",
                        "chunk_size": 1500,
                        "chunk_overlap": 300,
                    }
                },
            }
        )
    if "pdf" in extensions:
        extractors.append(
            {
                "file_patterns": ["pdf"],
                "pdf_extractor": {
                    "pdf_parser": {"method": "table_struct", "mode": "single"},
                    "markdown_smart_splitter": {
                        "chunk_size": 1500,
                        "chunk_overlap": 300,
                        "allow_oversize": False,
                        "headers_to_split_on": "",
                    },
                },
            }
        )

    return {
        "project_id": ctx.project_id,
        "knowledgebase_configuration": {
            "name": ctx.kb_name,
            "description": f"Knowledge Base '{ctx.kb_name}' (auto-created by setup pipeline)",
            "auth_configuration": {"forward_auth": False, "api_key": False},
            "database_configuration": {"support_hybrid_search": False},
        },
        "knowledgebase_version_configuration": {
            "name": "version-1",
            "description": "",
            "embedder_configuration": {
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "model_source": "MODEL_SOURCE_FOUNDATION_MODELS",
            },
            "telemetry_configuration": {
                "logging": {
                    "logaas_log_group_id": _resolve_log_group_id(ctx),
                },
            },
            "data_source_configuration": {
                "cloud_ru_evolution_object_storage_source": {
                    "bucket_name": ctx.bucket_name,
                    "paths": [""],
                    "object_storage_scan_options": {
                        "recursive": True,
                        "file_extensions": extensions,
                        "max_depth": 0,
                    },
                }
            },
            "extractors_configuration": {"extractors": extractors},
        },
    }


def step_create_kb(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 8: Create a Knowledge Base via BFF API (httpx, requires browser token)."""
    step = "create-kb"

    if not ctx.kb_name:
        return ctx.record(
            make_error(step, "--kb-name is required")
        )
    if not ctx.project_id:
        return ctx.record(
            make_error(step, "project_id required -- run extract-info first")
        )

    payload = _build_kb_payload(ctx)

    if ctx.dry_run:
        ctx.kb_id = ctx.kb_id or "dry-run-kb-id"
        return ctx.record(
            {"step": step, "kb_name": ctx.kb_name, "payload": payload, "dry_run": True}
        )

    status, data = _bff_request(
        "POST",
        "/u-api/managed-rag/user-plane/api/v2/knowledge-bases",
        ctx.token,
        body=payload,
        timeout=60.0,
    )

    if status not in (200, 201):
        return ctx.record(
            make_error(step, f"Failed to create KB: {json.dumps(data)}", status)
        )

    ctx.kb_id = (
        data.get("knowledgebase_id")
        or data.get("id")
        or data.get("knowledgebase", {}).get("id")
        or data.get("knowledgebase", {}).get("knowledgebase_id")
    )
    kb_status = (
        data.get("status")
        or data.get("knowledgebase", {}).get("status")
        or "UNKNOWN"
    )

    return ctx.record({"step": step, "kb_id": ctx.kb_id, "status": kb_status})


def step_wait_active(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 9: Poll until the KB becomes KNOWLEDGEBASE_ACTIVE.

    Strategy:
    - If access-key is available, obtain SA IAM token and poll via public API.
    - Otherwise fall back to BFF polling via browser token.
    """
    step = "wait-active"

    if not ctx.kb_id:
        return ctx.record(
            make_error(step, "kb_id required -- run create-kb first")
        )
    if not ctx.project_id:
        return ctx.record(
            make_error(step, "project_id required")
        )

    if ctx.dry_run:
        ctx.search_url = f"https://{ctx.kb_id}.managed-rag.inference.cloud.ru"
        return ctx.record(
            {"step": step, "kb_id": ctx.kb_id, "dry_run": True}
        )

    # Decide polling strategy: IAM token (preferred) or BFF fallback
    use_iam = False
    iam_token = None
    try:
        iam_token = ctx.ensure_iam_token()
        use_iam = True
    except RuntimeError:
        pass

    if use_iam:
        poll_host = RAG_API_HOST
        poll_path = f"/v1/knowledge-bases/{ctx.kb_id}?project_id={ctx.project_id}"
        emit({"step": step, "info": "Using SA IAM token for polling via public API"})
    else:
        poll_host = CONSOLE_HOST
        poll_path = (
            f"/u-api/managed-rag/user-plane/api/v2/knowledge-bases/{ctx.kb_id}"
            f"?project_id={ctx.project_id}"
        )
        emit({
            "step": step,
            "info": "No SA IAM token available, falling back to BFF polling. "
                    "Warning: browser token may expire before KB becomes active.",
        })

    deadline = time.monotonic() + DEFAULT_KB_POLL_TIMEOUT
    last_status = "UNKNOWN"
    poll_count = 0

    while time.monotonic() < deadline:
        poll_count += 1

        if use_iam:
            headers = _auth_headers(iam_token)
            status_code, data = _api_request(
                poll_host, "GET", poll_path, headers=headers, timeout=30.0
            )
        else:
            status_code, data = _bff_request("GET", poll_path, ctx.token, timeout=30.0)

        if status_code == 401 and use_iam:
            # IAM token expired -- refresh
            try:
                ctx.iam_token = None
                iam_token = ctx.ensure_iam_token()
            except RuntimeError:
                pass
            time.sleep(DEFAULT_KB_POLL_INTERVAL)
            continue

        if status_code == 401 and not use_iam:
            return ctx.record(
                make_error(
                    step,
                    "Browser token expired during polling. Re-run with a fresh token, "
                    "or ensure access-key step completes so IAM token can be used.",
                    401,
                )
            )

        if status_code != 200:
            emit({"step": step, "poll": poll_count, "http_status": status_code})
            time.sleep(DEFAULT_KB_POLL_INTERVAL)
            continue

        last_status = (
            data.get("status")
            or data.get("knowledgebase", {}).get("status")
            or "UNKNOWN"
        )

        if last_status == "KNOWLEDGEBASE_ACTIVE":
            ctx.search_url = f"https://{ctx.kb_id}.managed-rag.inference.cloud.ru"
            return ctx.record(
                {
                    "step": step,
                    "status": last_status,
                    "search_url": ctx.search_url,
                    "polls": poll_count,
                }
            )

        # Emit progress every 4 polls (~60s)
        if poll_count % 4 == 0:
            emit(
                {
                    "step": step,
                    "progress": True,
                    "status": last_status,
                    "polls": poll_count,
                }
            )

        time.sleep(DEFAULT_KB_POLL_INTERVAL)

    return ctx.record(
        make_error(
            step,
            f"Timeout after {DEFAULT_KB_POLL_TIMEOUT}s. Last status: {last_status}",
        )
    )


def step_save_env(ctx: PipelineContext) -> Dict[str, Any]:
    """Step 10: Save .env file with credentials and KB info."""
    step = "save-env"

    env_path = ctx.output_env or DEFAULT_ENV_PATH

    env_content_lines = [
        "# Auto-generated by managed_rag setup pipeline",
        f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f'export CP_CONSOLE_KEY_ID="{ctx.key_id or ""}"',
        f'export CP_CONSOLE_SECRET="{ctx.key_secret or ""}"',
        f'export PROJECT_ID="{ctx.project_id or ""}"',
        f'export MANAGED_RAG_KB_ID="{ctx.kb_id or ""}"',
        f'export MANAGED_RAG_SEARCH_URL="{ctx.search_url or ""}"',
        "",
    ]
    env_content = "\n".join(env_content_lines)

    if ctx.dry_run:
        return ctx.record(
            {"step": step, "path": env_path, "content_preview": env_content, "dry_run": True}
        )

    env_file = pathlib.Path(env_path).expanduser()
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(env_content, encoding="utf-8")

    return ctx.record({"step": step, "path": str(env_file)})


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_REGISTRY: Dict[str, Any] = {
    "extract-info": step_extract_info,
    "ensure-sa": step_ensure_sa,
    "ensure-role": step_ensure_role,
    "create-access-key": step_create_access_key,
    "get-tenant-id": step_get_tenant_id,
    "ensure-bucket": step_ensure_bucket,
    "upload-docs": step_upload_docs,
    "create-kb": step_create_kb,
    "wait-active": step_wait_active,
    "save-env": step_save_env,
}


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def run_pipeline(ctx: PipelineContext) -> int:
    """Run all steps sequentially. Stop on first critical error."""
    for step_name in ALL_STEPS:
        step_fn = STEP_REGISTRY[step_name]
        try:
            result = step_fn(ctx)
        except Exception as exc:
            result = {"step": step_name, "error": str(exc)}
            ctx.record(result)

        if "error" in result:
            if step_name not in ("ensure-role", "upload-docs", "wait-active", "save-env"):
                emit(
                    {
                        "pipeline": "stopped",
                        "failed_step": step_name,
                        "results": ctx.results,
                    }
                )
                return 1

    emit({"pipeline": "complete", "results": ctx.results})
    return 0


def run_single_step(ctx: PipelineContext, step_name: str) -> int:
    """Run a single named step."""
    step_fn = STEP_REGISTRY.get(step_name)
    if not step_fn:
        emit(make_error("cli", f"Unknown step: {step_name}"))
        return 1
    try:
        result = step_fn(ctx)
    except Exception as exc:
        emit(make_error(step_name, str(exc)))
        return 1
    return 0 if "error" not in result else 1


# ---------------------------------------------------------------------------
# CLI entry points (called from managed_rag.py via commands registry)
# ---------------------------------------------------------------------------


def _build_context(args) -> PipelineContext:
    """Build PipelineContext from argparse namespace."""
    _setup_no_proxy()
    return PipelineContext(
        token=args.token,
        project_id=getattr(args, "project_id", None),
        customer_id=getattr(args, "customer_id", None),
        sa_name=getattr(args, "sa_name", DEFAULT_SA_NAME),
        bucket_name=getattr(args, "bucket_name", None),
        docs_path=getattr(args, "docs_path", None),
        kb_name=getattr(args, "kb_name", None),
        file_extensions=getattr(args, "file_extensions", DEFAULT_FILE_EXTENSIONS),
        output_env=getattr(args, "output_env", None),
        dry_run=getattr(args, "dry_run", False),
    )


def cmd_setup(args):
    """Full 10-step infrastructure setup pipeline."""
    ctx = _build_context(args)
    rc = run_pipeline(ctx)
    if rc != 0:
        raise SystemExit(rc)


def cmd_setup_step(args):
    """Single step execution."""
    ctx = _build_context(args)
    step_name = args.step
    rc = run_single_step(ctx, step_name)
    if rc != 0:
        raise SystemExit(rc)
