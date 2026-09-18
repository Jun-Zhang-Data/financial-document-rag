# Deployment

## 1. Local API, no database

This is the fastest developer path and uses the existing BM25 backend.

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=src uvicorn api:app --app-dir src --host 0.0.0.0 --port 8080
```

Test it:

```bash
curl http://localhost:8080/health/ready
curl -X POST http://localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{"question":"Why did Northstar EBITDA margin fall in 2024?","generate":false}'
```

## 2. Docker Compose: API + PostgreSQL/pgvector

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` if you want generated answers. Retrieval
works without them.

```bash
docker compose up --build
```

Compose starts PostgreSQL, runs the idempotent ingestion job, then starts the API on port
8080. The first build/run may download the SentenceTransformer model.

Optional local observability:

```bash
docker compose --profile observability up --build
```

This also starts Prometheus on port 9090 and Grafana on port 3000. Change the default
Grafana password before using that profile outside a throwaway local environment.

## 3. Kubernetes

Use managed PostgreSQL with pgvector enabled rather than running the database in the same
cluster unless you have an established stateful-database platform.

Build a production image with the embedding model baked in so runtime pods do not require
Hugging Face egress:

```bash
docker build \
  --build-arg PRELOAD_EMBEDDING_MODEL=1 \
  -t ghcr.io/YOUR_ORG/financial-document-rag:YOUR_TAG .
docker push ghcr.io/YOUR_ORG/financial-document-rag:YOUR_TAG
```

Replace the image placeholder in `deploy/k8s/deployment.yaml` and
`deploy/k8s/ingest-job.yaml`. Create the namespace and secret from your secret manager; do
not commit the example secret.

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secret.yaml
kubectl apply -k deploy/k8s
kubectl apply -f deploy/k8s/ingest-job.yaml
kubectl wait --for=condition=complete job/financial-rag-ingest -n financial-rag --timeout=10m
kubectl rollout status deployment/financial-rag -n financial-rag
```

The supplied deployment has two replicas, readiness/liveness probes, CPU-based HPA,
resource requests/limits, a PodDisruptionBudget, non-root execution, dropped Linux
capabilities, and a read-only root filesystem with writable cache/tmp volumes.

## 4. Secrets and production controls

Use your cloud secret manager or Vault for `DATABASE_URL`, `OPENAI_API_KEY`, and
`SERVICE_API_KEY`. Put TLS and enterprise identity at the ingress/API-gateway layer. The
built-in `X-API-Key` check is useful for service-to-service deployments but is not a full
workforce identity system.

Before production traffic, also configure database backups/PITR, network policies,
private database connectivity, log redaction, alert thresholds, and a document-retention
policy.

## 5. CI/CD

`.github/workflows/ci.yml` runs tests, Ruff, dependency auditing, and a container build.
Dependabot tracks Python and GitHub Actions updates. For deployment, publish the tested
image to your registry and let your existing GitOps system (for example Argo CD or Flux)
apply the Kubernetes manifests.
