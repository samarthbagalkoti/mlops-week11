PY=python
BENTONAME=iris-svc
PORT=3000

# --- VENV ---
.PHONY: setup
setup:
	@test -d .venv || $(PY) -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip
	. .venv/bin/activate; pip install -r requirements.txt

# --- TRAIN & TEST ---
.PHONY: train test
train:
	. .venv/bin/activate; $(PY) train.py

test:
	. .venv/bin/activate; pytest -q

# --- SERVE LOCALLY (no container) ---
.PHONY: serve
serve:
	. .venv/bin/activate; bentoml serve service.py:svc --port $(PORT)

# -------- Bento containerize helpers --------
# Change this if your service name differs
BENTONAME ?= w11d1_bento_hello
# Optional: set a default platform (used by Docker buildx via Bento)
PLATFORM ?= linux/amd64

.PHONY: bento.containerize bento.tag

# Internal helper: resolve the freshest "<name>:<version>" from `bentoml list`
bento.tag:
	@. .venv/bin/activate; \
	set -e; \
	BENTO_TAG=$$(bentoml list | awk -v name="$(BENTONAME)" '$$1==name {print $$1":"$$2; exit}'); \
	if [ -z "$$BENTO_TAG" ]; then \
		echo "ERROR: Could not find a Bento for '$(BENTONAME)'. Run 'make bento.build' first."; \
		exit 1; \
	fi; \
	echo "$$BENTO_TAG" > .bento_tag; \
	echo "Resolved BENTO_TAG=$$BENTO_TAG"

# Build the OCI image from the resolved Bento tag
bento.containerize: bento.tag
	@. .venv/bin/activate; \
	set -e; \
	BENTO_TAG=$$(cat .bento_tag); \
	echo "Containerizing $$BENTO_TAG"; \
	DOCKER_BUILDKIT=1 bentoml containerize $$BENTO_TAG --platform $(PLATFORM)

# --- DOCKER RUN (local) ---
.PHONY: docker.run
docker.run:
	# find built image and run on port 3000
	IMG=$$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^$(BENTONAME):' | head -n1); \
	echo "Running $$IMG on :$(PORT)"; \
	docker run --rm -p $(PORT):3000 $$IMG

# --- CURL HELPERS ---
.PHONY: curl.health curl.predict
curl.health:
	curl -s http://localhost:$(PORT)/healthz | jq .

curl.predict:
	curl -s http://localhost:$(PORT)/predict \
	  -H "Content-Type: application/json" \
	  -d '{"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2}' | jq .

# --- ECR (optional, set env vars first) ---
# export AWS_ACCOUNT_ID=123456789012
# export AWS_REGION=us-east-1
# export ECR_REPO=iris-svc
.PHONY: ecr.login docker.tag docker.push
ecr.login:
	aws ecr get-login-password --region $$AWS_REGION | docker login --username AWS --password-stdin $$AWS_ACCOUNT_ID.dkr.ecr.$$AWS_REGION.amazonaws.com

docker.tag:
	IMG=$$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^$(BENTONAME):' | head -n1); \
	TAG=$${IMG#$(BENTONAME):}; \
	docker tag $$IMG $$AWS_ACCOUNT_ID.dkr.ecr.$$AWS_REGION.amazonaws.com/$$ECR_REPO:$$TAG; \
	echo "Tagged as $$AWS_ACCOUNT_ID.dkr.ecr.$$AWS_REGION.amazonaws.com/$$ECR_REPO:$$TAG"

docker.push:
	docker push $$AWS_ACCOUNT_ID.dkr.ecr.$$AWS_REGION.amazonaws.com/$$ECR_REPO:latest

