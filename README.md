# End-to-End MLOps Pipeline — Olivetti Face Classifier

An automated MLOps pipeline that trains a **DecisionTreeClassifier** on the
**Olivetti faces dataset**, serves predictions through a **Flask** web app,
containerises everything with **Docker**, automates CI/CD via **GitHub Actions**,
and deploys at scale on **Kubernetes**.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI/CD pipeline
├── k8s/
│   └── deployment.yaml     # Kubernetes Deployment + NodePort Service
├── templates/
│   └── index.html          # Flask HTML front-end
├── train.py                # Model training script
├── test.py                 # Model evaluation script
├── app.py                  # Flask prediction server
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Branching Strategy

| Branch        | Purpose                                         |
|---------------|-------------------------------------------------|
| `main`        | Initial repo setup (README + .gitignore)        |
| `dev`         | Model development + CI workflow                 |
| `docker_cicd` | Docker build, Docker Hub push, K8s deployment   |

> `dev` and `docker_cicd` are **not** merged back into `main`.

---

## Step-by-Step Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- A Docker Hub account
- `kubectl` + a running cluster (Minikube, kind, or cloud)

---

### Step 1 — Clone & install

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
pip install -r requirements.txt
```

---

### Step 2 — Train the model (`dev` branch)

```bash
git checkout dev
python train.py
```

Expected output:
```
Loading Olivetti faces dataset...
Dataset loaded: 400 samples, 4096 features, 40 classes
Train size : 280 samples
Test  size : 120 samples
Training DecisionTreeClassifier...
Training complete.
Model saved to savedmodel.pth
```

---

### Step 3 — Evaluate the model

```bash
python test.py
```

Expected output:
```
Loading model from savedmodel.pth...

Test Accuracy: XX.XX%

Detailed Classification Report:
...
```

---

### Step 4 — Run the Flask app locally

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000), upload any 64×64 grayscale
face image, and click **Predict**.

---

### Step 5 — Docker (`docker_cicd` branch)

```bash
git checkout docker_cicd

# Build (trains model inside the container automatically)
docker build -t <YOUR_DOCKERHUB_USERNAME>/olivetti-face-classifier:latest .

# Run locally
docker run -p 5000:5000 <YOUR_DOCKERHUB_USERNAME>/olivetti-face-classifier:latest

# Push to Docker Hub
docker login
docker push <YOUR_DOCKERHUB_USERNAME>/olivetti-face-classifier:latest
```

---

### Step 6 — Kubernetes Deployment

> Edit `k8s/deployment.yaml` and replace `YOUR_DOCKERHUB_USERNAME` with your
> actual Docker Hub username before applying.

```bash
# Apply the Deployment and NodePort Service
kubectl apply -f k8s/deployment.yaml

# Verify 3 replicas are running
kubectl get pods
kubectl get svc olivetti-classifier-service
```

Access the app:
- **Minikube**: `minikube service olivetti-classifier-service --url`
- **Cloud node**: `http://<NODE_IP>:30080`

#### Demonstrate self-healing (3 replicas always running)

```bash
# Delete one pod — Kubernetes will immediately recreate it
kubectl delete pod <POD_NAME>

# Watch the replacement spin up
kubectl get pods -w
```

---

## CI/CD — GitHub Actions

The workflow file `.github/workflows/ci.yml` runs automatically on every push.

| Job                   | Trigger          | Steps                                      |
|-----------------------|------------------|--------------------------------------------|
| `check_working_repo`  | all branches     | Checkout → Install → `train.py` → `test.py`|
| `build_and_push_docker` | `docker_cicd` only | Build image → Push to Docker Hub       |

### Required GitHub Secrets

| Secret               | Value                          |
|----------------------|--------------------------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username       |
| `DOCKERHUB_TOKEN`    | Docker Hub access token        |

Set these under **Settings → Secrets and variables → Actions** in your repo.

---

## Dataset & Model

| Item       | Detail                                      |
|------------|---------------------------------------------|
| Dataset    | `sklearn.datasets.fetch_olivetti_faces`     |
| Samples    | 400 (40 subjects × 10 images each)          |
| Features   | 4096 (64×64 pixel grayscale values)         |
| Classes    | 40                                          |
| Split      | 70% train / 30% test (stratified)           |
| Model      | `sklearn.tree.DecisionTreeClassifier`       |
| Serialiser | `joblib` → `savedmodel.pth`                 |

---

## Links

- **GitHub Repository**: `https://github.com/<YOUR_USERNAME>/<YOUR_REPO>`
- **Docker Hub**: `https://hub.docker.com/r/<YOUR_DOCKERHUB_USERNAME>/olivetti-face-classifier`
