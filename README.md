# HuggingFace Proxy Service

A production-ready proxy service for the HuggingFace API built with FastAPI, featuring structured JSON logging, graceful shutdown, and a simple interface.

## Features

- **FastAPI Framework**: High-performance async API built on Python 3.12
- **Structured Logging**: JSON-formatted logs to stdout for easy parsing
- **Graceful Shutdown**: Proper SIGTERM handling for Kubernetes deployments
- **Multi-stage Docker Build**: Optimized image with non-root user
- **Comprehensive Testing**: Full test suite with pytest
- **CI/CD Ready**: GitHub Actions workflow included

## Endpoints

### Health Check
```
GET /healthz
```
Returns service health status and version.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### List Outputs
```
GET /outputs
```
List all available models from HuggingFace.

**Response:**
```json
{
  "items": [
    {
      "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
      "visibility": "public",
      "gated": true
    },
    {
      "repo_id": "bert-base-uncased",
      "visibility": "public",
      "gated": false
    }
  ]
}
```

### Get Output Details
```
GET /outputs/{repo_id}
```
Get detailed information about a specific repository.

**Example:**
```
GET /outputs/meta-llama/Llama-3.1-8B-Instruct
```

**Response:**
```json
{
  "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
  "visibility": "public",
  "gated": true,
  "tags": ["text-generation", "llama", "pytorch"],
  "cached": false,
  "last_modified": "2024-07-23T14:48:00.000Z"
}
```

**Note:** The `cached` field is always `false` in the current implementation.

## Quick Start

### Using Docker (Recommended)

1. **Build the image:**
   ```bash
   make build
   ```

2. **Run the container:**
   ```bash
   make run
   ```

3. **Test the service:**
   ```bash
   curl http://localhost:8080/healthz
   ```

### Using Poetry (Development)

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Run development server:**
   ```bash
   make dev
   ```

3. **Access the API:**
   - API: http://localhost:8080
   - Docs: http://localhost:8080/docs
   - ReDoc: http://localhost:8080/redoc

## Configuration

Configure the service using environment variables with the `HF_PROXY_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_PROXY_PORT` | `8080` | Server port |
| `HF_PROXY_LOG_LEVEL` | `INFO` | Logging level |
| `HF_PROXY_LOG_JSON` | `true` | Enable JSON logging |
| `HF_PROXY_HF_TIMEOUT` | `30` | API request timeout (seconds) |

**Note:** HuggingFace authentication tokens are retrieved via third-party library call (`lib.call()`) rather than environment variables.

## Development

### Running Tests

```bash
# Run all tests
make test

# Run linting
make lint

# Format code
make format

# Run all checks
make check
```

### Project Structure

```
hf-proxy/
├── main.py              # Single-file FastAPI application
├── tests/               # Test suite
├── Dockerfile           # Multi-stage build
├── Makefile             # Development commands
└── pyproject.toml       # Poetry dependencies
```

**Simplified Structure**: All application code is consolidated into a single `main.py` file containing:
- Configuration and settings
- Structured logging setup
- Pydantic models
- HuggingFace service client
- FastAPI endpoints and application

## Docker Image

The Docker image is optimized for production:
- **Multi-stage build** reduces final image size
- **Non-root user** (`appuser`) for security
- **Health checks** for container orchestration
- **Graceful shutdown** on SIGTERM

### Building and Pushing

```bash
# Build image
make build

# Tag and push to registry
export REGISTRY=ghcr.io/your-org
export IMAGE_TAG=v1.0.0
make build-hook-image
make push-hook-image
```

## CI/CD

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. Runs linting and tests
2. Builds the Docker image
3. Tests the container health
4. Pushes to registry (on main branch)

## Deployment

### Kubernetes Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hf-proxy
  template:
    metadata:
      labels:
        app: hf-proxy
    spec:
      containers:
      - name: hf-proxy
        image: hf-proxy:latest
        ports:
        - containerPort: 8080
        env:
        - name: HF_PROXY_LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make install` | Install dependencies with Poetry |
| `make test` | Run tests with pytest |
| `make lint` | Run linting with ruff |
| `make format` | Format code with ruff |
| `make clean` | Clean build artifacts |
| `make build` | Build Docker image |
| `make push` | Push Docker image to registry |
| `make run` | Run Docker container locally |
| `make dev` | Run development server with hot reload |
| `make check` | Run linting and tests |
| `make all` | Run all checks and build |

## License

[Add your license here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
