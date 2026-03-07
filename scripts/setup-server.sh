#!/bin/bash
# =============================================================================
# Dana AI Platform - Server Setup Script
# Target OS: Ubuntu 24.04 LTS (clean install)
#
# This script installs all dependencies and launches the entire platform:
# 1. System packages (Docker, Node.js 20, Python 3.12, nginx, certbot)
# 2. NVIDIA Container Toolkit (if GPU detected)
# 3. Firewall configuration
# 4. .env file setup
# 5. Docker Compose build & launch
# 6. Health checks
#
# Usage:
#   chmod +x scripts/setup-server.sh
#   sudo ./scripts/setup-server.sh
#
# Options:
#   --skip-gpu       Skip NVIDIA GPU setup
#   --skip-firewall  Skip firewall configuration
#   --no-start       Build images but don't start services
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR ]${NC} $1"; }
log_step()  { echo -e "\n${CYAN}${BOLD}==> Step $STEP/$TOTAL_STEPS: $1${NC}"; }

# Parse arguments
SKIP_GPU=false
SKIP_FIREWALL=false
NO_START=false
for arg in "$@"; do
    case $arg in
        --skip-gpu)       SKIP_GPU=true ;;
        --skip-firewall)  SKIP_FIREWALL=true ;;
        --no-start)       NO_START=true ;;
    esac
done

# Root check
if [ "$EUID" -ne 0 ]; then
    log_error "Please run with sudo: sudo $0"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
DANA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "======================================================"
echo "  Dana AI Platform - Server Setup"
echo "  Ubuntu 24.04 LTS"
echo "======================================================"
echo ""
echo "  Project directory: $DANA_DIR"
echo "  User: $ACTUAL_USER"
echo ""

TOTAL_STEPS=7
STEP=0

# ===================== 1. System Update & Packages =====================
STEP=1
log_step "Updating system and installing packages"

apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    make \
    jq \
    htop \
    unzip \
    software-properties-common \
    nginx \
    certbot \
    python3-certbot-nginx

log_ok "System packages installed (including nginx + certbot)"

# ===================== 2. Docker =====================
STEP=2
log_step "Installing Docker"

if command -v docker &> /dev/null; then
    log_ok "Docker already installed: $(docker --version)"
else
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    usermod -aG docker "$ACTUAL_USER"
    log_ok "Docker installed"
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose not available!"
    exit 1
fi
log_ok "Docker Compose $(docker compose version --short) ready"

# ===================== 3. NVIDIA Container Toolkit =====================
STEP=3
log_step "Checking NVIDIA GPU"

if [ "$SKIP_GPU" = true ]; then
    log_warn "GPU setup skipped (--skip-gpu)"
elif command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log_info "GPU detected: $GPU_NAME ($GPU_MEM)"

    if ! command -v nvidia-ctk &> /dev/null; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

        apt-get update -qq
        apt-get install -y -qq nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        log_ok "NVIDIA Container Toolkit installed"
    else
        log_ok "NVIDIA Container Toolkit already installed"
    fi
else
    log_warn "No NVIDIA GPU detected. inference-worker and finetuning require GPU."
fi

# ===================== 4. Node.js & Python =====================
STEP=4
log_step "Installing Node.js 20 and Python 3.12"

if command -v node &> /dev/null && node --version | grep -q "v20"; then
    log_ok "Node.js $(node --version) already installed"
else
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y -qq nodejs
    log_ok "Node.js $(node --version) installed"
fi

if python3 --version 2>&1 | grep -q "3.12"; then
    log_ok "Python 3.12 ready"
else
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3-pip
    log_ok "Python 3.12 installed"
fi

# ===================== 5. Firewall =====================
STEP=5
log_step "Configuring firewall"

if [ "$SKIP_FIREWALL" = true ]; then
    log_warn "Firewall setup skipped (--skip-firewall)"
elif command -v ufw &> /dev/null; then
    ufw allow 22/tcp     # SSH
    ufw allow 80/tcp     # HTTP
    ufw allow 443/tcp    # HTTPS
    ufw allow 3000/tcp   # Landing page (direct, dev only)
    ufw allow 8000/tcp   # API Gateway (direct, dev only)
    ufw --force enable
    log_ok "Firewall configured (ports: 22, 80, 443, 3000, 8000)"
else
    log_warn "ufw not found, skipping firewall"
fi

# ===================== 6. Environment Setup =====================
STEP=6
log_step "Checking environment configuration"

cd "$DANA_DIR"

if [ ! -f "$DANA_DIR/.env" ]; then
    if [ -f "$DANA_DIR/.env.example" ]; then
        cp "$DANA_DIR/.env.example" "$DANA_DIR/.env"
        log_warn ".env created from .env.example - EDIT IT with production values!"
        log_warn "  nano $DANA_DIR/.env"
    else
        log_error ".env file not found and no .env.example available!"
        log_error "Create a .env file with required variables before running this script."
        exit 1
    fi
else
    log_ok ".env file found"
fi

# Validate critical env vars
set +u
source "$DANA_DIR/.env"
set -u

MISSING=""
for var in POSTGRES_PASSWORD REDIS_PASSWORD JWT_SECRET_KEY RABBITMQ_PASSWORD; do
    if [ -z "${!var:-}" ]; then
        MISSING="$MISSING $var"
    fi
done

if [ -n "$MISSING" ]; then
    log_error "Missing required .env variables:$MISSING"
    log_error "Edit your .env file and re-run this script."
    exit 1
fi
log_ok "Environment variables validated"

# Warn about placeholder values
if [ "${ZARINPAL_MERCHANT_ID:-}" = "YOUR_MERCHANT_ID_HERE" ]; then
    log_warn "ZarinPal merchant ID is a placeholder - payment won't work"
fi
if [ "${SMTP_PASSWORD:-}" = "YOUR_SMTP_PASSWORD_HERE" ]; then
    log_warn "SMTP password is a placeholder - email won't work"
fi

# ===================== 7. Build & Launch =====================
STEP=7
log_step "Building and launching services"

# Create model directory if referenced
MODEL_DIR="${MODEL_PATH:-/models}"
MODEL_PARENT=$(dirname "$MODEL_DIR")
if [ ! -d "$MODEL_PARENT" ]; then
    mkdir -p "$MODEL_PARENT"
    log_info "Created model directory: $MODEL_PARENT"
fi

# Build all images
log_info "Building Docker images (this may take several minutes)..."
docker compose build --parallel

if [ "$NO_START" = true ]; then
    log_ok "Images built successfully. Skipping start (--no-start)."
else
    log_info "Starting services..."
    docker compose up -d

    # Wait for services to initialize
    log_info "Waiting for services to become healthy (30s)..."
    sleep 30

    echo ""
    echo "======================================================"
    echo "  Service Status"
    echo "======================================================"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || docker compose ps

    echo ""

    # Health checks
    check_service() {
        local name=$1
        local url=$2
        if curl -sf --max-time 5 "$url" > /dev/null 2>&1; then
            log_ok "$name: healthy"
        else
            log_warn "$name: still starting..."
        fi
    }

    check_service "API Gateway     " "http://localhost:8000/health"
    check_service "Auth Service    " "http://localhost:8001/health"
    check_service "Billing Service " "http://localhost:8003/health"
    check_service "Model Registry  " "http://localhost:8005/health"
    check_service "Landing Page    " "http://localhost:3000"
    check_service "Grafana         " "http://localhost:3001"
    check_service "Prometheus      " "http://localhost:9090/-/healthy"
fi

# ===================== Summary =====================

SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo ""
echo "  Web:"
echo "  ────────────────────────────────────────"
echo "  Landing Page:     http://$SERVER_IP:3000"
echo "  API Gateway:      http://$SERVER_IP:8000"
echo "  API Docs:         http://$SERVER_IP:8000/docs"
echo ""
echo "  Monitoring:"
echo "  ────────────────────────────────────────"
echo "  Grafana:          http://$SERVER_IP:3001  (admin / see .env)"
echo "  Prometheus:       http://$SERVER_IP:9090"
echo "  RabbitMQ:         http://$SERVER_IP:15672"
echo "  MinIO Console:    http://$SERVER_IP:9001"
echo "  Metabase:         http://$SERVER_IP:3002"
echo ""
echo "  Commands:"
echo "  ────────────────────────────────────────"
echo "  View logs:        docker compose logs -f"
echo "  View one svc:     docker compose logs -f auth-service"
echo "  Stop all:         docker compose down"
echo "  Restart all:      docker compose up -d"
echo "  Status:           docker compose ps"
echo "  Run tests:        make test"
echo "  Rebuild:          docker compose build --parallel && docker compose up -d"
echo ""
echo "  Next Steps:"
echo "  ────────────────────────────────────────"
echo "  1. Point DNS (dana.ir, api.dana.ir) -> $SERVER_IP"
echo "  2. SSL:       sudo certbot --nginx -d dana.ir -d api.dana.ir"
echo "  3. Edit .env: set ZARINPAL_MERCHANT_ID, SMTP credentials"
echo "  4. Models:    sudo ./scripts/finetune.sh download"
echo "  5. Fine-tune: sudo ./scripts/finetune.sh train --dataset your_data.jsonl"
echo ""
