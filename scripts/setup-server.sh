#!/bin/bash
# =============================================================================
# اسکریپت راه‌اندازی کامل سرور دانا
# سیستم‌عامل: Ubuntu 24.04 LTS (تمیز)
#
# این اسکریپت تمام نیازمندی‌ها را از صفر نصب و پلتفرم را راه‌اندازی می‌کند:
# - Docker و Docker Compose
# - Git و کلون پروژه
# - ساخت و اجرای تمام سرویس‌ها
# - تنظیم فایروال
# - تنظیم SSL (اختیاری)
#
# استفاده:
#   chmod +x scripts/setup-server.sh
#   sudo ./scripts/setup-server.sh
# =============================================================================

set -euo pipefail

# رنگ‌ها برای خروجی
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[اطلاع]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[موفق]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[هشدار]${NC} $1"; }
log_error() { echo -e "${RED}[خطا]${NC} $1"; }

# ===================== بررسی پیش‌نیازها =====================

if [ "$EUID" -ne 0 ]; then
    log_error "لطفاً با sudo اجرا کنید: sudo $0"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
DANA_DIR="/home/$ACTUAL_USER/dana"

echo ""
echo "=============================================="
echo "  🚀 راه‌اندازی پلتفرم هوش مصنوعی دانا"
echo "  Ubuntu 24.04 LTS - نصب از صفر"
echo "=============================================="
echo ""

# ===================== ۱. به‌روزرسانی سیستم =====================

log_info "مرحله ۱/۸: به‌روزرسانی سیستم..."
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
    software-properties-common
log_ok "سیستم به‌روزرسانی شد"

# ===================== ۲. نصب Docker =====================

log_info "مرحله ۲/۸: نصب Docker..."
if command -v docker &> /dev/null; then
    log_warn "Docker از قبل نصب شده"
else
    # اضافه کردن مخزن رسمی Docker
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # اضافه کردن کاربر به گروه docker
    usermod -aG docker "$ACTUAL_USER"
    log_ok "Docker نصب شد"
fi

# بررسی Docker Compose
if docker compose version &> /dev/null; then
    log_ok "Docker Compose $(docker compose version --short) آماده است"
else
    log_error "Docker Compose نصب نشد!"
    exit 1
fi

# ===================== ۳. نصب NVIDIA Container Toolkit (اختیاری) =====================

log_info "مرحله ۳/۸: بررسی NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU شناسایی شد. نصب NVIDIA Container Toolkit..."

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log_ok "NVIDIA Container Toolkit نصب شد"
else
    log_warn "GPU NVIDIA شناسایی نشد. سرویس inference-worker بدون GPU اجرا خواهد شد."
fi

# ===================== ۴. نصب Node.js =====================

log_info "مرحله ۴/۸: نصب Node.js 20 LTS..."
if command -v node &> /dev/null; then
    log_warn "Node.js از قبل نصب شده: $(node --version)"
else
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y -qq nodejs
    log_ok "Node.js $(node --version) نصب شد"
fi

# ===================== ۵. نصب Python 3.12 =====================

log_info "مرحله ۵/۸: بررسی Python 3.12..."
if python3 --version 2>&1 | grep -q "3.12"; then
    log_ok "Python 3.12 آماده است"
else
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3-pip
    log_ok "Python 3.12 نصب شد"
fi

# ===================== ۶. کلون یا به‌روزرسانی پروژه =====================

log_info "مرحله ۶/۸: آماده‌سازی کد پروژه..."
if [ -d "$DANA_DIR/.git" ]; then
    log_warn "پروژه از قبل موجود است. به‌روزرسانی..."
    cd "$DANA_DIR"
    sudo -u "$ACTUAL_USER" git pull origin main 2>/dev/null || true
else
    log_info "پوشه پروژه: $DANA_DIR"
    if [ ! -d "$DANA_DIR" ]; then
        log_error "پوشه $DANA_DIR وجود ندارد. لطفاً پروژه را کلون کنید."
        log_info "git clone <REPO_URL> $DANA_DIR"
        # اگر پروژه لوکال هست، ادامه بده
    fi
fi

cd "$DANA_DIR"

# ===================== ۷. تنظیم فایروال =====================

log_info "مرحله ۷/۸: تنظیم فایروال..."
if command -v ufw &> /dev/null; then
    ufw allow 22/tcp    # SSH
    ufw allow 80/tcp    # HTTP
    ufw allow 443/tcp   # HTTPS
    ufw allow 8000/tcp  # API Gateway
    ufw allow 3000/tcp  # Landing Page
    ufw --force enable
    log_ok "فایروال تنظیم شد (پورت‌های ۲۲، ۸۰، ۴۴۳، ۸۰۰۰، ۳۰۰۰ باز)"
else
    log_warn "ufw نصب نیست. فایروال تنظیم نشد."
fi

# ===================== ۸. ساخت و اجرا =====================

log_info "مرحله ۸/۸: ساخت و اجرای سرویس‌ها با Docker Compose..."

# بررسی وجود .env
if [ ! -f "$DANA_DIR/.env" ]; then
    log_error "فایل .env یافت نشد! لطفاً فایل .env را تنظیم کنید."
    exit 1
fi

# ساخت تمام سرویس‌ها
docker compose build --parallel

# اجرا
docker compose up -d

# انتظار برای آماده شدن
log_info "منتظر آماده شدن سرویس‌ها..."
sleep 10

# بررسی سلامت
echo ""
echo "=============================================="
echo "  وضعیت سرویس‌ها:"
echo "=============================================="
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""

# بررسی API Gateway
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    log_ok "API Gateway: آماده ✓"
else
    log_warn "API Gateway: هنوز در حال راه‌اندازی..."
fi

# بررسی Auth Service
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    log_ok "Auth Service: آماده ✓"
else
    log_warn "Auth Service: هنوز در حال راه‌اندازی..."
fi

echo ""
echo "=============================================="
echo "  🎉 نصب با موفقیت انجام شد!"
echo "=============================================="
echo ""
echo "  آدرس‌های دسترسی:"
echo "  ─────────────────────────────"
echo "  صفحه اصلی:      http://$(hostname -I | awk '{print $1}'):3000"
echo "  API Gateway:     http://$(hostname -I | awk '{print $1}'):8000"
echo "  API Docs:        http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "  RabbitMQ Panel:  http://$(hostname -I | awk '{print $1}'):15672"
echo "  Grafana:         http://$(hostname -I | awk '{print $1}'):3001"
echo "  MinIO Console:   http://$(hostname -I | awk '{print $1}'):9001"
echo ""
echo "  دستورات مفید:"
echo "  ─────────────────────────────"
echo "  مشاهده لاگ‌ها:    docker compose logs -f"
echo "  توقف سرویس‌ها:   docker compose down"
echo "  راه‌اندازی مجدد:  docker compose up -d"
echo "  وضعیت سرویس‌ها:  docker compose ps"
echo ""
echo "  ⚠️ کارهای باقیمانده (دستی):"
echo "  ─────────────────────────────"
echo "  ۱. تنظیم دامنه و DNS"
echo "  ۲. دریافت گواهی SSL با certbot"
echo "  ۳. تنظیم درگاه پرداخت (زرین‌پال)"
echo "  ۴. دانلود وزن‌های مدل Qwen3-235B"
echo ""
