#!/bin/bash
# Deployment script for Qwen3 Reranker Service

set -e

# Configuration
COMPOSE_FILE="compose.yaml"
PROD_COMPOSE_FILE="compose.prod.yaml"
CPU_COMPOSE_FILE="compose.cpu.yaml"
CONFIG_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_usage() {
    echo "Usage: $0 [dev|prod|cpu|stop|logs|test]"
    echo "  dev   - Start development environment"
    echo "  prod  - Start production environment (GPU with flash_attn)"
    echo "  cpu   - Start CPU-only production environment"
    echo "  stop  - Stop all containers"
    echo "  logs  - Show container logs"
    echo "  test  - Run API tests"
}

check_requirements() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: Docker Compose plugin is not installed${NC}"
        exit 1
    fi
}

setup_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${YELLOW}Creating .env from template...${NC}"
        cp config.env.example "$CONFIG_FILE"
        echo -e "${GREEN}✅ Created $CONFIG_FILE - please review and modify as needed${NC}"
    fi
}

setup_hf_cache() {
    if [ ! -d "./hf_cache" ]; then
        echo -e "${YELLOW}Creating hf_cache directory...${NC}"
        mkdir -p ./hf_cache
        chmod 777 ./hf_cache
        echo -e "${GREEN}✅ Created hf_cache directory with 777 permissions${NC}"
    fi
}

setup_compose_file() {
    local mode="$1"

    case "$mode" in
        "prod")
            echo -e "${YELLOW}Setting up compose.yaml for production...${NC}"
            cp "$PROD_COMPOSE_FILE" "$COMPOSE_FILE"
            echo -e "${GREEN}✅ Copied $PROD_COMPOSE_FILE to $COMPOSE_FILE${NC}"
            ;;
        "cpu")
            echo -e "${YELLOW}Setting up compose.yaml for CPU-only...${NC}"
            cp "$CPU_COMPOSE_FILE" "$COMPOSE_FILE"
            echo -e "${GREEN}✅ Copied $CPU_COMPOSE_FILE to $COMPOSE_FILE${NC}"
            ;;
        *)
            # For dev mode, we keep the original compose.yaml
            ;;
    esac
}

start_dev() {
    echo -e "${GREEN}🚀 Starting development environment...${NC}"
    setup_config
    setup_hf_cache
    docker compose -f "$COMPOSE_FILE" up --build -d
    echo -e "${GREEN}✅ Development environment started${NC}"
    echo -e "${YELLOW}API available at: http://localhost:8004${NC}"
}

start_prod() {
    echo -e "${GREEN}🚀 Starting production environment...${NC}"
    setup_config
    setup_hf_cache
    setup_compose_file "prod"
    docker compose -f "$COMPOSE_FILE" up --build -d
    echo -e "${GREEN}✅ Production environment started${NC}"
    echo -e "${YELLOW}API available at: http://localhost:8004${NC}"
}

start_cpu() {
    echo -e "${GREEN}🚀 Starting CPU-only production environment...${NC}"
    setup_config
    setup_hf_cache
    setup_compose_file "cpu"
    docker compose -f "$COMPOSE_FILE" up --build -d
    echo -e "${GREEN}✅ CPU-only production environment started${NC}"
    echo -e "${YELLOW}API available at: http://localhost:8004${NC}"
}

stop_all() {
    echo -e "${YELLOW}🛑 Stopping all containers...${NC}"
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    docker compose -f "$PROD_COMPOSE_FILE" down 2>/dev/null || true
    docker compose -f "$CPU_COMPOSE_FILE" down 2>/dev/null || true
    echo -e "${GREEN}✅ All containers stopped${NC}"
}

show_logs() {
    echo -e "${GREEN}📋 Container logs:${NC}"
    if docker ps --format "table {{.Names}}" | grep -q "qwen3-reranker-prod"; then
        docker logs -f qwen3-reranker-prod
    elif docker ps --format "table {{.Names}}" | grep -q "qwen3-reranker-cpu"; then
        docker logs -f qwen3-reranker-cpu
    elif docker ps --format "table {{.Names}}" | grep -q "qwen3-reranker"; then
        docker logs -f qwen3-reranker
    else
        echo -e "${RED}No Qwen reranker containers running${NC}"
    fi
}

run_tests() {
    echo -e "${GREEN}🧪 Running API tests...${NC}"
    if command -v python3 &> /dev/null; then
        python3 test_api.py
    elif command -v python &> /dev/null; then
        python test_api.py
    else
        echo -e "${RED}Error: Python is not installed${NC}"
        exit 1
    fi
}

# Main
check_requirements

case "${1:-}" in
    "dev")
        start_dev
        ;;
    "prod")
        start_prod
        ;;
    "cpu")
        start_cpu
        ;;
    "stop")
        stop_all
        ;;
    "logs")
        show_logs
        ;;
    "test")
        run_tests
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
