#!/bin/bash

# =============================================================================
# Hunyuan3D-Part 智能启动脚本 v2.0
# 功能：端口检测、进程管理、GPU监控、模型检查
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 配置参数
APP_PORT=7860
APP_FILE="app.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --app)
            APP_FILE="$2"
            shift 2
            ;;
        --port)
            APP_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --app FILE    指定要运行的Python文件 (默认: app.py)"
            echo "  --port PORT   指定端口号 (默认: 7860)"
            echo "  --help|-h     显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done
LOG_FILE="logs/app_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/app.pid"

# 创建日志目录
mkdir -p logs

# 日志函数
log_info() {
    echo -e "${CYAN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1" >> "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $1" >> "$LOG_FILE"
}

# 显示启动横幅
show_banner() {
    echo -e "${PURPLE}"
    echo "=================================="
    echo "  ☯️ Hunyuan3D-Part 启动器 v2.0"
    echo "  🚀 智能3D模型分析工具"
    echo "=================================="
    echo -e "${NC}"
}

# 检查系统要求
check_system() {
    log_info "检查系统环境..."

    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi

    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_success "检测到NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
            log_info "GPU: $line"
        done
    else
        log_warn "未检测到NVIDIA GPU，将使用CPU模式"
    fi

    # 检查磁盘空间
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        log_warn "可用磁盘空间不足10GB，可能影响模型下载"
    fi
}

# 检查模型文件
check_models() {
    log_info "检查模型文件..."

    local p3sam_model="models/Hunyuan3D-Part/p3sam.pt"
    local xpart_model="models/Hunyuan3D-Part/xpart.pt"

    if [ -f "$p3sam_model" ]; then
        local p3sam_size=$(du -h "$p3sam_model" | cut -f1)
        log_success "P3-SAM模型已存在 ($p3sam_size)"
    else
        log_warn "P3-SAM模型未找到，应用启动时将自动下载"
    fi

    if [ -f "$xpart_model" ]; then
        local xpart_size=$(du -h "$xpart_model" | cut -f1)
        log_success "XPart模型已存在 ($xpart_size)"
    else
        log_warn "XPart模型未找到，应用启动时将自动下载"
    fi
}

# 检查端口占用
check_port() {
    log_info "检查端口 $APP_PORT 使用情况..."

    if lsof -Pi :$APP_PORT -sTCP:LISTEN -t >/dev/null; then
        log_warn "端口 $APP_PORT 被占用"

        # 获取占用进程信息
        local pid=$(lsof -ti:$APP_PORT)
        local process_info=$(ps -p $pid -o pid,ppid,user,comm 2>/dev/null || echo "Unknown process")

        log_info "占用进程信息: $process_info"

        # 询问是否终止进程
        log_info "自动清理占用端口的进程..."

        # 优雅关闭
        kill -TERM $pid 2>/dev/null
        sleep 3

        # 强制关闭
        if lsof -Pi :$APP_PORT -sTCP:LISTEN -t >/dev/null; then
            log_warn "优雅关闭失败，强制终止进程..."
            kill -KILL $pid 2>/dev/null
            sleep 2
        fi

        # 验证端口释放
        if lsof -Pi :$APP_PORT -sTCP:LISTEN -t >/dev/null; then
            log_error "无法释放端口 $APP_PORT"
            exit 1
        else
            log_success "端口 $APP_PORT 已自动释放"
        fi
    else
        log_success "端口 $APP_PORT 可用"
    fi
}

# 检查并清理旧的进程
cleanup_old_processes() {
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            log_warn "发现旧的应用进程 (PID: $old_pid)，正在清理..."
            kill -TERM "$old_pid" 2>/dev/null
            sleep 2
            if ps -p "$old_pid" > /dev/null 2>&1; then
                kill -KILL "$old_pid" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# 监控GPU使用情况
monitor_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU使用情况监控..."
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read gpu_util mem_used mem_total temp; do
            log_info "GPU利用率: ${gpu_util}%, 显存: ${mem_used}MB/${mem_total}MB, 温度: ${temp}°C"
        done
    fi
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."

    # Hugging Face镜像
    export HF_ENDPOINT=${HF_ENDPOINT:-"http://hf.x-gpu.com"}
    export HUGGINGFACE_HUB_CACHE="./cache"

    # CUDA设置
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

    # PyTorch设置
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

    # OMP设置
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"4"}

    log_info "HF_ENDPOINT: $HF_ENDPOINT"
    log_info "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
}

# 启动应用
start_application() {
    log_info "启动Hunyuan3D-Part应用..."

    # 创建输出目录
    mkdir -p P3-SAM/results/gradio

    echo -e "${WHITE}"
    echo "🚀 应用启动中..."
    echo "📡 Web界面将在 http://localhost:$APP_PORT 启动"
    echo "📝 日志文件: $LOG_FILE"
    echo "🛑 按 Ctrl+C 停止应用"
    echo -e "${NC}"

    # 启动应用并记录PID
    python "$APP_FILE" 2>&1 | tee -a "$LOG_FILE" &
    local app_pid=$!
    echo $app_pid > "$PID_FILE"

    log_success "应用已启动 (PID: $app_pid)"

    # 等待应用启动
    log_info "等待应用初始化..."
    sleep 5

    # 检查应用是否正常启动
    if ps -p $app_pid > /dev/null 2>&1; then
        if curl -s http://localhost:$APP_PORT > /dev/null 2>&1; then
            log_success "应用启动成功! 访问地址: http://localhost:$APP_PORT"
        else
            log_info "应用正在初始化，请稍候..."
        fi
    else
        log_error "应用启动失败，请查看日志: $LOG_FILE"
        exit 1
    fi

    # 等待用户中断
    wait $app_pid
}

# 清理函数
cleanup() {
    log_info "正在清理资源..."

    if [ -f "$PID_FILE" ]; then
        local app_pid=$(cat "$PID_FILE")
        if ps -p "$app_pid" > /dev/null 2>&1; then
            log_info "正在停止应用 (PID: $app_pid)..."
            kill -TERM "$app_pid" 2>/dev/null
            sleep 5
            if ps -p "$app_pid" > /dev/null 2>&1; then
                kill -KILL "$app_pid" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi

    log_success "清理完成"
    exit 0
}

# 主函数
main() {
    # 设置信号处理
    trap cleanup SIGINT SIGTERM

    # 显示启动横幅
    show_banner

    # 执行检查和启动流程
    check_system
    check_models
    cleanup_old_processes
    check_port
    setup_environment
    monitor_gpu
    start_application
}

# 脚本帮助信息
show_help() {
    echo "Hunyuan3D-Part 启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -p, --port     指定端口 (默认: 7860)"
    echo "  -g, --gpu      指定GPU设备 (默认: 0)"
    echo "  --no-gpu       禁用GPU，使用CPU模式"
    echo ""
    echo "示例:"
    echo "  $0                    # 默认启动"
    echo "  $0 -p 8080           # 使用端口8080"
    echo "  $0 -g 1              # 使用GPU 1"
    echo "  $0 --no-gpu          # CPU模式"
    echo ""
}

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            APP_PORT="$2"
            shift 2
            ;;
        -g|--gpu)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --no-gpu)
            export CUDA_VISIBLE_DEVICES=""
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查应用文件是否存在
if [ ! -f "$APP_FILE" ]; then
    log_error "应用文件 $APP_FILE 不存在"
    exit 1
fi

# 启动主函数
main