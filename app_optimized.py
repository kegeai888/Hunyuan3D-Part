"""
Hunyuan3D-Part 优化版本 - RTX 4090 GPU加速
集成Flash Attention、混合精度、torch.compile等优化技术
"""

import gradio as gr
import os
import sys
import numpy as np
import trimesh
from pathlib import Path
import torch
import pytorch_lightning as pl
import spaces
import time
import logging
from contextlib import contextmanager

# 导入GPU优化模块
from gpu_optimizer import RTX4090Optimizer, FlashAttentionOptimizer, gpu_inference_mode

sys.path.append('P3-SAM')
from demo.auto_mask import AutoMask
sys.path.append('XPart')
from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file

# 初始化优化器
gpu_optimizer = RTX4090Optimizer()
flash_attn_optimizer = FlashAttentionOptimizer()

# 全局变量
automask = None
_PIPELINE = None
output_path = 'P3-SAM/results/gradio'
os.makedirs(output_path, exist_ok=True)

# 性能监控
performance_logger = logging.getLogger("PerformanceMonitor")
performance_logger.setLevel(logging.INFO)

def setup_performance_logging():
    """设置性能监控日志"""
    if not performance_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] PERF - %(message)s'
        )
        handler.setFormatter(formatter)
        performance_logger.addHandler(handler)

@contextmanager
def timer(operation_name: str):
    """性能计时器"""
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        elapsed = end_time - start_time
        memory_diff = (end_memory - start_memory) / (1024**2)  # MB

        performance_logger.info(
            f"{operation_name}: {elapsed:.2f}s, Memory: {memory_diff:+.1f}MB"
        )

def initialize_models():
    """延迟加载模型，使用GPU优化"""
    global automask, _PIPELINE

    with timer("模型初始化"):
        if automask is None:
            with timer("P3-SAM模型加载"):
                automask = AutoMask()
                # 优化P3-SAM模型
                if hasattr(automask, 'model'):
                    automask.model = gpu_optimizer.optimize_model(automask.model)

        if _PIPELINE is None:
            with timer("XPart模型加载"):
                pl.seed_everything(2026, workers=True)
                cfg_path = str(Path(__file__).parent / "XPart/partgen/config" / "infer.yaml")
                config = get_config_from_file(cfg_path)

                assert hasattr(config, "ckpt") or hasattr(
                    config, "ckpt_path"
                ), "ckpt or ckpt_path must be specified in config"

                _PIPELINE = PartFormerPipeline.from_pretrained(
                    config=config,
                    verbose=True,
                    ignore_keys=config.get("ignore_keys", []),
                )

                # GPU优化
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _PIPELINE.to(device=device, dtype=torch.float16)  # 使用FP16

                # 编译优化主要模型组件
                if hasattr(_PIPELINE, 'model') and torch.cuda.is_available():
                    _PIPELINE.model = gpu_optimizer.optimize_model(_PIPELINE.model)

    gpu_optimizer.log_performance_summary()

def is_supported_3d_file(filename):
    """检查3D文件格式支持"""
    if filename is None:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.glb', '.ply', '.obj']

def get_file_info(filename):
    """获取文件信息"""
    if filename is None:
        return "未选择文件"
    try:
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        return f"{os.path.basename(filename)} ({file_size:.1f} MB)"
    except:
        return f"{os.path.basename(filename)}"

@spaces.GPU
def segment_mesh_optimized(mesh_path, postprocess=True, postprocess_threshold=0.95, seed=42, progress=gr.Progress()):
    """P3-SAM 3D模型分割 - GPU优化版本"""

    with timer("完整分割流程"):
        try:
            # 初始化模型
            progress(0.1, desc="🚀 初始化GPU优化模型...")
            with timer("模型初始化"):
                initialize_models()

            # 验证输入
            if mesh_path is None:
                return None, None, None, "❌ 请先上传3D模型文件"

            if not is_supported_3d_file(mesh_path):
                return None, None, None, "❌ 仅支持 .glb, .ply, .obj 格式"

            progress(0.2, desc="📂 加载3D模型...")
            with timer("3D模型加载"):
                mesh = trimesh.load(mesh_path, force='mesh', process=False)

            progress(0.4, desc="🔍 执行GPU加速分割算法...")

            # 使用GPU优化推理
            with gpu_inference_mode():
                with timer("P3-SAM推理"):
                    aabb, face_ids, mesh = automask.predict_aabb(
                        mesh, seed=seed, is_parallel=False,
                        post_process=postprocess, threshold=postprocess_threshold
                    )

            progress(0.7, desc="🎨 生成分割结果...")
            with timer("结果后处理"):
                # 生成颜色映射 - 优化版本
                color_map = {}
                unique_ids = np.unique(face_ids)

                # 使用GPU加速的随机数生成
                if torch.cuda.is_available():
                    colors = torch.rand((len(unique_ids), 3), device='cuda') * 255
                    colors = colors.cpu().numpy()
                else:
                    colors = np.random.rand(len(unique_ids), 3) * 255

                for i, unique_id in enumerate(unique_ids):
                    if unique_id == -1:
                        continue
                    color_map[unique_id] = colors[i]

                # 优化的面片颜色分配
                face_colors = np.array([
                    [50, 50, 50] if fid == -1 else color_map[fid]
                    for fid in face_ids
                ], dtype=np.uint8)

                mesh_save = mesh.copy()
                mesh_save.visual.face_colors = face_colors

            progress(0.9, desc="💾 保存结果...")
            with timer("文件保存"):
                timestamp = int(time.time())
                file_path = os.path.join(output_path, f'segment_gpu_{timestamp}.glb')
                mesh_save.export(file_path)
                face_id_save_path = os.path.join(output_path, f'face_id_gpu_{timestamp}.npy')
                np.save(face_id_save_path, face_ids)

            # 统计分割信息
            num_parts = len(unique_ids) - (1 if -1 in unique_ids else 0)
            gpu_status = gpu_optimizer.get_gpu_status()

            status_msg = (
                f"✅ GPU加速分割完成！共分割出 {num_parts} 个部分\n"
                f"🚀 GPU利用率: {gpu_status.get('gpu_utilization_percent', 0):.1f}% | "
                f"🔥 温度: {gpu_status.get('temperature_celsius', 0):.1f}°C"
            )

            gr_state = [(aabb, mesh_path)]

            progress(1.0, desc="🎉 完成!")
            return file_path, face_id_save_path, gr_state, status_msg

        except Exception as e:
            gpu_optimizer.clear_memory_if_needed()  # 错误时清理内存
            return None, None, None, f"❌ 分割失败: {str(e)}"

@spaces.GPU(duration=150)
def generate_parts_optimized(mesh_path, seed=42, gr_state=None, progress=gr.Progress()):
    """XPart 零件生成 - GPU优化版本"""

    with timer("完整生成流程"):
        try:
            progress(0.1, desc="🔍 验证输入...")
            if mesh_path is None:
                return None, None, None, "❌ 请先上传3D模型文件"

            if gr_state is None or len(gr_state) == 0 or gr_state[0][0] is None:
                return None, None, None, "❌ 请先执行模型分割"

            if mesh_path != gr_state[0][1]:
                return None, None, None, "❌ 模型已更改，请重新执行分割"

            progress(0.2, desc="🚀 初始化GPU优化生成模型...")
            with timer("生成模型初始化"):
                initialize_models()

            aabb = gr_state[0][0]

            progress(0.3, desc="🎲 设置随机种子...")
            try:
                pl.seed_everything(int(seed), workers=True)
            except Exception:
                pl.seed_everything(2026, workers=True)

            progress(0.5, desc="⚡ GPU加速零件生成（预计1-2分钟）...")

            # 使用GPU优化推理
            with gpu_inference_mode():
                with timer("XPart推理"):
                    additional_params = {"output_type": "trimesh"}

                    # 启用混合精度推理
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = _PIPELINE(
                            mesh_path=mesh_path,
                            aabb=aabb,
                            octree_resolution=512,
                            **additional_params,
                        )

            progress(0.8, desc="💾 保存GPU加速生成结果...")
            with timer("结果保存"):
                timestamp = int(time.time())
                obj_path = os.path.join(output_path, f'generated_gpu_{timestamp}.glb')
                out_bbox_path = os.path.join(output_path, f'bbox_gpu_{timestamp}.glb')
                explode_path = os.path.join(output_path, f'exploded_gpu_{timestamp}.glb')

                obj_mesh.export(obj_path)
                out_bbox.export(out_bbox_path)
                explode_object.export(explode_path)

            # 获取性能统计
            gpu_status = gpu_optimizer.get_gpu_status()

            status_msg = (
                f"✅ GPU加速零件生成完成！\n"
                f"🚀 GPU利用率: {gpu_status.get('gpu_utilization_percent', 0):.1f}% | "
                f"💾 显存使用: {gpu_status.get('memory_usage_percent', 0):.1f}% | "
                f"🔥 温度: {gpu_status.get('temperature_celsius', 0):.1f}°C"
            )

            progress(1.0, desc="🎉 完成!")
            return obj_path, out_bbox_path, explode_path, status_msg

        except Exception as e:
            gpu_optimizer.clear_memory_if_needed()  # 错误时清理内存
            return None, None, None, f"❌ 生成失败: {str(e)}"

def reset_all_optimized():
    """重置所有状态 - 包含GPU内存清理"""
    gpu_optimizer.clear_memory_if_needed()
    return None, None, None, None, None, None, None, [(None, None)], "🔄 已重置所有内容（含GPU内存清理）"

def get_gpu_info():
    """获取GPU信息显示"""
    gpu_status = gpu_optimizer.get_gpu_status()
    if "error" in gpu_status:
        return "🖥️ CPU模式运行"

    return (
        f"🚀 {gpu_status['gpu_name']} | "
        f"💾 显存: {gpu_status['memory_usage_percent']:.1f}% | "
        f"🔥 温度: {gpu_status['temperature_celsius']:.1f}°C | "
        f"⚡ 利用率: {gpu_status['gpu_utilization_percent']:.1f}%"
    )

# 优化版CSS样式（包含GPU状态显示）
optimized_css = """
/* 原有CSS样式基础上添加GPU状态样式 */
:root {
    --primary-color: #1e3a8a;
    --secondary-color: #3b82f6;
    --accent-color: #8b5cf6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --gpu-color: #06b6d4;
    --bg-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
}

.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gpu-status {
    background: linear-gradient(135deg, var(--gpu-color), var(--accent-color));
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3);
}

.performance-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
    background: var(--success-color);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* 其他原有样式保持不变 */
.main-header {
    background: var(--bg-gradient);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.status-indicator {
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    font-weight: 500;
    border-left: 4px solid;
}

.status-success {
    background: #ecfdf5;
    color: #065f46;
    border-color: var(--success-color);
}

.status-error {
    background: #fef2f2;
    color: #991b1b;
    border-color: var(--error-color);
}

.status-info {
    background: #eff6ff;
    color: #1e40af;
    border-color: var(--primary-color);
}

.primary-btn {
    background: var(--bg-gradient) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(30, 58, 138, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(30, 58, 138, 0.4) !important;
}

.model-viewer {
    border-radius: 0.75rem;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

.section-title {
    color: var(--primary-color);
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

.param-group {
    background: #f8fafc;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}

.examples-gallery {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    margin-top: 2rem;
}

/* 示例图片样式 */
.examples-gallery img, .example-image img {
    border-radius: 0.75rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    border: 2px solid transparent;
}

.examples-gallery img:hover, .example-image img:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(30, 58, 138, 0.3);
    border-color: var(--primary-color);
}

.examples-gallery .gr-column {
    margin: 0.5rem;
}

.examples-gallery strong {
    color: var(--primary-color);
    font-size: 0.9rem;
}

.example-image {
    position: relative;
    overflow: hidden;
    border-radius: 0.75rem;
}

.example-image::after {
    content: "点击加载";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, rgba(30, 58, 138, 0.9), transparent);
    color: white;
    text-align: center;
    padding: 0.5rem;
    opacity: 0;
    transition: opacity 0.3s ease;
    font-size: 0.8rem;
}

.example-image:hover::after {
    opacity: 1;
}
"""

# 初始化性能日志
setup_performance_logging()

# 创建优化版Gradio界面
with gr.Blocks(css=optimized_css, title="Hunyuan3D-Part RTX GPU加速版") as demo:
    # 顶部标题区域
    with gr.Row():
        gr.HTML("""
        <div class="main-header">
            <h1>⚡ Hunyuan3D-Part RTX GPU加速版</h1>
            <p>🚀 AI驱动的3D模型分析工具 | Flash Attention + 混合精度 + torch.compile优化</p>
            <p>🚀 二次开发构建by科哥 bug反馈微信：312088415</p>
        </div>
        """)

    # GPU状态显示区域
    with gr.Row():
        gpu_status_display = gr.HTML(
            f'<div class="gpu-status"><span class="performance-indicator"></span>{get_gpu_info()}</div>',
            elem_classes=["gpu-status"]
        )

    # 状态显示区域
    with gr.Row():
        status_display = gr.HTML(
            '<div class="status-indicator status-info">🔵 RTX 4090 GPU就绪，请上传3D模型开始加速处理</div>',
            elem_classes=["status-display"]
        )

    # 主要内容区域
    with gr.Row():
        # 左侧控制面板
        with gr.Column(scale=2):
            with gr.Group():
                gr.HTML('<div class="section-title">📁 3D模型上传</div>')
                input_mesh = gr.Model3D(
                    label="支持格式：GLB, PLY, OBJ",
                    clear_color=[0.95, 0.95, 0.95, 1.0],
                    elem_classes=["model-viewer"]
                )

            # P3-SAM分割参数
            with gr.Group():
                gr.HTML('<div class="section-title">🔍 P3-SAM GPU加速分割</div>')
                with gr.Group(elem_classes=["param-group"]):
                    with gr.Row():
                        segment_btn = gr.Button(
                            "⚡ GPU加速分割",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                        reset_btn = gr.Button(
                            "🔄 重置+清理GPU",
                            variant="secondary"
                        )

                    with gr.Row():
                        p3sam_seed = gr.Number(
                            value=42,
                            label="🎲 随机种子",
                            minimum=0,
                            maximum=9999,
                            step=1
                        )
                        postprocess = gr.Checkbox(
                            value=True,
                            label="✨ 后处理优化"
                        )

                    postprocess_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="🎯 后处理阈值"
                    )

            # XPart生成参数
            with gr.Group():
                gr.HTML('<div class="section-title">🛠️ XPart GPU加速生成</div>')
                with gr.Group(elem_classes=["param-group"]):
                    generate_btn = gr.Button(
                        "🚀 GPU加速生成",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    xpart_seed = gr.Number(
                        value=42,
                        label="🎲 生成种子",
                        minimum=0,
                        maximum=9999,
                        step=1
                    )

        # 右侧结果展示区域
        with gr.Column(scale=3):
            with gr.Tabs():
                # 分割结果选项卡
                with gr.TabItem("🔍 GPU加速分割结果"):
                    with gr.Group():
                        gr.HTML('<div class="section-title">P3-SAM GPU分割结果</div>')
                        segmented_mesh = gr.Model3D(
                            label="GPU加速分割后的3D模型",
                            clear_color=[0.05, 0.05, 0.1, 1.0],
                            elem_classes=["model-viewer"]
                        )
                        face_id_file = gr.File(
                            label="📄 面片ID文件下载"
                        )

                # 生成结果选项卡
                with gr.TabItem("🛠️ GPU加速生成结果"):
                    with gr.Group():
                        gr.HTML('<div class="section-title">XPart GPU零件生成结果</div>')

                        with gr.Row():
                            generated_parts = gr.Model3D(
                                label="🎯 GPU生成的零件",
                                clear_color=[0.05, 0.05, 0.1, 1.0],
                                elem_classes=["model-viewer"]
                            )
                            parts_with_bbox = gr.Model3D(
                                label="📦 带边界框的零件",
                                clear_color=[0.05, 0.05, 0.1, 1.0],
                                elem_classes=["model-viewer"]
                            )

                        exploded_view = gr.Model3D(
                            label="💥 分解视图",
                            clear_color=[0.05, 0.05, 0.1, 1.0],
                            elem_classes=["model-viewer"]
                        )

    # 示例图库
    with gr.Group(elem_classes=["examples-gallery"]):
        gr.HTML('<div class="section-title">🎨 示例模型库</div>')
        gr.HTML('<p style="color: #64748b; margin-bottom: 1rem;">点击示例快速体验RTX 4090 GPU加速处理</p>')

        # 创建带名称和图片的示例展示
        examples_info = [
            {"name": "女战士", "model": "P3-SAM/demo/assets/Female_Warrior.glb", "image": "P3-SAM/demo/assets/Female_Warrior.png"},
            {"name": "悬浮岛", "model": "P3-SAM/demo/assets/Suspended_Island.glb", "image": "P3-SAM/demo/assets/Suspended_Island.png"},
            {"name": "甲虫车", "model": "P3-SAM/demo/assets/Beetle_Car.glb", "image": "P3-SAM/demo/assets/Beetle_Car.png"},
            {"name": "锦鲤", "model": "XPart/data/Koi_Fish.glb", "image": "XPart/data/Koi_Fish.png"},
            {"name": "摩托车", "model": "XPart/data/Motorcycle.glb", "image": "XPart/data/Motorcycle.png"},
            {"name": "高达", "model": "XPart/data/Gundam.glb", "image": "XPart/data/Gundam.png"},
            {"name": "电脑桌", "model": "XPart/data/Computer_Desk.glb", "image": "XPart/data/Computer_Desk.png"},
            {"name": "咖啡机", "model": "XPart/data/Coffee_Machine.glb", "image": "XPart/data/Coffee_Machine.png"},
        ]

        # 创建图片点击区域
        with gr.Row():
            example_images = []
            for i, info in enumerate(examples_info[:4]):  # 第一行4个
                with gr.Column(scale=1):
                    gr.HTML(f'<div style="text-align: center; margin-bottom: 0.5rem;"><strong>{info["name"]}</strong></div>')
                    img = gr.Image(
                        info["image"],
                        show_label=False,
                        height=120,
                        width=120,
                        elem_classes=["example-image"],
                        elem_id=f"example-img-{i}"
                    )
                    example_images.append(img)

        with gr.Row():
            for i, info in enumerate(examples_info[4:], 4):  # 第二行4个
                with gr.Column(scale=1):
                    gr.HTML(f'<div style="text-align: center; margin-bottom: 0.5rem;"><strong>{info["name"]}</strong></div>')
                    img = gr.Image(
                        info["image"],
                        show_label=False,
                        height=120,
                        width=120,
                        elem_classes=["example-image"],
                        elem_id=f"example-img-{i}"
                    )
                    example_images.append(img)

        # 传统的Examples组件作为备用
        example_inputs = gr.Examples(
            examples=[[info["model"]] for info in examples_info],
            inputs=[input_mesh],
            label="或点击下方按钮选择模型",
            examples_per_page=8,
            cache_examples=False
        )

    # 隐藏状态
    gr_state = gr.State(value=[(None, None)])

    # 事件绑定函数
    def update_file_info(mesh_path):
        info = get_file_info(mesh_path)
        gpu_info = get_gpu_info()
        return (
            f'<div class="status-indicator status-info">📁 {info}</div>',
            f'<div class="gpu-status"><span class="performance-indicator"></span>{gpu_info}</div>'
        )

    def update_status_after_segment(status_msg):
        gpu_info = get_gpu_info()
        if status_msg.startswith("✅"):
            status_class = "status-success"
        else:
            status_class = "status-error"
        return (
            f'<div class="status-indicator {status_class}">{status_msg}</div>',
            f'<div class="gpu-status"><span class="performance-indicator"></span>{gpu_info}</div>'
        )

    def update_status_after_generate(status_msg):
        gpu_info = get_gpu_info()
        if status_msg.startswith("✅"):
            status_class = "status-success"
        else:
            status_class = "status-error"
        return (
            f'<div class="status-indicator {status_class}">{status_msg}</div>',
            f'<div class="gpu-status"><span class="performance-indicator"></span>{gpu_info}</div>'
        )

    # 文件上传事件
    input_mesh.change(
        fn=update_file_info,
        inputs=[input_mesh],
        outputs=[status_display, gpu_status_display]
    )

    # 分割事件
    segment_btn.click(
        fn=segment_mesh_optimized,
        inputs=[input_mesh, postprocess, postprocess_threshold, p3sam_seed],
        outputs=[segmented_mesh, face_id_file, gr_state, status_display]
    ).then(
        fn=update_status_after_segment,
        inputs=[status_display],
        outputs=[status_display, gpu_status_display]
    )

    # 生成事件
    generate_btn.click(
        fn=generate_parts_optimized,
        inputs=[input_mesh, xpart_seed, gr_state],
        outputs=[generated_parts, parts_with_bbox, exploded_view, status_display]
    ).then(
        fn=update_status_after_generate,
        inputs=[status_display],
        outputs=[status_display, gpu_status_display]
    )

    # 示例图片点击事件
    def load_example_model(model_path):
        """加载示例模型"""
        return model_path, f'<div class="status-indicator status-info">📁 已选择示例模型: {os.path.basename(model_path)}</div>'

    # 为每个示例图片添加点击事件
    examples_models = [info["model"] for info in examples_info]
    for i, (img, model_path) in enumerate(zip(example_images, examples_models)):
        img.select(
            fn=lambda model=model_path: load_example_model(model),
            outputs=[input_mesh, status_display]
        )

    # 重置事件
    reset_btn.click(
        fn=reset_all_optimized,
        outputs=[
            input_mesh, segmented_mesh, face_id_file,
            generated_parts, parts_with_bbox, exploded_view,
            status_display, gr_state, gpu_status_display
        ]
    )

if __name__ == '__main__':
    # 启动时显示GPU状态
    gpu_optimizer.logger.info("=== RTX 4090 GPU加速版启动 ===")
    gpu_status = gpu_optimizer.get_gpu_status()

    if "error" not in gpu_status:
        gpu_optimizer.logger.info(f"GPU: {gpu_status['gpu_name']}")
        gpu_optimizer.logger.info(f"显存: {gpu_status['memory_total_mb']:.0f}MB")
        gpu_optimizer.logger.info(f"计算能力: {gpu_status['compute_capability']}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )