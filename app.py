import gradio as gr
import os
import sys
import argparse
import numpy as np
import trimesh
from pathlib import Path
import torch
import pytorch_lightning as pl
import spaces
import time

sys.path.append('P3-SAM')
from demo.auto_mask import AutoMask
sys.path.append('XPart')
from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file

# 全局变量
automask = None
_PIPELINE = None
output_path = 'P3-SAM/results/gradio'
os.makedirs(output_path, exist_ok=True)

def initialize_models():
    """延迟加载模型，提升启动速度"""
    global automask, _PIPELINE
    if automask is None:
        automask = AutoMask()

    if _PIPELINE is None:
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
        device = "cuda"
        _PIPELINE.to(device=device, dtype=torch.float32)

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
        return f"📁 {os.path.basename(filename)} ({file_size:.1f} MB)"
    except:
        return f"📁 {os.path.basename(filename)}"

@spaces.GPU
def segment_mesh(mesh_path, postprocess=True, postprocess_threshold=0.95, seed=42, progress=gr.Progress()):
    """P3-SAM 3D模型分割"""
    try:
        # 初始化模型
        progress(0.1, desc="初始化模型...")
        initialize_models()

        # 验证输入
        if mesh_path is None:
            return None, None, None, "❌ 请先上传3D模型文件"

        if not is_supported_3d_file(mesh_path):
            return None, None, None, "❌ 仅支持 .glb, .ply, .obj 格式"

        progress(0.2, desc="加载3D模型...")
        mesh = trimesh.load(mesh_path, force='mesh', process=False)

        progress(0.4, desc="执行分割算法...")
        aabb, face_ids, mesh = automask.predict_aabb(
            mesh, seed=seed, is_parallel=False,
            post_process=postprocess, threshold=postprocess_threshold
        )

        progress(0.7, desc="生成分割结果...")
        # 生成颜色映射
        color_map = {}
        unique_ids = np.unique(face_ids)
        for i in unique_ids:
            if i == -1:
                continue
            part_color = np.random.rand(3) * 255
            color_map[i] = part_color

        face_colors = []
        for i in face_ids:
            if i == -1:
                face_colors.append([50, 50, 50])  # 深灰色表示未分割区域
            else:
                face_colors.append(color_map[i])

        face_colors = np.array(face_colors).astype(np.uint8)
        mesh_save = mesh.copy()
        mesh_save.visual.face_colors = face_colors

        progress(0.9, desc="保存结果...")
        file_path = os.path.join(output_path, f'segment_{int(time.time())}.glb')
        mesh_save.export(file_path)
        face_id_save_path = os.path.join(output_path, f'face_id_{int(time.time())}.npy')
        np.save(face_id_save_path, face_ids)

        # 统计分割信息
        num_parts = len(unique_ids) - (1 if -1 in unique_ids else 0)
        status_msg = f"✅ 分割完成！共分割出 {num_parts} 个部分"

        gr_state = [(aabb, mesh_path)]

        progress(1.0, desc="完成!")
        return file_path, face_id_save_path, gr_state, status_msg

    except Exception as e:
        return None, None, None, f"❌ 分割失败: {str(e)}"

@spaces.GPU(duration=150)
def generate_parts(mesh_path, seed=42, gr_state=None, progress=gr.Progress()):
    """XPart 零件生成"""
    try:
        progress(0.1, desc="验证输入...")
        if mesh_path is None:
            return None, None, None, "❌ 请先上传3D模型文件"

        if gr_state is None or len(gr_state) == 0 or gr_state[0][0] is None:
            return None, None, None, "❌ 请先执行模型分割"

        if mesh_path != gr_state[0][1]:
            return None, None, None, "❌ 模型已更改，请重新执行分割"

        progress(0.2, desc="初始化生成模型...")
        initialize_models()

        aabb = gr_state[0][0]

        progress(0.3, desc="设置随机种子...")
        try:
            pl.seed_everything(int(seed), workers=True)
        except Exception:
            pl.seed_everything(2026, workers=True)

        progress(0.5, desc="生成零件（预计2-3分钟）...")
        additional_params = {"output_type": "trimesh"}
        obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = _PIPELINE(
            mesh_path=mesh_path,
            aabb=aabb,
            octree_resolution=512,
            **additional_params,
        )

        progress(0.8, desc="保存生成结果...")
        timestamp = int(time.time())
        obj_path = os.path.join(output_path, f'generated_{timestamp}.glb')
        out_bbox_path = os.path.join(output_path, f'bbox_{timestamp}.glb')
        explode_path = os.path.join(output_path, f'exploded_{timestamp}.glb')

        obj_mesh.export(obj_path)
        out_bbox.export(out_bbox_path)
        explode_object.export(explode_path)

        progress(1.0, desc="完成!")
        return obj_path, out_bbox_path, explode_path, "✅ 零件生成完成！"

    except Exception as e:
        return None, None, None, f"❌ 生成失败: {str(e)}"

def reset_all():
    """重置所有状态"""
    return None, None, None, None, None, None, None, [(None, None)], "🔄 已重置所有内容"

# 自定义CSS样式
custom_css = """
/* 主题色彩 */
:root {
    --primary-color: #1e3a8a;
    --secondary-color: #3b82f6;
    --accent-color: #8b5cf6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --bg-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
}

/* 全局样式 */
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 标题样式 */
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

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0.5rem 0 0 0;
}

/* 状态指示器 */
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

/* 按钮样式 */
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

.secondary-btn {
    background: white !important;
    color: var(--primary-color) !important;
    border: 2px solid var(--primary-color) !important;
    padding: 0.75rem 2rem !important;
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: var(--primary-color) !important;
    color: white !important;
}

/* 卡片样式 */
.section-card {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.section-title {
    color: var(--primary-color);
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

/* 3D模型查看器 */
.model-viewer {
    border-radius: 0.75rem;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

/* 进度条 */
.progress-container {
    background: white;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 参数控制 */
.param-group {
    background: #f8fafc;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}

/* 示例图库 */
.examples-gallery {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    margin-top: 2rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }

    .main-header p {
        font-size: 1rem;
    }

    .section-card {
        padding: 1rem;
    }
}

/* 自定义滚动条 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-color);
}
"""

# 创建Gradio界面
with gr.Blocks(css=custom_css, title="Hunyuan3D-Part 智能3D模型分析工具") as demo:
    # 顶部标题区域
    with gr.Row():
        gr.HTML("""
        <div class="main-header">
            <h1>☯️ Hunyuan3D-Part</h1>
            <p>🚀 智能3D模型分割与零件生成系统 | AI驱动的3D内容创作工具</p>
        </div>
        """)

    # 状态显示区域
    with gr.Row():
        status_display = gr.HTML(
            '<div class="status-indicator status-info">🔵 系统就绪，请上传3D模型开始处理</div>',
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
                file_info_display = gr.HTML("请上传3D模型文件")

            # P3-SAM分割参数
            with gr.Group():
                gr.HTML('<div class="section-title">🔍 P3-SAM 分割参数</div>')
                with gr.Group(elem_classes=["param-group"]):
                    with gr.Row():
                        segment_btn = gr.Button(
                            "🚀 开始分割",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                        reset_btn = gr.Button(
                            "🔄 重置",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
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
                        label="🎯 后处理阈值",
                        info="更高的值合并更少的零件"
                    )

            # XPart生成参数
            with gr.Group():
                gr.HTML('<div class="section-title">🛠️ XPart 生成参数</div>')
                with gr.Group(elem_classes=["param-group"]):
                    generate_btn = gr.Button(
                        "⚡ 生成零件",
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
                with gr.TabItem("🔍 分割结果", elem_id="segment-tab"):
                    with gr.Group():
                        gr.HTML('<div class="section-title">P3-SAM 分割结果</div>')
                        segmented_mesh = gr.Model3D(
                            label="分割后的3D模型",
                            clear_color=[0.05, 0.05, 0.1, 1.0],
                            elem_classes=["model-viewer"]
                        )
                        face_id_file = gr.File(
                            label="📄 面片ID文件下载",
                            visible=True
                        )

                # 生成结果选项卡
                with gr.TabItem("🛠️ 生成结果", elem_id="generate-tab"):
                    with gr.Group():
                        gr.HTML('<div class="section-title">XPart 零件生成结果</div>')

                        with gr.Row():
                            generated_parts = gr.Model3D(
                                label="🎯 生成的零件",
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
        gr.HTML('<p style="color: #64748b; margin-bottom: 1rem;">点击下方示例快速体验系统功能</p>')

        example_inputs = gr.Examples(
            examples=[
                ['P3-SAM/demo/assets/Female_Warrior.glb'],
                ['P3-SAM/demo/assets/Suspended_Island.glb'],
                ['P3-SAM/demo/assets/Beetle_Car.glb'],
                ['XPart/data/Koi_Fish.glb'],
                ['XPart/data/Motorcycle.glb'],
                ['XPart/data/Gundam.glb'],
                ['XPart/data/Computer_Desk.glb'],
                ['XPart/data/Coffee_Machine.glb'],
            ],
            inputs=[input_mesh],
            label="点击选择示例模型"
        )

    # 隐藏状态
    gr_state = gr.State(value=[(None, None)])

    # 事件绑定
    def update_file_info(mesh_path):
        info = get_file_info(mesh_path)
        return f'<div class="status-indicator status-info">📁 {info}</div>'

    def update_status_after_segment(status_msg):
        if status_msg.startswith("✅"):
            return f'<div class="status-indicator status-success">{status_msg}</div>'
        else:
            return f'<div class="status-indicator status-error">{status_msg}</div>'

    def update_status_after_generate(status_msg):
        if status_msg.startswith("✅"):
            return f'<div class="status-indicator status-success">{status_msg}</div>'
        else:
            return f'<div class="status-indicator status-error">{status_msg}</div>'

    # 文件上传事件
    input_mesh.change(
        fn=update_file_info,
        inputs=[input_mesh],
        outputs=[status_display]
    )

    # 分割事件
    segment_btn.click(
        fn=segment_mesh,
        inputs=[input_mesh, postprocess, postprocess_threshold, p3sam_seed],
        outputs=[segmented_mesh, face_id_file, gr_state, status_display]
    ).then(
        fn=update_status_after_segment,
        inputs=[status_display],
        outputs=[status_display]
    )

    # 生成事件
    generate_btn.click(
        fn=generate_parts,
        inputs=[input_mesh, xpart_seed, gr_state],
        outputs=[generated_parts, parts_with_bbox, exploded_view, status_display]
    ).then(
        fn=update_status_after_generate,
        inputs=[status_display],
        outputs=[status_display]
    )

    # 重置事件
    reset_btn.click(
        fn=reset_all,
        outputs=[
            input_mesh, segmented_mesh, face_id_file,
            generated_parts, parts_with_bbox, exploded_view,
            status_display, gr_state
        ]
    )

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )