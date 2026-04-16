import gradio as gr
import numpy as np
from PIL import Image
import torch
import sys
import os

sys.path.append('./src')

# 导入您提供的函数
from src.model.MultiU_NetModel import MultiU_Net
from src.model.MaskTransform import multi_class_post_process
from src.model.MaskVisualization import segmentation_visualizer
from src.dataset.FeatureExtraction import extract_features
from src.space.ModelLoadAndWork import load_model, run_inference

BEST_PERM = [1, 2, 0, 3, 4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
try:
    model = load_model()
    model_loaded = True
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model_loaded = False


def process_image(input_image):
    """处理图像分割的主函数"""
    if input_image is None:
        return None, "请上传一张图片"

    if not model_loaded:
        return None, "模型未加载成功，请检查模型文件"

    try:
        # 确保输入图像是numpy数组格式 (H, W, 3)
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)

        # 确保是RGB三通道图像
        if len(input_image.shape) == 2:  # 灰度图
            input_image = np.stack([input_image] * 3, axis=-1)
        elif input_image.shape[-1] != 3:  # 不是RGB格式
            raise ValueError(f"输入图像必须是RGB格式，当前形状: {input_image.shape}")

        # 如果值范围在 [0, 255]，转换到 [0, 1]
        if input_image.max() > 1.0:
            input_image = input_image / 255.0

        print(f"处理前图像形状: {input_image.shape}, 值范围: [{input_image.min():.3f}, {input_image.max():.3f}]")

        # 运行推理
        result = run_inference(model, input_image)

        # 确保结果是正确的格式用于显示
        if result.max() <= 1.0:
            result_display = (result * 255).astype(np.uint8)
        else:
            result_display = result.astype(np.uint8)

        log_message = f"图像处理成功！原始尺寸: {input_image.shape}, 分割结果尺寸: {result_display.shape}"
        return result_display, log_message
    except Exception as e:
        error_msg = f"处理图像时发生错误: {str(e)}"
        import traceback
        error_msg += f"\n详细错误信息:\n{traceback.format_exc()}"
        return None, error_msg


def create_interface():
    custom_css = """
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .gradio-container {
            max-width: 1400px !important;
            width: 95% !important;
            margin: 20px auto !important;
            padding: 30px !important;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            border-radius: 20px !important;
            border: 4px solid transparent !important;
            border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1 !important;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3) !important;
        }
        .main-title {
            text-align: center !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-size: 2.8em !important;
            font-weight: bold !important;
            margin-bottom: 15px !important;
            padding: 10px 0 !important;
        }
        .subtitle {
            text-align: center !important;
            color: #667eea !important;
            margin-bottom: 30px !important;
            font-size: 1.3em !important;
        }
        .image-container {
            display: flex !important;
            justify-content: space-between !important;
            gap: 30px !important;
            margin-bottom: 25px !important;
            flex-wrap: nowrap !important;
        }
        .image-column {
            flex: 1 !important;
            min-width: 48% !important;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            padding: 20px !important;
            border-radius: 15px !important;
            border: 3px solid transparent !important;
            border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1 !important;
        }
        .image-component {
            min-height: 450px !important;
            border-radius: 10px !important;
            width: 100% !important;
            object-fit: contain !important;
        }
        .button-row {
            display: flex !important;
            justify-content: center !important;
            margin: 25px 0 !important;
        }
        .primary-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 30px !important;
            height: 60px !important;
            font-size: 1.3em !important;
            font-weight: bold !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5) !important;
            transition: all 0.3s ease !important;
            min-width: 250px !important;
        }
        .primary-button:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 25px rgba(102, 126, 234, 0.7) !important;
        }
        .log-container {
            width: 100% !important;
            margin-top: 25px !important;
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7ff 100%) !important;
            border-radius: 15px !important;
            border: 3px solid transparent !important;
            border-image: linear-gradient(135deg, #3498db 0%, #2980b9 100%) 1 !important;
            padding: 20px !important;
        }
        .log-box {
            background: transparent !important;
            border: none !important;
            font-family: monospace !important;
            font-size: 1.1em !important;
        }
        .footer {
            text-align: center !important;
            color: #7f8c8d !important;
            margin-top: 35px !important;
            font-size: 1.1em !important;
        }
        .label-wrap {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-weight: bold !important;
            font-size: 1.1em !important;
        }

        /* 强制桌面布局 */
        @media (max-width: 1200px) {
            .image-container {
                flex-direction: row !important;
            }
            .image-column {
                min-width: 48% !important;
            }
        }

        @media (max-width: 992px) {
            .image-container {
                flex-direction: row !important;
                gap: 20px !important;
            }
            .image-column {
                min-width: 48% !important;
            }
        }

        @media (max-width: 768px) {
            .image-container {
                flex-direction: row !important;
                gap: 15px !important;
            }
            .image-column {
                min-width: 48% !important;
            }
        }

        /* 确保在各种屏幕下都保持横向布局 */
        .wrap {
            flex-wrap: nowrap !important;
        }
    """

    with gr.Blocks(
            title="图像分割模型演示",
            css=custom_css
    ) as demo:
        gr.Markdown("<h1 class='main-title'>MultiU-Net 图像分割演示</h1>")
        gr.Markdown("<p class='subtitle'>上传一张图片，查看模型的分割结果</p>")

        with gr.Row(elem_classes="image-container wrap"):
            with gr.Column(scale=1, min_width=500, elem_classes="image-column"):
                input_image = gr.Image(label="上传原始图像", type="numpy", height=450, interactive=True)
            with gr.Column(scale=1, min_width=500, elem_classes="image-column"):
                segmented_image = gr.Image(label="分割结果", interactive=False, height=450)

        with gr.Row(elem_classes="button-row"):
            confirm_btn = gr.Button("开始分割", elem_classes="primary-button")

        with gr.Row(elem_classes="log-container"):
            log_output = gr.Textbox(label="处理日志", interactive=False, lines=5, max_lines=12, elem_classes="log-box")

        # 设置事件处理
        confirm_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[segmented_image, log_output]
        )

        gr.Markdown("<p class='footer'>Powered by MultiU-Net & Gradio | 2026</p>")

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )