import gradio as gr
import numpy as np

from inference_script.shared import OUT_H
from touch_detection_alg.depth import colorize_depth, predict_depth


def run_depth_estimation(image: np.ndarray) -> np.ndarray:
    return colorize_depth(predict_depth(image))


tab = gr.Interface(
    fn=run_depth_estimation,
    inputs=gr.Image(label="Input Image", type="numpy", height=OUT_H),
    outputs=gr.Image(label="Depth Map", height=OUT_H),
    flagging_mode="never",
)
