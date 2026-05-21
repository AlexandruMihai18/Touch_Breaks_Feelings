import gradio as gr

from inference_script.shared import OUT_H
from inference_script.src.deprecated.pipelines import run_dual_sam_seggpt

tab = gr.Interface(
    fn=run_dual_sam_seggpt,
    inputs=[
        gr.ImageEditor(
            label="Object 1 → layer 1  |  Object 2 → layer 2",
            type="numpy",
            height=OUT_H,
        ),
        gr.Image(label="Image 1", type="numpy", height=OUT_H),
        gr.Image(label="Image 2", type="numpy", height=OUT_H),
    ],
    outputs=[
        gr.Image(label="Prompt — blue=obj1 · orange=obj2", height=OUT_H),
        gr.Image(label="Image 1 — SegGPT predicted classes", height=OUT_H),
        gr.Image(label="Image 2 — SegGPT predicted classes", height=OUT_H),
    ],
    flagging_mode="never",
)
