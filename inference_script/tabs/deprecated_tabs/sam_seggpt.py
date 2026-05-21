import gradio as gr

from inference_script.shared import OUT_H
from inference_script.src.pipelines import run_sam_and_seggpt

tab = gr.Interface(
    fn=run_sam_and_seggpt,
    inputs=[
        gr.ImageEditor(label="Draw Scribble/Mask", type="numpy", height=OUT_H),
        gr.Image(label="Image 1", type="numpy", height=OUT_H),
        gr.Image(label="Image 2", type="numpy", height=OUT_H),
    ],
    outputs=[
        gr.Image(label="Prompt — SAM mask + points", height=OUT_H),
        gr.Image(label="Image 1 — SegGPT predicted mask", height=OUT_H),
        gr.Image(label="Image 2 — SegGPT predicted mask", height=OUT_H),
    ],
    flagging_mode="never",
)
