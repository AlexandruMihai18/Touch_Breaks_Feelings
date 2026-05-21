import gradio as gr

from inference_script.shared import OUT_H
from inference_script.src.pipelines import run_touch_demo

with gr.Blocks() as tab:
    gr.Markdown(
        "## Touch Detection Demo\n"
        "Scribble the two objects in the **context** image (layer 1 = obj1, layer 2 = obj2). "
        "Optionally supply a **query** image — masks are predicted via SegGPT. "
        "Each column shows: SAM/SegGPT masks → initial dilation zone → depth map → depth-filtered touch (v2)."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Context image")
            ctx_editor = gr.ImageEditor(
                label="Scribble: Object 1 → layer 1  |  Object 2 → layer 2",
                type="numpy",
                height=OUT_H,
            )
        with gr.Column():
            gr.Markdown("### Query image")
            qry_image = gr.Image(
                label="Query image — masks predicted by SegGPT from context",
                type="numpy",
                height=OUT_H,
            )

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        with gr.Column():
            ctx_sam_out = gr.Image(
                label="Context — SAM masks (blue=obj1 · orange=obj2)",
                height=OUT_H,
                interactive=False,
            )
            ctx_init_out = gr.Image(
                label="Context — Initial dilation touch zone",
                height=OUT_H,
                interactive=False,
            )
            ctx_depth_out = gr.Image(
                label="Context — Depth map",
                height=OUT_H,
                interactive=False,
            )
            ctx_final_out = gr.Image(
                label="Context — Final touch region (depth-filtered v2)",
                height=OUT_H,
                interactive=False,
            )
        with gr.Column():
            qry_seggpt_out = gr.Image(
                label="Query — SegGPT masks (blue=obj1 · orange=obj2)",
                height=OUT_H,
                interactive=False,
            )
            qry_init_out = gr.Image(
                label="Query — Initial dilation touch zone",
                height=OUT_H,
                interactive=False,
            )
            qry_depth_out = gr.Image(
                label="Query — Depth map",
                height=OUT_H,
                interactive=False,
            )
            qry_final_out = gr.Image(
                label="Query — Final touch region (depth-filtered v2)",
                height=OUT_H,
                interactive=False,
            )

    run_btn.click(
        fn=run_touch_demo,
        inputs=[ctx_editor, qry_image],
        outputs=[
            ctx_sam_out, ctx_init_out, ctx_depth_out, ctx_final_out,
            qry_seggpt_out, qry_init_out, qry_depth_out, qry_final_out,
        ],
    )
