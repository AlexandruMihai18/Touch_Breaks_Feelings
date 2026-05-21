import gradio as gr

from inference_script.shared import FRAME_H, OUT_H
from inference_script.src.pipelines import run_video_touch_inference
from inference_script.src.deprecated.sampling import sample_video_frames

N_FRAMES = 5  # number of frames sampled from video


with gr.Blocks() as tab:
    gr.Markdown(
        "### Video Touch Detection\n"
        "Upload a prompt image and a video. "
        "Draw **Object 1** on layer 1 and **Object 2** on layer 2. "
        "Sample frames, then run inference."
    )
    with gr.Row():
        vid_editor = gr.ImageEditor(
            label="Prompt — Object 1 (layer 1)  ·  Object 2 (layer 2)",
            type="numpy",
            height=OUT_H,
        )
        video_input = gr.Video(label="Input video", height=OUT_H)
    with gr.Row():
        resample_btn = gr.Button("Sample / Resample Frames", variant="secondary")
    with gr.Row():
        frame_previews = [
            gr.Image(
                type="numpy", label=f"Frame {i + 1}", height=FRAME_H, interactive=False
            )
            for i in range(N_FRAMES)
        ]
    with gr.Row():
        vid_dilation = gr.Slider(
            minimum=5, maximum=100, value=10, step=5, label="Touch region dilation (px)"
        )
        vid_depth_threshold = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.2,
            step=0.05,
            label="Depth similarity threshold (relative)",
        )
        run_btn = gr.Button("Run Inference", variant="primary")
    gr.Markdown("#### Results")
    out_prompt = gr.Image(label="Prompt — SAM masks + touch", height=OUT_H)
    with gr.Row():
        out_frames = [
            gr.Image(label=f"Frame {i + 1} — SegGPT + touch", height=OUT_H)
            for i in range(N_FRAMES)
        ]

    resample_btn.click(
        fn=sample_video_frames, inputs=[video_input], outputs=frame_previews
    )
    run_btn.click(
        fn=run_video_touch_inference,
        inputs=[vid_editor, vid_dilation, vid_depth_threshold] + frame_previews,
        outputs=[out_prompt] + out_frames,
    )
