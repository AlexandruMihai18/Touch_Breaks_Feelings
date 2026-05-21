"""Touch detection — unified app. Run with: python app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from inference_script.tabs.annotate import tab as tab_annotate
from inference_script.tabs.depth_repair import tab as tab_depth_repair
from inference_script.tabs.greatest_hits import tab as tab_greatest_hits
from inference_script.tabs.augmentation import tab as tab_augmentation
from inference_script.tabs.review import tab as tab_review
from inference_script.tabs.touch_detection import tab as tab_touch
from inference_script.tabs.touch_tuner import tab as tab_touch_tuner

demo = gr.TabbedInterface(
    [
        tab_touch,
        tab_review,
        tab_augmentation,
        tab_annotate,
        tab_greatest_hits,
        tab_depth_repair,
        tab_touch_tuner,
    ],
    [
        "Touch Detection Pipeline",
        "Review",
        "Augmentation Preview",
        "Manual Annotations",
        "Annotate Greatest Hits",
        "Annotate Epic Kitchen",
        "Touch Dilation Tuner",
    ],
)

if __name__ == "__main__":
    demo.launch()
