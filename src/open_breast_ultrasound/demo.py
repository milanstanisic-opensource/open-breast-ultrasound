"""Local Gradio demo. It does not create a public link unless asked."""

from __future__ import annotations

import os
from pathlib import Path

DISCLAIMER = """Open Breast Ultrasound — Research Prototype

The three numbers below are model scores for a whole-image classifier with classes benign, malignant, and normal. They are not a diagnosis. They do not mean that cancer was detected, and they are not a statement of clinical reliability.

The model does not localize or segment a lesion. A score is not evidence that a finding sits in a particular part of the image. This prototype has not been shown to improve early detection. It is not a medical device and must not be used to decide care, to reassure a patient, or to replace a qualified clinician.

Upload only images you have permission to use. Checking that a file is a readable image is not a check that the image is an appropriate or de-identified medical input.

Uploaded files are handled in the demo process's temporary working storage. This software does not promise zero storage. Stop the process and remove that temporary directory if you need the file gone.
"""


def build_interface(runtime):
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr

    from .inference import format_scores, predict_image

    def predict(filepath):
        if not filepath:
            return {}, "No image was provided."
        scores = predict_image(runtime, Path(filepath))
        text = format_scores(scores, runtime) + "\n\n" + DISCLAIMER
        return scores, text

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Breast ultrasound image"),
        outputs=[
            gr.Label(label="Model scores (not a diagnosis)", num_top_classes=3),
            gr.Textbox(label="What these scores are, and are not", lines=16),
        ],
        title="Open Breast Ultrasound — Research Prototype",
        description=DISCLAIMER,
        flagging_mode="never",
        analytics_enabled=False,
        article=(
            "Whole-image research scores only. "
            "No Grad-CAM or other heatmap is shown, because a heatmap would not be proof of localization."
        ),
    )


def launch_demo(
    experiment_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    allow_public_share: bool = False,
) -> None:
    """Start a local demo.

    ``allow_public_share`` is off by default. Turn it on only for images that
    cannot identify a person, and only when you accept a third-party tunnel.
    """
    from .inference import load_runtime

    if allow_public_share:
        print(
            "Public Gradio sharing is enabled. Use it only for non-identifying research images. "
            "Do not upload clinical images that could identify a person."
        )
    runtime = load_runtime(Path(experiment_dir))
    app = build_interface(runtime)
    app.launch(
        server_name=host,
        server_port=int(port),
        share=bool(allow_public_share),
        show_api=False,
        show_error=True,
    )
