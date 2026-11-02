"""Gradio demo (used locally with ``bcdc demo`` and on Hugging Face Spaces via spaces/app.py)."""
from __future__ import annotations

from pathlib import Path

from .inference import DISCLAIMER, Predictor, load_image
from .report import build_report

HEADER = f"""# BCDC: brain-tumour MRI classifier (research prototype)

Upload a **2D brain MRI slice** (PNG, JPEG or **DICOM**). The model:

1. checks that the image looks like its training MRIs, and refuses otherwise;
2. gives calibrated probabilities for **glioma, meningioma and an unspecified tumour class**;
3. gives a **prediction set** that contains the true class 90% of the time on data like its test set;
4. writes a short background report from cited sources.

> ⚠️ **{DISCLAIMER}** The model has **no "no tumour" class** and was trained on one public dataset.
> Uploaded images are processed in memory and not stored.
"""


def build_demo(predictor: Predictor):
    import gradio as gr

    def run(path):
        if not path:
            return None, {}, "Upload an image to start."
        try:
            img, info = load_image(Path(path).read_bytes(), Path(path).name)
        except (ValueError, RuntimeError) as e:
            return None, {}, f"**Could not read this file:** {e}"
        pred = predictor.predict(img, info)
        report = build_report(pred, predictor.display_names)["markdown"]
        label = {} if pred.rejected else {predictor.name(c): p for c, p in pred.probabilities.items()}
        return img, label, report

    with gr.Blocks(title="BCDC brain-tumour MRI classifier") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column():
                upload = gr.File(label="MRI slice (PNG, JPEG, BMP, TIFF or DICOM)", type="filepath",
                                 file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"])
                button = gr.Button("Analyse", variant="primary")
                preview = gr.Image(label="Image as seen by the model", interactive=False)
            with gr.Column():
                probs = gr.Label(label="Calibrated probabilities", num_top_classes=len(predictor.class_names))
                report = gr.Markdown()
        button.click(run, inputs=upload, outputs=[preview, probs, report])
        upload.change(run, inputs=upload, outputs=[preview, probs, report])
    return demo
