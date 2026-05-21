import torch

from .models import INFERENCE_SEED, device, image_processor, seggpt_model


def run_seggpt(target_pil, prompt_img_pil, prompt_mask_pil, num_labels=None):
    """Run SegGPT on target_pil conditioned on (prompt_img_pil, prompt_mask_pil).

    prompt_mask_pil : 2-D numpy array or PIL image used as the prompt mask.
    num_labels      : number of foreground classes (excluding background).
                      When set, the processor uses a palette to convert the 2-D
                      label map to RGB, and post-processing recovers per-class
                      indices via nearest-neighbour colour matching.
                      When None, the mask is replicated across channels and the
                      output is a grayscale intensity map.

    Returns a 2-D numpy array (H, W) of predicted class indices (or intensities
    when num_labels is None).
    """
    processor_args = dict(
        images=target_pil,
        prompt_images=prompt_img_pil,
        prompt_masks=prompt_mask_pil,
        return_tensors="pt",
    )
    if num_labels is not None:
        processor_args["num_labels"] = num_labels
    inputs = image_processor(**processor_args).to(device)

    torch.manual_seed(INFERENCE_SEED)
    with torch.inference_mode():
        out = seggpt_model(**inputs)

    post_processor_args = dict(
        target_sizes=[target_pil.size[::-1]],
    )
    if num_labels is not None:
        post_processor_args["num_labels"] = num_labels

    return (
        image_processor.post_process_semantic_segmentation(out, **post_processor_args)[
            0
        ]
        .cpu()
        .numpy()
    )
