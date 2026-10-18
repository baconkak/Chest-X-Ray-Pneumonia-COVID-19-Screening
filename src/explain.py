import numpy as np
import torch


def grad_cam_resnet(model, x):
    """Positive-logit Grad-CAM. A visualization, not a lesion annotation."""
    activations = []
    handle = model.layer4.register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        model.eval(); model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(x)
            features = activations[0]
            gradients = torch.autograd.grad(logits[:, 0].sum(), features)[0]
            weights = gradients.mean((2, 3), keepdim=True)
            cam = (weights*features).sum(1).relu()
            cam = torch.nn.functional.interpolate(cam[:, None], size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()
