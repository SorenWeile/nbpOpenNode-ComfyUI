# nbpOpenNode-ComfyUI

A custom ComfyUI node for generating images with Google Gemini image models, authenticated via your ComfyUI Cloud API key — no separate Google API key required.

## Node: NBP Gemini Image 2

Generates (and optionally edits) images using Google's Gemini image models through the ComfyUI Vertex AI proxy. Auth is handled automatically by your ComfyUI account credentials.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | String | Yes | Describe the image to generate or edits to apply |
| `model` | Dropdown | Yes | Gemini model to use (see models below) |
| `seed` | Int | Yes | Fixed seed for reproducibility (best-effort) |
| `aspect_ratio` | Dropdown | Yes | Output aspect ratio; `auto` matches the input image |
| `resolution` | Dropdown | Yes | `1K`, `2K`, or `4K` — 2K/4K use Gemini's native upscaler |
| `response_modalities` | Dropdown | Yes | `IMAGE` for image only; `IMAGE+TEXT` also returns model reasoning |
| `images` | IMAGE | No | Optional reference image(s) for editing or style guidance (up to 14, use Batch Images node for multiple) |
| `system_prompt` | String | No | System-level instructions shaping the model's behavior |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `IMAGE` | IMAGE | Generated image batch |
| `STRING` | STRING | Any text returned alongside the image (empty if `IMAGE` modality) |

### Supported models

| Model | Notes |
|-------|-------|
| `gemini-3-pro-image-preview` | Highest quality, higher cost |
| `gemini-3.1-flash-image-preview` | Fast, lower cost (Nano Banana 2) |
| `gemini-2.5-flash-image-preview` | Previous generation preview |
| `gemini-2.5-flash-image` | Previous generation stable |

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes/` directory:
   ```
   custom_nodes/
   └── nbpOpenNode-ComfyUI/
       ├── __init__.py
       └── nbp_gemini_image2.py
   ```
2. Restart ComfyUI.
3. The node will appear under **api node → image → Gemini** as **NBP Gemini Image 2**.

### Requirements

No additional packages required — all dependencies ship with ComfyUI.

## Authentication

This node uses the ComfyUI Cloud API. You must be logged in to your ComfyUI account within the application. The node will use your account's API key automatically; no separate Google or Vertex AI credentials are needed.
