"""
NBP OpenNode — Gemini Image 2 Generation Node
Generates images via Google Vertex AI (Gemini) using the ComfyUI Cloud API key.
Auth is handled automatically via IO.Hidden.api_key_comfy_org / auth_token_comfy_org.
"""

import base64
from io import BytesIO

import torch
from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiPart,
    GeminiRole,
    GeminiSystemInstructionContent,
    GeminiTextPart,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    validate_string,
)
from comfy_api_nodes.apis.gemini import GeminiInlineData, GeminiFileData, GeminiMimeType

GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)


async def _create_image_parts(cls, images: Input.Image) -> list[GeminiPart]:
    """Upload up to 10 images as URLs via the ComfyAPI proxy, rest as inline data."""
    total = get_number_of_images(images)
    num_url = min(total, 10)
    urls = await upload_images_to_comfyapi(cls, images, max_images=num_url)
    parts: list[GeminiPart] = [
        GeminiPart(fileData=GeminiFileData(mimeType=GeminiMimeType.image_png, fileUri=url))
        for url in urls
    ]
    for idx in range(num_url, total):
        parts.append(
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.image_png,
                    data=tensor_to_base64_string(images[idx]),
                )
            )
        )
    return parts


def _extract_images(response: GeminiGenerateContentResponse) -> torch.Tensor:
    """Pull all image parts out of the response and stack them into a batch tensor."""
    tensors: list[torch.Tensor] = []
    if not response.candidates:
        raise ValueError("Gemini returned no candidates.")
    for candidate in response.candidates:
        if candidate.content is None or candidate.content.parts is None:
            continue
        for part in candidate.content.parts:
            if part.inlineData and part.inlineData.mimeType.value.startswith("image/"):
                img_bytes = base64.b64decode(part.inlineData.data)
                tensors.append(bytesio_to_image_tensor(BytesIO(img_bytes)))
            elif part.fileData and part.fileData.mimeType.value.startswith("image/"):
                # fileData images are downloaded asynchronously — collect URIs for later
                tensors.append(None)  # placeholder; handled in execute()
    if not tensors:
        raise ValueError(
            "Gemini did not return an image. "
            "Try 'IMAGE+TEXT' mode to see the model's reasoning."
        )
    return tensors


async def _collect_images(response: GeminiGenerateContentResponse) -> torch.Tensor:
    """Async version that also downloads fileData images."""
    tensors: list[torch.Tensor] = []
    if not response.candidates:
        raise ValueError("Gemini returned no candidates.")
    for candidate in response.candidates:
        if candidate.content is None or candidate.content.parts is None:
            continue
        for part in candidate.content.parts:
            if part.inlineData and part.inlineData.mimeType.value.startswith("image/"):
                img_bytes = base64.b64decode(part.inlineData.data)
                tensors.append(bytesio_to_image_tensor(BytesIO(img_bytes)))
            elif part.fileData and part.fileData.mimeType.value.startswith("image/"):
                tensors.append(await download_url_to_image_tensor(part.fileData.fileUri))
    if not tensors:
        raise ValueError(
            "Gemini did not return an image. "
            "Try 'IMAGE+TEXT' mode to see the model's reasoning."
        )
    return torch.cat(tensors, dim=0)


def _extract_text(response: GeminiGenerateContentResponse) -> str:
    parts: list[str] = []
    if not response.candidates:
        return ""
    for candidate in response.candidates:
        if candidate.content is None or candidate.content.parts is None:
            continue
        for part in candidate.content.parts:
            if part.text:
                parts.append(part.text)
    return "\n".join(parts)


class NBPGeminiImage2(IO.ComfyNode):
    """
    Generate or edit images via Google Gemini (Vertex AI) using the
    ComfyUI Cloud API key — no separate Google API key required.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NBPGeminiImage2Node",
            display_name="NBP Gemini Image 2",
            category="api node/image/Gemini",
            description=(
                "Generate or edit images with Google Gemini image models "
                "via the ComfyUI Cloud API. Auth is handled automatically "
                "through your ComfyUI account."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the image you want to generate or the edits to apply.",
                ),
                IO.Combo.Input(
                    "model",
                    options=[
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image-preview",
                        "gemini-2.5-flash-image-preview",
                        "gemini-2.5-flash-image",
                    ],
                    default="gemini-3-pro-image-preview",
                    tooltip="Which Gemini image model to use.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "Fixed seed makes the model try to reproduce the same result. "
                        "Determinism is best-effort — temperature and model changes still vary output."
                    ),
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    tooltip=(
                        "'auto' matches the input image aspect ratio; "
                        "if no image is provided a square is generated."
                    ),
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    default="1K",
                    tooltip="Output resolution. 2K/4K uses the Gemini native upscaler.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    default="IMAGE",
                    tooltip=(
                        "'IMAGE' for image-only output; "
                        "'IMAGE+TEXT' also returns the model's reasoning."
                    ),
                    advanced=True,
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). Use Batch Images for multiple (up to 14).",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="System-level instructions that shape the model's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(tooltip="Generated image(s)."),
                IO.String.Output(tooltip="Any text the model returned alongside the image."),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        images: Input.Image | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("Maximum 14 reference images are supported.")
            parts.extend(await _create_image_parts(cls, images))

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(
                parts=[GeminiTextPart(text=system_prompt)], role=None
            )

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[GeminiContent(role=GeminiRole.user, parts=parts)],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(
                        ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]
                    ),
                    imageConfig=image_config,
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
        )

        output_image = await _collect_images(response)
        output_text = _extract_text(response)
        return IO.NodeOutput(output_image, output_text)


NODE_CLASS_MAPPINGS = {
    "NBPGeminiImage2Node": NBPGeminiImage2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NBPGeminiImage2Node": "NBP Gemini Image 2",
}
