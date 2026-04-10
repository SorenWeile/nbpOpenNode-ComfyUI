"""
NBP OpenNode — Gemini Composite Reference Node
Sends structured reference images (product angles, character/outfit, additional references)
to a Gemini image model via the ComfyUI Cloud API key.
"""

import base64
from io import BytesIO

import torch
from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiFileData,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiInlineData,
    GeminiMimeType,
    GeminiPart,
    GeminiRole,
    GeminiSystemInstructionContent,
    GeminiTextPart,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    sync_op,
    tensor_to_base64_string,
    validate_string,
)

COMPOSITE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "You will receive structured reference images grouped by role:\n"
    "  - Product images show the subject from different angles.\n"
    "  - Character/outfit images show the person and clothing to feature.\n"
    "  - Additional reference images provide style, mood, environment, or other context.\n"
    "Use ALL provided references faithfully. Maintain product accuracy, character likeness, "
    "and outfit details. Apply mood and environment from additional references.\n"
    "Prioritize generating the visual representation above any text or conversational requests."
)


def _image_to_inline_part(image_tensor: Input.Image) -> GeminiPart:
    """Convert a single ComfyUI IMAGE tensor (B,H,W,C) to an inline GeminiPart."""
    return GeminiPart(
        inlineData=GeminiInlineData(
            mimeType=GeminiMimeType.image_png,
            data=tensor_to_base64_string(image_tensor[0]),
        )
    )


def _add_group(
    parts: list[GeminiPart],
    images: list[Input.Image | None],
    group_label: str,
    item_label_prefix: str,
) -> None:
    """
    Append a labelled group of images to parts.
    Skips None entries. Only adds the group header if at least one image is present.
    """
    provided = [(i + 1, img) for i, img in enumerate(images) if img is not None]
    if not provided:
        return
    parts.append(GeminiPart(text=f"--- {group_label} ---"))
    for idx, img in provided:
        parts.append(GeminiPart(text=f"{item_label_prefix} {idx}"))
        parts.append(_image_to_inline_part(img))


async def _collect_images(response: GeminiGenerateContentResponse) -> torch.Tensor:
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


class NBPGeminiComposite(IO.ComfyNode):
    """
    Structured multi-reference image generation node.
    Accepts product angle shots, character/outfit references, and additional
    context images — each group is labelled so the model understands their role.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NBPGeminiCompositeNode",
            display_name="NBP Gemini Composite",
            category="api node/image/Gemini",
            description=(
                "Generate images from structured reference groups: product angles, "
                "character/outfit, and additional references. "
                "Auth via ComfyUI Cloud — no separate Google API key needed."
            ),
            inputs=[
                # ── Prompt ────────────────────────────────────────────────────
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the scene, composition, or edits to apply.",
                ),

                # ── Model & quality ───────────────────────────────────────────
                IO.Combo.Input(
                    "model",
                    options=[
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image-preview",
                        "gemini-2.5-flash-image-preview",
                        "gemini-2.5-flash-image",
                    ],
                    default="gemini-3-pro-image-preview",
                    tooltip="Gemini image model to use.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Best-effort reproducibility seed.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="1:1",
                    tooltip="Output aspect ratio.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    default="1K",
                    tooltip="Output resolution. 2K/4K uses the Gemini native upscaler.",
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Randomness. Lower = more predictable, higher = more creative.",
                ),
                IO.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Nucleus sampling threshold.",
                    advanced=True,
                ),
                IO.Int.Input(
                    "top_k",
                    default=40,
                    min=1,
                    max=100,
                    tooltip="Token pool size limit per step.",
                    advanced=True,
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    default="IMAGE",
                    tooltip="'IMAGE+TEXT' returns the model's reasoning alongside the image.",
                    advanced=True,
                ),

                # ── Product images (angles) ────────────────────────────────────
                IO.Image.Input(
                    "product_image_1",
                    optional=True,
                    tooltip="Product angle 1.",
                ),
                IO.Image.Input(
                    "product_image_2",
                    optional=True,
                    tooltip="Product angle 2.",
                ),
                IO.Image.Input(
                    "product_image_3",
                    optional=True,
                    tooltip="Product angle 3.",
                ),
                IO.Image.Input(
                    "product_image_4",
                    optional=True,
                    tooltip="Product angle 4.",
                ),

                # ── Character / outfit ─────────────────────────────────────────
                IO.Image.Input(
                    "character_image_1",
                    optional=True,
                    tooltip="Character or outfit reference 1.",
                ),
                IO.Image.Input(
                    "character_image_2",
                    optional=True,
                    tooltip="Character or outfit reference 2.",
                ),
                IO.Image.Input(
                    "character_image_3",
                    optional=True,
                    tooltip="Character or outfit reference 3.",
                ),
                IO.Image.Input(
                    "character_image_4",
                    optional=True,
                    tooltip="Character or outfit reference 4.",
                ),

                # ── Additional references ──────────────────────────────────────
                IO.Image.Input(
                    "reference_image_1",
                    optional=True,
                    tooltip="Additional reference 1 (style, mood, environment, etc.).",
                ),
                IO.Image.Input(
                    "reference_image_2",
                    optional=True,
                    tooltip="Additional reference 2.",
                ),
                IO.Image.Input(
                    "reference_image_3",
                    optional=True,
                    tooltip="Additional reference 3.",
                ),
                IO.Image.Input(
                    "reference_image_4",
                    optional=True,
                    tooltip="Additional reference 4.",
                ),
                IO.Image.Input(
                    "reference_image_5",
                    optional=True,
                    tooltip="Additional reference 5.",
                ),
                IO.Image.Input(
                    "reference_image_6",
                    optional=True,
                    tooltip="Additional reference 6.",
                ),

                # ── System prompt ──────────────────────────────────────────────
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=COMPOSITE_SYS_PROMPT,
                    optional=True,
                    tooltip="System-level instructions shaping the model's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(tooltip="Generated image."),
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
        temperature: float,
        top_p: float,
        top_k: int,
        response_modalities: str,
        product_image_1: Input.Image | None = None,
        product_image_2: Input.Image | None = None,
        product_image_3: Input.Image | None = None,
        product_image_4: Input.Image | None = None,
        character_image_1: Input.Image | None = None,
        character_image_2: Input.Image | None = None,
        character_image_3: Input.Image | None = None,
        character_image_4: Input.Image | None = None,
        reference_image_1: Input.Image | None = None,
        reference_image_2: Input.Image | None = None,
        reference_image_3: Input.Image | None = None,
        reference_image_4: Input.Image | None = None,
        reference_image_5: Input.Image | None = None,
        reference_image_6: Input.Image | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        _add_group(
            parts,
            [product_image_1, product_image_2, product_image_3, product_image_4],
            group_label="Product Images",
            item_label_prefix="product angle",
        )
        _add_group(
            parts,
            [character_image_1, character_image_2, character_image_3, character_image_4],
            group_label="Character / Outfit References",
            item_label_prefix="character/outfit reference",
        )
        _add_group(
            parts,
            [
                reference_image_1, reference_image_2, reference_image_3,
                reference_image_4, reference_image_5, reference_image_6,
            ],
            group_label="Additional References",
            item_label_prefix="additional reference",
        )

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
                    seed=seed,
                    temperature=temperature,
                    topP=top_p,
                    topK=top_k,
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
    "NBPGeminiCompositeNode": NBPGeminiComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NBPGeminiCompositeNode": "NBP Gemini Composite",
}
