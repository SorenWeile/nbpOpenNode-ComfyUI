import torch
import numpy as np
import requests
import base64
import io
from PIL import Image

class NanoBananaAPINode_V2:
    """
    A custom node for ComfyUI to interface with the 
    Google Gemini 2.5 Flash Image Preview (NanoBanana) API.
    
    This node supports:
    - Text and System Prompts
    - Up to 5 reference images (standard generation)
    - 1 Image + Mask (for Visual Editing / Inpainting)
    - All image inputs enabled at all times
    - Natural language labels for all inputs (e.g., "image 1", "mask")
    - Model version selection
    - Batch generation (candidateCount)
    - Full control over Safety Settings
    - Detailed error feedback
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input fields for the node in the ComfyUI interface.
        """
        safety_options = ["BLOCK_DEFAULT", "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH"]
        # Updated aspect ratio list based on official API docs
        aspect_ratio_options = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:4", "4:5"]
        
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "password": True, "default": ""}),
                "model_version": (["gemini-2.5-flash-image", "gemini-2.5-flash-image-preview"], {"default": "gemini-2.5-flash-image"}),
                "prompt": ("STRING", {"multiline": True, "default": "A majestic golden retriever, watercolor style"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful image generation assistant."}),
                "aspect_ratio": (aspect_ratio_options, {"default": "1:1"}),
                
                # --- Moved from "generation_config" to "required" ---
                # --- FIX 1: Seed max value changed to 32-bit integer max ---
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "candidate_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                
                # --- Moved from "safety_settings" to "required" ---
                "safety_harassment": (safety_options, {"default": "BLOCK_DEFAULT"}),
                "safety_hate_speech": (safety_options, {"default": "BLOCK_DEFAULT"}),
                "safety_sexual": (safety_options, {"default": "BLOCK_DEFAULT"}),
                "safety_dangerous": (safety_options, {"default": "BLOCK_DEFAULT"}),
                
                # --- NEW: Edit Mode Toggle ---
                "edit_mode_enabled": (["no", "yes"], {"default": "no"}),
            },
            "optional": {
                # --- NEW: Mask input added to optional for Image Editing ---
                "mask": ("MASK",),
                # Image 1 is now the primary image for both modes
                "image_1": ("IMAGE",), 
                # Remaining images are now ALWAYS enabled
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
            # --- Removed custom "generation_config" and "safety_settings" keys ---
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_batch", "text_output")
    FUNCTION = "generate_image_batch"
    CATEGORY = "API/Gemini"

    # --- tensor_to_base64 is UNCHANGED ---
    def tensor_to_base64(self, image_tensor):
        """
        Converts a ComfyUI IMAGE tensor (Batch, H, W, C) 
        in 0.0-1.0 float format to a base64-encoded PNG string.
        Also handles MASK tensor (Batch, H, W) for PNG encoding.
        """
        if image_tensor is None:
            return None
        
        try:
            # Handle MASK tensor (B, H, W, 1) or (B, H, W) by taking the first batch element
            if image_tensor.dim() == 4 and image_tensor.shape[3] == 1:
                img_tensor = image_tensor[0].squeeze(2) # Remove channel dim if present
            elif image_tensor.dim() == 3:
                img_tensor = image_tensor[0]
            elif image_tensor.dim() == 4 and image_tensor.shape[3] == 3:
                 img_tensor = image_tensor[0]
            else:
                raise ValueError("Tensor shape not recognized for image or mask conversion.")
            
            # If it's a mask (1-channel or squeezed), convert to an 8-bit grayscale image
            if img_tensor.dim() == 2:
                img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_np, 'L') # 'L' for grayscale
                format_type = "PNG"
            # If it's a standard RGB image
            elif img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_np, 'RGB')
                format_type = "PNG"
            else:
                raise ValueError("Tensor content (channels) not supported for image or mask encoding.")

            buffer = io.BytesIO()
            img_pil.save(buffer, format=format_type)
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"[NanoBanana Node] Error converting tensor to base64: {e}")
            return None
            
    # --- base64_to_tensor is UNCHANGED ---
    def base64_to_tensor(self, base64_data):
        """
        Converts a base64-encoded image string to a ComfyUI IMAGE tensor.
        """
        try:
            img_bytes = base64.b64decode(base64_data)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            return img_tensor.unsqueeze(0) # Add batch dimension
        except Exception as e:
            print(f"[NanoBanana Node] Error converting base64 to tensor: {e}")
            return None

    def generate_image_batch(self, api_key, model_version, prompt, system_prompt, aspect_ratio, 
                             seed, temperature, top_p, candidate_count,
                             safety_harassment, safety_hate_speech, safety_sexual, safety_dangerous,
                             edit_mode_enabled, # <--- INPUT
                             mask=None, # <--- INPUT
                             image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        The main execution function of the node.
        """
        # Create a dummy image tensor to return on failure, preventing workflow crashes
        dummy_image = torch.zeros((1, 1, 1, 3)) 

        if not api_key:
            return (dummy_image, "ERROR: API Key is required for NanoBanana (Gemini 2.5 Flash) Node.")

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_version}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # --- 1. Construct Payload Parts (Prompt + Images + Mask) ---
        
        # Prepend system prompt if provided
        full_prompt = prompt
        if system_prompt and system_prompt.strip() and system_prompt.lower() != "you are a helpful image generation assistant.":
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
        # Start the parts list with the main prompt
        parts = [{"text": full_prompt}]
        
        print(f"[NanoBanana Node] Using model: {model_version}")
        
        image_count = 0
        
        # --- NEW LOGIC START: Sequentially add all inputs with labels ---
        is_edit_mode = edit_mode_enabled == "yes"
        print(f"[NanoBanana Node] Edit Mode Enabled (for mask): {is_edit_mode}")
        
        # A. Handle Primary Image (image_1)
        if image_1 is not None:
            b64_image = self.tensor_to_base64(image_1)
            if b64_image:
                # Add the new natural language label first
                parts.append({"text": "image 1"}) 
                # Then add the image data
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_image
                    }
                })
                image_count += 1
                print("[NanoBanana Node] Added 'image 1' (with label) to request.")
            else:
                return (dummy_image, "ERROR: Failed to encode input image_1.")

        # B. Handle Mask (Only if Edit Mode is 'yes' AND mask is provided)
        if is_edit_mode and mask is not None:
            b64_mask = self.tensor_to_base64(mask)
            if b64_mask:
                # Add the new natural language label for the mask
                parts.append({"text": "mask for image 1"})
                # Then add the mask data
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png", # Use image/png for the mask
                        "data": b64_mask
                    }
                })
                print("[NanoBanana Node] Added 'mask' (with label) for Visual Editing.")
            else:
                return (dummy_image, "ERROR: Failed to encode input mask.")

        # C. Handle Remaining Reference Images (Now runs in ALL modes)
        image_inputs_refs = [image_2, image_3, image_4, image_5]
        for i, img_tensor in enumerate(image_inputs_refs):
            if img_tensor is not None:
                b64_image = self.tensor_to_base64(img_tensor)
                if b64_image:
                    # Add the new natural language label for this image
                    parts.append({"text": f"image {i+2}"}) 
                    # Then add the image data
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64_image
                        }
                    })
                    image_count += 1
                    print(f"[NanoBanana Node] Added 'image {i+2}' (with label) to request.")
                else:
                    return (dummy_image, f"ERROR: Failed to encode input image_{i+2}.")
        # --- NEW LOGIC END ---

        # --- [DEBUG LOG] ---
        print(f"[NanoBanana Node] Total parts being sent: {len(parts)} (Text + Labeled Images/Mask)")
        # --- [END DEBUG LOG] ---

        # --- 2. Construct Generation Config ---
        # --- FIX 2: Moved aspectRatio into a nested imageConfig object ---
        generation_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": float(temperature),
            "topP": float(top_p),
            "seed": int(seed),
            "candidateCount": int(candidate_count),
            "imageConfig": {
                "aspectRatio": aspect_ratio
            }
        }
        
        # --- 3. Construct Safety Settings (UNCHANGED) ---
        safety_settings_list = []
        safety_map = {
            "HARM_CATEGORY_HARASSMENT": safety_harassment,
            "HARM_CATEGORY_HATE_SPEECH": safety_hate_speech,
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": safety_sexual,
            "HARM_CATEGORY_DANGEROUS_CONTENT": safety_dangerous,
        }
        
        for category, threshold in safety_map.items():
            if threshold != "BLOCK_DEFAULT":
                safety_settings_list.append({
                    "category": category,
                    "threshold": threshold
                })
        
        # --- 4. Assemble Final Payload (UNCHANGED) ---
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }
        
        if safety_settings_list:
            # Only add safetySettings if the user changed them from the default
            payload["safetySettings"] = safety_settings_list

        # --- 5. Make API Call & Handle Errors (UNCHANGED) ---
        try:
            # Increased timeout for potentially long batch generations
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status() 
            result = response.json()

            # --- 6. Process Response (UNCHANGED) ---

            # 6a. Check for Pre-generation Prompt Block
            if "promptFeedback" in result and "blockReason" in result["promptFeedback"]:
                block_reason = result["promptFeedback"].get("blockReason", "Unknown")
                details = result["promptFeedback"].get("safetyRatings", "")
                return (dummy_image, f"ERROR: Prompt was blocked by safety filter.\nReason: {block_reason}\nDetails: {details}")

            if "candidates" not in result or not result["candidates"]:
                error_msg = result.get("error", {}).get("message", "No candidates returned from API.")
                return (dummy_image, f"ERROR: API Error: {error_msg}")

            output_images = []
            output_texts = []

            # 6b. Loop through all generated candidates (for batching)
            for i, candidate in enumerate(result["candidates"]):
                
                # Check for Post-generation Block
                finish_reason = candidate.get("finishReason")
                if finish_reason == "SAFETY":
                    ratings = candidate.get("safetyRatings", [])
                    error_detail = f"Candidate {i+1} blocked. Reason: SAFETY."
                    for rating in ratings:
                        error_detail += f"\n  - Category: {rating.get('category')}, Probability: {rating.get('probability')}"
                    output_texts.append(error_detail)
                    continue # Skip to the next candidate

                if finish_reason != "STOP":
                    # Other finish reasons: "MAX_TOKENS", "RECITATION", "OTHER"
                    output_texts.append(f"WARN: Candidate {i+1} finished unexpectedly. Reason: {finish_reason}")
                    continue

                if "content" not in candidate or "parts" not in candidate["content"]:
                    output_texts.append(f"WARN: Candidate {i+1} had invalid content structure.")
                    continue

                # 6c. Extract Image and Text from successful candidate
                base64_data = None
                text_part = ""
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part and part["inlineData"]["mimeType"].startswith("image/"):
                        base64_data = part["inlineData"]["data"]
                    if "text" in part:
                        text_part += part["text"] + "\n"
                
                # 6d. Check for Soft Refusal (Text-only response)
                if not base64_data:
                    output_texts.append(f"INFO: Candidate {i+1} returned no image. Text response: {text_part.strip()}")
                    continue
                
                # 6e. Convert and store successful image
                img_tensor = self.base64_to_tensor(base64_data)
                if img_tensor is not None:
                    output_images.append(img_tensor)
                    if text_part.strip():
                        output_texts.append(f"Candidate {i+1} Text: {text_part.strip()}")
                else:
                    output_texts.append(f"ERROR: Candidate {i+1} image data could not be decoded.")

            # --- 7. Collate Final Outputs (UNCHANGED) ---
            
            # If no images were successfully generated, return error/info text
            if not output_images:
                final_text = "\n".join(output_texts) or "ERROR: No images were generated and no text was returned."
                return (dummy_image, final_text)

            # If images were generated, stack them into a batch
            batch_tensor = torch.cat(output_images, dim=0)
            final_text = "\n".join(output_texts) or f"Image(s) generated successfully ({len(output_images)} of {candidate_count})."
            
            return (batch_tensor, final_text)

        except requests.exceptions.HTTPError as http_err:
            error_details = http_err.response.text
            return (dummy_image, f"ERROR: HTTP Error: {http_err.response.status_code}\nDetails: {error_details}")
        except requests.exceptions.Timeout:
            return (dummy_image, f"ERROR: API request timed out after 120 seconds. Try a smaller batch or check network.")
        except Exception as e:
            return (dummy_image, f"ERROR: An unexpected error occurred: {e}")

# --- Node Mappings (UNCHANGED) ---
NODE_CLASS_MAPPINGS = {
    "NanoBanana_Gemini_2_5_Flash_V2": NanoBananaAPINode_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana_Gemini_2_5_Flash_V2": "NanoBanana (Gemini 2.5 Flash) API V2"
}
