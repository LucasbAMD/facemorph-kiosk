"""
generator.py — AMD-Adapt AI Scene Style Transfer

Best quality mode: SDXL Base + ControlNet Depth + IP-Adapter FaceID
  - Extracts depth map from camera frame to preserve structure
  - Uses SDXL Base (not Turbo) for highest quality output
  - IP-Adapter FaceID preserves facial identity for styles that need it
  - ~25-40s per generation but dramatically better results

Fallback mode: SDXL Turbo img2img
  - Fast (~5s) but lower quality
  - Used when ControlNet models are not available
"""

import cv2
import numpy as np
import threading
import time
import warnings
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", message=".*upcast_vae.*")
warnings.filterwarnings("ignore", message=".*enable_vae_slicing.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*The following part of your input was truncated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_TURBO_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                   "sd_xl_turbo_1.0_fp16.safetensors")

# HuggingFace model IDs (downloaded to cache by setup_models.py)
SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEPTH_ESTIMATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# IP-Adapter FaceID — preserves facial identity during style transfer
IP_ADAPTER_FACEID_REPO = "h94/IP-Adapter-FaceID"
IP_ADAPTER_FACEID_WEIGHT = "ip-adapter-faceid_sdxl.bin"

# ── Common negative terms — split across dual CLIP encoders ───────────────────
# neg1 = identity/body issues (CLIP-L, <=77 tokens)
# neg2 = quality/content issues (CLIP-G, <=77 tokens)
_NEG_IDENTITY = (
    "beard, mustache, facial hair, goatee, stubble, "
    "mask, helmet, face covering, face paint, "
    "different face, different person, different race, different gender, "
    "nudity, exposed chest, bare chest, cleavage, topless, nsfw"
)
_NEG_QUALITY = (
    "deformed hands, extra fingers, fused fingers, missing fingers, "
    "bad anatomy, blurry, low quality, lowres, "
    "text, watermark, deformed, ugly"
)

# ── Style definitions ─────────────────────────────────────────────────────────
# Each style uses dual SDXL prompts to stay within 77-token CLIP limits:
#   prompt   = subject/identity (CLIP-L, 77 tokens)
#   prompt_2 = scene/style/background (CLIP-G, 77 tokens)
# controlnet_scale ~0.50-0.58 preserves face/pose/layout from depth map.
# strength ~0.78-0.85 controls how much the image deviates from original.

STYLES = {
    "avatar": {
        "label": "Avatar",
        "prompt": (
            "Na'vi from James Cameron Avatar movie, "
            "blue skin with darker blue tiger stripe markings on face and body, "
            "wide flat nose, large luminous yellow-green eyes with cat-slit pupils, "
            "small glowing bioluminescent freckles, long braided hair, "
            "tall pointed elf-like ears, tribal bone and leather necklace"
        ),
        "prompt_2": (
            "cinematic still from Avatar 2009 movie, planet Pandora, "
            "lush bioluminescent rainforest, giant glowing willow trees, "
            "floating mountains in background, blue-green misty atmosphere, "
            "Weta Digital CGI quality, James Cameron cinematography, "
            "ultra detailed, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", human skin, pink skin, white skin, normal skin, "
            "pale skin, grey alien, green alien, generic alien, "
            "smooth skin without stripes, plain blue skin"
        ),
        "negative_2": _NEG_QUALITY + ", realistic photo, office, indoor room, plain wall",
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.5,
        "controlnet": {"strength": 0.88, "guidance": 14.0, "steps": 40,
                        "controlnet_scale": 0.42},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "claymation": {
        "label": "Claymation",
        "prompt": (
            "same person's identical face sculpted in smooth clay, "
            "preserving exact jaw nose eyes and expression, clean-shaven, "
            "round chunky clay body, visible fingerprint textures, "
            "big expressive clay eyes, smooth clay hair matching their hairstyle"
        ),
        "prompt_2": (
            "incredible stop-motion claymation scene, "
            "miniature clay furniture and props on detailed tabletop set, "
            "painted cardboard sky backdrop, cotton ball clouds, "
            "warm studio spotlight, soft shadows, "
            "Aardman animations, Wallace and Gromit quality, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, real human, real skin, photograph",
        "negative_2": _NEG_QUALITY + ", office, plain wall",
        "controlnet": {"strength": 0.78, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.56},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    "anime": {
        "label": "Anime",
        "prompt": (
            "same person redrawn as anime character, preserving their exact face shape "
            "nose jaw and eyes, same hairstyle same hair color same expression, "
            "thick black ink outlines, flat cel-shaded coloring, "
            "anime eyes with light reflections"
        ),
        "prompt_2": (
            "stunning 2D anime illustration, NOT a photograph, "
            "vibrant anime cityscape at sunset background, "
            "cherry blossom trees, glowing lanterns, dramatic clouds, "
            "manga illustration, vibrant colors, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, real person, photograph, real skin texture",
        "negative_2": _NEG_QUALITY + ", 3d render, realistic lighting, office, plain background",
        "controlnet": {"strength": 0.78, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.58},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "prompt": (
            "same person identical face preserved exactly, same eyes nose jaw, "
            "subtle neon circuit tattoo lines on cheeks, "
            "chrome cybernetic accent near temple, LED strips in hair, "
            "futuristic jacket with illuminated trim"
        ),
        "prompt_2": (
            "incredible cyberpunk scene at night, "
            "neon-lit rain-soaked city alley, massive holographic billboards, "
            "pink and cyan neon signs, puddles reflecting neon, "
            "steam vents, flying vehicles, "
            "Blade Runner cinematic style, neon noir, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", natural lighting, daytime, sunny",
        "negative_2": _NEG_QUALITY + ", office, plain room",
        "controlnet": {"strength": 0.80, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.53},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "oilpainting": {
        "label": "Oil Painting",
        "prompt": (
            "same person's identical face painted with visible brushstrokes, "
            "preserving exact facial features eyes nose mouth and expression, "
            "rich thick impasto technique, warm golden Rembrandt lighting, "
            "dramatic chiaroscuro shadows and glowing highlights"
        ),
        "prompt_2": (
            "magnificent classical oil painting on canvas, "
            "grand Renaissance palace interior background, "
            "rich velvet curtains, ornate gold frames, marble columns, "
            "candlelight flickering, old master museum painting, "
            "baroque masterpiece, gallery quality artwork"
        ),
        "negative": _NEG_IDENTITY + ", photograph, digital art, modern",
        "negative_2": _NEG_QUALITY + ", plain background, office",
        "controlnet": {"strength": 0.80, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.55},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    "pixelart": {
        "label": "Pixel Art",
        "prompt": (
            "same person as pixel art character, identical face shape and features, "
            "same hairstyle and hair color, "
            "visible square pixels, limited retro color palette, dithering"
        ),
        "prompt_2": (
            "retro 16-bit pixel art video game screenshot, "
            "colorful RPG game level background, pixel art trees, 8-bit clouds, "
            "tiled ground, HUD elements and health bar, "
            "classic SNES Final Fantasy style, nostalgic pixel art, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, smooth, high resolution, photograph",
        "negative_2": _NEG_QUALITY,
        "controlnet": {"strength": 0.83, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "comicbook": {
        "label": "Comic Book",
        "prompt": (
            "same person's exact face drawn in comic ink style, no mask, "
            "face fully visible preserving their real eyes nose mouth jaw, "
            "heavy black ink outlines around features, halftone dot shading, "
            "bright saturated primary colors, same hairstyle"
        ),
        "prompt_2": (
            "bold dynamic comic book illustration, "
            "dramatic speed lines and pop art bursts background, "
            "Ben-Day dots pattern, bold color blocks, "
            "action comic panel layout, "
            "classic comic art style, dynamic composition, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", superhero mask, domino mask, eye mask, face covered",
        "negative_2": _NEG_QUALITY + ", photorealistic, photograph, plain background",
        "controlnet": {"strength": 0.76, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.60},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    "steampunk": {
        "label": "Steampunk",
        "prompt": (
            "same person identical face preserved exactly, same eyes nose jaw mouth, "
            "clean-shaven face matching the real person precisely, "
            "ornate brass goggles pushed up on forehead, "
            "Victorian leather coat with copper gears and clockwork accessories"
        ),
        "prompt_2": (
            "incredible steampunk scene, "
            "grand Victorian workshop background, "
            "massive spinning brass gears, steam pipes, pressure gauges, "
            "leather-bound books, amber gaslight glow, "
            "copper and bronze palette, warm dramatic lighting, "
            "industrial revolution aesthetic, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", added beard, added facial hair, rugged face",
        "negative_2": _NEG_QUALITY + ", modern, futuristic, neon, office",
        "controlnet": {"strength": 0.78, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.58},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "watercolor": {
        "label": "Watercolor",
        "prompt": (
            "same person's exact face and features painted in watercolor, "
            "preserving their real eyes nose mouth jaw and expression, "
            "soft translucent watercolor washes, visible paper texture, "
            "delicate wet-on-wet paint bleeds, same hairstyle"
        ),
        "prompt_2": (
            "beautiful watercolor painting on textured cotton paper, "
            "soft floral garden background with bleeding colors, "
            "splashes of diluted pigment, artistic drips at edges, "
            "gentle pastel tones, fine art watercolor, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photograph, digital art, oil painting",
        "negative_2": _NEG_QUALITY + ", office, plain background, harsh colors",
        "controlnet": {"strength": 0.76, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.58},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    "zombie": {
        "label": "Zombie",
        "prompt": (
            "same person's exact face transformed into a zombie, "
            "keeping their real facial structure eyes nose and jaw, "
            "pale green decaying skin, dark hollow eye sockets, "
            "torn clothing, same hairstyle but messy and dirty"
        ),
        "prompt_2": (
            "cinematic horror movie scene, dark foggy graveyard at night, "
            "creepy dead trees, cracked tombstones, full moon, "
            "eerie green fog, dramatic horror lighting, "
            "movie quality, highly detailed, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", cute, happy colors, bright, cheerful",
        "negative_2": _NEG_QUALITY + ", office, plain background, daytime",
        "controlnet": {"strength": 0.82, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.52},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "fantasy": {
        "label": "Fantasy Elf",
        "prompt": (
            "same person's exact face as a high fantasy elf, "
            "preserving their real eyes nose mouth and expression, "
            "elegant pointed elf ears, ethereal glowing skin, "
            "ornate silver circlet on forehead, flowing elven robes"
        ),
        "prompt_2": (
            "epic high fantasy scene, enchanted ancient forest, "
            "magical golden light filtering through giant trees, "
            "glowing fireflies, mystical stone ruins covered in moss, "
            "fantasy art style, highly detailed, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", modern clothing, technology, urban",
        "negative_2": _NEG_QUALITY + ", office, plain background, contemporary",
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.55},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "popart": {
        "label": "Pop Art",
        "prompt": (
            "same person's exact face in bold pop art style, "
            "preserving their real facial features and expression, "
            "flat bright neon color blocks on face, thick black outlines, "
            "stylized high contrast portrait, same hairstyle"
        ),
        "prompt_2": (
            "iconic pop art screen print, four-color offset print style, "
            "vibrant pink yellow cyan and orange color palette, "
            "repeating pattern background with bold graphic shapes, "
            "gallery art, graphic design masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, subtle colors, muted",
        "negative_2": _NEG_QUALITY + ", office, plain background, photograph",
        "controlnet": {"strength": 0.78, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.56},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "ice": {
        "label": "Ice & Frost",
        "prompt": (
            "same person's exact face with frost and ice effects, "
            "preserving their real eyes nose mouth and expression, "
            "crystalline ice forming on skin and hair, frozen eyelashes, "
            "pale blue-white frosty skin, icicles hanging from clothing"
        ),
        "prompt_2": (
            "magical frozen winter wonderland scene, "
            "massive ice crystals and frozen waterfalls background, "
            "shimmering aurora borealis in dark sky, falling snowflakes, "
            "blue and white color palette, cinematic, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", warm colors, summer, tropical, green",
        "negative_2": _NEG_QUALITY + ", office, plain background, indoor",
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.55},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "neon": {
        "label": "Neon Glow",
        "prompt": (
            "same person's exact face with vivid neon glow effect, "
            "preserving their real facial features and expression, "
            "glowing neon light outlines tracing face and body contours, "
            "electric blue and hot pink light on skin, same hairstyle"
        ),
        "prompt_2": (
            "dramatic neon light portrait in pitch black darkness, "
            "vibrant electric blue pink and purple neon tubes, "
            "light painting streaks, lens flares, glowing particles, "
            "ultraviolet blacklight atmosphere, stunning, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", natural lighting, daytime, bright ambient",
        "negative_2": _NEG_QUALITY + ", office, plain background, washed out",
        "controlnet": {"strength": 0.78, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.56},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
}

STYLE_ORDER = [
    "avatar", "anime", "cyberpunk", "claymation",
    "oilpainting", "comicbook", "pixelart", "steampunk",
    "watercolor", "zombie", "fantasy", "popart", "ice", "neon",
]


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_depth_estimator = None
_depth_processor = None
_face_analyzer = None  # InsightFace for IP-Adapter FaceID
_ip_adapter_loaded = False
_pipe_lock = threading.Lock()
_pipe_ready = False
_pipe_mode = "none"  # "controlnet" or "turbo"


def _try_load_ip_adapter(pipe):
    """Optionally load IP-Adapter FaceID + InsightFace for face-preserving styles."""
    global _face_analyzer, _ip_adapter_loaded

    try:
        from insightface.app import FaceAnalysis

        print("[Generator]   Loading InsightFace for face embedding...")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))

        print("[Generator]   Loading IP-Adapter FaceID for SDXL...")
        pipe.load_ip_adapter(
            IP_ADAPTER_FACEID_REPO,
            subfolder="",
            weight_name=IP_ADAPTER_FACEID_WEIGHT,
        )
        # Default scale to 0 so non-IP-Adapter styles are unaffected
        pipe.set_ip_adapter_scale(0.0)

        with _pipe_lock:
            _face_analyzer = app
        _ip_adapter_loaded = True
        print("[Generator]   IP-Adapter FaceID ready! Avatar style will preserve your face.")

    except ImportError:
        print("[Generator]   InsightFace not installed — IP-Adapter FaceID disabled.")
        print("[Generator]   Install with: pip install insightface onnxruntime-gpu")
    except Exception as e:
        print(f"[Generator]   IP-Adapter FaceID not available: {e}")
        print("[Generator]   Avatar style will use ControlNet-only mode.")


def _extract_face_embedding(frame_bgr):
    """Extract face embedding from a BGR frame using InsightFace.
    Returns a torch tensor of shape (1, 1, 512) or None if no face found."""
    import torch

    with _pipe_lock:
        app = _face_analyzer
    if app is None:
        return None

    faces = app.get(frame_bgr)
    if not faces:
        print("[Generator]   No face detected for IP-Adapter — skipping face ID")
        return None

    # Use the largest face (most prominent in frame)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)
    return embedding.to(dtype=torch.float16, device="cuda")


def _load_pipeline():
    """Load best available pipeline. Tries ControlNet first, falls back to Turbo."""
    global _pipe, _pipe_ready, _pipe_mode
    global _depth_estimator, _depth_processor

    try:
        import torch

        # ── Try ControlNet + SDXL Base (best quality) ─────────────────────
        try:
            from diffusers import (
                StableDiffusionXLControlNetImg2ImgPipeline,
                ControlNetModel,
            )
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            print("[Generator] Loading ControlNet Depth + SDXL Base...")
            print("[Generator]   Loading depth estimator (Depth-Anything-V2)...")
            depth_proc = AutoImageProcessor.from_pretrained(DEPTH_ESTIMATOR_ID)
            depth_model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_ESTIMATOR_ID)
            depth_model = depth_model.to("cuda")
            depth_model.eval()

            print("[Generator]   Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_DEPTH_ID,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            print("[Generator]   Loading SDXL Base pipeline...")
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                SDXL_BASE_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = pipe.to("cuda")

            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            with _pipe_lock:
                _pipe = pipe
                _depth_estimator = depth_model
                _depth_processor = depth_proc
            _pipe_mode = "controlnet"
            _pipe_ready = True
            print("[Generator] Ready! mode=ControlNet+SDXL_Base (best quality)")

            # ── Try loading IP-Adapter FaceID (optional enhancement) ───
            _try_load_ip_adapter(pipe)
            return

        except Exception as e:
            print(f"[Generator] ControlNet not available: {e}")
            print("[Generator] Falling back to SDXL Turbo...")

        # ── Fallback: SDXL Turbo ──────────────────────────────────────────
        if not SDXL_TURBO_PATH.exists():
            print(f"[Generator] ERROR: No models found!")
            print(f"[Generator]   Run: python setup_models.py")
            return

        from diffusers import StableDiffusionXLImg2ImgPipeline

        print(f"[Generator] Loading SDXL Turbo from {SDXL_TURBO_PATH}...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(SDXL_TURBO_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        with _pipe_lock:
            _pipe = pipe
        _pipe_mode = "turbo"
        _pipe_ready = True
        print("[Generator] Ready! mode=SDXL_Turbo (fast)")

    except Exception as e:
        print(f"[Generator] ERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


def is_ready():
    return _pipe_ready and _pipe is not None


def get_mode():
    return _pipe_mode


# ── Depth estimation ──────────────────────────────────────────────────────────
def _estimate_depth(pil_image):
    """Extract depth map from image using Depth-Anything-V2."""
    import torch

    with _pipe_lock:
        proc = _depth_processor
        model = _depth_estimator

    inputs = proc(images=pil_image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    # Normalize to 0-255
    depth = depth.squeeze().float()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_np = depth.cpu().numpy()
    depth_np = (depth_np * 255).astype(np.uint8)

    # Resize to match input image size
    w, h = pil_image.size
    depth_resized = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to 3-channel RGB PIL image for ControlNet
    depth_3ch = np.stack([depth_resized] * 3, axis=-1)
    return Image.fromarray(depth_3ch)


# ── Generation ────────────────────────────────────────────────────────────────
def generate_scene(frame, style_key):
    """
    Transform entire camera frame into the given style.
    Uses ControlNet+depth if available, otherwise Turbo img2img.
    Returns (cv2_image, error_string).
    """
    if not is_ready():
        return None, "AI pipeline loading — please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        mode = _pipe_mode
        params = style.get(mode, style["turbo"])
        prompt = style["prompt"]
        prompt_2 = style.get("prompt_2", prompt)
        neg_prompt = style["negative"]
        neg_prompt_2 = style.get("negative_2", neg_prompt)

        # Prepare source image at 1024x1024 for SDXL
        h, w = frame.shape[:2]
        source_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        print(f"[Generator] Style={style_key} Mode={mode} "
              f"Steps={params['steps']} Strength={params['strength']} "
              f"Frame={w}x{h}")

        start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        # ── IP-Adapter FaceID: extract face embedding if style needs it ──
        use_ip = (style.get("use_ip_adapter", False)
                  and _ip_adapter_loaded and mode == "controlnet")
        face_embeds = None
        if use_ip:
            print("[Generator]   Extracting face embedding (IP-Adapter FaceID)...")
            face_embeds = _extract_face_embedding(frame)
            if face_embeds is not None:
                ip_scale = style.get("ip_adapter_scale", 0.6)
                pipe.set_ip_adapter_scale(ip_scale)
                print(f"[Generator]   Face found — IP-Adapter scale={ip_scale}")
            else:
                pipe.set_ip_adapter_scale(0.0)

        with torch.inference_mode():
            if mode == "controlnet":
                # Extract depth map for structural preservation
                print("[Generator]   Estimating depth...")
                depth_image = _estimate_depth(source_pil)

                pipe_kwargs = dict(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=neg_prompt,
                    negative_prompt_2=neg_prompt_2,
                    image=source_pil,
                    control_image=depth_image,
                    strength=params["strength"],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    controlnet_conditioning_scale=params["controlnet_scale"],
                    generator=generator,
                )
                if face_embeds is not None:
                    pipe_kwargs["ip_adapter_image_embeds"] = [face_embeds]

                result = pipe(**pipe_kwargs).images[0]

                # Reset IP-Adapter scale so next style isn't affected
                if use_ip:
                    pipe.set_ip_adapter_scale(0.0)
            else:
                # Turbo fallback
                result = pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=neg_prompt,
                    negative_prompt_2=neg_prompt_2,
                    image=source_pil,
                    strength=params["strength"],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    generator=generator,
                ).images[0]

        elapsed = time.time() - start
        print(f"[Generator] Done in {elapsed:.1f}s")

        # Resize back to original aspect ratio
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_cv = cv2.resize(result_cv, (w, h),
                               interpolation=cv2.INTER_LANCZOS4)
        return result_cv, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Generation failed: {e}"


# ── ComfyBridge-compatible wrapper ────────────────────────────────────────────
class ComfyBridge:
    """Drop-in replacement maintaining the same interface as main.py expects."""

    def __init__(self):
        self._result = None
        self._status = "idle"
        self._message = ""
        self._thread = None
        self.available = False
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _init_pipeline(self):
        _load_pipeline()
        self.available = is_ready()
        if self.available:
            print(f"[Generator] AI engine ready (mode={get_mode()})")
        else:
            print("[Generator] AI engine failed to load")

    def check_available(self):
        self.available = is_ready()
        return self.available

    def get_mode(self):
        return get_mode()

    def generate(self, frame, style_key):
        if self._status == "generating":
            return False
        self._status = "generating"
        self._result = None
        self._message = "Transforming scene..."
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), style_key),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, frame, style_key):
        img, err = generate_scene(frame, style_key)
        if img is not None:
            self._result = img
            self._status = "done"
            self._message = "Done!"
        else:
            self._status = "error"
            self._message = err or "Generation failed"

    def get_status(self):
        return {"status": self._status, "message": self._message}

    def get_result(self):
        if self._result is not None:
            img = self._result.copy()
            self._status = "idle"
            self._result = None
            return img
        return None

    def reset(self):
        self._status = "idle"
        self._result = None
        self._message = ""
