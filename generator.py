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
# Suppress harmless ROCm/PyTorch warnings on AMD GPUs
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention support on Navi3.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention on Navi3.*")
# Suppress InsightFace deprecation warning
warnings.filterwarnings("ignore", message=".*`estimate` is deprecated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_TURBO_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                   "sd_xl_turbo_1.0_fp16.safetensors")

# HuggingFace model IDs (downloaded to cache by setup_models.py)
# Juggernaut XL v9 — dramatically better quality than vanilla SDXL Base
SDXL_BASE_ID = "RunDiffusion/Juggernaut-XL-v9"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEPTH_ESTIMATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# IP-Adapter FaceID — preserves facial identity during style transfer
IP_ADAPTER_FACEID_REPO = "h94/IP-Adapter-FaceID"
IP_ADAPTER_FACEID_WEIGHT = "ip-adapter-faceid_sdxl.bin"

# Real-ESRGAN — neural 2x upscaler for sharper output
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)
REALESRGAN_MODEL_PATH = Path.home() / "kiosk_models" / "RealESRGAN_x2plus.pth"

# ── Common negative terms — split across dual CLIP encoders ───────────────────
# neg1 = identity/body issues (CLIP-L, <=77 tokens)
# neg2 = quality/content issues (CLIP-G, <=77 tokens)
_NEG_IDENTITY = (
    "beard, mustache, facial hair, goatee, stubble, "
    "mask, helmet, face covering, face paint, "
    "different face, different person, different race, different gender, "
    "altered skin tone, darker skin, darkened skin, blackened skin, "
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
    # ── AVATAR ────────────────────────────────────────────────────────
    "avatar": {
        "label": "Avatar",
        "prompt": (
            "same person as a Na'vi from Avatar movie, "
            "blue skin with subtle darker blue tiger stripe markings, "
            "same facial structure and expression but with Na'vi features, "
            "luminous yellow-green eyes with cat-slit pupils, "
            "small glowing bioluminescent freckles on cheeks, "
            "long braided hair, pointed elf-like ears, same face preserved"
        ),
        "prompt_2": (
            "cinematic still from Avatar 2009 movie, planet Pandora, "
            "lush bioluminescent rainforest, giant glowing willow trees, "
            "floating mountains in background, blue-green misty atmosphere, "
            "Weta Digital CGI quality, James Cameron cinematography, "
            "ultra detailed, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", random alien, generic alien, grey alien, green alien, "
            "monster, creature, non-humanoid, animal face, "
            "smooth skin without stripes, plain blue skin, "
            "unrecognizable face, distorted features"
        ),
        "negative_2": _NEG_QUALITY + ", realistic photo, office, indoor room, plain wall",
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.6,
        "controlnet": {"strength": 0.82, "guidance": 13.0, "steps": 40,
                        "controlnet_scale": 0.45},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    # ── CLAYMATION ────────────────────────────────────────────────────
    "claymation": {
        "label": "Claymation",
        "prompt": (
            "same person sculpted as a clay figure, "
            "chunky exaggerated clay features, fingerprint impressions in clay, "
            "light colored clay matching original skin tone exactly, "
            "big round clay eyes, clay hair shaped in thick textured strands, "
            "same face same identity same skin tone"
        ),
        "prompt_2": (
            "stop-motion animation movie scene, detailed miniature clay set, "
            "tiny clay props and furniture on tabletop, "
            "painted paper backdrop of rolling hills and blue sky, "
            "cotton ball clouds, warm studio spotlight with soft shadows, "
            "Wallace and Gromit claymation quality, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real human, real skin, photograph, "
            "smooth skin, smooth plastic, airbrushed, "
            "dark skin, dark clay, brown clay, black clay, "
            "animal, sheep, lamb, wool, fur"
        ),
        "negative_2": _NEG_QUALITY + ", office, plain wall, indoor room, white background",
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.55,
        "controlnet": {"strength": 0.72, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    # ── ANIME – One Piece ────────────────────────────────────────────
    "anime_op": {
        "label": "Anime-OP",
        "prompt": (
            "person redrawn in One Piece anime style by Eiichiro Oda, "
            "bold black ink outlines, large expressive eyes with shiny highlights, "
            "exaggerated facial expression, vibrant flat cel-shaded coloring, "
            "dynamic confident pose, same hairstyle and hair color, same skin tone"
        ),
        "prompt_2": (
            "One Piece anime illustration, vibrant pirate adventure scene, "
            "tropical ocean harbor with pirate ships in background, "
            "bright blue sky with dramatic clouds, palm trees and island, "
            "Toei Animation style, shonen manga art, "
            "colorful and energetic, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real person, photograph, "
            "real skin texture, 3d render"
        ),
        "negative_2": (
            _NEG_QUALITY + ", realistic lighting, office, plain background, "
            "white wall, indoor room"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.4,
        "controlnet": {"strength": 0.82, "guidance": 13.0, "steps": 35,
                        "controlnet_scale": 0.32},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── ANIME – Dragon Ball ──────────────────────────────────────────
    "anime_db": {
        "label": "Anime-DB",
        "prompt": (
            "same person redrawn in Dragon Ball Z anime style by Akira Toriyama, "
            "sharp angular face with bold black outlines, intense determined eyes, "
            "spiky dramatic hair with highlighted streaks, "
            "muscular heroic proportions, vibrant cel-shaded coloring, "
            "same face same identity same skin tone, same hair color"
        ),
        "prompt_2": (
            "Dragon Ball Z anime illustration, epic battle arena scene, "
            "rocky desert landscape with towering cliffs and boulders, "
            "dramatic energy aura glowing around character, "
            "power-up lightning and ki energy effects, bright blue sky, "
            "Toei Animation Dragon Ball style, shonen anime art, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real person, photograph, "
            "real skin texture, 3d render, soft rounded features"
        ),
        "negative_2": (
            _NEG_QUALITY + ", realistic lighting, office, plain background, "
            "white wall, indoor room"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.4,
        "controlnet": {"strength": 0.82, "guidance": 13.0, "steps": 35,
                        "controlnet_scale": 0.32},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── CYBERPUNK ─────────────────────────────────────────────────────
    "cyberpunk": {
        "label": "Cyberpunk",
        "prompt": (
            "same person with subtle neon circuit tattoo lines on cheeks, "
            "chrome cybernetic implant near temple, LED strips woven in hair, "
            "futuristic jacket with glowing illuminated trim, "
            "reflective cyberpunk visor pushed up on forehead, "
            "same face same identity same skin tone same skin color"
        ),
        "prompt_2": (
            "cyberpunk city at night, rain-soaked neon-lit alley, "
            "massive holographic billboards and advertisements, "
            "pink and cyan neon signs, wet street reflecting neon colors, "
            "steam vents and flying vehicles overhead, "
            "Blade Runner 2049 cinematic style, neon noir, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", natural lighting, daytime, sunny, nature",
        "negative_2": _NEG_QUALITY + ", office, plain room, indoor room, white wall",
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.6,
        "controlnet": {"strength": 0.75, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── COMIC BOOK ────────────────────────────────────────────────────
    "comicbook": {
        "label": "Comic Book",
        "prompt": (
            "same person drawn in bold comic book ink style, "
            "face fully visible with heavy black ink outlines around all features, "
            "halftone dot shading on skin, bright saturated primary colors, "
            "same pose same body position, same hairstyle, same skin tone"
        ),
        "prompt_2": (
            "bold dynamic comic book page illustration, "
            "dramatic explosion and speed lines background, "
            "Ben-Day dots pattern, bold POW and ZAP effects, "
            "cityscape rooftop at sunset behind, "
            "classic Marvel comic art style, dynamic composition, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", superhero mask, domino mask, eye mask, "
            "face covered, helmet"
        ),
        "negative_2": _NEG_QUALITY + ", photorealistic, photograph, office, plain background",
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.45,
        "controlnet": {"strength": 0.78, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    # ── PIXAR 3D ────────────────────────────────────────────────────
    "pixar": {
        "label": "Pixar 3D",
        "prompt": (
            "same person as a Pixar 3D animated character, "
            "smooth stylized 3D render with slightly oversized head, "
            "big expressive glossy eyes, soft rounded features, "
            "subtle subsurface skin scattering, perfectly smooth skin, "
            "same face same identity same skin tone same hair color"
        ),
        "prompt_2": (
            "Pixar Disney 3D animated movie scene, "
            "colorful vibrant stylized room with warm lighting, "
            "soft depth of field background blur, global illumination, "
            "Pixar RenderMan quality, The Incredibles and Coco art style, "
            "high quality 3D animation still frame, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", real person, photograph, 2d, flat, "
            "anime, cartoon, hand drawn, sketch"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "uncanny valley, creepy, horror, dark"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.5,
        "controlnet": {"strength": 0.78, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.42},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── SIMPSONS ─────────────────────────────────────────────────────
    "simpsons": {
        "label": "Simpsons",
        "prompt": (
            "same person drawn as a character from The Simpsons TV show, "
            "bright yellow skin, large round white eyes with black pupils, "
            "overbite mouth, four fingers on each hand, "
            "same hairstyle and hair color, same face same identity, "
            "Matt Groening cartoon art style"
        ),
        "prompt_2": (
            "The Simpsons animated TV show scene, "
            "colorful Springfield living room with worn couch, "
            "simple flat colored background, bright cheerful lighting, "
            "classic Simpsons animation cel style, "
            "clean bold outlines, flat cartoon coloring, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real person, photograph, "
            "3d render, realistic skin, normal skin color"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "dark, gritty, realistic"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.4,
        "controlnet": {"strength": 0.82, "guidance": 13.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── GTA LOADING SCREEN ───────────────────────────────────────────
    "gta": {
        "label": "GTA",
        "prompt": (
            "same person drawn in Grand Theft Auto game loading screen art style, "
            "bold stylized illustration with heavy ink outlines, "
            "saturated colors, slightly exaggerated features, "
            "confident attitude pose, same face same identity same skin tone, "
            "same hairstyle, GTA V character portrait"
        ),
        "prompt_2": (
            "Grand Theft Auto V loading screen illustration, "
            "Los Santos city skyline with palm trees and sunset, "
            "warm orange and pink gradient sky, "
            "bold graphic novel illustration style, "
            "Rockstar Games official art style, vibrant, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, photograph, "
            "anime, cartoon, 3d render, soft"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "dark, muted colors, desaturated"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.5,
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.40},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── MINECRAFT ────────────────────────────────────────────────────
    "minecraft": {
        "label": "Minecraft",
        "prompt": (
            "same person as a Minecraft character with blocky voxel body, "
            "square head with pixelated face texture, "
            "blocky rectangular arms and legs, cubic proportions, "
            "pixel art skin texture on all surfaces, "
            "same face same identity same skin tone, Minecraft Steve style"
        ),
        "prompt_2": (
            "Minecraft video game screenshot, "
            "blocky voxel landscape with grass blocks and oak trees, "
            "bright blue sky with square white clouds, "
            "pixelated sunlight, crafting table and torches nearby, "
            "Minecraft Java Edition graphics, iconic voxel world, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, smooth skin, round features, "
            "high resolution, realistic proportions, curved surfaces"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, realistic, "
            "smooth textures, anti-aliased"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.4,
        "controlnet": {"strength": 0.85, "guidance": 13.0, "steps": 35,
                        "controlnet_scale": 0.32},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    # ── TIM BURTON ───────────────────────────────────────────────────
    "timburton": {
        "label": "Tim Burton",
        "prompt": (
            "same person in Tim Burton gothic art style, "
            "pale white skin with dark sunken eye circles, "
            "exaggerated thin spindly proportions, wild messy hair, "
            "large haunted expressive eyes, slightly elongated face, "
            "same face same identity same skin tone"
        ),
        "prompt_2": (
            "Tim Burton movie scene, dark gothic whimsical world, "
            "twisted bare trees with spiral branches, "
            "crooked Victorian houses on a misty hill, "
            "purple and grey moonlit sky, Corpse Bride and "
            "Nightmare Before Christmas art style, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, photograph, "
            "bright colors, cheerful, realistic proportions, "
            "normal eyes, tanned skin"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "sunny, daylight, tropical"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.5,
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.40},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── SOUTH PARK ────────────────────────────────────────────────────
    "southpark": {
        "label": "South Park",
        "prompt": (
            "same person as a South Park cartoon character, "
            "simple construction paper cutout style, round head, "
            "dot eyes, simple oval mouth, stubby body with no neck, "
            "flat colored shapes with wobbly paper-cut edges, "
            "same face same identity same skin tone same hair color"
        ),
        "prompt_2": (
            "South Park animated TV show scene, "
            "snowy small mountain town with simple buildings, "
            "flat construction paper cutout art style, "
            "bright simple colors, Comedy Central South Park quality, "
            "Trey Parker and Matt Stone animation style, masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real person, photograph, "
            "3d render, detailed features, realistic proportions"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "dark, realistic, detailed background"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.35,
        "controlnet": {"strength": 0.85, "guidance": 13.0, "steps": 35,
                        "controlnet_scale": 0.30},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    # ── DISNEY RENAISSANCE (2D) ──────────────────────────────────────
    "disney": {
        "label": "Disney 2D",
        "prompt": (
            "same person drawn as a classic Disney animated character, "
            "smooth clean ink outlines, large expressive Disney eyes, "
            "soft rounded features, warm skin tones, "
            "flowing animated hair with highlights, "
            "same face same identity same skin tone same hair color"
        ),
        "prompt_2": (
            "classic Disney Renaissance 2D animated movie scene, "
            "magical kingdom castle in golden sunset background, "
            "warm glowing light, enchanted sparkles in the air, "
            "The Lion King and Aladdin hand-drawn animation quality, "
            "beautiful 2D cel animation, Disney masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, real person, photograph, "
            "3d render, CGI, Pixar, modern 3D animation"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "dark, gritty, realistic"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.45,
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.38},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── DARK SOULS / MEDIEVAL ────────────────────────────────────────
    "darksouls": {
        "label": "Dark Souls",
        "prompt": (
            "same person as a Dark Souls character portrait, "
            "dark fantasy medieval knight aesthetic, "
            "dramatic rim lighting on weathered face, "
            "intricate engraved armor pauldrons visible at shoulders, "
            "same face same identity same skin tone"
        ),
        "prompt_2": (
            "dark medieval fantasy landscape, "
            "crumbling gothic cathedral ruins with fog, "
            "bonfire glowing in the distance, "
            "FromSoftware Dark Souls concept art style, "
            "dark moody atmosphere, epic fantasy masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", cartoon, anime, bright colors, "
            "cheerful, modern, clean"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "bright, colorful, modern city"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.50,
        "controlnet": {"strength": 0.80, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.40},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    # ── POP ART ──────────────────────────────────────────────────────
    "popart": {
        "label": "Pop Art",
        "prompt": (
            "same person in bold Andy Warhol pop art style, "
            "flat vibrant neon color blocks, "
            "strong black outlines separating color areas, "
            "Ben-Day halftone dot pattern on skin, "
            "high contrast graphic poster look, "
            "same face same identity same skin tone"
        ),
        "prompt_2": (
            "Andy Warhol silkscreen print background, "
            "solid bold color field in hot pink or electric blue, "
            "Roy Lichtenstein comic halftone dots, "
            "1960s pop art gallery print, "
            "Museum of Modern Art quality, pop art masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, photograph, "
            "soft, muted colors, painterly, watercolor"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, "
            "detailed background, landscape, scenery"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.45,
        "controlnet": {"strength": 0.82, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    # ── ART DECO / GATSBY ────────────────────────────────────────────
    "artdeco": {
        "label": "Art Deco",
        "prompt": (
            "same person in 1920s Art Deco illustration style, "
            "sharp geometric lines, stylized angular features, "
            "gold and black color palette with metallic sheen, "
            "clean symmetrical composition, "
            "elegant fashion illustration look, "
            "same face same identity same skin tone"
        ),
        "prompt_2": (
            "lavish Art Deco interior background, "
            "geometric gold sunburst patterns, marble floors, "
            "ornate Chrysler Building inspired architecture, "
            "Great Gatsby 1920s glamour, champagne gold tones, "
            "luxury vintage poster masterpiece"
        ),
        "negative": (
            _NEG_IDENTITY + ", photorealistic, photograph, "
            "cartoon, anime, soft, painterly, messy"
        ),
        "negative_2": (
            _NEG_QUALITY + ", office, plain background, white wall, "
            "modern, dark, gritty, natural"
        ),
        "use_ip_adapter": True,
        "ip_adapter_scale": 0.45,
        "controlnet": {"strength": 0.80, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
}

STYLE_ORDER = [
    "avatar", "anime_op", "anime_db", "pixar", "disney",
    "simpsons", "southpark", "gta", "minecraft",
    "timburton", "cyberpunk", "claymation",
    "comicbook", "darksouls", "popart", "artdeco",
]


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_depth_estimator = None
_depth_processor = None
_face_analyzer = None  # InsightFace for IP-Adapter FaceID
_ip_adapter_loaded = False
_upscaler = None  # Real-ESRGAN 2x upscaler
_pipe_lock = threading.Lock()
_pipe_ready = False
_pipe_mode = "none"  # "controlnet" or "turbo"


def _try_load_upscaler():
    """Optionally load Real-ESRGAN 2x upscaler for sharper output."""
    global _upscaler

    try:
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if not REALESRGAN_MODEL_PATH.exists():
            print("[Generator]   Real-ESRGAN model not found — skipping upscaler")
            print(f"[Generator]   Expected at: {REALESRGAN_MODEL_PATH}")
            return

        print("[Generator]   Loading Real-ESRGAN 2x upscaler...")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path=str(REALESRGAN_MODEL_PATH),
            model=model,
            half=True,
            device="cuda",
        )

        with _pipe_lock:
            _upscaler = upsampler
        print("[Generator]   Real-ESRGAN 2x ready! Output will be upscaled to 2048x2048.")

    except ImportError:
        print("[Generator]   realesrgan/basicsr not installed — no upscaling.")
        print("[Generator]   Install with: pip install realesrgan")
    except Exception as e:
        print(f"[Generator]   Real-ESRGAN not available: {e}")


def _upscale_image(cv2_image):
    """Upscale a BGR image using Real-ESRGAN. Returns upscaled image or original."""
    with _pipe_lock:
        upscaler = _upscaler
    if upscaler is None:
        return cv2_image

    try:
        output, _ = upscaler.enhance(cv2_image, outscale=2)
        return output
    except Exception as e:
        print(f"[Generator]   Upscale failed: {e}")
        return cv2_image


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
            subfolder=None,
            weight_name=IP_ADAPTER_FACEID_WEIGHT,
            image_encoder_folder=None,
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
    Returns a torch tensor of shape (2, 1, 512) or None if no face found."""
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
    embedding = embedding.to(dtype=torch.float16, device="cuda")
    # Concatenate with zeros (negative embed) so diffusers can chunk(2)
    neg_embedding = torch.zeros_like(embedding)
    return torch.cat([embedding, neg_embedding], dim=0)


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
                DPMSolverMultistepScheduler,
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

            print(f"[Generator]   Loading base model ({SDXL_BASE_ID})...")
            try:
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    SDXL_BASE_ID,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
            except OSError:
                # Model doesn't have fp16 variant, load without it
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    SDXL_BASE_ID,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )
            # ── Swap scheduler to DPM++ 2M Karras (sharper, more coherent) ──
            print("[Generator]   Setting DPM++ 2M Karras scheduler...")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
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

            # ── Try loading optional enhancements ───
            _try_load_upscaler()
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

        # ── Global identity preservation — appended to ALL styles ──
        _ID_SUFFIX = ", preserve original skin tone and skin color exactly"
        _NEG_SUFFIX = ", changed skin color, wrong skin tone, wrong ethnicity"
        prompt += _ID_SUFFIX
        prompt_2 += _ID_SUFFIX
        neg_prompt += _NEG_SUFFIX
        neg_prompt_2 += _NEG_SUFFIX

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

        gen_elapsed = time.time() - start
        print(f"[Generator] Generated in {gen_elapsed:.1f}s")

        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # ── Upscale with Real-ESRGAN for sharper output ──────────────
        if _upscaler is not None:
            print("[Generator]   Upscaling 2x with Real-ESRGAN...")
            up_start = time.time()
            result_cv = _upscale_image(result_cv)
            print(f"[Generator]   Upscaled to {result_cv.shape[1]}x"
                  f"{result_cv.shape[0]} in {time.time()-up_start:.1f}s")

        # Resize back to original aspect ratio
        result_cv = cv2.resize(result_cv, (w, h),
                               interpolation=cv2.INTER_LANCZOS4)

        elapsed = time.time() - start
        print(f"[Generator] Total time: {elapsed:.1f}s")
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
