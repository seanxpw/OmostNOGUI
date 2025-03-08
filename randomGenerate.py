import os
import os
from itertools import product
# 设置可见的CUDA设备
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None
OUTPUT_FOLDER = "/home/sean/Omost/outputs/random"
OUTPUT_FOLDER = "/kaggle/working/outputs/random"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
import lib_omost.memory_management as memory_management

import torch
torch.cuda.set_device(0)
import numpy as np
# import gradio as gr
import tempfile

# gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
# os.makedirs(gradio_temp_dir, exist_ok=True)

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList

import lib_omost.canvas as omost_canvas


# SDXL

sdxl_name = 'SG161222/RealVisXL_V4.0'
# sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'

tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

# LLM

# llm_name = 'lllyasviel/omost-phi-3-mini-128k-8bits'
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
# llm_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits'

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
    token=HF_TOKEN,
    device_map="auto"  # This will load model to gpu with an offload system
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    token=HF_TOKEN
)

memory_management.unload_all_models(llm_model)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def chat_fn(message: str, history,seed:int, temperature: float = 0.5, top_p: float = 0.9, max_new_tokens: int = 4096):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    # for user, assistant in history:
    #     if isinstance(user, str) and isinstance(assistant, str):
    #         if len(user) > 0 and len(assistant) > 0:
    #             conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('User stopped generation')
            return True
        else:
            return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    def interrupter():
        streamer.user_interrupted = True
        return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=llm_model.generate, kwargs=generate_kwargs)
    t.start()
    outputs = []
    for text in streamer:
        outputs.append(text)
    
    # print(outputs)
    t.join()


    return  "".join(outputs), interrupter


@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None

    # print("postchat history",history)

    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    #print(f"processed canvas",canvas_outputs)

    return canvas_outputs, None, None


@torch.inference_mode()
def diffusion_fn( canvas_outputs, seed, image_name,image_folder = OUTPUT_FOLDER,num_samples = 1, image_width = 600, image_height =800,
                 highres_scale = 1.2, steps = 30, cfg = 7.0, highres_steps = 20, highres_denoise = 0.4,negative_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality, wrong number, small objects'):

    use_initial_latent = False
    #use_initial_latent = True
    eps = 0.05
    # print(f"canvas ouput {canvas_outputs}")

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
            initial_latent=latents,
            strength=highres_denoise,
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        # unique_hex = uuid.uuid4().hex
        image_path = os.path.join(image_folder, f"{image_name}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)

    return 
import random
import os
import logging

# Example object pool (you can customize per group)
import random

import random

# Expanded, flexible object categories (mixing general types with specific items)
object_categories = [
    "a few books", "some scattered papers", "a phone", "a water bottle", "a couple of chairs",
    "a small lamp", "a potted plant", "a laptop", "some kitchen utensils", "a backpack", 
    "a pair of shoes", "a bowl of fruit", "a jacket draped over a chair", "some toys", 
    "a television remote", "a handbag", "a camera", "a coffee mug", "a clock", 
    "a painting leaning against the wall", "a desk lamp", "a tablet", "a pair of headphones",
    "some clothes folded neatly", "a set of keys", "a framed photo", "some pens and pencils",
    "a stack of magazines", "a remote-controlled car", "a small speaker", "a pair of sunglasses"
]

# Extended scenario templates to increase variety
scenario_templates = [
    "Generate a top-down view of a cozy indoor space with {}.",
    "Create an image showing a bird's-eye view of a cluttered tabletop containing {}.",
    "Show a neatly arranged desk or table with {}.",
    "Generate a simple top-down view of an indoor scene with {}.",
    "Imagine a work-from-home setup seen from above, with {}.",
    "Generate a casual living space from a top-down view, featuring {}.",
    "Create a well-lit tabletop scene viewed from above, containing {}.",
    "Show a home corner or shelf with {} from a top-down perspective.",
    "Render a relaxed indoor environment with {}.",
    "Generate a creative, top-down visual story using {}.",
    "Show a small section of a living room or bedroom from above, featuring {}.",
    "Generate a visual inventory scene with {} laid out neatly.",
    "Create an artist's workspace from above, with {}.",
    "Depict a student's study area viewed from a top-down perspective, containing {}.",
    "Show an indoor tabletop arrangement prepared for photography, including {}.",
    "Generate an image with your creative thinking of a scene with {}."
]

def generate_vague_prompt():
    # Randomly select how many objects will appear (5 to 20 objects)
    num_objects = random.randint(5, 20)
    selected_objects = random.sample(object_categories, num_objects)

    # Assign each object a random count (1 to 3 times each)
    object_counts = {}
    total_objects = 0

    for obj in selected_objects:
        count = random.randint(1, 3)
        object_counts[obj] = count
        total_objects += count
        if total_objects >= 20:
            break  # Stop once we hit around 20 objects (not items, total pieces)

    # Convert to descriptive text like "2 water bottles"
    object_descriptions = []
    for obj, count in object_counts.items():
        if count == 1:
            object_descriptions.append(obj)
        else:
            object_descriptions.append(f"{count} {obj.replace('a ', '').replace('some ', '').replace('a pair of ', '')}{'s' if count > 1 else ''}")

    # Select a random scenario template
    scenario_template = random.choice(scenario_templates)

    # Combine into final prompt
    object_list_str = ", ".join(object_descriptions)
    full_prompt = scenario_template.format(object_list_str)


    return full_prompt, object_counts

import json

for i in range(1, 51):
    prompt, object_counts = generate_vague_prompt()  # Fix: unpack the tuple

    seed = random.randint(1, 100000)
    logging.info(f"Generating image {i}/100 with seed {seed}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Object counts: {object_counts}")

    # Assume chat_fn and post_chat exist for generation (replace with your actual generation functions)
    output_gen = chat_fn(message=prompt, history=None, seed=seed)
    outputs = list(output_gen)

    post_history = [(prompt, outputs[0])]
    canvas_state, _, _ = post_chat(post_history)

    if canvas_state is None:
        logging.warning(f"Canvas state is None for image {i}. Skipping...")
        continue

    # Save description prompt
    image_folder = os.path.join(OUTPUT_FOLDER, f"image_{i}_{seed}")
    os.makedirs(image_folder, exist_ok=True)

    description_path = os.path.join(image_folder, f"description_{i}_{seed}.txt")
    with open(description_path, "w") as file:
        file.write(prompt)

    # Save object counts to a JSON file for later analysis
    counts_path = os.path.join(image_folder, f"object_counts_{i}_{seed}.json")
    with open(counts_path, "w") as file:
        json.dump(object_counts, file, indent=4)

    # Run diffusion process to generate image
    image_name = f"image_{i}_{seed}"
    diffusion_fn(canvas_state, seed, image_folder=image_folder, image_name=image_name, steps=28 + len(object_counts))

    logging.info(f"Image {i} generated and saved.")

