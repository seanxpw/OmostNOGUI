import os
import os
from itertools import product
# 设置可见的CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None
OUTPUT_FOLDER = "outputs/newEX20"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
import lib_omost.memory_management as memory_management

import torch
torch.cuda.set_device(0)
import numpy as np
import gradio as gr
import tempfile

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)

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

    print(f"processed canvas",canvas_outputs)

    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)


@torch.inference_mode()
def diffusion_fn( canvas_outputs, seed, image_name,image_folder = OUTPUT_FOLDER,num_samples = 1, image_width = 896, image_height =1152,
                 highres_scale = 1.2, steps = 30, cfg = 7.0, highres_steps = 20, highres_denoise = 0.4,negative_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality, wrong number, small objects'):

    use_initial_latent = False
    use_initial_latent = True
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


from itertools import product

def generate_combinations(group, group_name):
    items = list(group.values())
    combinations = []
    names = []  # This will store the simplified names
    total_counts = []  # This will store the total counts of items for each combination
    yolo_name_lists = []  # This will store the name_list for is_match_yolo
    yolo_count_lists = []  # This will store the corresponding_num_list for is_match_yolo
    
    # 更贴合的室内场景描述
    group_scenarios = {
        "group_0": "A private home garage with only a clean concrete floor and plain walls, featuring",
        "group_1": "A quiet indoor pet corner with nothing on the floor and a plain wall behind, featuring",
        "group_2": "A traveler's neatly packed luggage on a flat, uncluttered surface against a neutral background, containing",
        "group_3": "A modern tech-filled home office desk with a plain wall behind and no extra items, featuring",
        "group_4": "A clean, well-organized dining table set on a smooth, wooden surface with a plain light-colored wall behind, set with",
    }
    
    ending = ('which should be close to camera, clearly visible, placed separately without any additional items, and viewed from a top-down angle. '
              'The background should be minimal and uncluttered, with plain walls or neutral surfaces. '
              'The camera should be positioned directly above the objects, ensuring a flat, evenly lit, and well-centered composition.')
    
    scenario = group_scenarios.get(group_name, "In this scene, there are")
    
    # 排列组合（数量从 0 到 max_count，每种物体的数量独立）
    for counts in product(range(0, 3), repeat=len(items)):  # 这里从 0 开始，允许物体数量为 0
        # 计算物体的总数
        total_count = sum(counts)
        
        # 排除所有数量为 0 的组合
        if total_count == 0:
            continue
        
        combination = []
        name_parts = []  # For generating the simplified name
        name_list = []  # For is_match_yolo input
        count_list = []  # For is_match_yolo input
        
        for count, item in zip(counts, items):
            if count > 0:
                # For the name, just use the singular form of the item and count
                name_parts.append(f"{count}{item.lower()}")  # Convert to lowercase for consistency
                plural = f"{item}s" if count > 1 else item
                combination.append(f"{count} {plural}")
                name_list.append(item.lower())  # Add the item name to the name_list
                count_list.append(count)  # Add the count to the count_list
        
        description = (f"generate an image {scenario} that has the exact number of {', '.join(combination)} "
                       f"{ending} Ensure to match the exact number of {', '.join(combination)} and create a short clear canvas.add_local_description ")
        combinations.append(description)
        
        # Generate a simplified name like "1bike1person"
        simplified_name = ''.join(name_parts)
        names.append(simplified_name)
        total_counts.append(total_count)  # 添加总数到列表中
        yolo_name_lists.append(name_list)  # 添加到 name_list 列表
        yolo_count_lists.append(count_list)  # 添加到 count_list 列表
    
    return combinations, names, total_counts, yolo_name_lists, yolo_count_lists


# 示例输入
groups = {
    # "group_0": {
    #     0: "man",
    #     1: "car",
    #     2: "bench"
    # },
    "group_1": {
        0: "bird",
        1: "boy",
        2: "dog",
    },
    # "group_2": {
    #     0: "backpack",
    #     1: "umbrella",
    #     2: "suitcase"
    # },
    # "group_3": {
    #     0: "tv",
    #     1: "laptop",
    #     2: "cell phone"
    # },
    # "group_4": {
    #     0: "cup",
    #     1: "fork",
    #     2: "knife",
    #     3: "spoon",
    #     4: "bowl"
    # }
}
def concatenate_words(word_list):
    # 使用join将单词列表拼接成字符串
    return "".join(word_list)
import logging
import random

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("output.log"),  # 日志文件
        logging.StreamHandler()  # 控制台输出
    ]
)

def is_match_yolo(yolo_model, photo_path, name_list, corresponding_num_list):
    """
    Check if the names match the corresponding number of occurrences in a YOLO model's prediction.

    Args:
        yolo_model (YOLO): The YOLO model instance for detection.
        photo_path (str): Path to the image for prediction.
        name_list (list): List of class names to match.
        corresponding_num_list (list): List of expected counts for each class name.

    Returns:
        bool: True if the counts match, False otherwise.
    """
    if not isinstance(name_list, list) or not isinstance(corresponding_num_list, list):
        raise TypeError("name_list and corresponding_num_list must both be lists.")
    
    if len(name_list) != len(corresponding_num_list):
        raise ValueError("The length of name_list and corresponding_num_list must be the same.")
    
    # Run the YOLO model on the provided image
    try:
        results = yolo_model(photo_path)  # predict on an image
    except Exception as e:
        raise RuntimeError(f"Error during YOLO prediction: {e}")
    
    # Initialize a dictionary to count occurrences of the target names
    detected_count = {name: 0 for name in name_list}
    names = yolo_model.names
    names[0] = "boy"

    # Count detections
    for r in results:
        for c in r.boxes.cls:
            class_name = names[int(c)]
            if class_name in detected_count:
                detected_count[class_name] += 1

    # Compare the detected counts with the expected counts
    for i, name in enumerate(name_list):
        if detected_count.get(name, 0) != corresponding_num_list[i]:
            return False

    return True


from ultralytics import YOLO

# Load a model
yolo_model = YOLO("yolo11l.pt")  # load an official model

if __name__ == '__main__':
    for group_name, group in groups.items():
        logging.info(f"Combinations for {group_name}:")
        combinations, names, counts, yolo_name_lists, yolo_count_lists = generate_combinations(group, group_name)

        for combo, simplified_name, count, name_list, count_list in zip(combinations, names, counts, yolo_name_lists, yolo_count_lists):
            rounds = 1
            while True:
                if rounds > 20:
                    break
                
                seed = random.randint(1, 100000)
                logging.info(f"Processing combination: {combo} with seed: {seed}")

                # Call chat_fn
                output_gen = chat_fn(message=combo, history=None, seed=seed)
                outputs = list(output_gen)

                post_history = [(combo, outputs[0])]

                # Call post_chat and print canvas_state
                canvas_state, _, _ = post_chat(post_history)
                print(f"Canvas state for '{combo}': {canvas_state} (type: {type(canvas_state)})")

                if canvas_state is None:
                    logging.warning(f"Canvas state is None for {combo}. Retrying...")
                    continue  # Retry the current combo

                folder = os.path.join(OUTPUT_FOLDER, f"{simplified_name}")
                os.makedirs(folder, exist_ok=True)

                # Save output to a file
                simplified_name_full = f"{simplified_name}_{rounds}_{seed}"
                file_path = os.path.join(folder, simplified_name_full)
                with open(file_path, "w") as file:
                    file.write(outputs[0])
                logging.info(f"Output saved to {simplified_name_full}.txt.")

                # Call diffusion function
                diffusion_fn(canvas_state, seed, image_folder=folder, image_name=simplified_name_full, steps=28 + count)
                logging.info(f"Diffusion function called for {simplified_name} with seed: {seed}.")

                # Perform YOLO detection and check match
                image_path = os.path.join(folder, f"{simplified_name_full}.png")
                if is_match_yolo(yolo_model=yolo_model, photo_path=image_path, name_list=name_list, corresponding_num_list=count_list):
                    break
                else:
                    logging.warning(f"YOLO detection did not match for {simplified_name}. Retrying...")
                    rounds += 1
                    continue
