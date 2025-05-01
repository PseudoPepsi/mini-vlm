import os
import gc
import torch
import time
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
from transformers import Qwen2_5_VLProcessor
from PIL import Image
from tqdm import tqdm
import base64
from io import BytesIO

def find_files(dirs,path="data/sft"):
    """
    遍历目录，查找所有文件
    """
    files = []
    for dir in dirs:
        base_path = os.path.join(path, dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

import os
from PIL import Image

def load_png_images(image_folder):
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    all_images = {}
    for image_id, png_file in tqdm(enumerate(png_files), total=len(png_files), desc="loading images"):
        image_path = os.path.join(image_folder, png_file)
        image = Image.open(image_path)
        all_images[image_id] = image
    return all_images

def get_grouped_images(results, all_images):
    grouped_images = []
    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        grouped_images.append(all_images[doc_id])
    return grouped_images

def process_ranker_results(results, grouped_images, top_k=3, log=False):
    new_grouped_images = []
    for i, doc in enumerate(results.top_k(top_k)):
        if log:
            print(f"Rank {i}:")
            print("Document ID:", doc.doc_id)
            print("Document Score:", doc.score)
            print("Document Base64:", doc.base64[:30] + "...")
            print("Document Path:", doc.image_path)
        new_grouped_images.append(grouped_images[doc.doc_id])
    return new_grouped_images

def images_to_base64(images):
    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(img_base64)
    return base64_images

def pdf_folder_to_images(
    input_folder: str,
    output_folder: str = "data/mrag/images",
    dpi: int = 300,
    index_name: str = "page",
):
    """
    遍历 input_folder 下所有 pdf，把每一页转换成图片并保存。

    Args:
        input_folder (str): 存放 pdf 的文件夹
        output_folder (str): 图片输出目录
        dpi (int): 转图分辨率
        index_name (str): save_images_to_local 中对应的索引键
    """
    # 收集所有 pdf 页面的 PIL 图像
    dataset = []
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="converting pdf pages"):
        pdf_path = os.path.join(input_folder, pdf_file)
        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"{pdf_file} 转换失败: {e}")
            continue
        # 封装成字典，保持与 save_images_to_local 兼容
        dataset.extend([{index_name: page} for page in pages])
    # 统一保存
    save_images_to_local(dataset, index=index_name, output_folder=output_folder)


def save_images_to_local(dataset, index='page', output_folder="data/mrag/images"):
    os.makedirs(output_folder, exist_ok=True)
    for image_id, image_data in tqdm(enumerate(dataset), total=len(dataset), desc="preprocessing images"):
        image = image_data[index]
        if isinstance(image, str):
            image = Image.open(image)
        image = image.resize((448, 448))
        output_path = os.path.join(output_folder, f"image_{image_id}.png")
        image.save(output_path, format="PNG")
    print(f"All images saved to {output_folder}.")

# this implementation has label issues: https://github.com/om-ai-lab/VLM-R1/issues/123
def collate_func_deprecated(examples, processor):
    import IPython;IPython.embed();
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # 1) Mask padding tokens in labels
    if isinstance(processor, Qwen2_5_VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor, refer to https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # 151655
    
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # 2 Mask image token IDs in the labels
        
    batch["labels"] = labels  
    return batch 

def format_data_chartqa(sample):
    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
    return [
        {
            "role": "system",
            "content": [{"type": "text","text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "image","image": sample["image"],},
                        {"type": "text","text": sample['query'],}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text","text": sample["label"][0]}],
        },
    ]

def collate_func(examples, processor):
    # Get the texts and images, and apply the chat template
    # example:
    # messages = [
    #    {
    #        "role": "system",
    #        "content": [{"type": "text","text": "You are a good boy."}],
    #    },
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     },
    #      {
    #            "role": "assistant",
    #            "content": [{"type": "text","text": "nothing"}],
    #      },
    # ]
    # [{'role': 'system',
    # 'content': [{'type': 'text', 'text': 'You are a good boy.'}]},
    # {'role': 'user',
    # 'content': [{'type': 'image',
    #     'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    # {'type': 'text', 'text': 'Describe this image.'}]},
    # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'nothing'}]}]
    # processor.apply_chat_template(messages, tokenize=False) => '<|im_start|>system\nYou are a good boy.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\nnothing<|im_end|>\n'
    # process_vision_info(messages)[0] => [<PIL.Image.Image image mode=RGB size=2044x1372>]
    """
    refering to https://github.com/QwenLM/Qwen/blob/main/finetune.py preprocess function.
    """
    IGNORE_TOKEN_ID = -100
    texts = [
        processor.apply_chat_template(example, tokenize=False) 
        for example in examples
    ]
    image_inputs = [
        process_vision_info(example)[0] 
        for example in examples
    ]
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )

    input_ids = batch["input_ids"]
    labels = input_ids.clone() 
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = IGNORE_TOKEN_ID
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]
    for tok_id in image_tokens:
        labels[labels == tok_id] = IGNORE_TOKEN_ID

    # system/user/assistant segmentation
    im_start_id = processor.tokenizer("<|im_start|>").input_ids[0]
    im_end_id   = processor.tokenizer("<|im_end|>").input_ids[0]
    system_ids    = processor.tokenizer("system\n").input_ids
    user_ids      = processor.tokenizer("user\n").input_ids
    assistant_ids = processor.tokenizer("assistant\n").input_ids
    newline_id = processor.tokenizer("\n").input_ids[0]

    batch_size, seq_len = input_ids.shape
    for b_idx in range(batch_size):
        ids = input_ids[b_idx]
        labs = labels[b_idx]

        i = 0
        while i < seq_len:
            if ids[i] != im_start_id:
                if ids[i] == im_end_id:
                    i += 1
                elif ids[i]==newline_id:
                    i += 1
                else:
                    labs[i] = IGNORE_TOKEN_ID
                    i += 1
                continue

            i_next = i + 1  
            seg_role = None

            def match_subseq(main_ids, start_idx, pattern):
                end_idx = start_idx + len(pattern)
                if end_idx > len(main_ids):
                    return False
                return list(main_ids[start_idx:end_idx].cpu().numpy()) == pattern

            if match_subseq(ids, i_next, system_ids):
                seg_role = "system"
                seg_role_len = len(system_ids)
            elif match_subseq(ids, i_next, user_ids):
                seg_role = "user"
                seg_role_len = len(user_ids)
            elif match_subseq(ids, i_next, assistant_ids):
                seg_role = "assistant"
                seg_role_len = len(assistant_ids)
                
            i += 1
            # no role find, default
            if seg_role is None:
                while i < seq_len and ids[i] != im_end_id:
                    labs[i] = IGNORE_TOKEN_ID
                    i += 1
                if i < seq_len and ids[i] == im_end_id:
                    i += 1
                continue

            # seg_role in ["system", "user", "assistant"]
            if seg_role in ["system", "user"]:
                for _ in range(seg_role_len):
                    if i >= seq_len:
                        break
                    labs[i] = IGNORE_TOKEN_ID
                    i += 1
                while i < seq_len and ids[i] != im_end_id:
                    labs[i] = IGNORE_TOKEN_ID
                    i += 1
                if i < seq_len and ids[i] == im_end_id:
                    i += 1
                while i < seq_len and ids[i]==newline_id:
                    i += 1
                continue

            # seg_role == "assistant"
            for _ in range(seg_role_len):
                if i >= seq_len:
                    break
                labs[i] = IGNORE_TOKEN_ID
                i += 1
            while i < seq_len and ids[i] != im_end_id:
                i += 1
            if i < seq_len and ids[i] == im_end_id:
                i += 1
            while i < seq_len and ids[i]==newline_id:
                i += 1

    batch["labels"] = labels
    return batch


def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")