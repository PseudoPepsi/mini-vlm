from datasets import load_dataset
from byaldi import RAGMultiModalModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from rerankers import Reranker
from utils.utils import find_files,pdf_folder_to_images,clear_memory,save_images_to_local,load_png_images,images_to_base64,get_grouped_images,process_ranker_results

MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/qwen2.5vl"
PDF_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/pdf"
RERANK_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/MonoQwen2-VL-v0.1"
RETRIEVAL_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/colqwen2-v1.0"
DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/pdfvqa"
IMAGE_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/images"
TMP_PATH = "/archive/share/cql/aaa/tmp"
SUBSET = -1
PREPROCESS = -1
INPUT_PDF = 1

if PREPROCESS>0:
    directories = ['data']
    data_files = find_files(directories,DATA_PATH)
    dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
    if SUBSET > 0:
        dataset = dataset.select(range(SUBSET))
    save_images_to_local(dataset,'page',IMAGE_PATH)
if INPUT_PDF>0:
    pdf_folder_to_images(input_folder=PDF_PATH)

all_images = load_png_images(IMAGE_PATH)

retrieval_model = RAGMultiModalModel.from_pretrained(RETRIEVAL_MODEL_PATH)
retrieval_model.index(input_path=IMAGE_PATH, index_name="paper_index", store_collection_with_index=False, overwrite=True)

# import IPython;IPython.embed();

reranker_model = Reranker(RERANK_MODEL_PATH, device="cuda")

vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
vl_model.eval()
min_pixels = 224 * 224
max_pixels = 448 * 448
vl_model_processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
)

retrival_top_k = 5
reranker_top_k = 3
max_new_tokens = 500

clear_memory()

while True:
    text_query = input("query:")
    results = retrieval_model.search(text_query, k=retrival_top_k)
    grouped_images = get_grouped_images(results, all_images)
    base64_list = images_to_base64(grouped_images)
    results = reranker_model.rank(text_query, base64_list)
    grouped_images = process_ranker_results(results, grouped_images, top_k=reranker_top_k)

    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in grouped_images]
            + [{"type": "text", "text": text_query}],
        }
    ]
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("response:",output_text[0])