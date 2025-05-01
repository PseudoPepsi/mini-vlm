import torch
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from utils.utils import find_files,format_data_chartqa
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

MODEL_APTH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/results/sft-2/checkpoint-500"
DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/sft"
TMP_PATH = "/archive/share/cql/aaa/tmp"
SUBSET = -1

directories = ['data']
data_files = find_files(directories,DATA_PATH)[:1]
dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
if SUBSET > 0:
    train_dataset = dataset.select(range(SUBSET))

train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
train_dataset, eval_dataset = train_val_dataset.train_test_split(test_size=0.1, seed=42).values()

train_dataset = [format_data_chartqa(sample) for sample in train_dataset]
eval_dataset = [format_data_chartqa(sample) for sample in eval_dataset]
test_dataset = [format_data_chartqa(sample) for sample in test_dataset]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_APTH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/qwen2.5vl")

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    print(sample)
    sample[1]['content'][0]['image'].save("logs/eval_test.png")
    text_input = processor.apply_chat_template(
        sample[:2],
        tokenize=False,
        add_generation_prompt=True
    )
    print("text_input",text_input)
    image_inputs, _ = process_vision_info(sample)
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]  

output = generate_text_from_sample(model, processor, test_dataset[0])
print(output)
import IPython;IPython.embed()