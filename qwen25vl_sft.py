import torch
import wandb
from trl import SFTConfig,SFTTrainer
from functools import partial
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from utils.utils import find_files,format_data_chartqa,collate_func,clear_memory
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

MODEL_APTH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/qwen2.5vl"
DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/sft"
TMP_PATH = "/archive/share/cql/aaa/tmp"
OUTPUT_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/results/sft-2"
SUBSET = -1

directories = ['data']
data_files = find_files(directories,DATA_PATH)
dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
if SUBSET > 0:
    train_dataset = dataset.select(range(SUBSET))

train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
train_dataset, eval_dataset = train_val_dataset.train_test_split(test_size=0.1, seed=42).values()

train_dataset = [format_data_chartqa(sample) for sample in train_dataset]
eval_dataset = [format_data_chartqa(sample) for sample in eval_dataset]
test_dataset = [format_data_chartqa(sample) for sample in test_dataset]

clear_memory()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_APTH,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_APTH)
collate_fn = partial(collate_func, processor=processor)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=4,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir=OUTPUT_PATH,  
    num_train_epochs=4,  
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,  
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,  
    # Optimizer and scheduler settings
    optim="adamw_torch_fused", 
    learning_rate=2e-4,  
    lr_scheduler_type="constant",
    # Logging and evaluation
    logging_steps=5, 
    eval_steps=500,  
    eval_strategy="steps",  
    save_strategy="steps",  
    save_steps=50,  
    metric_for_best_model="eval_loss",  
    # Mixed precision and gradient settings
    bf16=True,  
    max_grad_norm=1.0,  
    warmup_ratio=0.03, 
    report_to="wandb",  
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},
    max_seq_length=800,  # Maximum sequence length for input
    remove_unused_columns = False
)

wandb.init(
    project="qwen25vl-sft-ChartQA-new",
    name="qwen25vl-sft-ChartQA-new",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model()