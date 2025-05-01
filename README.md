# Mini Qwen

> 多模态大模型项目代码，by 古希腊掌管代码的神



## Python Environment

- 安装环境（除了grpo）

```bash
pip install -r requirements.txt
```


## Download Data

> 所以需要的数据都写在这个脚本里了，如果需要分次下载请进入脚本注释

```bash
bash download_data.sh
```

## Downlaod Model

> 所以需要的模型都写在这个脚本里了，如果需要分次下载请进入脚本注释

```bash
bash download_models.sh
```

## SFT

- single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python qwen25vl_sft.py
```

- multi GPU

```bash
accelerate launch --config_file accelerate_config.yaml qwen25vl_sft.py

# 后台运行，最好改成绝对路径
nohup accelerate launch --config_file accelerate_config.yaml qwen25vl_sft.py > logs/output_pt.log 2>&1 &
```