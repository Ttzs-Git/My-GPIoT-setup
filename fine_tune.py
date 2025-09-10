"""
使用LoRA技术给Llama-2-13b进行微调
"""

# In[]
## 基础操作: 导包+设置地址
# 导包
import os
import torch
from datasets import load_from_disk # 用于加载预处理号的数据集
from trl import SFTTrainer # 用于监督微调的高级训练器
from peft import LoraConfig, prepare_model_for_kbit_training # LoRA核心库
from transformers import ( #  Hugging Face 生态核心
    AutoModelForCausalLM, # 自动加载因果语言模型
    AutoTokenizer, # 自动加载分词器
    BitsAndBytesConfig, # 量化配置
    TrainingArguments, # 训练参数配置
)


# 定义路径(基础模型的路径，数据集的路径和输出的路径)
base_model = "meta-llama/Llama-2-13b-chat-hf" # 基础模型
# new_model = "GPIoT_Code_Generation"
new_model = "GPIoT_Task_Decomposition" # 新模型保存路径
final_dir = new_model+"/final_checkpoint" # 最终checkpoint路径
os.makedirs(final_dir, exist_ok=True)# 确保输出目录存在

# dataset = load_from_disk("dataset/Code_Generation_dataset")
dataset = load_from_disk("dataset/Task_Decomposition_dataset")# 记载数据集

# 用于测试
dataset = dataset.select(range(100))  # 只取前100个样本
print(f"⚠️ 警告：已将数据集缩减为 {len(dataset)} 个样本，仅用于测试！")

# 检查数据集类型
if isinstance(dataset, dict) or "test" in dataset:
    eval_dataset = dataset["test"]
else:
    eval_dataset = None  # 或划分一部分作为验证集
    print("⚠️ 警告: 数据集没有 'test' split，将不使用验证集。")

# In[]
## 模型的配置与量化

compute_dtype = getattr(torch, "float16") # 配置计算数据类型
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, # 启用8bit加载
    bnb_8bit_quant_type="nf8", # 使用“nf8”量化格式，对LLM效果更好
    bnb_8bit_compute_dtype=compute_dtype, # 计算时使用float16
    bnb_8bit_use_double_quant=False, # 不使用双重量化
)# 配置 8-bit量化

model = AutoModelForCausalLM.from_pretrained(
    base_model,# 基础模型
    quantization_config=quantization_config,# 量化配置
    device_map="auto",# 自动将模型层配置到可用GPU
)# 加载并量化基础模型

model.config.use_cache = False # 关闭缓存，因为训练时梯度检查点会与之冲突
model = prepare_model_for_kbit_training(model) # 关闭缓存，因为训练时梯度检查点会与之冲突

# In[]
## LoRA与训练参数配置

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token # 将填充符设为结束符
tokenizer.padding_side = "right" # 右填充，这是因果语言模型的常规做法

peft_parameters = LoraConfig(
    r=64,# 低秩矩阵的秩，决定适配器容量（越大越强，但也越占显存）
    lora_alpha=16,# 缩放因子，控制适配器更新幅度
    lora_dropout=0.001, # 防止过拟合的dropout率
    bias="lora_only", # 只训练适配器部分的偏置项，不训练原始模型的偏置
    task_type="CAUSAL_LM",# 任务类型：因果语言建模（自回归生成）
    # target_modules=["q_proj", "v_proj"],  # ← 只在 query 和 value 投影层添加 LoRA
)# 配置LoRA参数

# 配置训练参数
training_params = TrainingArguments(
    output_dir=final_dir,# 模型和日志输出目录
    num_train_epochs=1,# 训练1个epoch
    per_device_train_batch_size=1, # 每个GPU每次处理1个样本（因显存限制）
    per_device_eval_batch_size=1,# 评估时同理
    gradient_accumulation_steps=8,# 梯度累积8步，模拟 batch_size=8，保证训练稳定性
    optim="paged_adamw_32bit", # noinspection SpellChecking # 优化器，适合量化模型，能处理分页内存
    save_steps=20,# 每20步保存一次checkpoint
    logging_steps=5, # 每5步在控制台打印一次loss
    learning_rate=2e-4,# 学习率
    weight_decay=0.001,# 权重衰减，防止过拟合
    fp16=True,  # 3090 完美支持 # 启用混合精度训练，加速并省显存
    bf16=False, # 不使用bf16（3090支持不佳）
    max_grad_norm=0.3,# 梯度裁剪，防止梯度爆炸
    max_steps=-1,# 不设置最大步数，由epoch数决定
    warmup_ratio=0.03,# 学习率预热比例
    group_by_length=True,# 按序列长度分组，提高填充效率
    lr_scheduler_type="cosine",# 学习率调度器：余弦退火
    report_to="none",# 不报告给任何平台（如W&B）
    evaluation_strategy="steps",  # 每 N 步评估一次# 按步数进行评估
    eval_steps=100,  # 每 100 步评估
)
# In[]
## 启动训练与保存

# 创建监督微调训练器
trainer = SFTTrainer(
    model=model,  # 已量化的、冻结的、准备好的基础模型
    train_dataset=dataset,# 训练数据集
    eval_dataset=eval_dataset,# 评估数据集（如果有的话）
    peft_config=peft_parameters,# LoRA 配置
    dataset_text_field="data",# 数据集中包含文本的字段名
    max_seq_length=1024, # 最大序列长度，防止过长样本导致OOM
    tokenizer=tokenizer,# 分词器
    args=training_params,# 训练参数
    packing=False, # 不将多个短样本打包成一个长样本
)

# 打印可训练参数信息（验证LoRA是否生效）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params}, 总参数: {total_params}, 比例: {trainable_params/total_params:.2%}")

# 正式启动训练！
trainer.train()

# 保存模型
trainer.model.save_pretrained(final_dir) # type: ignore # 只会保存 adapter_model.safetensors 和 adapter_config.json
trainer.tokenizer.save_pretrained(final_dir) # 保存tokenizer配置

# 清理显存
del trainer
torch.cuda.empty_cache()
print(f"🎉 训练完成！模型已保存至: {final_dir}")