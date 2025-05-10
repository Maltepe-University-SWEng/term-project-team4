from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset
)
import torch

# 1) Model ve Tokenizer
model_id = "redrussianarmy/gpt2-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
# CPU modunu zorla
device = torch.device("cpu")
model.to(device)

# 2) TextDataset ile veri yükleme
# Jokes.txt içinde her cümle veya paragraf ayrı satıra yazılmışsa uygun çalışır
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="Jokes.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 3) Eğitim Ayarları
training_args = TrainingArguments(
    output_dir="./turkish-fıkra-model",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    fp16=False,
    no_cuda=True,
    gradient_checkpointing=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 4) Eğitimi Başlat
trainer.train()

# 5) İnference Örneği
prompt = "Temel bir gün"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,
    top_p=0.95
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 6) Modeli Kaydet
model.save_pretrained("turkish-fıkra-model")
tokenizer.save_pretrained("turkish-fıkra-model")
