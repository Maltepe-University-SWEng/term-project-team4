from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Cihaz seçimi (GPU varsa kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GenerateJoke():
    # Eğittiğin modeli yükle (örnek olarak yol verildi, senin model yoluna göre değiştir)
    model_path = "./turkish-fıkra-model"  # Eğitilmiş modelin yolu
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # Prompt – kullanıcıdan alınabilir hale getirilebilir (şu an sabit)
    prompt = prompt = "Aşağıda hiçbir örnek veya başlık yazma, sadece tek bir anlamlı Türkçe fıkra yaz. "
    "- Noktalama işaretleri kullanma (.,;:!? tırnak vs. yok)"
    "- Fıkra tek satırda olsun, iki cümle veya net bir punchline içersin"
    "- Tekrar eden kelime veya ifadeler olmasın"
    "Şimdi fıkrayı yaz"



    # Prompt'u tokenleştir
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Metin üretimi
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    # Çıktıyı çöz
    joke = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Sonucu yazdır
    print(joke)
    
    return joke