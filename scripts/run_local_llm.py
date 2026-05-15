import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORY_PATH = PROJECT_ROOT / "corpus" / "T01_story.txt"
CHARACTER_INVENTORY_PATH = PROJECT_ROOT / "inputs" / "T01_character_inventory.json"

OUTPUT_DIR = PROJECT_ROOT / "predictions" / "local_llm"

# Start with one model first. Then add others.

MODELS = [
    #"qwen2.5:14b",
    #"hf.co/ogulcanaydogan/Turkish-LLM-14B-Instruct-GGUF:Q4_K_M",
    "hf.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF:Q4_K_M",
    #"RefinedNeuro/Turkcell-LLM-7b-v1",
    #"alibayram/turkish-gemma-9b-v0.1"
]

PROMPT_TYPE = "candidate_based"


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def build_candidate_prompt(story_text: str, character_inventory_json: str) -> str:
    return f"""
Sen Türkçe halk anlatıları üzerinde çalışan dikkatli bir NLP anotasyon asistanısın.

Sana iki girdi verilecek:

1. Paragraf ID’leriyle bölünmüş Türkçe anlatı
2. Aday karakter envanteri JSON’u

Görevin, verilen aday karakterler arasından her paragrafta hangi karakterlerin var olduğunu belirlemek ve ana ilişkileri çıkarmaktır.

ÖNEMLİ:
- Sadece geçerli JSON döndür.
- Markdown kullanma.
- Açıklama yazma.
- JSON dışında hiçbir metin üretme.
- Yeni karakter ekleme.
- Sadece verilen character_inventory içindeki karakterleri kullan.
- paragraph_presence ve relations alanlarında sadece verilen char_id değerlerini kullan.

PARAGRAF VARLIĞI KURALI:
Bir karakter şu durumlarda paragrafta var kabul edilir:
- karakterin adı geçiyorsa,
- alias kullanılıyorsa,
- karakter eyleme katılıyorsa,
- karakter konuşuyorsa,
- karaktere hitap ediliyorsa,
- karakter zamirle veya bağlamla açıkça belirtiliyorsa,
- önceki paragraftan gelen açık özne devamlılığı varsa.

Zamirleri alias olarak düşünme, ancak bağlamdan açıkça anlaşılıyorsa paragraf varlığına dahil et.

Ancak sadece mekân geçmesi karakter varlığı oluşturmaz.

İLİŞKİ KURALLARI:
Sadece ana ilişkileri çıkar.
Metinsel kanıtı olmayan ilişki ekleme.
Her ilişki için evidence_paragraphs ve Türkçe evidence_text yaz.

Kullanılabilecek relation_type değerleri:
- enemy/conflict
- helper
- family
- romantic/marriage
- authority
- servant/master
- betrayal
- reward/benefactor
- ownership/caretaker
- acquaintance
- same_group
- unknown

Kullanılabilecek valence değerleri:
- positive
- negative
- neutral
- mixed
- unknown

Kullanılabilecek strength değerleri:
- weak
- moderate
- strong

ÇIKTI ŞEMASI:
Aşağıdaki yapıda geçerli JSON üret:

{{
  "text_id": "T01",
  "system": "local_llm_candidate_based",
  "paragraph_presence": [
    {{
      "paragraph_id": "P001",
      "characters_present": ["C01"]
    }}
  ],
  "relations": [
    {{
      "source": "C01",
      "target": "C02",
      "relation_type": "",
      "valence": "",
      "strength": "",
      "evidence_paragraphs": ["P001"],
      "evidence_text": ""
    }}
  ]
}}

KURALLAR:
- character_inventory içinde olmayan karakterleri kullanma.
- characters_present içinde sadece char_id kullan.
- source ve target alanlarında sadece char_id kullan.
- evidence_text Türkçe yazılmalıdır.
- JSON dışında hiçbir şey yazma.

HİKÂYE:
<<<
{story_text}
>>>

CHARACTER_INVENTORY_JSON:
<<<
{character_inventory_json}
>>>
""".strip()


def run_ollama_model(model_name: str, prompt: str) -> str:
    """
    Runs an Ollama model using the command line:
    ollama run model_name prompt
    """
    print(f"\nRunning model: {model_name}")

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"Ollama model failed: {model_name}")

    return result.stdout.strip()


def extract_json(raw_output: str) -> str:
    """
    Tries to extract the JSON object from the model output.
    Useful if the model accidentally adds text before/after JSON.
    """
    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return raw_output[start:end + 1]


def save_output(model_name: str, prompt_type: str, raw_output: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_model_name = (
        model_name
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
    )

    raw_path = OUTPUT_DIR / f"T01_{safe_model_name}_{prompt_type}_raw.txt"
    json_path = OUTPUT_DIR / f"T01_{safe_model_name}_{prompt_type}_output.json"

    raw_path.write_text(raw_output, encoding="utf-8")

    try:
        json_text = extract_json(raw_output)
        parsed = json.loads(json_text)

        json_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"Saved valid JSON to: {json_path}")

    except Exception as e:
        print(f"Could not parse valid JSON for {model_name}.")
        print(f"Raw output saved to: {raw_path}")
        print("Error:", e)


def main():
    story_text = load_text(STORY_PATH)
    character_inventory_json = load_text(CHARACTER_INVENTORY_PATH)

    prompt = build_candidate_prompt(
        story_text=story_text,
        character_inventory_json=character_inventory_json
    )

    for model_name in MODELS:
        raw_output = run_ollama_model(model_name, prompt)
        save_output(model_name, PROMPT_TYPE, raw_output)


if __name__ == "__main__":
    main()