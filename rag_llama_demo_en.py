from datasets import load_dataset
import torch
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 使用 Llama 3.1-8B Instruct（需 Hugging Face 權限）
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 如果測試跑不動，也可以暫時改成小一點的模型：
# MODEL_ID = "gpt2"

# ==========================================
# 2. 工具函數: Wikipedia RAG
# ==========================================

'''
def search_wikipedia(query: str, num_results: int = 3, lang: str = "en") -> str:
    """
    用 Wikipedia 官方 API 做簡單 RAG：
    1) 先用 search API 找到相關條目
    2) 再用 pageid 抓每個條目的摘要（extract）
    3) 回傳整理好的文字，給 Llama 當 Background Context

    lang 可以改成:
      - "en": 英文維基
      - "ja": 日文維基
      - "zh": 中文維基
    """
    try:
        # Step 1: search
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json"
        }
        r = requests.get(search_url, params=search_params, timeout=10)
        data = r.json()

        if "query" not in data or "search" not in data["query"]:
            print("[Wiki] No search results.")
            return ""

        context_text = ""
        for item in data["query"]["search"]:
            title = item.get("title", "")
            pageid = item.get("pageid", None)

            # Step 2: 用 pageid 抓摘要
            extract = ""
            if pageid is not None:
                detail_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": pageid,
                    "exintro": True,        # 只要開頭
                    "explaintext": True,    # 純文字
                    "format": "json"
                }
                r2 = requests.get(search_url, params=detail_params, timeout=10)
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                page = pages.get(str(pageid), {})
                extract = page.get("extract", "")

            context_text += f"- Title: {title}\n  Snippet: {extract[:300]}\n"

        return context_text.strip()

    except Exception as e:
        print(f"[Wiki] Error: {e}")
        return ""

'''
'''
def search_wikipedia(query: str, num_results: int = 3, lang: str = "en") -> str:
    """
    用 Wikipedia 官方 API 做簡單 RAG：
    1) 先用 search API 找到相關條目
    2) 再用 pageid 抓每個條目的摘要（extract）
    3) 回傳整理好的文字，給 Llama 當 Background Context
    """
    print(f"[Wiki] query = {query!r}")

    try:
        # Step 1: search
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }

        r = requests.get(search_url, params=search_params, timeout=10)
        print(f"[Wiki] HTTP status = {r.status_code}")
        # 看一下前面幾個字，確認是不是 JSON
        print(f"[Wiki] raw text (前80字) = {r.text[:80]!r}")

        data = r.json()

        if "query" not in data or "search" not in data["query"]:
            print("[Wiki] No search results in JSON.")
            return ""

        context_lines = []

        for item in data["query"]["search"]:
            title = item.get("title", "")
            pageid = item.get("pageid")

            extract = ""
            if pageid is not None:
                detail_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": pageid,
                    "exintro": True,        # 只要開頭
                    "explaintext": True,    # 純文字
                    "format": "json",
                }
                r2 = requests.get(search_url, params=detail_params, timeout=10)
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                page = pages.get(str(pageid), {})
                extract = page.get("extract", "")

            line = f"- Title: {title}\n  Snippet: {extract[:300]}"
            context_lines.append(line)

        context_text = "\n".join(context_lines)
        return context_text

    except Exception as e:
        print(f"[Wiki] Error: {e}")
        # fallback：不要讓整個 pipeline 掛掉，給一段假的背景
        fallback = f"""
- Title: 模擬背景（Wiki 連線失敗）
  Snippet: 原本要從 Wikipedia 查詢「{query[:20]}」，但目前環境無法正常取得結果。
        """.strip()
        return fallback
'''

def search_wikipedia(query: str, num_results: int = 3, lang: str = "en") -> str:
    """
    用 Wikipedia 官方 API 做簡單 RAG：
    1) 先用 search API 找到相關條目
    2) 再用 pageid 抓每個條目的摘要（extract）
    3) 回傳整理好的文字，給 Llama 當 Background Context
    """
    print(f"[Wiki] query = {query!r}")

    # 官方建議要帶 user-agent，避免被擋
    headers = {
        "User-Agent": "NTU-ADL-FinalProject/0.1",
        "Accept": "application/json",
    }

    try:
        # Step 1: search
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }

        r = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        print(f"[Wiki] HTTP status = {r.status_code}")
        print(f"[Wiki] raw text (前80字) = {r.text[:80]!r}")

        # 如果不是 200，或看起來不像 JSON，就直接 fallback
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            raise RuntimeError(f"Unexpected response from Wikipedia: status={r.status_code}")

        data = r.json()

        if "query" not in data or "search" not in data["query"]:
            print("[Wiki] No search results in JSON.")
            return ""

        context_lines = []

        for item in data["query"]["search"]:
            title = item.get("title", "")
            pageid = item.get("pageid")

            extract = ""
            if pageid is not None:
                detail_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": pageid,
                    "exintro": True,        # 只要開頭
                    "explaintext": True,    # 純文字
                    "format": "json",
                }
                r2 = requests.get(search_url, params=detail_params, headers=headers, timeout=10)
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                page = pages.get(str(pageid), {})
                extract = page.get("extract", "")

            line = f"- Title: {title}\n  Snippet: {extract[:300]}"
            context_lines.append(line)

        context_text = "\n".join(context_lines)
        return context_text

    except Exception as e:
        print(f"[Wiki] Error: {e}")
        # 🔁 fallback：不要讓整個 pipeline 掛掉，給一段假的背景
        fallback = f"""
- Title: 模擬背景（Wiki 未取得正常結果）
  Snippet: 原本要從 Wikipedia 查詢「{query[:20]}」，但目前環境無法正常取得 JSON 回應。
        """.strip()
        return fallback

# ==========================================
# 3. 工具類別: Llama 3.1 生成器
# ==========================================

class LlamaEngine:
    def __init__(self, model_id: str):
        print(f"Loading model: {model_id} (4-bit quantization if supported)...")

        # 使用 4-bit 量化以節省 VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

    def generate_fake_news(self, real_news, context: str) -> str:
        """
        根據真實新聞 + RAG 背景資訊，產生「假新聞版本」。
        real_news: dict, { "title": str, "text": str }
        context: str, 來自 Wikipedia 的背景摘要
        """
        title = real_news["title"]
        content = real_news["text"][:1000]  # 避免太長

        system_prompt = (
            "You are a sophisticated writer. Your task is to rewrite a real news story "
            "to introduce believable factual errors or alter key entities (names, locations, events) "
            "while maintaining the journalistic tone. "
            "The goal is to create a piece of 'Fake News' that is plausible enough to fool fact-checkers. "
            "This is only for research and model training, not for real-world publishing."
        )

        user_prompt = f"""
### Background Information (RAG Context from Wikipedia):
{context}

### Original Real News:
Title: {title}
Content: {content}

### Task:
Please rewrite the news above.
1. Use the Background Information to add realistic details but twist the main facts.
2. Keep the style professional, like a news article.
3. Output format:
Title: [New Title]
Body: [New Body]
        """.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 將對話格式轉成模型的 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(prompt)
        generated_text = outputs[0]["generated_text"]

        # 如果是 Llama-3.1 的 chat 模板，可能會包含特殊 token，這裡做個簡單切割
        split_tok = "<|start_header_id|>assistant<|end_header_id|>"
        if split_tok in generated_text:
            generated_text = generated_text.split(split_tok)[-1].strip()

        return generated_text

# ==========================================
# 4. 主程序
# ==========================================

def main():
    # --- A. 載入真實新聞 Dataset ---
    print("Loading Dataset (CNN/DailyMail, test[0])...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:1]")  # 只拿 1 筆

    # --- B. 初始化 Llama 引擎 ---
    llama = LlamaEngine(MODEL_ID)

    # --- C. 處理每一則新聞（這裡只有 1 則） ---
    for i, news_item in enumerate(dataset):
        print(f"\n{'='*20} Processing News {i+1} {'='*20}")

        # CNN/DailyMail 的欄位是 'article'
        article_text = news_item["article"]
        original_snippet = article_text[:300].replace("\n", " ")
        formatted_news = {
            "title": "Breaking News",  # 如果沒有 title 欄位，就先給一個 placeholder
            "text": article_text
        }

        print(f"\n[Original News Snippet]:\n{original_snippet}")

        # --- D. RAG：用 Wikipedia 當外部知識來源 ---
        search_query = formatted_news["text"][:50].replace("\n", " ")
        print(f"\n[RAG] Using Wikipedia with query:\n{search_query}")

        # lang="en" 用英文維基；之後可以改成 "ja" / "zh"
        rag_context = search_wikipedia(search_query, num_results=3, lang="en")
        print(f"\n[RAG Context from Wikipedia]:\n{rag_context}")

        # --- E. 生成假新聞 ---
        fake_news = llama.generate_fake_news(formatted_news, rag_context)

        print(f"\n[Generated Fake News]:\n{fake_news}")
        print("-" * 80)


if __name__ == "__main__":
    main()
