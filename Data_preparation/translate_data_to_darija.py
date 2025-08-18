from datasets import load_dataset

data = load_dataset('AbderrahmanSkiredj1/xsample-tulu-3-mig-darija-split-31k3-tobe-translated', split="train")
import time
import copy
import google.generativeai as genai

# ──────────────────────────
# ❶  SET UP GEMINI MODELS
# ──────────────────────────
API_KEY = "AIblablablameeCrWg"

genai.configure(api_key=API_KEY)

_model_lite = genai.GenerativeModel("gemini-2.0-flash-lite")
_model_pro  = genai.GenerativeModel("gemini-2.0-flash")  # a bit stronger, slightly slower
#_model_pro  = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")  # a bit stronger, slightly slower


def _ask_llm(prompt: str, pro: bool = False) -> str:
    """Call Gemini and return the plain-text response."""
    model = _model_pro if pro else _model_lite
    return model.generate_content([prompt]).text



DUMMY_RESPONSE = "###GEMINIERROR###"      # fallback when every retry fails

def _safe_call(prompt: str,
               pro: bool = False,
               max_retries: int = 3,
               base_delay: float = 1.0) -> str:
    """
    Call Gemini with retries and exponential back-off.
    If *all* attempts are blocked or error out, return DUMMY_RESPONSE.
    """
    for attempt in range(max_retries):
        try:
            return _ask_llm(prompt, pro=pro)
        except Exception as err:            # broad catch OK for this utility
            if attempt == max_retries - 1:  # last try exhausted → give up
                print(f"[failed after {max_retries} tries] {err} – returning dummy")
                return DUMMY_RESPONSE
            wait = base_delay * (2 ** attempt)   # 2 s, 4 s, 8 s, …
            print(f"[retry {attempt+1}/{max_retries}] {err} – sleeping {wait}s")
            time.sleep(wait)

# ──────────────────────────
# ❸  HIGH-LEVEL HELPERS
# ──────────────────────────
def translate_text(text: str,
                   prompt_template: str,
                   pro: bool = False,
                   **retry_kwargs) -> str:
    """Translate a *single* text block and return the translated string."""
    full_prompt = f"{prompt_template.rstrip()}\n\n{text}"
    return _safe_call(full_prompt, pro=pro, **retry_kwargs)


def translate_messages(messages: list[dict],
                       prompt_template: str,
                       pro: bool = False,
                       deep_copy: bool = True,
                       **retry_kwargs) -> list[dict]:
    """
    Translate every “content” field in a classic chat-style messages list.
    Keeps roles / order intact and returns a *new* list unless deep_copy=False.
    """
    msgs = copy.deepcopy(messages) if deep_copy else messages
    for msg in msgs:
        if "content" in msg and isinstance(msg["content"], str):
            msg["content"] = translate_text(
                msg["content"],
                prompt_template=prompt_template,
                pro=pro,
                **retry_kwargs,
            )
    return msgs


# ──────────────────────────
# ❹  EXAMPLE USAGE
# ──────────────────────────


TRANSLATION_PROMPT = """
**Translate the following text into Moroccan Darija, written in Arabic script, following these strict instructions:**

---

### ❌ **DO NOT TRANSLATE the following elements (keep them exactly as they are):**

1. **🧑‍💻 Code blocks**:
   Keep *all programming code* (Python, JS, etc.) **unchanged**.
   ✅ **Translate comments inside code blocks** to Darija (Arabic script).

2. **🧮 Math formulas and LaTeX**:
   Preserve **all mathematical notation and LaTeX equations**. Do not alter them in any way.

3. **🔧 Technical Terms**:
   Keep technical/scientific terms in **English** *unless* they have a **known Arabic equivalent**.
   ✅ For example:

   * Use **"صيغة"** for “formula”,
   * Use **"معادلة"** for “equation”.
   * Use **"دالة"** for “function”.
     ❌ But **do not translate** terms like “API”, “eigenvalue”, “html”, etc.

4. **🌍 Proper Names**:
   Do **not translate** names of people, tools, or software (e.g., Python, Newton, GitHub, etc.).

---

### ✅ **TRANSLATE into Moroccan Darija (Arabic script):**

* Regular sentences, explanations, and natural narrative text.
* Comments inside code blocks (e.g., `# comment here`).
* Use a **natural mix of Moroccan Darija and Modern Standard Arabic** where appropriate:

  * Use Darija for informal or conversational parts.
  * Use Standard Arabic for formal or widely recognized scientific terms. Do your best to find most suitable translations.
  * If translating certain phrases would sound awkward, unclear, or artificial in Darija, then translate them in Standard Arabic.

---

❌ **DO NOT include any preface like “Here is your translation,”:**
- Do NOT include any preface like “Here is your translation,” no headings, no quotes, no backticks, no tags, no extra commentary.
- Do NOT repeat these instructions.
- Output ONLY the translated text. Do NOT add any preface, headings, labels, explanations, quotes, backticks, or extra text. If you are about to write anything other than the translation, stop.
- Do not put your translation in a json format.

---

### 📌 GOAL:

Make the content **clear and naturally understandable** to Moroccan Arabic speakers while **respecting the technical integrity** of the original text.
Return the translated text without saying anything else.

---

### ✅ **Examples:**

**Example 1 (Code + Explanation)**

**Input:**

> The 'Function' calculate\_mean(data) is used to compute the average of a list of numbers.
> // This function iterates through the list
> // and calculates the sum.
> Python
>
> ```python
> def calculate_mean(data_list):  
>     # Calculate sum of elements  
>     total_sum = sum(data_list)  
>     # Calculate number of elements  
>     count = len(data_list)  
>     # Return the mean  
>     return total_sum / count  
> ```
>
> Its 'Complexity' is O(n).

**Output:**

> ال 'Function' calculate\_mean(data) كتستعمل باش نحسبو المعدل ديال شي ليستة ديال الأرقام.
> // هاد الدالة كدور على العناصر
> // وكتحسب المجموع.
> Python
>
> ```python
> def calculate_mean(data_list):  
>     # حساب مجموع العناصر  
>     total_sum = sum(data_list)  
>     # حساب عدد العناصر  
>     count = len(data_list)  
>     # إرجاع المعدل  
>     return total_sum / count  
> ```
>
> ال 'Complexity' ديالها هي O(n).

---

**Example 2 (Math Formula)**

**Input:**

> Consider the quadratic equation $ax^2 + bx + c = 0$. The solutions can be found using the formula:
> $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$. This is known as the quadratic formula.

**Output:**

> ناخدو المعادلة من الدرجة الثانية $ax^2 + bx + c = 0$. الحلول ديالها كتلقاو بهاد الصيغة:
> $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$. هادي معروفة بصيغة المعادلة من الدرجة الثانية.

---

**Example 3 (Fully Regular Text)**

**Input:**

> This chapter introduces the basic concepts. Please read it carefully.

**Output:**

> هاد الجزء كيعرف بالمفاهيم الأساسية. عافاك قراه مزيان.

---

**Now, translate the following text:**
"""

import os, json, time, copy
from typing import List, Dict

# ──────────────────────────
# ❶  YOUR HF DATASET HERE
#     (it must already be in memory as `data`,
#      or you can load it from disk if you saved it)
# ──────────────────────────
from datasets import Dataset  # pip install datasets

# If you loaded it elsewhere, just make sure DATASET points to it ↓
DATASET: Dataset = data

from tqdm import tqdm
# ──────────────────────────
# ❹  DATASET-LEVEL TRANSLATION WITH INCREMENTAL SAVE
# ──────────────────────────
def translate_dataset_hf(ds: Dataset,
                         prompt_template: str,
                         output_path: str = "data_darija.jsonl",
                         pro: bool = False,
                         flush_every: int = 1,
                         **retry_kw):
    """
    Translate the 'dialogs' field of each row and append the result to `output_path`.
    Automatically resumes by counting existing lines in the JSONL file.
    """
    # how many samples already done?
    done = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"Resuming – {done} / {len(ds)} rows already translated")

    # open file in append mode (utf-8, no ASCII escapes)
    with open(output_path, "a", encoding="utf-8") as fout:
        for idx in tqdm(range(done, len(ds)), initial=done,total=len(ds),desc="Translating",unit="sample"):
            sample = ds[idx]
            # translate dialogs list
            sample["dialogs"] = translate_messages(
                sample["dialogs"],
                prompt_template=prompt_template,
                pro=pro,
                **retry_kw
            )

            # write as a single JSON line
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # flush every N rows (default 1 → safest)
            if flush_every == 1 or (idx + 1) % flush_every == 0:
                fout.flush()

            # progress info
            if (idx + 1) % 50 == 0 or idx + 1 == len(ds):
                print(f"✓ {idx + 1} / {len(ds)} translated")

    print("✅ Translation finished – file saved at", output_path)

translate_dataset_hf(
    DATASET,
    prompt_template=TRANSLATION_PROMPT,
    output_path="data_darija_tulu.jsonl",
    pro=True,          # True → gemini-2.0-flash (stronger, slower, pricier)
    max_retries=5,
    base_delay=1.0,
    flush_every=1,      # change to e.g. 10 for faster I/O, but riskier
)

#ds_darija = load_dataset("json", data_files="data_darija.jsonl", split="train")
