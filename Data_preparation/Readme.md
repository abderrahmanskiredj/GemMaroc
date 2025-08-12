## Step 1 – Clean the Tulu MIG 50k dataset

We keep only English utterances and remove any translation instructions.
For each dialog:

* Retain only the first 2,048 tokens.
* Discard any exchanges beyond this limit.
* If truncation would result in a partial (incomplete) utterance, remove that utterance entirely to avoid cutting it mid-sentence.

Refer to **`Filter_tulu.ipynb`** for implementation details.

Here’s a cleaner, more precise version of your steps:

---

## Step 2 – Perform the Translation

Translate 30% of the dataset into **Darija**, keeping the remaining 70% in **English**.

---

## Step 3 – Apply Chat Templates and Enforce Token Limit

* Apply the Gemma chat formatting/templates.
* Ensure the total length of instructions does not exceed **2,048 tokens**.
* Retain only the first 2,048 tokens in each dialog.
* Discard any exchanges beyond this limit.
* If truncation results in an incomplete utterance, remove that utterance entirely to avoid mid-sentence cuts.
