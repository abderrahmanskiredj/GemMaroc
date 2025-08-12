# 🌍💬 GemMaroc — State-of-the-Art Moroccan Darija LLM

<p align="center">
  <img src="assets/gemmaroc_logo.png" alt="GemMaroc Banner" width="400">
</p> <!-- Optional if you have a banner image -->

**GemMaroc** is an **open-source, instruction-tuned** Large Language Model for **Moroccan Arabic (Darija)**, built on **Gemma** and fine-tuned with an *efficient, low-carbon* methodology.

In just **2.5 months** since its release (**May 22, 2025**), **GemMaroc** has been downloaded **16,733+ times** — proving the demand for inclusive AI that empowers underrepresented languages.

---

## 🚀 Highlights

- **State-of-the-art** performance on Darija benchmarks:
  - **61.6%** on *DarijaMMLU* (matches the best published results)
  - **62.33%** on *DarijaHellaSwag* (**+12.1** points over Atlas-Chat)
- **Balanced bilingual reasoning** — retains strong English reasoning (84.2% GSM8K)
- **Ultra-efficient training**:
  - Just **≈50 GPU·hours on H100** (~**$200** on standard cloud)
  - ~**2%** of the energy footprint of comparable baselines
- **Completely open**:
  - Models, datasets, training scripts, and evaluation code — all available

---

## 📊 Full Evaluation Results

\begin{table*}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccccccccccccccc}
\hline
Language                          & \multicolumn{9}{c}{Darija}                                                                                                                                                                        & \multicolumn{5}{c}{English}                                                                                                                                  \\ \cline{2-15} 
\multirow{2}{*}{Model/Benchmark}  & Size (B) & DarijaMMLU     & DarijaHellaSwag & Sentiment Analysis & \begin{tabular}[c]{@{}c@{}}GSM8K-\\ darija-gen\end{tabular} & \multicolumn{4}{c}{Summarization}                                & MMLU           & TruthfulQA     & HellaSwag      & \begin{tabular}[c]{@{}c@{}}GSM8K\\ @5\end{tabular} & \begin{tabular}[c]{@{}c@{}}GSM8K-\\ gen\end{tabular} \\
                                  & Metrics  & Acc            & Acc             & Acc                & Acc                                                         & chrF           & ROUGE-1        & ROUGE-L       & BERTScore      & Acc            & bleu\_acc      & Acc            & Acc                                                & Acc                                                  \\ \hline
Gemma3-4b-it                      & 4        & 32.8           & 36.3            & 58.94              & 52.77                                                       & 27.22          & 8.38           & 8.19          & 37.23          & 51.1           & 40.88          & 47.65          & 74.75                                              & 89.08                                                \\ \hline
\textbf{GemMaroc-4b-LIMA (ours)}  & 4        & 34.95          & 39.27           & 42.1               & 30.05                                                       & 26.14          & 6.95           & 7.04          & 34.32          & 29.28          & 40.15          & 44.21          & 51.24                                              & 65.42                                                \\ \hline
\textbf{GemMaroc-4b-Deita (ours)} & 4        & 42.67          & 44.26           & 60.8               & 34.12                                                       & 27.16          & 7.4            & 7.34          & 38.48          & 51.35          & 44.55          & 68.97          & 53.15                                              & 61.64                                                \\ \hline
\textbf{GemMaroc-4b-Tulu (ours)}  & 4        & 47.53          & 47.13           & 53.29              & 37.91                                                       & 28.46          & 8.89           & 8.76          & 39.27          & 54.14          & 43.33          & 73.95          & 55.95                                              & 71.57                                                \\ \hline
ALLaM-Instruct-7b                 & 7        & 59.49          & 50.09           & 47.33              & 40.33                                                       & 10.27          & 1.68           & 1.68          & 12.28          & 58.31          & 42.11          & 75.2           & 49.28                                              & 68.61                                                \\ \hline
Atlas-Chat-9B                     & 9        & 58.32          & 43.65           & 81.85              & 66.69                                                       & 32.07          & 9.5            & 9.45          & 47             & 69.09          & \textbf{67.56} & 73.35          & 73.01                                              & 77.03                                                \\ \hline
Atlas-Chat-27B                    & 27       & \textbf{61.95} & 48.37           & \textbf{73}        & 71.04                                                       & \textbf{32.75} & \textbf{10.53} & 10.42         & \textbf{47.82} & 72.06          & 43.82          & 77.84          & 82.03                                              & 82.34                                                \\ \hline
Gemma-3-27b-it                    & 27       & 55.65          & 49.13           & 60.27              & 82.71                                                       & 28.33          & 10.28          & 9.95          & 38.17          & \textbf{78.12} & \textbf{63.05} & \textbf{86.02} & \textbf{95.9}                                      & \textbf{95.60}                                       \\ \hline
\textbf{GemMaroc-27b-Tulu (ours)} & 27       & \textbf{61.61} & \textbf{62.33}   & 59.25              & \textbf{84.46}                                              & 28.34          & 9              & \textbf{11.2} & \textbf{39.5}  & \textbf{73.6}  & 55.45          & \textbf{82.06} & \textbf{84.23}                                     & \textbf{93.18}                                       \\ \hline
\end{tabular}
}
\caption{Unified leaderboard comprising: (1) Darija-centric benchmarks, and (2) English-centric benchmarks across reasoning, mathematics, and truthfulness.}
\label{end_results}
\end{table*}

---

## 📂 Repository Structure

```

GemMaroc/
│
├── training/         # LoRA fine-tuning scripts & configs
├── evaluation/       # Automated evaluation pipelines & benchmarks
├── data/             # Scripts for dataset preparation & translation
├── results/          # Leaderboards, metrics, and plots
├── demos/            # Example usage
└── README.md

```

---

## 🛠️ How to Use in LM Studio

To run the model locally in [LM Studio](https://lmstudio.ai):

1. Open LM Studio.
2. Go to **Models** → **Download Model**.
3. Search for:

```

AbderrahmanSkiredj1/GemMaroc-27b-it-GGUF

````

4. Download the desired quantization variant.
5. Load the model and start chatting in Darija or English.

---

## 📚 Training Details

* **Base model**: Gemma 3-27B
* **Method**: LoRA fine-tuning on high-quality translated & reasoning-rich instructions (TULU)
* **Compute**: ≈50 GPU·h (H100), cost ≈ \$200
* **Green AI**: ~26 kWh energy use (~10 kg CO₂e)

---

## 📥 Downloads

* **Total downloads across all variants**: **16,733+** in the first 2.5 months

---

## 📜 Citation

If you use GemMaroc in your research, please cite:

```bibtex
@misc{gemmaroc2025,
  title   = {GemMaroc: Unlocking Darija Proficiency in LLMs with Minimal Data},
  author  = {Skiredj, Abderrahman et al.},
  year    = {2025}
}
````

---

## 🤝 Contributing

We welcome contributions! Whether it's:

* Adding new Darija benchmarks
* Improving training scripts
* Expanding datasets

Fork the repo, make your changes, and submit a pull request.

---

### 💡 Final Note

GemMaroc proves that **quality over quantity** works — delivering **state-of-the-art Darija** with minimal resources, full transparency, and open access.
We hope it inspires more work for underrepresented languages worldwide.
