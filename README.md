# Beyond Context: Enhancing LLM Comprehension through Extended Memory

This repository explores state-of-the-art techniques for extending the context windows of large language models (LLMs). By leveraging architectures such as the Recurrent Memory Transformer and drawing insights from Google's [Inifini Attention](https://arxiv.org/pdf/2404.07143), we aim to deepen our understanding of how extended context windows affect model performance—especially in terms of metrics like context recall, faithfulness, and overall responsiveness.

## Project Objectives

- **Investigate Extended Contexts:**  
  Analyze how large context windows influence LLM performance in retaining, processing, and leveraging extensive inputs.

- **Recurrent Memory Transformer (RMT):**  
  Explore the RMT architecture that employs custom memory cells (`MemoryCell` and `RecurrentWrapper`) to propagate context across segmented inputs.

- **Inifini Attention Insights:**  
  Utilize concepts from Google's Inifini Attention to further enhance context management in LLM systems.

- **Comprehensive Evaluation:**  
  Assess system performance using a variety of metrics to ensure robust and reliable deployment in production environments.

## Evaluation Metrics

Before launching your LLM system into production, it is essential to evaluate it against a suite of critical metrics:

- **Answer Relevancy**  
- **Prompt Alignment**  
- **Correctness**  
- **Hallucination**  
- **Contextual Relevancy**  
- **Responsible Metrics**  
- **Task-Specific Metrics**  

## Highlights

- **Innovative Architecture:**  
  Combines the RMT approach with efficient low-rank adaptation (LoRA) to optimize performance while extending context windows.

- **Robust Experimentation:**  
  Utilizes the Wikitext-2 dataset for extensive experiments, evaluating performance across various memory configurations.

- **Comprehensive Analysis:**  
  Detailed insights and experimental findings are documented in the accompanying report (`NLP_Final_Report.pdf`).

## Quick Start

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Vishwa44/nlp_rmt.git
   ```
2. **Install Dependencies**
Ensure you have Python 3.7+ installed, then run:

```bash
pip install numpy torch tqdm datasets wandb transformers matplotlib
```
# License
This project is licensed under the MIT License. See the LICENSE file for details.
