# Indian-Vernacular-Language-Assistant using Llama-3.2-1B-Instruct


A multilingual AI assistant fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)** on **Meta Llama-3.2-1B-Instruct** to support multiple Indian languages.

The project demonstrates an end-to-end Generative AI workflow including dataset preparation, parameter-efficient fine-tuning, LoRA adapter training, inference, and deployment through a Gradio web application.


## 🚀 Features

- Fine-tuned Meta Llama-3.2-1B-Instruct
  
- Parameter-Efficient Fine-Tuning using QLoRA
  
- LoRA Adapter-based inference
- 
- Supports multiple Indian languages:
  - English
  - 
  - Hindi
  - 
  - Marathi
  - 
  - Tamil
- Interactive Gradio Interface
- 
- Automatic CPU/GPU detection
- 
- Hugging Face integration
- 
- Memory-efficient inference
- 
- Configurable generation parameters
## 🏗️ Project Architecture


                 User
                  │
                  ▼
          Gradio Web Interface
                  │
                  ▼
        Prompt Construction
                  │
                  ▼
      Llama-3.2-1B-Instruct
                  │
          + LoRA Adapter
                  │
                  ▼
        Generated Response
