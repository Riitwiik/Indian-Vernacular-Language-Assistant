"""

Indian Multilingual Assistant - Gradio Inference App


This app loads the base Llama-3.2-1B-Instruct model together with
the LoRA adapter trained in `training.ipynb`, and serves it through
a simple Gradio web interface supporting 4 Indian languages.

Run locally:   python app.py
Run in Docker: docker build -t indian-assistant . && docker run -p 7860:7860 indian-assistant

The app automatically:
  - Detects GPU (falls back to CPU if unavailable)
  - Handles CUDA Out-Of-Memory gracefully
  - Uses the latest Transformers generation API
"""

import os
import gc
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
load_dotenv()


torch.set_num_threads(8)      # or 4 if you have a 4-core CPU
torch.set_num_interop_threads(2)
print(torch.get_num_threads())
# Base model used during fine-tuning
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Path to the LoRA adapter saved by training.ipynb
# Override with the ADAPTER_PATH environment variable if needed.
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./lora_adapters")

# Hugging Face token (required because Llama is a gated model)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
print("HF_TOKEN loaded:", bool(HF_TOKEN))

# Languages supported by the assistant
LANGUAGES = [
    "English",
    "Hindi",
    "Marathi",
    "Tamil",
    "Telugu",
    "Bengali",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
]



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Compute dtype: {COMPUTE_DTYPE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")



def load_model_and_tokenizer():
    """
    Load the base Llama model in 4-bit precision (when GPU is available)
    and attach the LoRA adapter if one exists at ADAPTER_PATH.
    """
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    # Llama has no default pad token; reuse EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config (only used on GPU)
    bnb_config = None
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=COMPUTE_DTYPE if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    # Enable KV cache for faster inference
    model.config.use_cache = True
    

    # Attach the LoRA adapter if it exists
    if os.path.isdir(ADAPTER_PATH) and os.path.exists(
        os.path.join(ADAPTER_PATH, "adapter_config.json")
    ):
        print(f"[INFO] Loading LoRA adapter from {ADAPTER_PATH} ...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("[OK] LoRA adapter loaded.")
    else:
        print(f"[WARN] No LoRA adapter found at {ADAPTER_PATH}.")
        print("       Using base model only. Run training.ipynb first to train an adapter.")

    model.eval()
    return model, tokenizer


# Load once at startup
model, tokenizer = load_model_and_tokenizer()



def generate_response(instruction, language, temperature, max_tokens):
    """
    Generate a response for the given instruction.

    Args:
        instruction: The user's question or command.
        language:    The language to respond in.
        temperature: Sampling temperature (0.1 = safe, 1.5 = creative).
        max_tokens:  Maximum number of new tokens to generate.

    Returns:
        The model's response as a string.
    """
    # Guard against empty input
    if not instruction or not instruction.strip():
        return "Please enter an instruction first."

    try:
        # Build the prompt. If a non-English language is selected, we
        # prepend a small instruction asking the model to respond in that language.
        if language != "English":
            full_instruction = f"Respond in {language}. {instruction}"
        else:
            full_instruction = instruction

        prompt = f"### Instruction:\n{full_instruction}\n\n### Response:\n"

        # Tokenize and move to the model's device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate - using the latest Transformers generation API.
        # `do_sample=False` when temperature is very low, to make
        # responses deterministic.
        do_sample = float(temperature) > 0.01

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature) if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clear GPU cache to keep memory usage low across calls
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return response if response else "(model produced an empty response)"

    except torch.cuda.OutOfMemoryError:
        # Gracefully handle GPU OOM instead of crashing the app
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return (
            "GPU ran out of memory. Try reducing 'Maximum Tokens' "
            "or restart the app with a smaller model."
        )
    except Exception as e:
        return f"Error during generation: {type(e).__name__}: {str(e)}"


def clear_inputs():
    """Clear both the instruction and response boxes."""
    return "", ""



with gr.Blocks(title="Indian Multilingual Assistant") as demo:
    gr.Markdown(
        "# Indian Multilingual Assistant\n"
        "Fine-tuned **Llama-3.2-1B-Instruct** with QLoRA, supporting "
        "10 Indian languages.\n\n"
        "Enter any instruction, pick a response language, and click Generate."
    )

    with gr.Row():
        # ---------- Left column: inputs ----------
        with gr.Column(scale=1):
            instruction_box = gr.Textbox(
                label="Instruction",
                placeholder=(
                    "e.g. Explain what machine learning is in one simple sentence."
                ),
                lines=4,
                max_lines=10,
            )
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Response Language",
                info="The model will respond in this language.",
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Lower = focused, Higher = creative.",
            )
            max_tokens_slider = gr.Slider(
                minimum=32,
                maximum=512,
                value=256,
                step=32,
                label="Maximum Tokens",
                info="Maximum length of the generated response.",
            )
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")

        
        with gr.Column(scale=1):
            response_box = gr.Textbox(
                label="Response",
                lines=15,
                max_lines=30,
                interactive=False,
                placeholder="The model's response will appear here.",
            )

    # Wire up buttons
    generate_btn.click(
        fn=generate_response,
        inputs=[instruction_box, language_dropdown, temperature_slider, max_tokens_slider],
        outputs=response_box,
    )
    clear_btn.click(
        fn=clear_inputs,
        outputs=[instruction_box, response_box],
    )

    # Allow pressing Enter in the instruction box to trigger generation
    instruction_box.submit(
        fn=generate_response,
        inputs=[instruction_box, language_dropdown, temperature_slider, max_tokens_slider],
        outputs=response_box,
    )



if __name__ == "__main__":
    # server_name="0.0.0.0" makes the app accessible from outside the container
    # (required for Docker and Hugging Face Spaces).
    #demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
    )
