import argparse
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        if extract_ans.isdigit():
            return int(extract_ans)
        else:
            return None
    else:
        return None

def load_model_and_tokenizer(model_path, lora_adapter_path=None, device='auto'):
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    if lora_adapter_path:
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer

def batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.0, do_sample=False):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    stop_words = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", 
                  "Instruction:", "Instruction", "Response:", "Response"]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_config
        )

    input_length = inputs["input_ids"].shape[1]
    generated_texts = []
    
    for output in outputs:
        generated_part = output[input_length:]
        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
        
        for stop_word in stop_words:
            if stop_word in generated_text:
                generated_text = generated_text.split(stop_word)[0]
        
        generated_texts.append(generated_text.strip())
    
    return generated_texts

def gsm8k_test(model_path, data_path, lora_adapter_path=None, start=0, end=1, batch_size=4, device='auto'):

    model, tokenizer = load_model_and_tokenizer(model_path, lora_adapter_path, device)
    
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    
    gsm8k_questions = []
    gsm8k_answers = []
    
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if start <= idx < end:
                question = problem_prompt.format(instruction=item["question"])
                gsm8k_questions.append(question)
                temp_ans = item['answer'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
                gsm8k_answers.append(temp_ans)
    
    print(f'Loaded {len(gsm8k_questions)} questions for testing')
    results = []
    for i in tqdm(range(0, len(gsm8k_questions), batch_size), desc="Testing"):
        batch_questions = gsm8k_questions[i:i+batch_size]
        batch_answers = gsm8k_answers[i:i+batch_size]
        generated_texts = batch_generate(
            model, tokenizer, batch_questions,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        for question, generated_text, correct_answer in zip(batch_questions, generated_texts, batch_answers):
            predicted_answer = extract_answer_number(generated_text)
            if predicted_answer is not None:
                is_correct = predicted_answer == correct_answer
                results.append(is_correct)
            else:
                results.append(False)
        
    
    accuracy = sum(results) / len(results)
    
    print(f'\n=== Test Results ===')
    print(f'Total questions: {len(results)}')
    print(f'Correct answers: {sum(results)}')
    print(f'Accuracy: {accuracy*100:.2f}%')

def main():
    parser = argparse.ArgumentParser(description="GSM8K Test with Transformers")
    parser.add_argument("--model", type=str, default="../models/meta-llama/Llama-2-7b-hf", 
                       help="Model path (base model or merged model)")
    parser.add_argument("--lora_adapter", type=str, default="./lora_model/checkpoints/epoch_3",
                       help="LoRA adapter path")
    parser.add_argument("--data_file", type=str, default="./gsm8k_test.jsonl",
                       help="GSM8K test data file")
    parser.add_argument("--start", type=int, default=0, 
                       help="Start index")
    parser.add_argument("--end", type=int, default=100, 
                       help="End index")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for model (auto, cpu, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    print("=== GSM8K Test Configuration ===")
    print(f"Model: {args.model}")
    print(f"LoRA adapter: {args.lora_adapter}")
    print(f"Data file: {args.data_file}")
    print(f"Test range: {args.start} to {args.end}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 35)
    
    gsm8k_test(
        model_path=args.model,
        data_path=args.data_file,
        lora_adapter_path=args.lora_adapter,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        device=args.device
    )
        
if __name__ == "__main__":
    main()