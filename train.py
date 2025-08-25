import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import prepare_model_for_kbit_training
from tqdm.auto import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_path: str = "meta-llama/Llama-2-7b-hf"
    max_seq_len: int = 1024
    
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

    dataset_name: str = "meta-math/MetaMathQA"
    max_samples: int = -1
    
    epochs: int = 3
    batch_size: int = 4
    accumulation_steps: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    output_path: str = "./trained_model"
    save_interval: int = 1000
    log_interval: int = 50
    use_fp16: bool = True
    random_seed: int = 42

class MathDataProcessor:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    
    def load_data(self, dataset_name: str = 'meta-math/MetaMathQA', max_samples: int = -1) -> List[Dict]:
        logger.info(f"Loading dataset: {dataset_name}")
        math_dataset = load_dataset(dataset_name)['train'].to_list()
        data = []
        for item in math_dataset:
            processed_item = {
                'instruction': item['query'].split('\n')[0], 
                'output': item['response']
            }
            data.append(processed_item)
        if max_samples > 0:
            data = data[:max_samples]
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def prepare_training_data(self, data_list: List[Dict]) -> Tuple[List[str], List[str]]:
        sources = []
        targets = []
        for example in data_list:
            source = self.prompt_template.format_map(example)
            sources.append(source)
            target = f"{example['output']}{self.tokenizer.eos_token}"
            targets.append(target)
        
        return sources, targets
    
    def tokenize_sample(self, text: str, is_target: bool = False) -> Dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        if is_target:
            labels = encoded['input_ids'].copy()
            labels = [-100 if token_id == self.tokenizer.pad_token_id else token_id 
                     for token_id in labels]
            encoded['labels'] = labels
            
        return encoded

class MathDataset(Dataset):
    def __init__(self, data_list: List[Dict], processor: MathDataProcessor):
        self.processor = processor
        self.sources, self.targets = processor.prepare_training_data(data_list)
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> List[Dict]:
        samples = []
        for source, target in tqdm(zip(self.sources, self.targets), 
                                  desc="Processing samples", 
                                  total=len(self.sources)):
            try:
                full_text = source + target
                source_tokens = self.processor.tokenize_sample(source)
                full_tokens = self.processor.tokenize_sample(full_text)
                labels = full_tokens['input_ids'].copy()
                source_length = len(source_tokens['input_ids'])
                for i in range(min(source_length, len(labels))):
                    labels[i] = -100
                sample = {
                    'input_ids': full_tokens['input_ids'],
                    'attention_mask': full_tokens['attention_mask'],
                    'labels': labels
                }
                
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
        
        logger.info(f"Successfully processed {len(samples)} samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

class CustomCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(len(sample['input_ids']) for sample in batch)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for sample in batch:
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            labels = sample['labels']
            pad_length = max_length - len(input_ids)
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            padded_attention_mask = attention_mask + [0] * pad_length
            padded_labels = labels + [-100] * pad_length
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long)
        }

class Trainer:
    def __init__(self, config: Config, model, tokenizer, train_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.collator = CustomCollator(tokenizer)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=4,
            pin_memory=True
        )
        self._setup_optimization()
        self.scaler = torch.amp.GradScaler() if config.use_fp16 else None
        self.global_step = 0
        self.current_epoch = 0
    
    def _setup_optimization(self):
        total_steps = (len(self.train_loader) * self.config.epochs // self.config.accumulation_steps)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.lr,
            eps=1e-8
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def compute_loss(self, outputs, labels):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.config.use_fp16:
                with torch.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    loss = self.compute_loss(outputs, batch['labels'])
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                loss = self.compute_loss(outputs, batch['labels'])
            
            loss = loss / self.config.accumulation_steps

            if self.config.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % self.config.log_interval == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_loss = total_loss / self.config.log_interval
                    
                    logger.info(
                        f"Step {self.global_step}: "
                        f"Loss={avg_loss:.4f}, LR={current_lr:.2e}"
                    )
                    total_loss = 0.0
                
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.accumulation_steps:.4f}",
                'step': self.global_step
            })
        
    
    def train(self):
        logger.info("Starting training...")
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.save_checkpoint(f"epoch_{epoch + 1}")

    
    def save_checkpoint(self, checkpoint_name: str):
        checkpoint_dir = Path(self.config.output_path) / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.config.use_lora:
            self.model.save_pretrained(checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save({
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

def setup_model_and_tokenizer(config: Config):
    logger.info(f"Loading model: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.max_seq_len,
        padding_side="right",
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        logger.info("Applying LoRA configuration...")
        model = prepare_model_for_kbit_training(model)
        target_modules = [m.strip() for m in config.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"LoRA applied with r={config.lora_r}, alpha={config.lora_alpha}")
        logger.info(f"Target modules: {target_modules}")
    
    return model, tokenizer

def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="MetaMath Fine-tuning with LoRA")
    
    parser.add_argument("--model_path", type=str, default="../models/meta-llama/Llama-2-7b-hf",
                       help="Path to the base model")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, 
                       default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                       help="LoRA target modules (comma separated)")
    parser.add_argument("--dataset_name", type=str, default="meta-math/MetaMathQA",
                       help="HuggingFace dataset name")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to use")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate (higher for LoRA)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--output_path", type=str, default="./lora_model",
                       help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=50,
                       help="Log every N steps")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--random_seed", type=int, default=2024,
                       help="Random seed")
    
    args = parser.parse_args()
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    return config

def main():
    config = parse_arguments()
    torch.manual_seed(config.random_seed)
    model, tokenizer = setup_model_and_tokenizer(config)
    processor = MathDataProcessor(tokenizer, config.max_seq_len)
    data_list = processor.load_data(config.dataset_name, config.max_samples)
    train_dataset = MathDataset(data_list, processor)
    trainer = Trainer(config, model, tokenizer, train_dataset)
    trainer.train()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()