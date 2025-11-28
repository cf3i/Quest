#!/usr/bin/env python3
"""
Simple test script for Qwen3 with Quest
Usage: python test_qwen3_simple.py --model_path /path/to/qwen3
"""

from transformers import AutoTokenizer
import torch
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3 with Quest")
    parser.add_argument("--model_path", type=str, help="Path to Qwen3 model", default="/home/feic/pjs/model/Qwen3-8B")
    parser.add_argument("--token_budget", type=int, default=1024, help="Token budget for Quest")
    parser.add_argument("--max_length", type=int, default=512, help="Max generation length")
    parser.add_argument("--prompt", type=str, default="In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?", help="Input prompt")
    args = parser.parse_args()

    # Setup
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float16
    torch.set_default_dtype(DTYPE)

    print(f"Loading model from: {args.model_path}")
    print(f"Device: {DEVICE}")
    print(f"Token budget: {args.token_budget}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("✅ Tokenizer loaded successfully")

        # Load model with Quest
        from quest.models.qwen3 import Qwen3ForCausalLM
        from quest.models.llama import LlamaForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(
        # model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            device_map=DEVICE,
            torch_dtype=DTYPE
        )
        print("✅ Model loaded successfully")

        # Initialize Quest Controller
        model.quest_init(
            page_size=16,
            max_seq_len=8192,
            token_budget=args.token_budget,
            dtype=DTYPE,
            device=DEVICE
        )
        print("✅ Quest Controller initialized")

        # Tokenize input
        inputs = tokenizer(args.prompt, return_tensors="pt").to(DEVICE)
        input_length = inputs.input_ids.shape[1]
        print(f"\nInput prompt: {args.prompt}")
        print(f"Input length: {input_length} tokens")

        # Generate
        print("\n" + "="*80)
        print("Generating...")
        print("="*80 + "\n")

        with torch.no_grad():
            generate_ids = model.generate(
                inputs.input_ids,
                max_length=args.max_length,
                use_cache=True,
                do_sample=False,  # Greedy decoding for reproducibility
            )

        # Decode output
        output_text = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(output_text)
        print("\n" + "="*80)
        print(f"✅ Generation completed! (Output length: {generate_ids.shape[1]} tokens)")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
