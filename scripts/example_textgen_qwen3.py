from transformers import AutoTokenizer
import torch
import argparse

# Update this to your Qwen3 model path
MODEL_PATH = "/home/feic/pjs/model/Qwen3-8B"  # Change this to your Qwen3 model path
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

RUNTIME_CFGS = [
    "quest",
    "hg",
]

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=RUNTIME_CFGS, default="quest")
parser.add_argument("--token_budget", type=int, default=1024)
args = parser.parse_args()

if args.method == "quest":
    from quest.models.qwen3 import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)

    # Init Quest Controller
    print("Initializing Quest Controller...")
    model.quest_init(page_size=16, max_seq_len=8192, token_budget=args.token_budget)
    print("Quest Controller initialized successfully!")
else:
    # Use HuggingFace's transformers (if Qwen3 is available in transformers)
    # Note: Qwen3 might not be in official transformers yet
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)
    except Exception as e:
        print(f"Error loading Qwen3 from transformers: {e}")
        print("Using Quest Qwen3 implementation instead...")
        from quest.models.qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)

# First Round - Testing Qwen3's Q/K Normalization
prompt = "在一个动物王国里，狮子是国王。一天，狮子宣布举办一场竞赛，选出最勤劳的动物。乌龟、兔子、猴子、斑马和长颈鹿都决定参加。经过一天的观察，狮子注意到所有动物都在努力工作，除了兔子在睡觉。那么为什么狮子选择兔子作为最勤劳的动物？"

# Alternative English prompt
# prompt = "In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?"

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
print(f"Input Sequence Length: {inputs.input_ids.shape[1]}")

print("\n" + "="*80)
print("Generating response with Qwen3 + Quest...")
print("="*80 + "\n")

generate_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=2048,
    use_cache=True,  # Managed by our InferenceController
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
)

output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)

print("\n" + "="*80)
print("Generation completed successfully!")
print("="*80)

# Optional: Test multi-turn conversation with quest_clear()
if args.method == "quest":
    print("\n" + "="*80)
    print("Testing multi-turn conversation with quest_clear()...")
    print("="*80 + "\n")

    # Clear KV cache for new conversation
    model.quest_clear()

    # Second round
    prompt2 = "你好，请介绍一下自己。"
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(DEVICE)
    print(f"Second Round - Input Sequence Length: {inputs2.input_ids.shape[1]}")

    generate_ids2 = model.generate(
        inputs2.input_ids,
        attention_mask=inputs2.attention_mask,
        max_length=512,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    output_text2 = tokenizer.batch_decode(generate_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output_text2)

    print("\n" + "="*80)
    print("Multi-turn conversation test completed!")
    print("="*80)
