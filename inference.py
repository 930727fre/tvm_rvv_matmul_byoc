import numpy as np
import tvm
from tvm import relax, runtime
from tvm.relax import VirtualMachine
from transformers import WhisperProcessor, WhisperTokenizer

from datetime import datetime
import csv
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy import signal

# Global aggregation dict: {Name: [total_duration_us, total_count]}
profile_agg = defaultdict(lambda: [0, 0])
start_time_all = datetime.now()
print("Start of all:", start_time_all)

# === 初始化空的 16 個 KV（給 prefill 和 step-by-step 共用）===
def init_zero_past_kv(num_layers=4, num_heads=6, head_dim=64,
                      decoder_seq_len=0, encoder_seq_len=1500, dtype="float32"):
    shape_decoder = (1, num_heads, decoder_seq_len, head_dim)
    shape_encoder = (1, num_heads, encoder_seq_len, head_dim)
    kvs = []
    for _ in range(num_layers):
        kvs += [
            tvm.nd.array(np.zeros(shape_decoder, dtype=dtype)),  # self.key
            tvm.nd.array(np.zeros(shape_decoder, dtype=dtype)),  # self.value
            tvm.nd.array(np.zeros(shape_encoder, dtype=dtype)),  # cross.key
            tvm.nd.array(np.zeros(shape_encoder, dtype=dtype))   # cross.value
        ]
    return kvs

def insert_profile_report(csv_str):
    """Insert one profile report (CSV string format) into the global aggregator."""
    reader = csv.DictReader(csv_str.strip().splitlines())
    for row in reader:
        name = row["Name"]
        duration = float(row["Duration (us)"])
        count = int(row["Count"])
        profile_agg[name][0] += duration
        profile_agg[name][1] += count

# === 載入模型與 tokenizer ===
processor = WhisperProcessor.from_pretrained("./")
tokenizer = WhisperTokenizer.from_pretrained("./")

# === 音訊轉 mel spectrogram ===


# === 1. Load audio ===
waveform, sr = sf.read("audio.wav")

# === 2. Resample to 16kHz if needed ===
target_sr = 16000
if sr != target_sr:
    num_samples = int(len(waveform) * target_sr / sr)
    waveform = signal.resample(waveform, num_samples)
    sr = target_sr

# === 3. Convert to mono ===
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

# === 4. Pass into Hugging Face processor (same as torchaudio flow) ===
inputs = processor(waveform, sampling_rate=16000, return_tensors="np")

# === 5. Get float32 mel features ===
mel = inputs.input_features.astype("float32")

print("Mel shape:", mel.shape)


# === Encoder ===
encoder_vm = VirtualMachine(runtime.load_module("./onnx/encoder_model.so"), tvm.cpu(), profile=True)

# === Profile code block ===

# Profile the encoder execution
# profile_report = encoder_vm.profile("main", tvm.nd.array(mel)).csv()
# insert_profile_report(profile_report)
# Convert to CSV and save to file
# with open("./profile_data/encoder.csv", "w") as f:
#     f.write(profile_report)

# === End of Profile code block ===

start_time = datetime.now()
print("Start of encoder:", start_time)
encoder_out = encoder_vm["main"](tvm.nd.array(mel))  # shape: (1, 1500, 384)
end_time = datetime.now()
print("End of encoder:", end_time)
print("Encoder takes: ", (end_time-start_time).total_seconds())

# === Decoder Step 0: Prefill ===
# Initialize decoder VM with profiling enabled


start_token = 50258
eos_token = tokenizer.eos_token_id
tokens = [start_token]
input_ids = np.array([[start_token]], dtype="int64")
past_kvs = init_zero_past_kv()
inputs = [tvm.nd.array(input_ids), encoder_out]

decoder_prefill_vm = VirtualMachine(
    runtime.load_module("./onnx/decoder_model.so"), 
    tvm.cpu(), 
    profile=True
)

# Initialize empty KV (self + cross) for prefill decoder

# === Decoder profiling ===


# print("\n=== Step 0 (Prefill) - Profiling ===")
# profile_report = decoder_prefill_vm.profile("main", *inputs).csv()
# insert_profile_report(profile_report)
# # Save CSV-formatted profiling report
# with open("./profile_data/decoder_prefill.csv", "w") as f:
#     f.write(profile_report)
# === End of Decoder profiling ===

# Get the actual output (without profiling)
start_time = datetime.now()
print("Start of decoder prefill:", start_time)
out = decoder_prefill_vm["main"](*inputs)
end_time = datetime.now()
print("End of decoder prefill:", end_time)
print("Decoder prefill takes: ", (end_time-start_time).total_seconds())


logits = out[0].numpy()
next_token = int(np.argmax(logits[0, -1]))

assert logits.ndim == 2 or logits.ndim == 3, "logits 維度不符"

tokens.append(next_token)
# print(f"⬆️ Next token: {next_token} ({tokenizer.decode([next_token])})")

# 將 decoder 回傳的 16 個 KV 擷取出來
decoder_kvs = list(out[1:])  # out[1]~out[16]

if next_token == eos_token:
    print("🛑 遇到 <eos>，結束解碼")
    transcript = tokenizer.decode(tokens, skip_special_tokens=True)
    print("\n📝 Transcription:\n", transcript)
    exit()

# === Decoder Step 1~N: step-by-step 解碼 ===

# === Decoder profiling ===
decoder_vm = VirtualMachine(
    runtime.load_module("./onnx/decoder_with_past_model.so"), 
    tvm.cpu(),
    profile=True  # Enable profiling
)
# === Decoder profiling ===



max_length = 64
all_reports = []  # Store all profiling reports

start_time = datetime.now()
print(f"Start of decoder token generation: {start_time}")

for step in range(1, max_length):
    # print(f"\n=== Step {step} ===")
    input_ids = np.array([[tokens[-1]]], dtype="int64")
    inputs = [tvm.nd.array(input_ids)] + decoder_kvs

    # Profile every step (optional: skip warm-up steps)
    # === Decoder profiling ===

    # profile_report = decoder_vm.profile("main", *inputs).csv()
    # insert_profile_report(profile_report)

    # Save with step number in filename

    # with open(f"./profile_data/decoder_step_{step}.csv", "w") as f:
    #     f.write(profile_report)
    # === Decoder profiling ===



    
    # Normal execution

    out = decoder_vm["main"](*inputs)
    end_time = datetime.now()


    logits = out[0].numpy()
    next_token = int(np.argmax(logits[0, -1]))
    tokens.append(next_token)
    # print(f"⬆️ Next token: {next_token} ({tokenizer.decode([next_token])})")

    if next_token == eos_token:
        print("🛑 遇到 <eos>，結束解碼")
        break

    # Update self-attention positions (index 0,1,4,5,8,9,12,13)
    for i, dst_idx in enumerate([0,1,4,5,8,9,12,13]):
        decoder_kvs[dst_idx] = out[i + 1]

print(f"End of decoder token generation: {end_time}")
print(f"Decoder token generation takes: {(end_time-start_time).total_seconds()}")

# === 最後輸出結果 ===
transcript = tokenizer.decode(tokens, skip_special_tokens=True)
print("\n📝 Transcription:\n", transcript)

"""Write aggregated results to a CSV file."""
# === profiling data aggregation === 
# with open("./profile_data/aggregation.csv", "w") as f:
#     f.write("Name,Total Duration (us),Total Count\n")
#     for name, (duration, count) in sorted(profile_agg.items(), key=lambda x: -x[1][0]):
#         f.write(f"{name},{duration},{count}\n")
# === profiling data aggregation === 
end_time_all = datetime.now()
print("End of all:", end_time_all)
print("All takes: ", (end_time_all-start_time_all).total_seconds())