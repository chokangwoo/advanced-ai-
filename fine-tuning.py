# @title Base 모델 불러오기
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import os

# 학습 파라미터 설정
max_seq_length = 2048  # 컨텍스트 길이: 2048 (메모리 부족 시 1024로 감소 가능)
dtype = None  # 자동 감지



# GGUF 변환을 고려하여 양자화되지 않은 원본 모델 사용
# 학습 시에만 메모리 절약을 위해 4bit로 로드하지만,
# 저장 시에는 LoRA 어댑터만 저장되어 나중에 원본 모델과 합쳐서 GGUF로 변환 가능
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",  # 양자화되지 않은 원본 모델 사용 (GGUF 변환 가능)
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True, # 학습 시 메모리 절약을 위해 4bit로 로드 (저장 시에는 영향 없음)
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)



# @title LORA 어뎁터 추가
# LoRA 파라미터 설정 (추천 값)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Rank: 16 (8보다 성능 향상, 메모리 사용량 적절) - 필요시 32로 증가 가능
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Alpha: r의 2배 (일반적으로 최적 비율)
    lora_dropout = 0,  # Dropout: 0이 최적화됨
    bias = "none",     # Bias: "none"이 최적화됨
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth",  # 메모리 절약 및 긴 컨텍스트 지원
    random_state = 3407,  # 재현성을 위한 시드
    use_rslora = False,   # Rank Stabilized LoRA: 필요시 True로 변경
    loftq_config = None,  # LoftQ: 필요시 설정
)

# @title 질의응답 데이터셋 로드 함수
def load_qa_dataset(file_path):
    """
    질의응답 데이터셋 txt 파일을 읽어서 학습 가능한 형식으로 변환

    Args:
        file_path: txt 파일 경로 (예: "/content/질의응답 데이터셋.txt")

    Returns:
        Dataset: HuggingFace Dataset 객체
    """
    print(f"데이터셋 파일 로드 중: {file_path}")

    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Q와 A를 분리하여 파싱
    qa_pairs = []
    lines = content.strip().split('\n')

    current_q = None
    current_a = None

    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄은 건너뛰기
            if current_q and current_a:
                qa_pairs.append({
                    'question': current_q,
                    'answer': current_a
                })
                current_q = None
                current_a = None
            continue

        # Q로 시작하는 줄
        if line.startswith('Q') and ':' in line:
            # 이전 Q-A 쌍 저장
            if current_q and current_a:
                qa_pairs.append({
                    'question': current_q,
                    'answer': current_a
                })
            # Q 번호와 내용 분리 (예: "Q1: 질문내용" -> "질문내용")
            parts = line.split(':', 1)
            if len(parts) == 2:
                current_q = parts[1].strip()
                current_a = None

        # A로 시작하는 줄
        elif line.startswith('A') and ':' in line:
            # A 번호와 내용 분리 (예: "A1: 답변내용" -> "답변내용")
            parts = line.split(':', 1)
            if len(parts) == 2:
                current_a = parts[1].strip()

        # Q나 A가 아닌 경우 (답변의 연속 부분일 수 있음)
        elif current_a is not None:
            current_a += " " + line
        elif current_q is not None and current_a is None:
            # Q 다음에 바로 내용이 오는 경우
            current_q += " " + line

    # 마지막 Q-A 쌍 저장
    if current_q and current_a:
        qa_pairs.append({
            'question': current_q,
            'answer': current_a
        })

    print(f"총 {len(qa_pairs)}개의 Q-A 쌍을 찾았습니다.")

    # messages 형식으로 변환 (chat template에 맞게)
    messages_list = []
    for qa in qa_pairs:
        messages = [
            {"role": "user", "content": qa['question']},
            {"role": "assistant", "content": qa['answer']}
        ]
        messages_list.append({"messages": messages})

    # HuggingFace Dataset으로 변환
    dataset = Dataset.from_list(messages_list)

    return dataset

# @title 데이터셋 경로 설정 및 로드
# 데이터셋 파일 경로 (Colab에서 사용 시 경로 수정 필요)
dataset_path = "/content/drive/MyDrive/energpt_data/질의응답 데이터셋.txt"

# 데이터셋 로드
dataset = load_qa_dataset(dataset_path)

# @title 데이터셋 포맷팅 함수
def formatting_prompts_func(examples):
    """
    messages 형식의 데이터를 tokenizer가 처리할 수 있는 텍스트로 변환
    """
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# 데이터셋 포맷팅 적용
dataset = dataset.map(formatting_prompts_func, batched = True)

# @title 데이터셋 샘플 확인
print("\n=== 데이터셋 샘플 ===")
print(f"총 데이터 개수: {len(dataset)}")
if len(dataset) > 0:
    print("\n첫 번째 샘플:")
    print(dataset[0]['text'][:500] + "..." if len(dataset[0]['text']) > 500 else dataset[0]['text'])

# @title 모델학습 SFT Trainer 불러오기
from trl import SFTConfig, SFTTrainer

# 학습 설정 (추천 값으로 최적화)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,  # 배치 크기: 메모리 부족 시 1로 감소 가능
        gradient_accumulation_steps = 4,  # 그래디언트 누적: 효과적인 배치 크기 = 2 * 4 = 8
        warmup_steps = 10,  # 워밍업: 데이터셋 크기의 3-5% 권장 (최소 10)
        num_train_epochs = 3,  # 에포크 수: 일반적으로 3-5 에포크 권장
        # max_steps = 100,  # 스텝 수로 제한하려면 주석 해제하고 원하는 값 지정 (num_train_epochs보다 우선)
        learning_rate = 2e-4,  # 학습률: 2e-4는 LoRA에 적합한 값
        logging_steps = 10,  # 로깅: 10 스텝마다 로그 출력 (더 자주 확인)
        optim = "adamw_8bit",  # 옵티마이저: 8bit AdamW (메모리 효율적)
        weight_decay = 0.01,  # 가중치 감쇠: 정규화를 위한 값
        lr_scheduler_type = "cosine",  # 스케줄러: cosine이 linear보다 일반적으로 더 좋음
        seed = 3407,  # 재현성을 위한 시드
        output_dir = "outputs",  # 출력 디렉토리
        save_steps = 100,  # 저장: 100 스텝마다 체크포인트 저장
        save_total_limit = 3,  # 최대 체크포인트 수: 디스크 공간 절약
        report_to = "none",  # WandB 등 사용 시 "wandb"로 변경
        fp16 = False,  # FP16: 메모리 절약 및 학습 속도 향상
        bf16 = True,  # BF16: A100 등에서 사용 가능 (fp16보다 안정적)
    ),
)

# @title 현재 메모리 상태 확인
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# @title 학습 시작
print("\n=== 학습 시작 ===")
trainer_stats = trainer.train()

# @title 최종 메모리 및 시간 통계 확인
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# @title 모델 실행을 통한 추론 테스트
print("\n=== 추론 테스트 ===")
messages = [
    {"role": "system", "content": "당신은 전력시장운영규칙에 대해 전문적으로 설명하는 도움이 되는 어시스턴트입니다."},
    {"role": "user", "content": "전력시장운영규칙의 가장 기본적인 목적은 무엇인가요?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to(model.device)

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=TextStreamer(tokenizer),
    do_sample=True,
    temperature=0.7,  # 창의성 조절: 0.7은 균형잡힌 값 (0.1-1.0 범위)
    top_p=0.9,  # Nucleus sampling: 더 일관된 출력을 위해 추가
    top_k=50,  # Top-k sampling: 다양성 조절
    pad_token_id=tokenizer.eos_token_id
)

# @title 모델 저장
print("\n=== 모델 저장 ===")
model_path = "/content/drive/MyDrive/energpt"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"모델 저장 완료: {model_path}")



