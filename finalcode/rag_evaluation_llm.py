import os
import re
import torch
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np  # [추가] 누락된 라이브러리
from tqdm import tqdm # [추가] 누락된 라이브러리



def load_llama_judge(model_name: str = 'meta-llama/Llama-3.2-3B-Instruct'):
    """
    로컬/서버 환경에서 Llama Judge 모델을 로드합니다.
    """
    print(f'🤖 Loading {model_name}...')

    # [수정] float32 -> bfloat16 (메모리 절약 및 속도 향상)
    # GPU가 Ampere 아키텍처(RTX 30xx, A100 등) 이상이면 bfloat16, 그 외엔 float16 권장
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    
    # [추가] Pad Token 설정 (경고 방지)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'✓ Model loaded (dtype: {dtype})')
    return model, tokenizer


def llm_judge_accuracy_llama(query: str, generated_answer: str, ground_truth: str, model, tokenizer, max_new_tokens: int = 256):
    """
    Llama를 사용한 답변 정확도 평가 (0-10점)
    """

    prompt = f"""You are an expert evaluator for question-answering systems.

Rate the accuracy of the predicted answer compared to the ground truth answer on a scale of 0-10.

Scoring guidelines:
- 10: Perfect match or semantically identical
- 8-9: Correct with minor wording differences
- 6-7: Mostly correct but missing some details
- 4-5: Partially correct
- 2-3: Incorrect but somewhat related
- 0-1: Completely wrong or unrelated

Question: {query}
Ground Truth: {ground_truth}
Predicted Answer: {generated_answer}

Provide your evaluation in this exact format:
Score: [number from 0-10]
Reasoning: [brief explanation in 1-2 sentences]
"""

    messages = [
        {'role': 'system', 'content': 'You are a precise and objective evaluator.'},
        {'role': 'user', 'content': prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # [수정] 점수 파싱 로직 강화 (Markdown bold 처리 등 대응)
    score_match = re.search(r'Score:\s*\*?(\d+(?:\.\d+)?)\*?', response, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else 0.0 # 못 찾으면 0점 처리 (안전하게)
    score = max(0.0, min(10.0, score))

    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()

    return {'score': score, 'reasoning': reasoning, 'raw_response': response}


def evaluate_qa_accuracy_llama(json_path: str, output_path: str, model_name: str = 'meta-llama/Llama-3.2-3B-Instruct'):
    """
    Llama를 사용한 QA 정확도 평가
    """
    # [추가] 문자열 경로를 Path 객체로 변환
    json_path = Path(json_path)
    output_path = Path(output_path)

    print('='*80)
    print('🤖 LLM Judge - Llama Evaluation')
    print('='*80)
    print(f'Model: {model_name}')
    print(f'Input: {json_path}')
    print(f'Output: {output_path}')
    print('='*80)

    model, tokenizer = load_llama_judge(model_name)

    with open(json_path, 'r', encoding='utf-8') as f:
        qa_records = json.load(f)

    print(f'✓ Loaded {len(qa_records)} QA records')

    evaluated_results = []

    for idx, record in enumerate(tqdm(qa_records, desc='Evaluating')):
        # [중요 수정] 이전 단계(RAG Pipeline)의 JSON Key와 일치시킴
        # 이전 코드 저장 키: "question", "predicted_answer", "ground_truth"
        query = record.get('question', record.get('query', '')) 
        generated_answer = record.get('predicted_answer', record.get('generated_answer', record.get('answer', '')))
        ground_truth = record.get('ground_truth', '')

        try:
            accuracy_eval = llm_judge_accuracy_llama(query, generated_answer, ground_truth, model, tokenizer)
        except Exception as e:
            print(f'Error at index {idx}: {e}')
            accuracy_eval = {'score': 0.0, 'reasoning': f'Error: {str(e)}', 'raw_response': ''}

        evaluated_results.append({
            'id': idx,
            'question': query,
            'predicted_answer': generated_answer,
            'ground_truth': ground_truth,
            'accuracy_score': accuracy_eval['score'],
            'reasoning': accuracy_eval['reasoning']
        })

        # 중간 저장 (10개마다)
        if (idx + 1) % 10 == 0:
            with open(output_path.with_suffix('.tmp.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluated_results, f, ensure_ascii=False, indent=2)

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, ensure_ascii=False, indent=2)

    scores = [r['accuracy_score'] for r in evaluated_results]

    print('' + '='*80)
    print('📊 Evaluation Summary')
    print('='*80)
    if scores:
        print(f'Total samples: {len(scores)}')
        print(f'Average score: {np.mean(scores):.2f} / 10')
        print(f'Std deviation: {np.std(scores):.2f}')
        print(f'Min score: {min(scores):.2f}')
        print(f'Max score: {max(scores):.2f}')
    else:
        print("No scores to evaluate.")
    print('='*80)
    print(f'✅ Results saved to: {output_path}')

    return evaluated_results

# %% 실행 예시
# 경로가 맞는지 확인 후 실행하세요.
if __name__ == "__main__":
    # 1. Baseline 평가
    # 2. Ours 평가
    if os.path.exists('title_full_results.json'): # 파일명 확인 필요 (예: rag_baseline_results.json과 구분이 필요하다면)
        results_ours = evaluate_qa_accuracy_llama(
            json_path='title_full_results.json', # 파일명을 본인의 설정에 맞게 변경하세요
            output_path='evaluation_results_title_full.json',
            model_name='meta-llama/Llama-3.2-1B-Instruct'
        )
