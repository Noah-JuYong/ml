"""
실습 1: 프롬프트 구성 요소별 결과 비교
- 동일 주제(텀블러 SNS)에 Role → Context → Task → Format 을 하나씩 추가하면서
  결과가 어떻게 달라지는지 체감하는 실습
- AWS Bedrock + Claude 사용
"""

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 자동 로드

# ──────────────────────────────────────────────
# AWS Bedrock 설정
# ──────────────────────────────────────────────
REGION   = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

bedrock = boto3.client(
    service_name = "bedrock-runtime",
    region_name  = REGION,
)

def llm_invoke(prompt_detail):
    res = bedrock.converse(
        modelId         = MODEL_ID,
        messages        = [{"role": "user", "content": [{"text": prompt_detail}]}],
        inferenceConfig = {"maxTokens": 600},
    )
    return res["output"]["message"]["content"][0]["text"]


# ──────────────────────────────────────────────
# 프롬프트 구성 단계별 정의
# ──────────────────────────────────────────────
prompts = {
    # 1단계: 아무 구성 없이 날것 요청
    "1단계 - 기본 요청": "텀블러 SNS 문구 써줘.",

    # 2단계: Role(페르소나) 추가
    "2단계 - Role 추가": """\
당신은 20년차 프로 카피라이터입니다.
텀블러 SNS 문구 써줘.""",

    # 3단계: Context(배경) 추가
    "3단계 - Context 추가": """\
당신은 20년차 프로 카피라이터입니다.

우리 회사에서 이번에 '24시간 얼음이 녹지 않는 셀럽형 텀블러'를 출시했습니다.
타겟 고객은 하루 종일 아이스 커피를 달고 사는 20대 직장인입니다.

홍보용 SNS 문구 써줘.""",

    # 4단계: Task(명확한 지시) 추가
    "4단계 - Task 추가": """\
당신은 20년차 프로 카피라이터입니다.

우리 회사에서 이번에 '24시간 얼음이 녹지 않는 셀럽형 텀블러'를 출시했습니다.
타겟 고객은 하루 종일 아이스 커피를 달고 사는 20대 직장인입니다.

타겟 고객에게 보낼 홍보용 SNS 초안을 작성해 주세요.
고객이 이 SNS를 보고 '구매하기' 버튼을 누르고 싶게 만들어야 합니다.""",

    # 5단계: Format/Constraints + XML 태그 → 완전한 Claude 특화 프롬프트
    "5단계 - Format + XML 추가 (완성)": """\
<role>
당신은 20년차 프로 카피라이터입니다.
</role>

<context>
제품: 24시간 얼음이 녹지 않는 셀럽형 텀블러
타겟: 하루 종일 아이스 커피를 달고 사는 20대 직장인
</context>

<instruction>
타겟 고객에게 보낼 홍보용 SNS 초안을 작성해 주세요.
고객이 이 SNS를 보고 '구매하기' 버튼을 누르고 싶게 만들어야 합니다.
</instruction>

<constraints>
- 톤앤매너: 유머러스하고 위트 있게, 너무 딱딱하지 않게
- 형식:
  - 도입부: 텀블러 장점 4개 포인트
  - 마무리: 구매 유도 문구
- 분량: 400자 이내
</constraints>""",
}

# ──────────────────────────────────────────────
# 단계별 호출 및 결과 출력
# ──────────────────────────────────────────────
for label, prompt in prompts.items():
    print(f"\n{'=' * 60}")
    print(f"[{label}]")
    print(f"{'─' * 60}")
    print(llm_invoke(prompt))

print(f"\n{'=' * 60}")
print("실습 완료: 단계별로 결과가 어떻게 달라졌는지 비교해 보세요.")
