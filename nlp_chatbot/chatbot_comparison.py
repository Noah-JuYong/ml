# ============================================================
# 실습 3: 토크나이저 교체 전후 챗봇 응답 품질 비교
#
# [System A] 직접 구축한 사전 + Keras Tokenizer + 코사인 유사도
# [System B] 사전 학습된 모델 (paraphrase-multilingual-MiniLM-L12-v2)
#
# 비교 포인트:
#   - 동일 질문에 대해 두 시스템이 어떤 답변을 선택하는지
#   - 유사도 점수(score)가 얼마나 차이 나는지
#   - 어떤 질문에서 System A가 틀리고 System B가 맞는지
#
# 실행 방법:
#   pip install -r requirements.txt
#   python chatbot_comparison.py
# ============================================================

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# NSMC (네이버 영화리뷰) 로드
from Korpora import Korpora

print("NSMC 로드 중...")
corpus = Korpora.load("nsmc")

train_df = pd.DataFrame({
    "sentence": corpus.train.get_all_texts(),
    "label":    corpus.train.get_all_labels()
})

# 전처리
train_df.dropna(inplace=True)
train_df.drop_duplicates(subset=["sentence"], inplace=True)

# ChatbotData 로드 (GitHub raw URL)
CHATBOT_CSV_URL = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
print("ChatbotData 로드 중...")
chatbot_df = pd.read_csv(CHATBOT_CSV_URL)
print(f"ChatbotData shape: {chatbot_df.shape}")
print(chatbot_df.head(3))

# ─────────────────────────────────────────────
# 2. [System A] 직접 구축 사전 기반 챗봇
#    tensorflow 없이 직접 구현
#    (Keras Tokenizer / pad_sequences와 동일한 동작)
# ─────────────────────────────────────────────

class SimpleTokenizer:
    """공백 분절 기반 토크나이저 (Keras Tokenizer 동일 동작)"""

    def __init__(self):
        self.word_index  = {}   # word -> index (1부터 시작, 0은 패딩 예약)
        self.index_word  = {}   # index -> word
        self._word_count = {}   # 사전 구축용 빈도 카운터

    def fit_on_texts(self, texts):
        for text in texts:
            for word in str(text).lower().split():
                self._word_count[word] = self._word_count.get(word, 0) + 1
        # 빈도 내림차순으로 인덱스 부여 (1부터)
        sorted_words = sorted(self._word_count, key=self._word_count.get, reverse=True)
        for i, word in enumerate(sorted_words, start=1):
            if word not in self.word_index:
                self.word_index[word] = i
                self.index_word[i]    = word

    def texts_to_sequences(self, texts):
        return [
            [self.word_index[w] for w in str(t).lower().split() if w in self.word_index]
            for t in texts
        ]


def pad_sequences(sequences, maxlen, padding="post"):
    """시퀀스를 maxlen 길이로 패딩/자르기"""
    result = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]
        if padding == "post":
            result[i, :len(seq)] = seq
        else:  # pre
            result[i, maxlen - len(seq):] = seq
    return result


print("\n[System A] 사전 구축 중...")

nlp_tokenizer = SimpleTokenizer()

# NSMC + 챗봇 Q/A 전부 포함하여 사전화
nlp_tokenizer.fit_on_texts(train_df.sentence)
nlp_tokenizer.fit_on_texts(chatbot_df.Q)
nlp_tokenizer.fit_on_texts(chatbot_df.A)

print(f"사전 크기: {len(nlp_tokenizer.word_index):,} 토큰")

MAXLEN = 50

def encode_custom(text: str) -> np.ndarray:
    """문장 -> 벡터 (패딩 포함)"""
    x = nlp_tokenizer.texts_to_sequences([text])
    x = pad_sequences(x, MAXLEN + 1, padding="post")
    return x  # shape: (1, MAXLEN+1)

# 챗봇 시트 Q 컬럼 전체 벡터화 (한 번만 수행)
print("챗봇 시트 벡터화 중 (System A)...")
chatbot_df["vec_a"] = chatbot_df.Q.apply(encode_custom)

def answer_system_a(q: str) -> tuple[str, float]:
    """System A: 직접 구축 사전으로 답변 + 최고 유사도 점수 반환"""
    if not q.strip():
        return "질문을 입력해주세요.", 0.0

    user_vec = encode_custom(q)

    scores = chatbot_df.vec_a.apply(
        lambda x: cosine_similarity(user_vec, x)[0][0]
    )

    best_idx   = scores.idxmax()
    best_score = float(scores[best_idx])
    answer     = chatbot_df.loc[best_idx, "A"]
    matched_q  = chatbot_df.loc[best_idx, "Q"]

    return answer, best_score, matched_q

# ─────────────────────────────────────────────
# 3. [System B] 사전 학습 모델 기반 챗봇
#    - paraphrase-multilingual-MiniLM-L12-v2
#    - 50개 언어 지원, 한국어 포함
#    - 강의에서 사용한 xlm-r 모델은 deprecated →
#      동일 계열 후속 권장 모델로 교체
# ─────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"\n[System B] 모델 로드 중: {MODEL_NAME}")
sbert_model = SentenceTransformer(MODEL_NAME)

# 인코딩 확인
test_vec = sbert_model.encode("테스트 문장입니다.")
print(f"임베딩 차원: {test_vec.shape}")  # (384,)

# 챗봇 시트 Q 컬럼 전체 벡터화 — 배치 처리 (한 번만 수행)
# apply로 1개씩 encode하면 ~15분, batch_size=64 배치 처리하면 ~2-3분
print("챗봇 시트 벡터화 중 (System B)... CPU 기준 약 2-3분 소요")
vec_b_matrix = sbert_model.encode(
    chatbot_df.Q.tolist(),
    batch_size=64,
    show_progress_bar=True,
)  # shape: (N, 384)

def answer_system_b(q: str) -> tuple[str, float, str]:
    """System B: 사전 학습 모델로 답변 + 최고 유사도 점수 반환"""
    if not q.strip():
        return "질문을 입력해주세요.", 0.0, ""

    user_vec = sbert_model.encode([q])              # (1, 384)
    scores   = cosine_similarity(user_vec, vec_b_matrix)[0]  # (N,)

    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    answer     = chatbot_df.loc[best_idx, "A"]
    matched_q  = chatbot_df.loc[best_idx, "Q"]

    return answer, best_score, matched_q

# ─────────────────────────────────────────────
# 4. 비교 테스트 — 터미널 출력
# ─────────────────────────────────────────────
TEST_QUESTIONS = [
    "오늘 기분이 너무 안 좋아",
    "사랑하는 사람이 생겼어",
    "헤어지고 나서 너무 힘들어",
    "친구가 없어서 외로워",
    "요즘 회사 스트레스가 심해",
]

print("\n" + "=" * 70)
print("비교 테스트 결과")
print("=" * 70)

for q in TEST_QUESTIONS:
    ans_a, score_a, match_a = answer_system_a(q)
    ans_b, score_b, match_b = answer_system_b(q)

    print(f"\n질문: {q}")
    print(f"  [A] score={score_a:.4f}  매칭Q: {match_a}")
    print(f"      답변: {ans_a}")
    print(f"  [B] score={score_b:.4f}  매칭Q: {match_b}")
    print(f"      답변: {ans_b}")
    print(f"  → 유사도 차이: {score_b - score_a:+.4f}  {'(B가 더 확신)' if score_b > score_a else '(A가 더 확신)'}")

# ─────────────────────────────────────────────
# 5. Gradio UI — 두 시스템 나란히 비교
# ─────────────────────────────────────────────
import gradio as gr

def compare(question: str):
    if not question.strip():
        return "질문을 입력해주세요.", "질문을 입력해주세요."

    ans_a, score_a, match_a = answer_system_a(question)
    ans_b, score_b, match_b = answer_system_b(question)

    result_a = (
        f"**답변:** {ans_a}\n\n"
        f"**유사도 점수:** {score_a:.4f}\n\n"
        f"**매칭된 질문:** {match_a}"
    )
    result_b = (
        f"**답변:** {ans_b}\n\n"
        f"**유사도 점수:** {score_b:.4f}\n\n"
        f"**매칭된 질문:** {match_b}"
    )
    return result_a, result_b

demo = gr.Interface(
    fn=compare,
    inputs=gr.Textbox(
        label="질문 입력",
        placeholder="예: 오늘 기분이 너무 안 좋아",
        lines=2,
    ),
    outputs=[
        gr.Markdown(label="[System A] 직접 구축 사전 (Keras Tokenizer)"),
        gr.Markdown(label="[System B] 사전 학습 모델 (multilingual-MiniLM)"),
    ],
    title="챗봇 토크나이저 품질 비교",
    description=(
        "**System A**: NSMC + ChatbotData로 직접 구축한 사전 → 공백 분절 → 코사인 유사도\n\n"
        "**System B**: `paraphrase-multilingual-MiniLM-L12-v2` (50개 언어, 384차원) → 코사인 유사도\n\n"
        "동일 질문에 대해 두 시스템의 답변과 유사도 점수를 비교하세요."
    ),
    examples=[
        ["오늘 기분이 너무 안 좋아"],
        ["사랑하는 사람이 생겼어"],
        ["헤어지고 나서 너무 힘들어"],
        ["친구가 없어서 외로워"],
        ["요즘 회사 스트레스가 심해"],
    ],
    flagging_mode="never",
)

demo.launch(inbrowser=True)
