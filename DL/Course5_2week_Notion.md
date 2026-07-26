

Week 2에서는 단어를 컴퓨터가 이해할 수 있는 의미 있는 밀집 벡터(Dense Vector)로 표현하는 **단어 임베딩(Word Embedding)** 기술과 대표적인 알고리즘(**Word2Vec, GloVe**), 그리고 이를 활용한 **감성 분석** 및 **편향(Bias) 제거 기법**을 배웁니다.

## 1. 단어 표현과 임베딩 (Word Representation)

### **1) One-Hot Encoding의 한계**

- 단어를 $V$ 크기의 원-핫 벡터로 표현하면, 두 단어 벡터 간의 내적(Inner Product)이 항상 0이 됩니다.
    
- 즉, `Apple`과 `Orange`가 비슷한 과일이라는 의미적 유사성(Semantic Similarity)을 모델이 전혀 학습하지 못합니다.
    

### **2) 단어 임베딩 (Word Embedding)**

- 단어를 고차원(예: 300차원)의 연속적인 수치 공간에 매핑합니다.
    
- 성별, 나이, 음식, 고유명사 등의 의미적 특성(Feature)이 각 차원에 녹아들어, 유사한 의미를 가진 단어끼리 **벡터 공간상에서 가까운 거리**에 위치하게 됩니다.
    

### **3) 임베딩의 유사도 및 유짚 연산 (Analogy Reasoning)**

- **유사도 측정 (Cosine Similarity):**
    
    $$\text{Sim}(u, v) = \frac{u^T v}{\vert{}\vert{}u\vert{}\vert{}_2 \vert{}\vert{}v\vert{}\vert{}_2}$$
    
- **유짚 연산 (Vector Analogies):**
    
    "Man이 Woman이면, King은 무엇인가?"라는 질문을 벡터 연산으로 해결합니다:
    
    $$e_{\text{man}} - e_{\text{woman}} \approx e_{\text{king}} - e_{\text{queen}}$$
    

## 2. 단어 임베딩 학습 알고리즘 (Word2Vec & GloVe)

### **1) Word2Vec: Skip-Gram 모델**

중심 단어(Target Word, $c$)가 주어졌을 때 주변 문맥 단어(Context Word, $t$)가 나타날 확률을 예측하도록 학습합니다.

- **Softmax 확률 식:**
    
    $$P(t\vert{}c) = \frac{\exp(\theta_t^T e_c)}{\sum_{w=1}^{\vert{}V\vert{}} \exp(\theta_w^T e_c)}$$
    
- **문제점:** 분모에서 전체 어휘 사전 $\vert{}V\vert{}$에 대해 합을 구해야 하므로 연산량이 매우 큽니다.
    

### **2) Negative Sampling (부정 음영 샘플링)**

Softmax의 무거운 연산량을 극복하기 위해, 문제를 **이진 분류(Binary Classification)** 문제로 전환합니다.

- 1개의 진짜 (Context, Target) 쌍 $\rightarrow$ Label 1
    
- $K$개의 무작위 가짜 (Context, Random Word) 쌍 $\rightarrow$ Label 0
    
- 매 단계마다 전체 사전을 연산하는 대신 $K+1$개의 이진 분류 문제만 풀어 연산 속도를 획기적으로 향상시킵니다.
    

### **3) GloVe (Global Vectors for Word Representation)**

단어 간 동시 등장 횟수(Co-occurrence Matrix, $X_{ij}$)를 전역적으로 활용하여 임베딩을 학습합니다.

$$\text{Cost} = \sum_{i=1}^{\vert{}V\vert{}} \sum_{j=1}^{\vert{}V\vert{}} f(X_{ij}) \left( \theta_i^T e_j + b_i + b_j' - \log X_{ij} \right)^2$$

## 3. 감성 분석 (Sentiment Analysis)

텍스트(리뷰, 트윗 등)를 입력받아 긍정/부정 정도나 별점을 예측하는 과제입니다.

- **Simple Averaging Model:** 모든 단어의 임베딩 벡터를 평균 낸 뒤 Softmax 분류기에 전달 (단어 순서를 무시한다는 단점 존재).
    
- **RNN-based Model:** Many-to-One 아키텍처를 적용하여 단어의 순서와 문맥("Not good"과 같은 부정 표현)을 반영하여 분류.
    

## 4. 임베딩의 편향 제거 (Debiasing Word Embeddings)

학습 데이터(인터넷 텍스트 등)에 존재하는 성별, 인종, 종교 등에 대한 편향(Bias)이 임베딩 공간에 그대로 반영되는 문제가 있습니다 (예: $\text{Man} : \text{Computer Programmer} \approx \text{Woman} : \text{Homemaker}$).

### **편향 제거 3단계 과정**

1. **Bias Direction 식별:** 성별 단어쌍들($e_{\text{he}} - e_{\text{she}}$, $e_{\text{male}} - e_{\text{female}}$)의 차이를 평균 내어 편향축(Bias Axis)을 구합니다.
    
2. **Neutralize (중립화):** 편향과 무관해야 하는 단어들(Doctor, Programmer, Nurse 등)을 편향축에 직교(Orthogonal)하도록 투영하여 성별 성분을 제거합니다.
    
3. **Equalize (균등화):** 성별 고유 단어쌍(Grandfather-Grandmother, Boy-Girl 등)이 중립 단어들로부터 동일한 거리를 갖도록 거리를 맞춥니다.
    

