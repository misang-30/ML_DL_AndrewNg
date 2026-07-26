
이 주차는 **Deep Learning Specialization 전체 과정의 최신 최종장**으로, RNN과 Attention의 한계를 극복하고 현대 Large Language Model(ChatGPT, BERT 등)의 기반이 된 **Transformer 아키텍처**를 완전히 파헤칩니다.

## 1. Transformer가 등장한 이유

- **RNN / LSTM의 한계:** 문장을 순차적(Sequential)으로 처리하기 때문에 **병렬 연산(Parallelization)이 불가능**하여 학습 속도가 매우 느립니다.
    
- **Transformer의 혁신:** 반복문(Loop)을 없애고 Self-Attention과 위치 인코딩(Positional Encoding)을 도입하여 문장 전체를 한 번에 병렬 처리함으로써 대규모 데이터 학습을 가능하게 했습니다.
    

## 2. Self-Attention (자체 어텐션 메커니즘)

문장 내의 각 단어가 **자신의 문맥을 이해하기 위해 문장 내 다른 모든 단어와 어떤 관계를 갖는지** 스스로 계산하는 핵심 연산입니다.

### **1) Query(Q), Key(K), Value(V) 벡터**

각 단어의 임베딩 벡터에 가중치 행렬 $W^Q, W^K, W^V$를 곱해 3가지 벡터를 생성합니다.

- **Query ($Q$):** "내가 찾고자 하는 정보/단어의 질문"
    
- **Key ($K$):** "각 단어가 가진 정체성/레이블"
    
- **Value ($V$):** "각 단어가 실제로 담고 있는 내용"
    

### **2) Scaled Dot-Product Attention 수식**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **$QK^T$:** Query와 Key의 내적을 통해 단어 간의 유사도(관련성) 점수를 계산합니다.
    
- **$\sqrt{d_k}$로 나누는 이유 (Scaling):** 차원 수가 커질수록 내적값이 너무 커져 Softmax의 기울기가 소멸(Gradient Vanishing)하는 현상을 방지합니다.
    

## 3. Multi-Head Attention (다중 헤드 어텐션)

단어 하나가 여러 가지 의미적 관계(예: 주어-동사 관계, 대명사-명사 관계, 시제 등)를 동시에 파악할 수 있도록 **Self-Attention 연산을 $h$개 병렬로 독립 수행**한 뒤 결과를 합칩니다.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

## 4. Positional Encoding (위치 인코딩)

Transformer는 RNN과 달리 단어를 한 번에 병렬로 입력받기 때문에 **단어의 순서(Order) 정보가 손실**됩니다. 이를 보완하기 위해 각 위치 $pos$에 고유한 주기 함수값을 더해줍니다.

- **삼각함수 기반 위치 인코딩:**
    
    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
    
    $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
    
- **최종 입력:** $\text{Input Vector} = \text{Word Embedding} + \text{Positional Encoding}$
    

## 5. Transformer 전체 구조 (Encoder & Decoder)

### **1) Encoder (인코더)**

- **Multi-Head Self-Attention:** 문장 전체 내 단어 간 관계 파악.
    
- **Feed-Forward Network (FFN):** 각 위치별 신경망 연산.
    
- **Residual Connection (잔차 연결) & Layer Normalization:** 학습 안정화 및 기울기 소멸 방지.
    

### **2) Masking (마스킹 메커니즘)**

- **Padding Mask:** 입력을 맞추기 위해 넣은 의미 없는 `<PAD>`(0) 토큰에 어텐션이 가지 않도록 가려줍니다.
    
- **Look-Ahead Mask (Causal Mask):** 디코더가 미래의 단어를 미리 보고 정답을 찌르는 것을 막기 위해, 현재 타임스텝 이후의 미래 단어 위치에 $-\infty$ (매우 작은 값)를 씌워 Softmax 확률을 0으로 만듭니다.
    

### **3) Decoder (디코더)**

- **Masked Multi-Head Attention:** 미래 단어를 보지 못하도록 마스킹 처리된 Self-Attention.
    
- **Encoder-Decoder Attention:** 디코더의 $Q$와 인코더의 $K, V$를 결합하여 번역할 원본 문맥을 참조.
    
- **Linear & Softmax:** 최종 단어 분포 출력.
    

