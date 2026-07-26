

Course 5의 마지막 주차에서는 기계 번역(Machine Translation)과 음성 인식(Speech Recognition)의 기반이 되는 **Seq2Seq 아키텍처**, 최적의 문장을 탐색하는 **Beam Search**, 기존 RNN의 한계를 극복한 **어텐션 메커니즘(Attention Mechanism)** 및 **Transformer의 초석**을 배웁니다.

## 1. Sequence-to-Sequence (Seq2Seq) 기본 아키텍처

한 시퀀스(예: 한국어 문장)를 다른 시퀀스(예: 영어 문장)로 변환하는 아키텍처로, 인코더(Encoder)와 **디코더(Decoder)** 두 부분으로 구성됩니다.

- **인코더 (Encoder):** 입력 시퀀스 $X^{\langle 1 \rangle}, \dots, X^{\langle T_x \rangle}$를 받아 전체 문맥을 요약한 고정 길이 벡터(Context Vector)로 압축합니다.
    
- **디코더 (Decoder):** 인코더가 넘겨준 문맥 벡터를 초기 상태로 입력받아, 출력 시퀀스 $Y^{\langle 1 \rangle}, \dots, Y^{\langle T_y \rangle}$를 한 단어씩 순차적으로 생성합니다.
    
- **이미지 캡셔닝 (Image Captioning):** 인코더로 CNN(AlexNet, ResNet 등)을 사용하여 이미지 특징을 추출한 뒤, 디코더 RNN에 넘겨 이미지를 설명하는 문장을 생성할 수 있습니다.
    

## 2. 빔 탐색 (Beam Search)

기계 번역 디코더에서 가장 확률이 높은 최적의 문장 $\hat{Y}$를 찾아내는 **조건부 언어 모델 탐색 알고리즘**입니다.

### **1) 왜 Greedy Search를 쓰지 않는가?**

매 타임스텝마다 가장 확률이 높은 단어 1개만 선택하는 탐색(Greedy Search)은 전체 문장 관점에서의 최적해(Global Optimum)를 보장하지 못합니다.

### **2) 빔 탐색 작동 원리**

- **빔 크기 (Beam Width, $B$):** 매 단계마다 가장 확률이 높은 **$B$개의 후보 시퀀스**를 동시에 유지합니다 (예: $B=3$).
    
- **단계별 동작:**
    
    1. 첫 번째 단어로 가장 확률이 높은 $B$개 단어를 선택합니다.
        
    2. 선택된 $B$개 단어 각각에 대해 두 번째 단어가 올 확률을 계산하고, 가장 확률이 높은 상위 $B$개 조합을 선택합니다.
        
    3. `<EOS>`(문장 끝) 토큰이 나올 때까지 이 과정을 반복합니다.
        

### **3) 길이 정규화 (Length Normalization)**

여러 단어의 확률을 계속 곱하면 $P(y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle}\vert{}x) = \prod_{t=1}^{T_y} P(y^{\langle t \rangle}\vert{}x, y^{\langle <t \rangle})$ 값은 문장이 길어질수록 0에 가깝게 매우 작아집니다 (Underflow 발생).

- **로그 변환 및 길이 정규화 수식:**
    
    $$\arg\max_Y \frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \vert{} x, y^{\langle <t \rangle})$$
    
    - $\alpha$ (보통 $0.7$ 수준): 짧은 문장만 선호하는 편향을 완화하는 하이퍼파라미터입니다.
        

## 3. 어텐션 메커니즘 (Attention Mechanism)

기존 Seq2Seq는 아무리 긴 문장이라도 **단 하나의 고정 길이 벡터**로 압축해야 하므로 문장이 길어지면 정보 손실이 발생했습니다. 어텐션은 디코더가 단어를 생성할 때마다 **인코더의 어느 부분에 집중(Attention)해야 하는지** 유동적으로 계산합니다.

### **1) 어텐션 가중치 및 문맥 벡터 연산**

- 디코더 타임스텝 $t$에서 인코더 타임스텝 $t'$의 은닉 상태 $a^{\langle t' \rangle}$에 부여할 가중치를 $\alpha^{\langle t, t' \rangle}$라 합니다.
    
- **소프트맥스 가중치:**
    
    $$\alpha^{\langle t, t' \rangle} = \frac{\exp(e^{\langle t, t' \rangle})}{\sum_{t'=1}^{T_x} \exp(e^{\langle t, t' \rangle})}$$
    
    - $\sum_{t'} \alpha^{\langle t, t' \rangle} = 1$
        
- **문맥 벡터 (Context Vector):**
    
    $$c^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}$$
    

## 4. 트랜스포머의 등장 (Transformer Intuition)

어텐션 개념을 극대화하여 **RNN의 순차적 연산(Self-Loop)을 완전히 제거**하고, 오직 어텐션 연산만으로 시퀀스를 처리하는 Transformer (Self-Attention)의 직관적 개념을 언급합니다.

- **Self-Attention:** 문장 내부의 단어들이 서로 어떤 관계를 맺고 있는지 파악합니다 (예: "The **animal** didn't cross the street because **it** was too tired"에서 `it`이 `animal`을 가리킴을 파악).
    
- **Query, Key, Value ($Q, K, V$):**
    
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    
- **병렬 처리 (Parallelization):** RNN처럼 시간 순서대로 기다릴 필요 없이 문장 전체를 한 번에 병렬 연산할 수 있어 학습 속도가 획기적으로 빨라집니다.
    

## 5. 음성 인식 및 오디오 모델 (Speech Recognition & Audio)

### **1) 음성 인식 (Speech Recognition)**

음성 파형(Audio Waveform)이나 스펙트로그램(Spectrogram)을 입력받아 텍스트 시퀀스로 변환하는 과제입니다.

- **CTC Loss (Connectionist Temporal Classification):**
    
    입력 타임스텝 수가 출력 텍스트 길이보다 훨씬 길 때 사용합니다 (예: `ttt-h-e---` $\rightarrow$ `the`).
    

### **2) 트리거 워드 탐지 (Trigger Word Detection)**

"Hey Siri", "OK Google"과 같은 특정 호출어를 감지하는 시스템입니다.

- 오디오 음성 데이터에 호출어가 끝나는 지점 뒤의 몇몇 타임스텝에 라벨 $1$을 부여하여 RNN을 학습시킵니다.
    

