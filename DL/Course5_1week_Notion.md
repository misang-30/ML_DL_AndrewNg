

Week 1에서는 텍스트, 음성, 시계열 데이터와 같이 순서가 중요한 시퀀스 데이터(Sequence Data)를 처리하기 위한 기본 아키텍처인 **RNN**부터, 장기 의존성 문제를 극복한 **GRU** 및 **LSTM**, 그리고 양방향/심층 신경망 구조까지 다룹니다.

## 1. 순환 신경망 (RNN - Recurrent Neural Networks)

### **1) 왜 일반 피드포워드 신경망(Standard NN)을 쓰지 않는가?**

- 입력과 출력의 길이가 샘플마다 다를 수 있습니다.
    
- 텍스트의 여러 위치에서 학습한 특징(Feature)을 공유하기 어렵습니다.
    
- 파라미터 수가 입력 데이터의 길이에 비례해 폭발적으로 증가합니다.
    

### **2) 순방향 전파 (Forward Propagation)**

이전 타임스텝의 은닉 상태(Hidden State) $a^{\langle t-1 \rangle}$와 현재 입력 $x^{\langle t \rangle}$를 전달받아 현재 상태 $a^{\langle t \rangle}$와 예측값 $\hat{y}^{\langle t \rangle}$를 계산합니다.

- **은닉 상태 업데이트:**
    
    $$a^{\langle t \rangle} = g_1(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$$
    
- **예측값 계산:**
    
    $$\hat{y}^{\langle t \rangle} = g_2(W_{ya} a^{\langle t \rangle} + b_y)$$
    
- _참고:_ 매 타임스텝마다 사용하는 가중치 행렬 $W_{aa}, W_{ax}, W_{ya}$는 모든 타임스텝에서 공유(Share)됩니다.
    

### **3) BPTT (Backpropagation Through Time)**

시간 순서의 역방향으로 손실(Loss)의 기울기를 전달하며 가중치를 업데이트합니다. 타임스텝 $t$에서의 손실 $\mathcal{L}^{\langle t \rangle}$을 합산한 전체 손실 $\mathcal{L} = \sum_t \mathcal{L}^{\langle t \rangle}$에 대해 역전파를 수행합니다.

## 2. 다양한 RNN 유형 (RNN Architectures)

입력 시퀀스 길이 $T_x$와 출력 시퀀스 길이 $T_y$의 관계에 따라 네트워크 구조가 달라집니다.

|**유형**|**설명**|**대표적인 응용 사례**|
|---|---|---|
|**One-to-One**|일반적인 신경망 구조 ($T_x=1, T_y=1$)|이미지 분류|
|**One-to-Many**|단일 입력에서 시퀀스 출력 ($T_x=1, T_y>1$)|음악 생성, 이미지 캡셔닝|
|**Many-to-One**|시퀀스 입력에서 단일 출력 ($T_x>1, T_y=1$)|감성 분석 (Sentiment Analysis)|
|**Many-to-Many ($T_x = T_y$)**|입력과 출력 길이가 같음|개체명 인식 (NER)|
|**Many-to-Many ($T_x \neq T_y$)**|인코더-디코더 구조|기계 번역 (Machine Translation)|

## 3. 언어 모델 및 시퀀스 생성 (Language Model & Sequence Generation)

언어 모델은 문장 $W = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \dots, y^{\langle T_y \rangle})$이 나타날 확률 $P(W)$를 추정합니다.

### **1) 토큰화 및 샘플링 (Sampling)**

- 문장을 단어 단위(또는 문자 단위)로 나누고 **Softmax**를 통해 각 타임스텝에서 다음에 올 단어의 확률 분포를 출력합니다.
    
- 학습이 완료된 후, 출력된 확률 분포에서 단어를 샘플링(Random Sampling)하여 다음 타임스텝의 입력으로 넣어주는 방식으로 새로운 텍스트를 무한히 생성해낼 수 있습니다.
    

### **2) 기울기 소멸과 게이트 메커니즘 (Vanishing Gradients)**

기본 RNN은 시퀀스가 길어질수록 앞쪽 타임스텝의 정보가 뒤쪽까지 전달되지 못하고 사라지는 **기울기 소멸(Vanishing Gradient)** 문제가 심각합니다. 이를 해결하기 위해 메모리 셀과 게이트 구조가 도입되었습니다.

## 4. GRU & LSTM (장기 기억을 위한 메커니즘)

### **1) GRU (Gated Recurrent Unit)**

RNN의 단순함과 LSTM의 장점을 절충한 효율적인 구조입니다.

- **업데이트 게이트 ($u_t$):** 이전 상태를 얼마나 유지할지 결정.
    
- **리셋 게이트 ($r_t$):** 이전 상태를 얼마나 무시할지 결정.
    
    $$\tilde{c}^{\langle t \rangle} = \tanh(W_c [r_t * c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$$
    
    $$c^{\langle t \rangle} = (1 - u_t) * c^{\langle t-1 \rangle} + u_t * \tilde{c}^{\langle t \rangle}$$
    

### **2) LSTM (Long Short-Term Memory)**

셀 상태(Cell State, $c^{\langle t \rangle}$)라는 전용 통로를 두어 수십~수백 타임스텝 뒤로도 정보를 손실 없이 전달합니다.

- **Forget Gate ($f_t$):** 이전 셀 상태 $c^{\langle t-1 \rangle}$에서 버릴 정보를 결정 ($\sigma$).
    
- **Input Gate ($i_t$):** 새로 들어온 정보 중 저장할 가치가 있는 것을 결정 ($\sigma$).
    
- **Output Gate ($o_t$):** 현재 셀 상태에서 은닉 상태 $a^{\langle t \rangle}$로 출력할 값을 결정 ($\sigma$).
    

## 5. BRNN & Deep RNN

### **1) 양방향 순환 신경망 (BRNN - Bidirectional RNN)**

- 과거의 정보뿐만 아니라 미래의 문맥(Future Context)까지 함께 고려해야 하는 과제(예: NER)에 사용됩니다.
    
- 정방향(Forward) RNN과 역방향(Backward) RNN을 병렬로 연결하여 두 은닉 상태를 결합합니다.
    
- _단점:_ 전체 시퀀스가 입력될 때까지 기다려야 하므로 실시간 음성 인식 등에는 적용하기 어렵습니다.
    

### **2) 심층 순환 신경망 (Deep RNN)**

- 여러 개의 RNN 레이어를 위로 쌓아 수직 방향의 표현력을 높인 구조입니다 ($L$개 레이어).
    
- 시간 축 계산이 복잡하므로 보통 3개 이하의 레이어를 쌓는 것이 일반적입니다.
    
