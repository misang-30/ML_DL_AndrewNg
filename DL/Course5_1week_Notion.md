

코스 5는 텍스트, 음성, 시계열 데이터처럼 **데이터의 '순서(Sequence)'가 중요한 문제**를 다룹니다. 1주차에서는 순환 신경망(RNN)의 기초부터 시작하여, 긴 문장을 기억하지 못하는 문제를 해결하기 위한 GRU와 LSTM 아키텍처까지 깊이 있게 배웁니다.

---

## 1. 순차 데이터(Sequence Data)의 특징과 표기법

### 왜 순차 모델(Sequence Model)인가?

기존의 DNN이나 CNN은 입력 데이터의 크기가 고정되어 있고, 입력 데이터들 사이에 시간적/공간적 순서 관계를 독립적으로 취급합니다. 하지만 현실의 많은 데이터는 길이가 제각각이며 앞뒤 맥락이 중요합니다.

* **음성 인식 (Audio to Text):** 입력(음성 Wave)과 출력(텍스트)의 길이가 다름.
* **감성 분류 (Text to Rating):** "이 영화 진짜 돈 아깝다" $\rightarrow$ 별점 1점 (입력은 시퀀스, 출력은 단일 값).
* **DNA 서열 분석, 기계 번역 등** 다양한 형태의 Many-to-One, One-to-Many, Many-to-Many 문제가 존재합니다.

### 수학적 표기법 (Notation)

문장 `"Harry Potter and Hermione Granger met in Hogwarts"`를 예시로 듭니다.

* $x^{\langle t \rangle}$: 입력 시퀀스의 $t$번째 소자(단어). (예: $x^{\langle 1 \rangle} =$ Harry, $x^{\langle 2 \rangle} =$ Potter)
* $T_x$: 입력 시퀀스의 총 길이 (위 문장에서는 8).
* $y^{\langle t \rangle}$: 출력 시퀀스의 $t$번째 소자.
* $T_y$: 출력 시퀀스의 총 길이.
* $x^{(i)\langle t \rangle}$: $i$번째 훈련 샘플의 $t$번째 단어.

### 단어의 표현: 원-핫 인코딩 (One-hot Encoding)

컴퓨터는 문자열을 그대로 이해할 수 없으므로, 사전(Vocabulary, 예: 10,000개의 단어장)을 만듭니다.

* 사전 안에서 'Harry'가 4075번째 단어라면, $x^{\langle 1 \rangle}$은 4075번째 원소만 `1`이고 나머지는 모두 `0`인 10,000차원의 **원-핫 벡터**가 됩니다.
* 사전에 없는 단어는 어휘 외 단어 토큰인 **`<UNK>`** (Unknown), 문장의 끝은 **`<EOS>`** (End of Sequence)로 표현합니다.

---

## 2. 순환 신경망 (RNN: Recurrent Neural Network)

기존의 표준 신경망(Standard Neural Network)에 원-핫 벡터들을 그대로 넣으면 두 가지 문제가 발생합니다. (1) 입력과 출력의 길이가 샘플마다 다를 때 대처가 안 되고, (2) 이미지나 텍스트의 다른 위치에서 배운 특징(예: 'Harry'가 사람 이름이라는 것)이 공유되지 않습니다. 이를 해결한 것이 **RNN**입니다.

### RNN의 구조와 작동 원리

RNN은 문장을 왼쪽에서 오른쪽으로 한 단어씩 읽어나갑니다.

1. 첫 번째 단어 $x^{\langle 1 \rangle}$을 입력받아 첫 번째 은닉 상태(Hidden State) $a^{\langle 1 \rangle}$을 계산하고, 이를 바탕으로 첫 번째 출력 $\hat{y}^{\langle 1 \rangle}$을 예측합니다.
2. 두 번째 단어를 처리할 때, $x^{\langle 2 \rangle}$뿐만 아니라 **직전 단계의 기억인 $a^{\langle 1 \rangle}$을 함께 입력**받아 $a^{\langle 2 \rangle}$를 계산합니다.
3. 이 과정을 $t = 1$부터 $t = T_x$까지 반복하며, **모든 시점(Time-step)에서 가중치 매개변수를 동일하게 공유**합니다.

> **초기화:** 맨 처음 시작할 때 필요한 $a^{\langle 0 \rangle}$은 대개 모든 원소가 0인 벡터로 초기화합니다.

### RNN의 정방향 전파(Forward Propagation) 수식

$$a^{\langle t \rangle} = g_1(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$$

$$\hat{y}^{\langle t \rangle} = g_2(W_{ya} a^{\langle t \rangle} + b_y)$$

* $W_{ax}$: 입력 $x$가 은닉층에 영향을 주는 가중치
* $W_{aa}$: 이전 은닉 상태 $a$가 다음 은닉 상태에 영향을 주는 가중치
* $W_{ya}$: 은닉 상태 $a$가 출력 $y$에 영향을 주는 가중치
* $g_1$: 주로 **$\tanh$** 또는 $\text{ReLU}$ 활성화 함수 사용
* $g_2$: 분류 문제에 따라 주로 **$\text{Sigmoid}$** 또는 **$\text{Softmax}$** 사용

이를 행렬 형태로 더 깔끔하게 압축하여 다음과 같이 표기하기도 합니다. (가중치 행렬을 가로로 이어 붙이는 방식)


$$a^{\langle t \rangle} = \tanh(W_a [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_a) \quad \text{where } W_a = [W_{aa} \; \vert{} \; W_{ax}]$$

---

## 3. 시간에 따른 역전파 (BPTT: Backpropagation Through Time)

RNN을 학습시키려면 손실 함수를 정의하고 역전파를 수행해야 합니다.

* **단일 시점 손실 (Loss):** 특정 시점 $t$에서의 예측 오차 (예: 이진 분류 시 Cross-Entropy)

$$\mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) = - y^{\langle t \rangle} \log \hat{y}^{\langle t \rangle} - (1 - y^{\langle t \rangle}) \log (1 - \hat{y}^{\langle t \rangle})$$


* **전체 손실 (Cost):** 모든 시점의 손실을 더한 값

$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}$$



### 역전파 흐름 (BPTT)

오차를 계산한 뒤, 가중치 $W_a, b_a, W_y, b_y$를 업데이트하기 위해 미분을 수행합니다. 이때 그래프의 화살표가 오른쪽(미래)에서 왼쪽(과거)으로 **시간의 흐름을 거슬러 올라가며 전파**되기 때문에 이를 BPTT(Backpropagation Through Time)라고 부릅니다.

---

## 4. 다양한 RNN 유형 (RNN Types)

입력 길이 $T_x$와 출력 길이 $T_y$의 관계에 따라 네트워크 구조가 달라집니다.

* **One-to-One:** 일반적인 표준 신경망 (시퀀스가 아님).
* **One-to-Many:** 하나의 입력으로 여러 출력을 만듦 (예: 이미지 캡셔닝: 이미지 1장 $\rightarrow$ 문장 출력).
* **Many-to-One:** 여러 입력을 받아 하나의 값을 출력 (예: 감성 분류: 문장 $\rightarrow$ 긍정/부정 별점).
* **Many-to-Many ($T_x = T_y$):** 각 시점마다 출력이 나옴 (예: 개체명 인식(NER): 문장의 각 단어가 사람 이름인지 장소인지 태깅).
* **Many-to-Many ($T_x \neq T_y$):** 입력을 끝까지 다 읽은 후 출력을 시작하는 인코더-디코더 구조 (예: 기계 번역).

---

## 5. 기본 RNN의 한계: 기울기 소실 (Vanishing Gradients)

단어와 단어 사이의 거리가 먼 **장기 의존성(Long-Term Dependencies)** 문제에서 기본 RNN은 치명적인 약점을 가집니다.

* 예: `"The **cat**, which already ate fish, ... , **was** full."` vs `"The **cats**, which already ate fish, ... , **were** full."`
* 'cat'이 단수형인지 복수형인지에 따라 저 멀리 뒤에 나오는 동사가 'was' 또는 'were'로 결정되어야 합니다.
* 하지만 기본 RNN은 구조상 $t$가 커질수록 앞쪽(과거)의 정보가 뒤쪽까지 전달되지 못하고 희석됩니다. 역전파 과정에서 레이어가 너무 길어지다 보니 기울기 소실(Vanishing Gradient)이 발생하여, 앞쪽 단어의 가중치를 거의 업데이트하지 못하기 때문입니다.
* *참고: 기울기 폭발(Exploding Gradient)도 일어날 수 있지만, 이는 기울기 클리핑(Gradient Clipping, 임계값을 넘으면 강제로 깎아내림)으로 비교적 쉽게 해결할 수 있습니다. 진짜 난제는 기울기 소실입니다.*

---

## 6. GRU (Gated Recurrent Unit)

기울기 소실 문제를 해결하고 아주 먼 과거의 정보도 기억할 수 있도록 보완된 하위 RNN 변형 구조입니다. 핵심은 '게이트(Gate)'라는 차단막을 두어 정보를 기억하거나 잊어버리는 것입니다.

### GRU의 메커니즘 (단순화된 버전)

GRU는 기억을 전담하는 메모리 셀 $c^{\langle t \rangle}$를 도입합니다. (GRU에서는 은닉 상태 $a^{\langle t \rangle}$와 $c^{\langle t \rangle}$의 값이 같습니다.)

1. **후보 메모리 셀 ($\tilde{c}^{\langle t \rangle}$):** 이번 단계에서 새롭게 기억할 만한 후보 지식입니다.

$$\tilde{c}^{\langle t \rangle} = \tanh(W_c [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$$


2. **업데이트 게이트 ($\Gamma_u$):** 0과 1 사이의 값(시그모이드 출력)을 가지며, **"과거의 기억을 얼마나 유지하고, 새로운 후보를 얼마나 반영할지"** 결정하는 문지기입니다.

$$\Gamma_u = \sigma(W_u [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u)$$


3. **최종 메모리 셀 결정 ($c^{\langle t \rangle}$):**

$$c^{\langle t \rangle} = \Gamma_u \times \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) \times c^{\langle t-1 \rangle}$$


* 만약 $\Gamma_u = 0$에 가깝다면, 새로운 정보는 무시하고 과거의 기억 $c^{\langle t-1 \rangle}$을 그대로 미래로 흘려보냅니다. 이 덕분에 기울기가 소실되지 않고 수백 단계 뒤까지 정보가 살아남을 수 있습니다.



### 풀 버전 GRU (Full GRU)

실제 GRU는 과거의 기억이 새로운 후보를 만들 때 얼마나 필요한지 조절하는 리셋 게이트($\Gamma_r$)가 하나 더 추가됩니다.


$$\tilde{c}^{\langle t \rangle} = \tanh(W_c [\Gamma_r * c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$$

---

## 7. LSTM (Long Short-Term Memory)

GRU보다 먼저 개발되었고, 훨씬 더 정교하게 정보를 제어하는 전통적인 강자입니다. GRU는 2개의 게이트를 쓰지만, LSTM은 **3개의 독립적인 게이트**를 사용합니다. 또한 은닉 상태 $a^{\langle t \rangle}$와 메모리 셀 $c^{\langle t \rangle}$이 서로 분리되어 흐릅니다.

### LSTM의 5가지 핵심 수식

$$\begin{aligned} \tilde{c}^{\langle t \rangle} &= \tanh(W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c) \quad &\text{(새로운 기억 후보)} \\ \Gamma_f &= \sigma(W_f [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f) \quad &\text{(망각 게이트 - 과거를 얼마나 지울까?)} \\ \Gamma_i &= \sigma(W_i [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_i) \quad &\text{(입력 게이트 - 현재를 얼마나 더할까?)} \\ \Gamma_o &= \sigma(W_o [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o) \quad &\text{(출력 게이트 - 외부로 얼마를 내보낼까?)} \end{aligned}$$

위 게이트들을 조합하여 진짜 메모리 값($c^{\langle t \rangle}$)과 다음 층으로 보낼 은닉 상태($a^{\langle t \rangle}$)를 구합니다.

$$c^{\langle t \rangle} = \Gamma_f * c^{\langle t-1 \rangle} + \Gamma_i * \tilde{c}^{\langle t \rangle}$$

$$a^{\langle t \rangle} = \Gamma_o * \tanh(c^{\langle t \rangle})$$

* **망각 게이트($\Gamma_f$)와 입력 게이트($\Gamma_i$)가 독립적**이기 때문에, GRU($\Gamma_u$와 $1-\Gamma_u$)보다 훨씬 더 유연하고 복잡한 기억 제어가 가능합니다. 정보가 끊기지 않고 고속도로처럼 곱하기 연산 없이 쭉 흐르는 라인($c^{\langle t-1 \rangle} \rightarrow c^{\langle t \rangle}$)이 존재하여 기울기 소실을 완벽히 방어합니다.

---

## 8. 양방향 RNN (BRNN) 및 다층 RNN (Deep RNNs)

### 양방향 RNN (Bidirectional RNN)

기존 RNN은 과거(왼쪽) 정보만 보고 미래를 예측합니다. 하지만 문맥을 파악하려면 미래(오른쪽) 정보도 봐야 합니다.

* 예: `"He said, Teddy bear is cute."` vs `"He said, Teddy Roosevelt was a president."`
* 'Teddy'가 곰 인형인지 대통령인지 알기 위해서는 뒤에 나오는 단어(`bear` 또는 `Roosevelt`)를 끝까지 읽어야 합니다.
* **원리:** 정방향으로 흐르는 RNN($\overrightarrow{a}$)과 역방향(문장 맨 끝 단어부터 거꾸로 읽음)으로 흐르는 RNN($\overleftarrow{a}$)을 동시에 둡니다. 특정 시점 $t$의 예측은 이 두 은닉 상태를 합쳐서($[\overrightarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle}]$) 결정합니다.
* **단점:** 문장 전체를 다 읽어야만 예측을 시작할 수 있으므로, 실시간 실시간 음성 인식 같은 곳에는 곧바로 쓰기 어렵습니다.

### 다층 RNN (Deep RNNs)

더 복잡한 함수를 학습하기 위해 RNN 층을 수직으로 여러 개 쌓아 올린 구조입니다. (예: $a^{[1]\langle t \rangle} \rightarrow a^{[2]\langle t \rangle} \rightarrow a^{[3]\langle t \rangle}$). 시간 축으로 이미 연산이 길기 때문에, 보통 일반 DNN처럼 100층씩 쌓지는 않고 대개 **3~4층 정도**만 쌓아도 충분히 깊은 모델이 됩니다.

---

1주차는 순차 데이터를 표기하는 엄격한 규칙부터 시작하여, **기본 RNN의 한계점(기울기 소실)을 인공지능 학자들이 게이트(GRU, LSTM) 개념을 도입해 어떻게 극적으로 돌파해 냈는가**를 관통하는 흐름을 담고 있습니다. 시퀀스 모델링의 가장 단단한 주춧돌이 되는 주차입니다!
