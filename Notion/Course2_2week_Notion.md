- 이번 2주차에서는 자신의 데이터셋($X, y$)을 이용해 신경망을 직접 학습(Training)시키는 전체적인 과정과 TensorFlow 코드 구현법을 배웁니다.
    
# < Neural Network Training>

## 1. TensorFlow를 활용한 신경망 학습 3단계 (3-Step Pipeline)

- 손글씨 숫자 인식(0 또는 1 분류) 예제를 바탕으로 한 신경망 학습 코드는 크게 **3단계**로 이루어집니다.

``` Python

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Step 1: 모델 구조 정의 (Model Architecture)
model = Sequential([
    Dense(units=25, activation='sigmoid'),  # Hidden Layer 1
    Dense(units=15, activation='sigmoid'),  # Hidden Layer 2
    Dense(units=1, activation='sigmoid'),  # Output Layer
])

# Step 2: 모델 컴파일 (Compile Model - 손실 함수 지정)
model.compile(loss=tf.keras.losses.BinaryCrossentropy())

# Step 3: 모델 학습 (Fit Model - 경사하강법 실행)
model.fit(X, y, epochs=100)
```

### 1). 각 단계별 상세 설명

|**단계**|**역할 및 주요 개념**|**세부 내용**|
|---|---|---|
|**Step 1. Model Definition**|**신경망 아키텍처 설계**|* `Sequential`을 사용해 각 레이어(층)를 순서대로 연결합니다.<br><br>  <br><br>* 추론(Inference) 시 입력 $X$가 출력 $\hat{y}$로 변환되는 계산 경로를 지정합니다.|
|**Step 2. Compile**|**손실 함수(Loss Function) 설정**|* 모델의 예측 오차를 측정할 기준을 지정합니다.<br><br>  <br><br>* 이진 분류(Binary Classification) 문제에서는 **`BinaryCrossentropy`** 손실 함수를 주로 사용합니다.|
|**Step 3. Fit**|**경사하강법(Gradient Descent) 실행**|* 손실(Cost)을 최적화(최소화)하기 위해 데이터 $X, y$를 입력받아 학습을 진행합니다.<br><br>  <br><br>* **`epochs`**: 경사하강법을 몇 번 반복하여 학습할지 지정하는 파라미터입니다 (예: `epochs=100`은 전체 데이터를 100번 반복하여 파라미터 업데이트).|

---
---
## 2. Training Details
### 1). 로지스틱 회귀 vs 인공신경망 학습 3단계 매핑

머신러닝과 딥러닝 알고리즘을 구축하는 과정은 [1] 모델 정의 $\rightarrow$ [2] 손실/비용 함수 설정 $\rightarrow$ [3] 최적화(최소화)라는 동일한 3단계 패러다임을 따릅니다.

| **단계**                               | **로지스틱 회귀 (Logistic Regression)**                                                                                                                                              | **인공신경망 (TensorFlow Implementation)**                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Step 1<br><br>  <br><br>Model Output | **수식 직접 정의**<br><br>  <br><br>$z = w \cdot x + b$<br><br>  <br><br>$f(x) = g(z) = \frac{1}{1 + e^{-z}}$                                                                        | **네트워크 아키텍처 정의 (`Sequential`)**<br><br>  <br><br>`model = Sequential([`<br><br>  <br><br>`Dense(25, activation='sigmoid'),`<br><br>  <br><br>`Dense(15, activation='sigmoid'),`<br><br>  <br><br>`Dense(1, activation='sigmoid')`<br><br>  <br><br>`])` |
| Step 2<br><br>  <br><br>Loss & Cost  | **손실/비용 함수 수식**<br><br>  <br><br>• Loss: $L(f(x), y) = -y\log f(x) - (1-y)\log(1-f(x))$<br><br>  <br><br>• Cost: $J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)})$ | **손실 함수 지정 (`compile`)**<br><br>  <br><br>`model.compile(`<br><br>  <br><br>`loss=tf.keras.losses.BinaryCrossentropy()`<br><br>  <br><br>`)`<br><br>  <br><br>_(전체 샘플에 대한 평균 비용 계산 자동 포함)_                                                              |
| Step 3<br><br>  <br><br>Minimize     | **경사하강법 수식 적용**<br><br>  <br><br>$w := w - \alpha \frac{\partial J}{\partial w}$<br><br>  <br><br>$b := b - \alpha \frac{\partial J}{\partial b}$                              | **학습 및 역전파 실행 (`fit`)**<br><br>  <br><br>`model.fit(X, y, epochs=100)`<br><br>  <br><br>*(내부적으로 **역전파(Backpropagation)*_를 통해 편미분값 계산)_                                                                                                                  |

### 2). 세부 단계별 깊이 있는 설명

#### Step 1: 입력-출력 관계 정의 (Model Architecture)

- TensorFlow 코드의 `Sequential` 내 각 레이어 명시는 **전체 신경망의 모든 가중치($W^{[l]}$)와 편향($b^{[l]}$) 파라미터를 정의**하는 과정입니다.
    
- 이를 통해 입력 $X$가 들어왔을 때 최종 출력 $A^{[L]}$ (또는 $f(x)$)를 계산해 내는 순전파(Forward Propagation)의 모든 경로가 완벽히 결정됩니다.
    

#### Step 2: 손실 함수(Loss)와 비용 함수(Cost)의 설정

- **이진 분류 (Binary Classification)**:
    
    - **Binary Cross-Entropy**: $L(f(x), y) = -y \log f(x) - (1 - y) \log (1 - f(x))$
        
    - 통계학의 'Cross-Entropy' 개념에서 유래하였으며, $y=0$ 또는 $1$인 이진 분류 문제에 사용됩니다.
        
    - TensorFlow 표기: `tf.keras.losses.BinaryCrossentropy()`
        
- **회귀 (Regression)**:
    
    - **Mean Squared Error (MSE)**: $L(f(x), y) = \frac{1}{2}(f(x) - y)^2$
        
    - 예측값과 실제값의 차이의 제곱을 최소화합니다.
        
    - TensorFlow 표기: `tf.keras.losses.MeanSquaredError()`
        
- **용어 구분**:
    
    - **Loss (손실)**: 단일 훈련 샘플 $(x^{(i)}, y^{(i)})$에 대한 예측 오차.
        
    - **Cost (비용, $J(W, B)$)**: 전체 $m$개 훈련 데이터셋 전체에 대한 Loss의 평균값.
        

#### Step 3: 비용 함수 최소화 (Optimization & Backpropagation)

- 신경망의 모든 레이어 $l$과 뉴런 $j$에 대해 파라미터를 업데이트합니다:
    
    $$W_{j}^{[l]} := W_{j}^{[l]} - \alpha \frac{\partial J(W, B)}{\partial W_{j}^{[l]}}$$
    
    $$b_{j}^{[l]} := b_{j}^{[l]} - \alpha \frac{\partial J(W, B)}{\partial b_{j}^{[l]}}$$
    
- TensorFlow의 `model.fit()` 함수는 내부적으로 **역전파(Backpropagation)** 알고리즘을 사용해 수많은 파라미터에 대한 편미분값($\frac{\partial J}{\partial W}, \frac{\partial J}{\partial b}$)을 자동으로 계산하고 업데이트합니다.
    


---
---
---
# < Activation Functions >
## 1. 시그모이드 대안

### 0). 시그모이드의 문제점
- **기존 방식**: 로지스틱 회귀에서 확장했기 때문에 은닉층(Hidden Layer)과 출력층(Output Layer)의 모든 노드에 **시그모이드(Sigmoid)** 함수를 사용했습니다.
    
- **문제점**: 시그모이드는 출력값이 항상 $0 \sim 1$ 사이로 제한됩니다.
	
- **예시 (의류 인지도 예측)**:

	- 인지도(Awareness)를 단순 이진값($0$ 또는 $1$)이나 $0 \sim 1$ 사이 확률로만 표현하기에는 한계가 있습니다.
    
	- 바이럴이 일어나거나 인지도가 폭발하는 경우, 인지도 값은 $0$부터 아주 큰 양수까지 자유롭게 가질 수 있어야 더 정확한 모델링이 가능합니다.

### 1). 주요 활성화 함수 3가지 비교

|**활성화 함수**|**수학적 수식 g(z)**|**출력 범위**|**주요 특징 및 특징**|
|---|---|---|---|
|**Sigmoid**|$g(z) = \frac{1}{1 + e^{-z}}$|$(0, 1)$|확률 값을 예측할 때 주로 사용.|
|**ReLU**<br><br>  <br><br>_(Rectified Linear Unit)_|$g(z) = \max(0, z)$|$[0, \infty)$|$z < 0$이면 $0$, $z \ge 0$이면 $z$ 값 그대로 출력.<br><br>  <br><br>딥러닝 은닉층에서 **가장 흔하게 사용**됨.|
|**Linear**|$g(z) = z$|$(-\infty, \infty)$|입력값을 그대로 반환.<br><br>  <br><br>실제 현업에서는 "활성화 함수를 쓰지 않는다(No activation function)"고 표현하기도 함.|

---
---
## 2. 활성 함수 선택

### 1). 출력층(Output Layer) 활성화 함수 선택

출력층의 활성화 함수는 예측하고자 하는 타깃 변수 $y$의 성격(타입 및 범위)에 따라 직관적으로 결정됩니다.

| **문제 유형**                                            | **타깃 라벨 (y)의 범주**                                         | **추천 활성화 함수**                          | **이유**                                          |
| ---------------------------------------------------- | --------------------------------------------------------- | -------------------------------------- | ----------------------------------------------- |
| **이진 분류 (Binary Classification)**                    | $y \in \{0, 1\}$                                          | **Sigmoid**                            | $0 \sim 1$ 사이의 값으로 출력되어 $y=1$일 확률을 표현하기에 완벽함    |
| **회귀 (Regression)**<br><br>  <br><br>_(음수/양수 모두 가능)_ | $y \in (-\infty, \infty)$<br><br>  <br><br>_(예: 주가 변동 폭)_ | **Linear**<br><br>  <br><br>_(또는 미사용)_ | 제한 없이 음수부터 양수까지 자유로운 연속값을 출력해야 하기 때문            |
| **회귀 (Regression)**<br><br>  <br><br>_(0 또는 양수만 가능)_ | $y \in [0, \infty)$<br><br>  <br><br>_(예: 주택 가격)_         | **ReLU**                               | 주택 가격처럼 음수가 될 수 없는 비음수(Non-negative) 데이터 예측에 적합 |

---
### 2). 은닉층(Hidden Layer) 활성화 함수 선택: 왜 ReLU가 기본인가?

초기 딥러닝에서는 은닉층에도 Sigmoid를 자주 썼으나, 현재 연구자와 실무자들은 ReLU(Rectified Linear Unit)를 디폴트(Default)로 사용합니다.

#### 이유 1: 빠른 연산 속도

- **Sigmoid**: 지수 연산($e^{-z}$), 나눗셈 등 복잡한 계산이 필요함.
    
- **ReLU**: $\max(0, z)$ 단순 비교 연산만 수행하므로 계산 비용이 매우 저렴함.
    

#### 이유 2: 경사 하강법(Gradient Descent) 학습 속도 향상

- **Sigmoid의 한계 (기울기 소멸)**: 그래프의 좌측($z \ll 0$)과 우측($z \gg 0$) 영역이 매우 평평(Flat)합니다. 기울기(Gradient)가 $0$에 가까워지면 경사 하강법에 의한 가중치 업데이트가 극도로 느려집니다.
    
- **ReLU의 장점**: 좌측($z < 0$) 영역만 평평하고, 양수 영역($z > 0$)에서는 기울기가 항상 $1$로 일정합니다. 따라서 비용 함수 $J(W, B)$의 경사가 완만해지는 현상이 줄어들어 **모델의 학습 속도가 훨씬 빠릅니다.**
    
---
### 3). TensorFlow 구현 예시

`tf.keras.layers.Dense` 구축 시 `activation` 파라미터로 손쉽게 지정합니다.

``` Python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    # 은닉층 (Hidden Layers): 기본적으로 ReLU 사용
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    
    # 출력층 (Output Layer): 문제 유형에 맞춰 선택
    # 1) 이진 분류인 경우: activation='sigmoid'
    # 2) 음수/양수 연속값 회귀인 경우: activation='linear'
    # 3) 비음수 연속값 회귀인 경우: activation='relu'
    Dense(units=1, activation='sigmoid')
])
```

---
---
## 3. 왜 활성 함수가 필요한가?

### 1). 핵심 결론

> **"선형 함수들의 합성(Linear Combination)은 결국 또 다른 하나의 선형 함수에 불과하다."**

- 모든 은닉층에 선형 활성화 함수($g(z) = z$)를 사용하면, 아무리 층(Layer)을 깊게 쌓아도 **단순한 선형 회귀(Linear Regression) 또는 로지스틱 회귀(Logistic Regression) 모델과 완벽히 동일**해집니다.
    
- 복잡하고 비선형적인 패턴을 학습하기 위해서는 은닉층에 반드시 비선형 활성화 함수(예: ReLU)를 써야 합니다.
    
---
### 2). 수학적 증명 (1개 은닉층 예시)

입력 $x$가 들어오는 간단한 신경망을 통해 증명해 보겠습니다.

```
Input x  --->  [ Hidden Layer 1 (a1) ]  --->  [ Output Layer 2 (a2) ]
```

#### (1) 각 층의 계산 수식

- **은닉층 1 ($a_1$)**: $a_1 = g(w_1 x + b_1)$
    
- **출력층 2 ($a_2$)**: $a_2 = g(w_2 a_1 + b_2)$
    

#### (2) 선형 활성화 함수($g(z) = z$)를 대입할 경우

$g(z) = z$이므로 활성화 함수 $g$를 제거하고 $a_1$ 수식을 $a_2$에 대입합니다.

$$a_2 = w_2 (w_1 x + b_1) + b_2$$

$$a_2 = (w_2 w_1) x + (w_2 b_1 + b_2)$$

여기서 $(w_2 w_1)$을 하나의 새로운 가중치 $w$, $(w_2 b_1 + b_2)$를 새로운 편향 $b$로 치환해 보면:

$$a_2 = w x + b$$

> **결과**: 은닉층을 거쳤음에도 불구하고, 출력은 단 하나의 레이어로 이루어진 **일반 선형 회귀($f(x) = wx + b$) 수식과 완전히 동일**해집니다.

---
### 3). 일반화: 다층 신경망에서의 케이스 비교

|**은닉층(Hidden Layers) 활성화 함수**|**출력층(Output Layer) 활성화 함수**|**최종 모델의 결과**|
|---|---|---|
|**Linear**|**Linear**|**선형 회귀 (Linear Regression)**와 완벽히 동일|
|**Linear**|**Sigmoid**|**로지스틱 회귀 (Logistic Regression)**와 완벽히 동일|
|**ReLU (비선형)**|**문제 유형에 맞게 선택**|**복잡한 비선형 패턴을 학습하는 다층 신경망(Neural Network)**|

---
### 4). 실무에서의 기본 규칙 (Rule of Thumb)

- **은닉층(Hidden Layer)의 권장 사항**:
    
    - 선형 활성화 함수(Linear)는 절대 쓰지 마세요.
        
    - 특별한 이유가 없다면 **`ReLU`를 디폴트로 사용**하는 것이 안전하고 성능이 좋습니다.


---
---
---
# < Multiclass Classification >
## 1. Multiclass란?

### 1).개념

- **이진 분류 (Binary Classification)**: 타깃 라벨 $y$가 오직 2개만 존재 ($y \in \{0, 1\}$).
    
- **다중 클래스 분류 (Multi-class Classification)**: 타깃 라벨 $y$가 **3개 이상의 불연속적인 카테고리(Discrete Categories)** 중 하나를 가질 때의 분류 문제 ($y \in \{1, 2, 3, 4, \dots\}$ 또는 $y \in \{0, 1, 2, \dots, K-1\}$).
    
### 2). 예시

|**분야**|**예시**|**가능한 출력 클래스 (y)**|
|---|---|---|
|**손글씨 숫자 인식**|우편번호(Zip Code) 자동 인식|$0, 1, 2, 3, 4, 5, 6, 7, 8, 9$ (총 10개)|
|**의료 진단**|환자 상태/질병 진단|질병 A, 질병 B, 질병 C, 건강함 (총 4개)|
|**제조업 품질 검사**|알약/부품 결함 시각 검사|스크래치(Scratch), 변색(Discoloration), 깨짐(Chip), 정상 (총 4개)|

### 3). 이진 분류 vs 다중 클래스 분류 비교

```
[이진 분류]                            [다중 클래스 분류]
  X2 ^                                   X2 ^
     |   O   O                              |   O  O     X  X
     |  O   O   X                           |   O  O     X  X
     |        X   X                         |     ▲  ▲   ■  ■
     |      X   X                           |     ▲  ▲   ■  ■
  ---+--------------> X1                 ---+--------------> X1
   (0과 1 두 영역 구분)                   (결정 경계가 공간을 여러 영역으로 분할)
```

|**구분**|**이진 분류 (Binary Classification)**|**다중 클래스 분류 (Multi-class Classification)**|
|---|---|---|
|**출력값 ($y$)**|$y \in \{0, 1\}$|$y \in \{1, 2, \dots, K\}$|
|**예측 목표**|$P(y=1 \mid x)$ (단일 확률)|$P(y=1 \mid x), P(y=2 \mid x), \dots, P(y=K \mid x)$<br><br>  <br><br>_(각 클래스별 발생 확률을 각각 추정)_|
|**결정 경계 (Decision Boundary)**|공간을 **2개** 구역으로 분할|공간을 **$K$개** 구역으로 분할|


---
---
## 2. Softmax에 대해
- 
- **핵심 결론**: $N=2$일 때 소프트맥스 회귀는 로지스틱 회귀와 수학적으로 완벽히 동일해지며, 따라서 소프트맥스는 로지스틱 회귀의 **일반화(Generalization) 버전**입니다.
### 1). 로지스틱 회귀에서 소프트맥스로의 확장

- **로지스틱 회귀 (2개 클래스)**:
    
    - 출력 $a_1 = P(y=1 \mid x)$를 계산하면, $a_2 = P(y=0 \mid x) = 1 - a_1$로 자동 결정됩니다.
        
    - 총 확률의 합: $a_1 + a_2 = 1$
        
- **소프트맥스 회귀 ($N$개 클래스)**:
    
    - 클래스가 $1, 2, \dots, N$개 존재할 때, 각 클래스에 속할 확률 $a_1, a_2, \dots, a_N$을 동시에 계산합니다.
        
    - 모든 확률의 합: $\sum_{j=1}^{N} a_j = 1$
        
---

### 2). 소프트맥스 수식 구조 (How it Works)

총 $N$개의 클래스가 있을 때 각 클래스 $j$에 대한 계산 과정은 다음과 같습니다.

#### (1단계): 선형 결합 계산 ($z_j$)

각 클래스별 독립된 가중치($W_j$)와 편향($b_j$)을 적용합니다.

$$z_j = W_j \cdot x + b_j \quad (j = 1, 2, \dots, N)$$

#### (2단계): 소프트맥스 활성화 함수 적용 ($a_j$)

지수 함수($e^{z}$)를 통해 모든 값을 양수로 만든 뒤, **전체 합으로 나누어 확률값(0~1)으로 정규화**합니다.

$$a_j = \frac{e^{z_j}}{\sum_{k=1}^{N} e^{z_k}} = P(y = j \mid x)$$

> **퀴즈 예시**: 만약 4개 클래스에서 $a_1 = 0.30$, $a_2 = 0.20$, $a_3 = 0.15$라면?
> 
> $\sum a_j = 1$이어야 하므로, **$a_4 = 1 - (0.30 + 0.20 + 0.15) = 0.35$**가 됩니다.

---

### 3). 소프트맥스의 손실 함수 (Loss Function)

실제 정답 라벨이 $y = j$일 때의 손실 함수(Cross-Entropy Loss)는 다음과 같이 정의됩니다.

$$\text{Loss}(a_1, \dots, a_N, y) = -\log(a_j) \quad (\text{단, } y=j\text{ 일 때})$$

- **작동 원리**:
    
    - 실제 정답 클래스 $j$에 대해 모델이 예측한 확률 $a_j$가 **1에 가까울수록** $\rightarrow$ 손실(Loss)은 **0에 가까워짐**.
        
    - 실제 정답 클래스 $j$에 대해 모델이 예측한 확률 $a_j$가 **0에 가까울수록** $\rightarrow$ 손실(Loss)은 **무한대($\infty$)로 급격히 증가**.
        
- **효과**: 모델이 실제 정답 클래스의 확률 $a_j$를 1에 최대한 가깝게 만들도록 강하게 유도(Incentivize)합니다.
    

---
---
## 3. Neural Network with Softmax output
### 1). 신경망 아키텍처 (예: 10개 숫자 인식 모델)

기존의 이진 분류 모델(출력 노드 1개)을 10개 숫자($0 \sim 9$)를 인식하는 다중 클래스 분류 모델로 확장하려면 **출력층의 노드 수를 10개로 변경**하고 **Softmax 활성화 함수**를 적용합니다.

```
[Input X] ---> [Hidden 1: 25 Units (ReLU)] ---> [Hidden 2: 15 Units (ReLU)] ---> [Output Layer: 10 Units (Softmax)]
                                                                                     │
                                                                                     └──> Outputs: a_1, a_2, ..., a_10
```

- **출력값**: $a_1, a_2, \dots, a_{10}$ 은 각각 입력 데이터 $x$가 $0, 1, \dots, 9$일 확률을 의미합니다.
    
- **확률의 합**: $a_1 + a_2 + \dots + a_{10} = 1$
    
### 2). Softmax만의 독특한 특징 (중요)

Sigmoid, ReLU, Linear 함수와 Softmax 함수는 결정적인 차이점이 있습니다.

- **Sigmoid / ReLU / Linear**:
    
    - 각 노드의 출력값 $a_i$는 오직 자기 자신 노드의 입력값 $z_i$에 의해서만 결정됩니다.
        
    - 예: $a_1 = g(z_1)$, $a_2 = g(z_2)$ (독립적 계산)
        
- **Softmax**:
    
    - $a_1$을 계산할 때 $z_1$뿐만 아니라 **모든 $z_1, z_2, \dots, z_{10}$ 값이 동시에 필요**합니다.
        
    - 수식: $a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + \dots + e^{z_{10}}}$
        
    - 즉, 모든 출력 활성화 값이 **서로 연결되어 동시에(Simultaneously) 계산**됩니다.
        

### 3). TensorFlow 기본 구현법 (개념적 코드)

``` Python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 모델 정의
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='softmax')  # 10개 클래스 출력을 위해 Softmax 적용
])

# 2. 컴파일 (손실 함수 지정)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

# 3. 학습
model.fit(X, Y, epochs=10)
```

---
---
## 4. Improved Implementation of softmax
- 컴퓨터의 부동소수점 오차 문제와 이를 해결하기 위해 
- TensorFlow에서 Softmax/Logistic Loss를 더욱 정확하게 계산하는 최적화 테크닉(`from_logits=True`)을 설명합니다.
- **핵심 결론**: 수치적 오차를 줄이고 안정적인 학습을 하려면 **출력층 활성화 함수를 `'linear'`로 두고 손실 함수에 `from_logits=True` 옵션을 지정**하는 것이 실무 표준입니다. (Logistic / Softmax 모두 적용 가능)
### 1). 문제가 발생하는 이유
- 부동소수점 수치 오차 (Numerical Round-off Error)

- 컴퓨터는 메모리가 한정되어 있어 실수를 **부동소수점(Floating-point)** 형태로 저장합니다. 이 과정에서 중간 계산 결과가 지나치게 크거나 작아지면 미세한 오차가 누적됩니다.

- **예시**: $x = \frac{2}{10000}$ 을 계산할 때,
    
    - **방식 A**: `2 / 10000` $\rightarrow$ `0.0002` (정확함)
        
    - **방식 B**: `(1 + 1/10000) - (1 - 1/10000)` $\rightarrow$ `0.00019999999999999575` (수치 오차 발생)
        

Softmax의 지수 함수($e^z$) 계산 시 $z$ 값이 커지거나 작아지면 수치 반올림 오차가 커져 **학습 불안정성**을 초과합니다.

### 2). 해결책
- 중간값 $A$ 생략하고 직접 결합하기

중간 활성화 값 $a$ (지수 함수 결과)를 굳이 중간 단계에서 독립적으로 구하지 않고, **$z$ 값(Logit)을 손실 함수(Loss)에 직접 전달**하면 TensorFlow가 수식을 내부적으로 더 안정적인 형태(재배치)로 변환해 연산합니다.

- **기존 방식**: $z \xrightarrow{\text{Softmax}} a \xrightarrow{\text{Loss}} \text{Loss}$
    
- **개선 방식 (from_logits=True)**: $z \xrightarrow[\text{통합 최적화 연산}]{\text{Softmax + Loss}} \text{Loss}$
    

### 3). TensorFlow 구현 비교 (★ 실무 필수 패턴)

#### (1).권장하지 않는 기존 방식 



``` Python
# 1. 모델 정의: 출력층에 직접 softmax 적용
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='softmax')  # <-- 중간값 a를 직접 계산
])

# 2. 컴파일
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
)
```

#### (2). 권장하는 개선된 방식 (구현 2: `from_logits=True`)

``` Python
# 1. 모델 정의: 출력층을 linear로 설정 (z 값인 Logit을 그대로 출력)
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')  # <-- Softmax 제거
])

# 2. 컴파일: from_logits=True 파라미터 추가
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
```

> 💡 **Logit(로짓)이란?**
> 
> 활성화 함수(Sigmoid나 Softmax)를 거치기 전의 **순수한 선형 계산 결과값($z = W \cdot a + b$)**을 의미합니다.

---

### 4). 예측(Inference) 시 차이점

`activation='linear'` 방식을 적용하면 모델의 출력값(`model.predict(X)`)은 확률값 $a$가 아닌 Logit $z$가 됩니다. 따라서 확률값을 구하려면 마지막에 Softmax(또나 Sigmoid)를 통과시켜 주어야 합니다.


``` Python
# 모델 예측 (결과는 Logit z_1 ~ z_10)
logits = model.predict(X_new)

# 확률값으로 변환
probabilities = tf.nn.softmax(logits)
```

---
---
## 5. Classification with multiple outputs

### 1). Multi-class vs Multi-label 핵심 비교

- **Multi-class Classification (다중 클래스 분류)**
    
    - **정의**: 여러 개의 후보 클래스 중 **단 하나의 정답**만 선택하는 문제 ($y$는 단일 숫자).
        
    - **예시**: 손글씨 숫자 인식 ($0 \sim 9$ 중 1개만 정답), 고양이/개/새 중 하나 분류.
        
    - **출력 방식**: 출력층 노드 $N$개 + **Softmax** 활성화 함수 ($\sum P = 1$).
        
- **Multi-label Classification (다중 라벨 분류)**
    
    - **정의**: 하나의 데이터(이미지)에 **여러 개의 정답이 동시에 존재**할 수 있는 문제 ($y$는 벡터).
        
    - **예시**: 자율주행 카메라 - [자동차 유무, 버스 유무, 보행자 유무]를 한 번에 검출.
        
    - **출력 벡터 예시**: $y = [1, 0, 1]$ (자동차 있음, 버스 없음, 보행자 있음).
        
---
### 2). Multi-label 구현 방법 2가지

#### 방법 A: 각각의 분류기를 별도로 학습 (3개의 독립된 모델)

- 모델 1: 자동차가 있는가? ($y_1 \in \{0, 1\}$)
    
- 모델 2: 버스가 있는가? ($y_2 \in \{0, 1\}$)
    
- 모델 3: 보행자가 있는가? ($y_3 \in \{0, 1\}$)
    
- **단점**: 모델을 3번 구축하고 학습시켜야 하므로 비효율적입니다.
    

#### 방법 B: 단일 신경망으로 동시에 예측 (★ 권장 방식)

하나의 신경망 출력층에 **여러 개의 독립된 이진 분류 노드**를 배치합니다.

```
[Input X] ---> [Hidden Layer 1] ---> [Hidden Layer 2] ---> [Output Layer (3 Units)]
                                                                │
                                                                ├──> Node 1 (Sigmoid): P(Car)
                                                                ├──> Node 2 (Sigmoid): P(Bus)
                                                                └──> Node 3 (Sigmoid): P(Pedestrian)
```
---
### 3). 신경망 구조 및 TensorFlow 코드 차이점

다중 라벨 분류는 결국 **여러 개의 이진 분류(Binary Classification) 문제를 한 번에 처리**하는 것입니다.


```Python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Multi-label 신경망 모델 정의
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    # 각 출력이 독립된 이진 분류이므로 Softmax 대신 Sigmoid 적용!
    Dense(3, activation='sigmoid')  
])

# 손실 함수도 각 출력별 이진 교차 엔트로피 적용
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam()
)
```

|**구분**|**Multi-class (다중 클래스)**|**Multi-label (다중 라벨)**|
|---|---|---|
|**출력 라벨 ($y$)**|단일 값 (예: $y = 2$)|벡터 (예: $y = [1, 0, 1]$)|
|**출력층 활성화 함수**|**Softmax** (노드 간 확률 합이 1)|**Sigmoid** (각 노드가 독립적인 0~1 확률)|
|**손실 함수 (Loss)**|`SparseCategoricalCrossentropy`|`BinaryCrossentropy`|

---
---
---
# < Additional Neural Network Concepts >

## 1. Advanced Optimization

### 1). 경사 하강법의 한계와 Adam의 필요성

기존 경사 하강법은 고정된 하나의 학습률(Global Learning Rate $\alpha$)을 전체 파라미터에 동일하게 적용합니다. 이로 인해 두 가지 비효율이 발생합니다.

```
[상황 1: 학습률이 너무 작음]                 [상황 2: 학습률이 너무 큼]
  * 먼 길을 아주 작은 걸음으로 계속 감.     * 최소점을 지나치며 지그재그로 진동
  * 속도가 매우 느림.                   * 발산하거나 수렴이 매우 더딤.
  ---------------------------------------------------------------
  해결책: 진행 방향을 보고 "학습률을 자동으로 키우거나 줄여줄 수 있다면?"
```

### 2). Adam 알고리즘의 핵심 직관 

**Adam**은 **Adaptive Moment Estimation**의 약자로, 핵심 개념은 "모든 파라미터마다 학습률을 다르게, 그리고 자동으로 조정한다"입니다.

- **동적 조정 원리**:
    
    1. **동일한 방향으로 계속 이동하는 파라미터 ($w_j$)**:
        
        "잘 가고 있네!" $\rightarrow$ 해당 파라미터의 학습률($\alpha_j$)을 **키워서 빠르게 보폭을 넓힙니다.**
        
    2. **지그재그로 진동하는 파라미터 ($w_k$)**:
        
        "너무 왔다 갔다 하네!" $\rightarrow$ 해당 파라미터의 학습률($\alpha_k$)을 **줄여서 안정적으로 수렴시킵니다.**
        

> 📌 **결과적으로**:
> 
> 모델이 $w_1 \sim w_{10}$ 및 $b$를 갖고 있다면, Adam은 각 파라미터에 맞춰 11개의 서로 다른 학습률($\alpha_1 \sim \alpha_{11}$)을 개별적으로 적응(Adapt)시키며 업데이트합니다.

### 3). TensorFlow 구현 방법

사용법은 매우 단순합니다. `model.compile()`에서 옵티마이저를 `tf.keras.optimizers.Adam`으로 지정해 주기만 하면 됩니다.

```Python
import tensorflow as tf

# 1. 모델 정의 (예시)
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')
])

# 2. Adam 옵티마이저 설정
# 초기 학습률(Initial Learning Rate)을 지정해 줍니다 (기본 권장값: 1e-3 또는 0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 3. 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### 4). 실무에서의 Adam 특징 및 조언

|**구분**|**기존 경사 하강법 (Standard GD)**|**Adam Optimization**|
|---|---|---|
|**학습률 조정**|고정된 단일 학습률 ($\alpha$)|파라미터별 자동 적응형 학습률 ($\alpha_j$)|
|**학습 속도**|상대적으로 느림|**훨씬 빠름 (Fast Convergence)**|
|**민감도**|학습률 초깃값 선택에 매우 민감|**민감도가 낮고 훨씬 견고함 (Robust)**|

- **초기 학습률 하이퍼파라미터 튜닝**:
    
	- Adam이 학습률을 자동으로 조절해 주더라도, 시작점(기본 학습률)에 따른 차이가 존재합니다. 보통 `1e-3` (0.001), `1e-2` (0.01), `1e-4` (0.0001) 등의 값을 시도해 보며 가장 학습이 잘 되는 값을 찾는 것이 좋습니다.
    
---
---
## 2. Additional Layer Types
- 기존에 사용하던 Dense Layer(완전 연결층) 외에, Convolutional Layer(합성곱 층)의 개념과 왜 이러한 레이어를 사용하는지를 다룬다.

### 1). Dense Layer vs. Convolutional Layer

- **Dense Layer (완전 연결층)**:
    
    - 이전 층의 모든 활성화값(All Activations)을 다음 층의 각 노드로 연결하여 계산합니다.
        
    - 모델이 강력하지만, 입력 데이터의 크기가 커질수록 파라미터 수가 급격히 늘어납니다.
        
- **Convolutional Layer (합성곱 층)**:
    
    - 각 뉴런이 입력 데이터의 특정 국소 영역(Limited Region/Window)만 바라보고 특징을 추출합니다.
        
    - 대표적으로 **얀 르쿤(Yann LeCun)** 교수님이 개발하고 대중화한 구조입니다.
        

### 2). Convolutional Layer의 주요 장점

1. **계산 속도 향상 (Speed Up Computation)**:
    
    - 연결된 가중치(Parameters)의 수가 적어져 계산 연산량이 크게 줄어듭니다.
        
2. **필요 데이터량 감소 및 과적합 예방 (Less Training Data & Less Overfitting)**:
    
    - 이미지나 신호 데이터의 공간적/시간적 국소 특징(Local Patterns)에 집중하므로 적은 데이터로도 학습이 잘 되며, 과적합(Overfitting) 발생 가능성이 낮아집니다.
        

### 3). 작동 예시 (EKG 심전도 데이터 기준)

강의에서는 이해를 돕기 위해 1D EKG(심전도) 시간 순서 데이터를 예시로 설명합니다.

- **입력**: $x_1 \sim x_{100}$ (100개의 시간별 심전도 측정값)
    
- **첫 번째 Convolutional Layer**:
    
    - **Neuron 1**: $x_1 \sim x_{20}$ 구간만 관찰
        
    - **Neuron 2**: $x_{11} \sim x_{30}$ 구간만 관찰 (작은 윈도우 이동)
        
    - **Neuron 3**: $x_{21} \sim x_{40}$ 구간만 관찰 ...
        
- **두 번째 Convolutional Layer**:
    
    - 이전 층에서 생성된 출력값들 중 일부 윈도우만 조합하여 더 상위 개념의 특징을 추출.
        
- **출력층 (Sigmoid)**:
    
    - 최종 추출된 특징들을 바탕으로 심장 질환 유무(0 또는 1)를 이진 분류.

---
---
---
# < Back Propagation >
## 1.미분


``` Python
import sympy as sp

# 1. 심볼 및 함수 정의
W = sp.Symbol('W')
J = W**2

# 2. 미분 계산 (J를 W에 대해 미분)
dJ_dW = sp.diff(J, W)  # 출력: 2*W

# 3. 특정 W 값 대입
result = dJ_dW.subs(W, 2)  # 출력: 4
```

---
---
## 2. Computation Graph

### 1). 핵심 개념 

> **계산 그래프(Computation Graph)**는 신경망의 복잡한 연산을 여러 노드로 쪼개어 나타낸 그림입니다.
> 
> - 순전파 (Forward Prop, 왼쪽 $\rightarrow$ 오른쪽)**: 입력값으로 **손실/비용 함수($J$)를 계산합니다.
>     
> - 역전파 (Backprop, 오른쪽 $\rightarrow$ 왼쪽)**: 체인 룰(Chain Rule)을 활용해 끝에서부터 거꾸로 올며 **모든 파라미터의 미분값(기울기)을 계산합니다.
>     

### 2). 예제 모델 구조 (선형 회귀 신경망)

강의에서는 1개의 출력 노드를 가진 아주 간단한 신경망 모델을 예시로 듭니다.

- **입력 및 파라미터 조건**:
    
    - 입력: $X = -2$, 실제 정답: $Y = 2$
        
    - 가중치 및 편향: $W = 2$, $B = 8$
        
- **연산 순서 (순전파)**:
    
    1. $C = W \cdot X = 2 \times (-2) = -4$
        
    2. $A = C + B = -4 + 8 = 4$ _(예측값)_
        
    3. $D = A - Y = 4 - 2 = 2$ _(오차)_
        
    4. $J = \frac{1}{2} D^2 = \frac{1}{2} \times 2^2 = \mathbf{2}$ _(최종 비용)_
        

### 3). 역전파(Backprop) 계산 과정 (오른쪽 $\rightarrow$ 왼쪽)

- 목표는 **"비용 $J$를 줄이기 위해 $W$와 $B$를 얼마큼 수정해야 하는가?"** 즉, $\frac{\partial J}{\partial W}$ 와 $\frac{\partial J}{\partial B}$를 구하는 것입니다.
- **역전파(Backpropagation)의 진짜 목적:**
    
    $W$와 $B$ 자체를 구하는 것이 아니라, "비용함수 $J$를 $W$와 $B$로 각각 편미분한 값(기울기, Gradient)"을 구하는 것입니다.
    
- **최종 파라미터($W, B$) 업데이트:**
    
    역전파가 구해준 그 미분값을 넘겨받아서, **경사하강법(Gradient Descent)이 $W$와 $B$의 값을 실제로 업데이트**합니다.

$$\begin{aligned} \text{Step 1 (오차 노드 미분):} \quad & \frac{\partial J}{\partial D} = D = \mathbf{2} \\ \text{Step 2 (예측값 노드 미분):} \quad & \frac{\partial J}{\partial A} = \frac{\partial J}{\partial D} \times 1 = \mathbf{2} \\ \text{Step 3 (중간/편향 미분):} \quad & \frac{\partial J}{\partial C} = \mathbf{2}, \quad \mathbf{\frac{\partial J}{\partial B} = 2} \\ \text{Step 4 (가중치 미분):} \quad & \mathbf{\frac{\partial J}{\partial W}} = \frac{\partial J}{\partial C} \times X = 2 \times (-2) = \mathbf{-4} \end{aligned}$$

- **결과 해석**:
    
    - $\frac{\partial J}{\partial B} = 2$: $B$를 $+0.001$ 올리면 $J$는 $+0.002$ 증가함.
        
    - $\frac{\partial J}{\partial W} = -4$: $W$를 $+0.001$ 올리면 $J$는 $-0.004$ 감소함 (기울기가 음수이므로 $W$를 키워야 함).
        

### 4). 왜 '오른쪽에서 왼쪽(역방향)'으로 계산할까? (효율성)

중간 연산 결과인 $\frac{\partial J}{\partial A}$ (값: 2)를 한 번 계산해 두면, 이를 재사용해서 **$W$의 미분값과 $B$의 미분값을 동시에 구해낼 수 있기 때문**입니다.

```
                  ┌──> [ W 미분 계산 ]  (중간값 2 재사용)
[J 미분] ──> [A 미분 (2)] ──┤
                  └──> [ B 미분 계산 ]  (중간값 2 재사용)
```

#### (1) 연산 복잡도 비교 (노드 $N$개, 파라미터 $P$개 기준)

- **일반적인 개별 미분 방식**: $N \times P$ 번의 연산 필요
    
    _(예: 노드 1만 개, 파라미터 10만 개 $\rightarrow$ **10억 번** 연산)_
    
- **역전파(Backpropagation) 방식**: $N + P$ 번의 연산으로 종료
    
    _(예: 노드 1만 개, 파라미터 10만 개 $\rightarrow$ **11만 번** 연산)_
    

> 💡 **결론**: 역전파 알고리즘은 거대한 신경망도 순식간에 학습시킬 수 있도록 계산량을 극적으로 줄여주는 **딥러닝의 핵심 기술**입니다. TensorFlow나 PyTorch 같은 프레임워크가 바로 이 방식을 통해 자동 미분을 수행합니다.


---
---
## 3. Larger Neural Network 예시


### 1). 신경망 모델 구조 (은닉층 1개)

- **입력 및 정답**: $x = 1$, $y = 5$
    
- **파라미터**: $w_1 = 2, b_1 = 0$, $w_2 = 3, b_2 = 1$
    
- **활성화 함수**: ReLU 계열 ($g(z) = \max(0, z)$)
    

####  순전파 (Forward Prop) 계산 단계

1. **은닉층**:
    
    - $z_1 = w_1 x + b_1 = 2(1) + 0 = 2$
        
    - $a_1 = g(z_1) = 2$
        
2. **출력층**:
    
    - $z_2 = w_2 a_1 + b_2 = 3(2) + 1 = 7$
        
    - $a_2 = g(z_2) = 7$ _(최종 예측값)_
        
3. **비용 함수 (Squared Error)**:
    
    - $J = \frac{1}{2}(a_2 - y)^2 = \frac{1}{2}(7 - 5)^2 = \mathbf{2}$
        

---

### 2). 계산 그래프로 나타낸 순전파/역전파 흐름

```
[입력 x, w1, b1] ──> [z1 = 2]──>[a1 = 2] ──> [z2 = 7] ──> [a2 = 7] ──> [비용 J = 2]
                                                                              │
  <── [∂J/∂w1 = 6] <─────────────── [역전파 (Chain Rule)] <───────────────────┘
```

#### 역전파 검증 예시 ($w_1$에 대한 미분)

- 역전파를 수행하면 $\frac{\partial J}{\partial w_1} = 6$이 나옵니다.
    
- **직관적 검증**: $w_1$을 $2$에서 $2.001$로 $\epsilon(0.001)$만큼 미세하게 올리면:
    
    - $a_1 \rightarrow 2.001$
        
    - $a_2 \rightarrow 7.003$
        
    - $J \rightarrow 2.00605$
        
- 비용 $J$가 약 $0.006$ (즉, $\mathbf{6} \times 0.001$)만큼 증가하므로, 기울기(미분값)가 **6**이 맞음을 직접 확인할 수 있습니다.
    
---

### 3). 역전파를 쓰는 이유: 압도적인 연산 효율성

|**구분**|**파라미터 하나씩 수동 변화 확인**|**역전파 (Backprop)**|
|---|---|---|
|**방향**|왼쪽 $\rightarrow$ 오른쪽|오른쪽 $\rightarrow$ 왼쪽|
|**연산량**|**$N \times P$** 번 (노드 수 $\times$ 파라미터 수)|**$N + P$** 번|
|**비고**|파라미터가 수억 개일 때 계산 불가능|거대한 신경망도 단 한 번의 과정으로 모든 기울기 계산|

- 역전파로 빠르게 구해낸 기울기들($\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial b_1}, \dots$)은 경사하강법(Gradient Descent)이나 **Adam** 같은 최적화 알고리즘에 전달되어 모델을 학습시킵니다.
    
---

### 4). 핵심 시사점: 자동 미분 (Automatic Differentiation)

> **"과거와 달리, 이제는 복잡한 미분 공식을 손으로 직접 풀 필요가 없습니다."**

- **과거**: 연구자들이 모든 미분 방정식을 종이에 직접 유도하고 코드로 일일이 구현해야 했습니다.
    
- **현재**: TensorFlow, PyTorch 같은 프레임워크가 계산 그래프를 기반으로 자동 미분(Autodiff)을 수행해 줍니다.
    
- **결과**: 개발자는 **순전파(Forward Prop)** 구조만 정의하면, 프레임워크가 알아서 역전파를 처리하므로 딥러닝 입문 및 활용에 필요한 수학적 장벽이 대폭 낮아졌습니다.

