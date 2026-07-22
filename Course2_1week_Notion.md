# < Intro>
## 1. 인공신경망(Neural Networks)의 기원과 역사

- **동기:** 수십 년 전, 인간/생물학적 뇌의 학습 방식과 사고 방식을 모방하는 소프트웨어를 만들기 위해 처음 발명되었습니다.
    
- **역사적 흐름:**
    
    - **1950년대:** 신경망에 대한 초기 연구 시작.
        
    - **1980년대 ~ 1990년대 초:** 필기체 숫자 인식(우편번호 정렬, 수표 금액 인식 등)에 성공적으로 적용되며 큰 인기를 얻음.
        
    - **1990년대 후반:** 다시 관심도가 떨어지며 침체기 진입.
        
    - **2005년 이후:** 딥러닝(Deep Learning)이라는 브랜드 재정립과 함께 화려하게 부활.
        
- **용어적 특징:** '인공신경망(Artificial Neural Networks)'과 '딥러닝(Deep Learning)'은 본질적으로 같은 의미를 지니지만, "딥(Deep)"이라는 명칭이 주는 강렬함 덕분에 브랜딩 측면에서 크게 성공했습니다.
    

## 2. 응용 분야의 혁신 및 확장

딥러닝의 부활 이후 여러 산업 분야를 차례대로 혁신했습니다.

1. **음성 인식 (Speech Recognition):** Li Deng, Geoff Hinton 등의 연구로 현대 딥러닝이 가장 먼저 큰 파급력을 불러일으킨 분야.
    
2. **컴퓨터 비전 (Computer Vision):** **2012년 ImageNet 모멘트**를 계기로 대중과 학계에 엄청난 쇼크를 주며 폭발적 성장.
    
3. **자연어 처리 (NLP):** 텍스트 이해 및 언어 모델 분야로 확산.
    
4. **다양한 분야로의 확장:** 기후 변화, 의료 영상 diagnostics, 온라인 광고, 제품 추천 시스템 등 현재 머신러닝의 대부분 영역에서 핵심 기술로 활용 중.
    

## 3. 생물학적 뇌 vs. 인공신경망 (Biological vs. Artificial)

### (1) 생물학적 뉴런 (Biological Neuron)

- **수상돌기 (Dendrites):** 다른 뉴런으로부터 전기적 신호(Input)를 받는 입력선.
    
- **세포체/핵 (Cell Body / Nucleus):** 입력받은 신호를 집계 및 계산 처리.
    
- **축삭 (Axon):** 계산된 전기 신호를 다른 뉴런으로 전달하는 출력선(Output wire).
    

### (2) 인공 뉴런 (Artificial Neuron)

- 생물학적 뉴런의 동작을 **극도로 단순화한 수학적 모델**.
    
- 하나의 뉴런은 숫자를 입력받아 계산을 수행한 후 새로운 숫자(Output)를 출력하며, 이 출력이 다음 뉴런의 입력이 됩니다.
    
- 실제 모델 구축 시 단일 뉴런이 아닌, **수많은 인공 뉴런을 동시에 시뮬레이션**하여 망을 형성합니다.
    

> **⚠️ 주의사항 (Andrew Ng의 조언):**
> 
> - 현시점에서 인공신경망은 생물학적 뇌가 작동하는 방식과 **사실상 거의 아무런 관련이 없습니다.**
>     
> - 뇌과학은 여전히 발전 단계이며 우리는 뇌의 작동 원리를 극히 일부만 알고 있습니다.
>     
> - 따라서 뇌를 맹목적으로 모방하기보다, **공학적 원리(Engineering Principles)**를 바탕으로 알고리즘 성능을 개선하는 방향으로 발전해 왔습니다.


## 4. 왜 최근에야 딥러닝이 폭발했을까?
- **데이터의 폭발적 증가 (Big Data):**
    
    - 인터넷, 스마트폰, 디지털화로 인해 종이 문서가 디지털 데이터로 전환되며 처리해야 할 데이터 양이 기하급수적으로 늘어남.
        
    - **기존 머신러닝 (Linear/Logistic Regression):** 데이터 양이 아무리 늘어나도 일정 수준 이상으로 성능이 올라가지 않고 정체됨.
        
- **신경망의 스케일링 특성:**
    
    - 데이터량이 많을 때, **신경망의 크기(뉴런의 개수/층의 깊이)를 키울수록 성능이 한계 없이 계속 상승**하는 현상을 발견함.
        
- **하드웨어의 발전 (GPU의 등장):**
    
    - 대규모 신경망을 학습시키려면 수많은 단순 연산을 빠르게 처리해야 함.
        
    - 원래 그래픽 처리용이었던 GPU(Graphics Processing Unit)가 딥러닝 연산에 매우 적합함이 밝혀지면서 대용량 모델 학습이 가능해짐.

---
## 5. Demand Prediction

### 1). 단일 뉴런(Neuron)과 로지스틱 회귀

- **뉴런의 정의:** 신경망에서 뉴런은 하나 이상의 숫자를 입력받아 간단한 계산을 거쳐 새로운 숫자를 출력하는 **'작은 컴퓨터'** 역할을 합니다.
    
- **활성화(Activation, $a$):** 뉴런이 출력하는 결과값을 의미하며, 뇌의 생물학적 뉴런이 다음 뉴런으로 신호를 보낼 때의 "활성화 정도"에서 유래된 용어입니다.
    
- **로지스틱 회귀와의 관계:** 하나의 뉴런은 가장 단순한 형태의 로지스틱 회귀(Logistic Regression) 모델로 볼 수 있습니다.
    
    - _예시:_ 티셔츠 가격($x$)을 입력받아 **시그모이드 함수**를 거쳐 "Top Seller가 될 확률($a$)"을 예측.
        
    - $a = \frac{1}{1 + e^{-(wx + b)}}$


### 2). 신경망의 구조 (티셔츠 판매 예측 예시)

여러 개의 뉴런을 연결하여 더 복잡한 예측 모델을 만들 수 있습니다.

|**구분**|**역할 및 특징**|**예시 요소**|
|---|---|---|
|**입력층 (Input Layer)**|모델에 입력되는 원본 특성(Feature) 벡터 $x$|가격, 배송비, 마케팅비, 재질 품질|
|**은닉층 (Hidden Layer)**|입력값을 조합하여 새로운 의미 있는 특성(Activation) 생성|구매 가능성, 브랜드 인지도, 체감 품질|
|**출력층 (Output Layer)**|은닉층의 출력을 받아 최종 예측값을 출력|최종 Top Seller 여부 (확률)|


### 3). 신경망의 핵심 강점: 자동 특징 공학 (Automated Feature Engineering)

- **기존 방식:** 사람이 직접 특성을 조합해야 했음 (예: 집값을 예측하기 위해 가로×세로를 곱해 '면적'이라는 특성을 직접 계산).
    
- **신경망 방식:**
    
    - 모든 입력 특성을 은닉층의 모든 뉴런에 연결해 줍니다.
        
    - 신경망이 학습 과정을 통해 **어떤 입력 데이터에 가중치를 두고 어떤 데이터를 무시할지 스스로 판단**하여 최적의 특성을 알아서 찾아냅니다.

### 4). 신경망 아키텍처 (Architecture)

- **다층 퍼셉트론 (Multi-layer Perceptron, MLP):** 은닉층이 여러 개 연결된 형태의 신경망.
    
- **아키텍처 결정 요소:** 은닉층을 몇 개로 만들지, 각 은닉층에 뉴런을 몇 개 둘지 결정하는 것이 신경망 구조 설계의 핵심이며, 이는 모델의 성능에 큰 영향을 미칩니다.




### 5). 이미지의 컴퓨터 내부 표현 (Input Representation)

- **입력 예시:** $1,000 \times 1,000$ 픽셀 크기의 얼굴 이미지.
    
- **컴퓨터 내 표현:** $1,000 \times 1,000$ 행렬(Matrix) 형태의 픽셀 밝기/강도(Pixel Intensity) 값.
    
    - **픽셀 값 범위:** $0$ ~ $255$ (예: 좌상단 $197$, 우하단 $214$ 등).
        
- **특성 벡터(Feature Vector) 변환:**
    
    - $1,000 \times 1,000$ 행렬을 일렬로 펼치면(Unroll) **총 100만 개($1,000,000$)의 요소**를 가진 입력 벡터 $x$가 됩니다.
        
- **최종 목표:** 100만 개의 픽셀 값을 입력받아 해당 인물의 정체(Identity) 또는 확률을 예측하는 모델 구축.
    

### 6). 계층별 특징 추출 과정 (Feature Hierarchy)

신경망을 수많은 얼굴 이미지로 학습시킨 후, 각 은닉층(Hidden Layer)의 뉴런이 무엇을 감지하도록 학습되었는지 시각화해보면 단계별로 점진적으로 고차원적 특징을 찾아냅니다.

```
[입력 픽셀 데이터] 
       ↓
[은닉층 1] 짧은 선 / 가장자리 (Edges)
       ↓
[은닉층 2] 얼굴의 부분적 요소 (Parts of Face: 눈, 코, 귀)
       ↓
[은닉층 3] 전체적인 얼굴 형태 (Coarser Face Shapes)
       ↓
[출력층] 최종 인물 식별 (Identity Detection)
```

|**구분**|**감지하는 특징 (Feature Detector)**|**수용 영역 (Region Size)**|
|---|---|---|
|**첫 번째 은닉층 ($L_1$)**|수직선, 기울어진 선 등 아주 **짧은 선이나 가장자리(Edges)** 감지|이미지의 매우 작은 영역(Small Window)|
|**두 번째 은닉층 ($L_2$)**|선들을 조합하여 **눈, 코, 귀 하단** 등 **얼굴의 부분적 요소(Parts of Face)** 감지|$L_1$보다 더 넓은 영역|
|**세 번째 은닉층 ($L_3$)**|부위들을 조합하여 **더 크고 완성된 형태의 얼굴 윤곽 및 모양** 감지|$L_2$보다 훨씬 더 넓은 영역|
|**출력층 (Output Layer)**|완성된 풍부한 특징 세트를 기반으로 최종 **인물의 정체(Probability of Identity)** 예측|이미지 전체 영역 고려|

### 7). 신경망의 자동 학습 능력 (Self-Learning)

- **자동 특징 학습:** 사용자가 신경망에게 "첫 번째 층에서는 선을 찾고, 두 번째 층에서는 눈과 코를 찾아라"라고 **지시한 적이 없습니다.**
    
- 신경망은 데이터만을 가지고 **스스로(All by itself)** 각 은닉층에 필요한 특징 감지기(Feature Detector)를 도출해 냅니다.
    

### 8). 데이터 변경에 따른 적응성 (자동차 감지 예시)

- 동일한 구조의 학습 알고리즘이라도 **자동차(Car) 이미지 데이터셋**을 주면 다음과 같이 스스로 학습 방향을 바꿉니다.
    
    - **1단계 층:** 선 및 가장자리 감지 (동일)
        
    - **2단계 층:** 바퀴, 범퍼, 창문 등 **자동차 부품(Parts of Cars)** 감지
        
    - **3단계 층:** 더 완전한 **자동차 전체 형태(Car Shapes)** 감지
        
- 즉, 데이터만 바꾸어 주면 **동일한 신경망 알고리즘이 해당 도메인에 맞는 완전히 새로운 특징을 스스로 학습**합니다.

----
----
----
# < Neural Network Layer>


## 1. 핵심 개념: 신경망 구조

현대의 신경망은 여러 개의 연속된 층(layer)으로 구성됩니다.

- **Layer 0 (입력층, Input Layer):** 입력 특성(feature)을 포함하는 초기 입력 벡터 $x$입니다.
    
- **Layer 1 (은닉층, Hidden Layer):** 여러 개의 뉴런(로지스틱 회귀 유닛)으로 구성된 중간 층입니다.
    
- **Layer 2 (출력층, Output Layer):** 네트워크의 최종 예측을 생성하는 마지막 층입니다 (이진 분류의 경우 보통 1개의 뉴런으로 구성).
    

## 2. Layer 1 (은닉층) 연산

영상 예시에서 **Layer 1**은 4개의 입력 특성($x$)을 받아 **3개의 뉴런**을 통해 처리합니다. 각 뉴런은 독립적인 로지스틱 회귀 유닛(logistic regression unit)처럼 작동합니다.

### Layer 1의 각 뉴런 $j$에 대하여:

1. **선형 결합 ($z$):** 입력 벡터 $x$와 해당 뉴런의 가중치(weight) 벡터 $w$의 내적(dot product)에 편향(bias) $b$를 더하여 계산합니다.
    
2. **활성화 ($a$):** 시그모이드(로지스틱) 활성화 함수 $g(z) = \frac{1}{1 + e^{-z}}$를 적용하여 $z$ 값을 0과 1 사이의 확률값으로 변환합니다.
    

### 수학적 표현:

- **Neuron 1:**
    $$a_1^{[1]} = g\left(w_1^{[1]} \cdot x + b_1^{[1]}\right) \quad \text{(예: 0.3)}$$
    
- **Neuron 2:**
    $$a_2^{[1]} = g\left(w_2^{[1]} \cdot x + b_2^{[1]}\right) \quad \text{(예: 0.7)}$$
    
- **Neuron 3:**
    $$a_3^{[1]} = g\left(w_3^{[1]} \cdot x + b_3^{[1]}\right) \quad \text{(예: 0.2)}$$
    

이 출력값들을 하나로 묶으면 Layer 1의 활성화 벡터(activation vector)가 완성됩니다:

$$\vec{a}^{[1]} = \begin{bmatrix} a_1^{[1]} \\ a_2^{[1]} \\ a_3^{[1]} \end{bmatrix} = \begin{bmatrix} 0.3 \\ 0.7 \\ 0.2 \end{bmatrix}$$

## 3. Layer 2 (출력층) 연산

Layer 1에서 계산된 출력 벡터 $\vec{a}^{[1]}$은 **Layer 2**의 직접적인 입력값이 됩니다.

이 예시의 이진 분류 문제에서는 Layer 2에 단 1개의 뉴런만 존재하므로 다음과 같이 연산합니다:

1. **출력층 연산:**
    
    $$a_1^{[2]} = g\left(w_1^{[2]} \cdot \vec{a}^{[1]} + b_1^{[2]}\right)$$
    
2. **최종 확률 도출:** 계산 결과가 만약 0.84라면, 이는 베스트셀러가 될 확률이 84%라는 의미입니다.
    

## 4. 최종 선택 단계: 임계값 설정 (이진 예측)

단순한 확률값이 아닌, 0 또는 1로 떨어지는 명확한 이진 예측값($\hat{y}$)을 원한다면 임계값(threshold)을 설정할 수 있습니다. 주로 0.5를 기준으로 합니다.

$$\hat{y} = \begin{cases} 1 & \text{if } a_1^{[2]} \ge 0.5 \\ 0 & \text{if } a_1^{[2]} < 0.5 \end{cases}$$

도출된 확률 0.84는 0.5 이상이므로, 신경망의 최종 예측은 1 (참)이 됩니다.


---
## 5. 다층 신경망(Multi-layer Neural Network)의 기본 구조

- **층(Layer) 개수 세는 법:**
    
    - 입력층(Input Layer)은 **Layer 0**으로 부르며, 전체 층 수를 셈할 때 **포함하지 않는 것**이 컨벤션(관례)입니다.
        
    - 영상의 예시 모델은 총 4개 층(4-layer neural network)으로 구성되어 있습니다.
        
        - **Layer 0:** 입력층 (Input Layer, $x$)
            
        - **Layer 1, 2, 3:** 은닉층 (Hidden Layers)
            
        - **Layer 4:** 출력층 (Output Layer)
            

### 1). Layer 3(세 번째 은닉층)의 연산 과정

Layer 3가 이전 층(Layer 2)으로부터 입력 벡터 $\vec{a}^{[2]}$를 받아 $3$개의 뉴런(Hidden units)을 통해 출력 벡터 $\vec{a}^{[3]}$를 만드는 연산 과정입니다.

#### 각 뉴런의 세부 연산:

- **Neuron 1:**
    
    $$a_1^{[3]} = g\left(\vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]}\right)$$
    
- **Neuron 2:**
    
    $$a_2^{[3]} = g\left(\vec{w}_2^{[3]} \cdot \vec{a}^{[2]} + b_2^{[3]}\right)$$
    
- **Neuron 3:**
    
    $$a_3^{[3]} = g\left(\vec{w}_3^{[3]} \cdot \vec{a}^{[2]} + b_3^{[3]}\right)$$
    

#### 결과 출력 벡터:

$$\vec{a}^{[3]} = \begin{bmatrix} a_1^{[3]} \\ a_2^{[3]} \\ a_3^{[3]} \end{bmatrix}$$



---
---
---
# < TensorFlow Implementation>
- 예시를 들어 설명한다.

## 1. 추론 방법

- **주요 프레임워크:** 딥러닝 분야의 대표 프레임워크 중 하나인 **TensorFlow**를 활용해 순전파(Inference/Forward Propagation) 코드를 구현합니다.
    
- **예시 시나리오:** Coffee Roasting (커피 원두 로스팅)
    
    - **입력 피처 ($x$):** 로스팅 온도(Temperature, °C)와 로스팅 시간(Duration, 분)
        
    - **타겟 ($y$):** 맛있는 커피가 되는지 여부 ($y=1$: Good Coffee, $y=0$: Bad Coffee)
        
    - **특징:** 온도가 너무 낮거나/높거나, 시간이 너무 짧거나/길면 맛이 없어지므로 특정 적정 영역(삼각형 범위) 내에서만 $y=1$이 됩니다.
        

### 1). 커피 로스팅 예제의 TensorFlow 코드 구현

TensorFlow의 `Dense` 레이어(전결합층/밀집층)를 사용하여 2층 구조 신경망의 추론 과정을 구현합니다.


``` Python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 1. 입력 데이터 준비 (온도 200도, 시간 17분)
x = np.array([[200.0, 17.0]])

# 2. Layer 1 (은닉층: 3개 뉴런, Sigmoid 활성화 함수)
layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)  # 예: [0.2, 0.7, 0.3] 형태의 활성화 벡터 출력

# 3. Layer 2 (출력층: 1개 뉴런, Sigmoid 활성화 함수)
layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)  # 예: [0.8] 형태의 스칼라/확률 출력

# 4. 임계값(Threshold) 적용을 통한 최종 예측 (y_hat)
if a2 >= 0.5:
    y_hat = 1
else:
    y_hat = 0
```

### 2). 손글씨 숫자 인식(Digit Classification) 예제 구현

64개 피처(8x8 이미지)를 입력받아 0 또는 1을 분류하는 3층 신경망 구조입니다.


```Python
# x: 64개 픽셀 강도 값을 담은 NumPy 배열

# Layer 1: 25개 유닛
layer_1 = Dense(units=25, activation='sigmoid')
a1 = layer_1(x)

# Layer 2: 15개 유닛
layer_2 = Dense(units=15, activation='sigmoid')
a2 = layer_2(a1)

# Layer 3: 1개 유닛 (출력층)
layer_3 = Dense(units=1, activation='sigmoid')
a3 = layer_3(a2)

# 최종 판단
y_hat = 1 if a3 >= 0.5 else 0
```

| **용어/문법**        | **설명**                                                       |
| ---------------- | ------------------------------------------------------------ |
| **`Dense`**      | 우리가 다루는 표준적인 신경망 레이어(가중치 $W$와 편향 $b$를 이용한 내적 및 활성화 함수 연산 수행) |
| **`units`**      | 해당 레이어에 존재하는 **뉴런(유닛)의 개수**                                  |
| **`activation`** | 해당 레이어에서 사용할 **활성화 함수** (예: `'sigmoid'`)                     |
| **`layer_1(x)`** | 레이어 객체 자체가 함수처럼 동작하여 입력 $x$를 받아 출력 활성화 값 $a$를 반환             |


---
---

## 2. Data in TensorFlow

### 1). 개요
NumPy와 TensorFlow는 생성된 시기와 개발 목적이 달라 데이터 표현 방식에 약간의 불일치가 존재합니다.

- **NumPy:** 파이썬 표준 선형대수 라이브러리
    
- **TensorFlow:** 구글 브레인(Google Brain) 팀에서 대규모 데이터의 고성능 연산을 위해 개발한 딥러닝 프레임워크
    
- **핵심 차이:** NumPy는 기본 1차원 배열(1D Array)을 흔히 사용하지만, **TensorFlow는 계산 효율성을 위해 모든 데이터를 2차원 행렬(2D Matrix/Tensor) 형태로 다루는 것을 선호**합니다.
    
---

### 2). 행렬(Matrix)과 차원 개념 정리

행렬의 차원은 **행(Row)의 개수 $\times$ 열(Column)의 개수**로 표기합니다.

|**차원**|**형태 예시**|**NumPy 생성 코드**|**설명**|
|---|---|---|---|
|**$2 \times 3$ 행렬**|2개 행, 3개 열|`np.array([[1, 2, 3], [4, 5, 6]])`|2차원 배열 (2D Array)|
|**$4 \times 2$ 행렬**|4개 행, 2개 열|`np.array([[1, 2], [3, 4], [5, 6], [7, 8]])`|2차원 배열 (2D Array)|
|**$1 \times 2$ 행렬**|1개 행, 2개 열|`np.array([[200, 17]])`|**행 벡터 (Row Vector)** / 대괄호 2개 `[[ ]]`|
|**$2 \times 1$ 행렬**|2개 행, 1개 열|`np.array([[200], [17]])`|**열 벡터 (Column Vector)** / 각 원소마다 대괄호|
|**1D Vector**|행/열 구분 없음|`np.array([200, 17])`|**1차원 배열** / 대괄호 1개 `[ ]`|

---
### 3). TensorFlow에서의 데이터 표현

TensorFlow 내부에서는 데이터를 다룰 때 고유의 자료구조인 `tf.Tensor`를 사용합니다.

- **Tensor란?**
    
    - 행렬 개념을 일반화한 전용 데이터 타입으로, 메모리 효율적인 대규모 연산을 지원합니다.
        
- **데이터 형태 (Shape) 및 타입:**
    
    - 커피 로스팅 예제 입력 데이터 `X`: **Shape = `(1, 2)`** ($1 \times 2$ 행렬)
        
    - 데이터 타입은 주로 **`float32`** (32비트 부동소수점 실수)를 사용합니다.
        
---
### 4). 신경망 각 레이어의 출력 형태 

커피 로스팅 예제에서 훈련 샘플 하나($x = [[200, 17]]$)가 입력될 때의 레이어별 출력 형태입니다.

#### (1) Layer 1 (은닉층: 3개 유닛)

- **연산:** `a1 = layer_1(x)`
    
- **출력 데이터:** **Shape = `(1, 3)`** ($1 \times 3$ 행렬)
    
- **출력 예시:**
    
    ```
    tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
    ```
    

#### (2) Layer 2 (출력층: 1개 유닛)

- **연산:** `a2 = layer_2(a1)`
    
- **출력 데이터:** **Shape = `(1, 1)`** ($1 \times 1$ 행렬)
    
- **출력 예시:**
    
    ```
    tf.Tensor([[0.8]], shape=(1, 1), dtype=float32)
    ```
    
---
### 5). Tensor $\leftrightarrow$ NumPy 배열 변환 방법

TensorFlow Tensor 형태의 데이터를 표준 NumPy 배열로 다시 변환하려면 **`.numpy()`** 메서드를 호출합니다.

``` Python
# Tensor를 NumPy 배열로 변환
a1_np = a1.numpy()
a2_np = a2.numpy()

# 출력 형태: np.ndarray([[0.8]])
```


---
---
## 3. Building a neural Network
### 1). 개요: 수동 연산에서 `Sequential` API로의 전환

이전 수업에서는 각 레이어를 수동으로 생성하고, $X \rightarrow a^{[1]} \rightarrow a^{[2]}$와 같이 이전 층의 활성화 값을 다음 층으로 직접 전달하며 순전파(Forward Propagation)를 수행했습니다.

TensorFlow에서는 이를 훨씬 간결하게 처리하기 위해 **`Sequential` 모델**을 사용합니다. 여러 레이어를 순서대로 연결하여 하나의 네트워크로 자동 엮어주는 방식입니다.

---
### 2). TensorFlow 신경망 구축 파이프라인

#### (1) 모델 정의 (Model Architecture)

변수에 레이어를 하나씩 할당하는 대신, `Sequential` 내부 리스트에 레이어를 직접 전달하는 방식을 선호합니다.

```Python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 커피 로스팅 예제 (2개 입력 -> 은닉층 3개 -> 출력층 1개)
model = Sequential([
    Dense(units=3, activation='sigmoid'),  # Layer 1
    Dense(units=1, activation='sigmoid')   # Layer 2
])
```

#### (2) 데이터 준비 (Data Representation)

- **입력 특성 ($X$)**: 2차원 행렬(2D Matrix) 형태 (예: $4 \times 2$ 크기의 NumPy 배열)
    
- **타겟 레이블 ($y$)**: 1차원 배열(1D Array) 형태 (예: `[1, 0, 0, 1]`)
    
```Python
import numpy as np

X = np.array([
    [200.0, 17.0],
    [120.0, 5.0],
    [300.0, 20.0],
    [110.0, 4.0]
])
y = np.array([1, 0, 0, 1])
```

#### (3) 학습 수행 (`compile` & `fit`)

수동으로 가중치를 업데이트할 필요 없이 단 두 줄의 코드로 학습이 완료됩니다.

- `model.compile(...)`: 손실 함수 및 최적화 기법 설정
    
- `model.fit(X, y)`: 데이터를 전달하여 모델 학습 실행
    

#### (4) 추론 및 예측 (`predict`)

새로운 데이터 `X_new`에 대한 예측은 레이어를 일일이 거칠 필요 없이 **`model.predict()`** 한 번으로 내부 순전파가 자동 처리됩니다.

```Python
X_new = np.array([[200.0, 17.0]])
prediction = model.predict(X_new)  # 최종 활성화 값 a2 반환
```
---
### 3). 주요 예제 코드 구조 비교

|**구분**|**커피 로스팅 예제**|**손글씨 숫자 분류 예제**|
|---|---|---|
|**은닉층 1**|`Dense(units=3, activation='sigmoid')`|`Dense(units=25, activation='sigmoid')`|
|**은닉층 2**|-|`Dense(units=15, activation='sigmoid')`|
|**출력층**|`Dense(units=1, activation='sigmoid')`|`Dense(units=1, activation='sigmoid')`|
|**전체 구조**|입력 $\rightarrow$ 3 $\rightarrow$ 1|입력 $\rightarrow$ 25 $\rightarrow$ 15 $\rightarrow$ 1|

---
---
---
# < Neural Network Implementation in Python>

## 1. 파이썬으로 바닥부터 구현
### 1). 단일 뉴런 및 Layer 1 연산 구현 (커피 로스팅 예제)

이번 예제에서는 연산 이해를 돕기 위해 2차원 행렬이 아닌 1차원 NumPy 배열(1D Array)을 사용합니다.

#### 입력 데이터 ($x$) 및 표현 규칙

- `w1_1`: Layer 1의 1번째 뉴런 가중치 벡터 ($w_1^{[1]}$)
    
- `b1_1`: Layer 1의 1번째 뉴런 편향 스칼라 ($b_1^{[1]}$)
    
#### Layer 1 (은닉층: 3개 뉴런) 연산 과정
```Python


import numpy as np

# Sigmoid 활성화 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 입력 데이터 x (온도, 시간)
x = np.array([200, 17])

# --- Neuron 1 연산 ---
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z1_1)

# --- Neuron 2 연산 ---
w1_2 = np.array([-3, 4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

# --- Neuron 3 연산 ---
w1_3 = np.array([5, -6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z1_3)

# Layer 1의 최종 활성화 벡터 a1 도출
a1 = np.array([a1_1, a1_2, a1_3])
```

### 2). Layer 2 (출력층) 연산 구현

Layer 1의 출력값인 `a1` 벡터를 입력받아 최종 출력값 `a2`를 계산합니다.

``` Python
# Layer 2의 파라미터 (1개 뉴런)
w2_1 = np.array([-5, 1, 2])
b2_1 = np.array([-3])

# 선형 결합 및 시그모이드 적용
z2_1 = np.dot(w2_1, a1) + b2_1
a2_1 = sigmoid(z2_1)

# Layer 2의 최종 출력 (a2)
a2 = a2_1
```

---
---
## 2. 일반적인  Forward Propagation
### 1). 일반화 구현의 목적

- 이전 수업처럼 각 뉴런마다 코드를 일일이 하드코딩하는 대신, **반복문(`for-loop`)과 행렬 표기법**을 활용해 뉴런의 개수나 층의 깊이에 상관없이 동작하는 일반화된 순전파 함수를 작성합니다.
    
- **디버깅 능력 향상**: TensorFlow나 PyTorch 내부에서 실제 어떤 방식으로 연산이 수행되는지 알면, 모델이 예상대로 동작하지 않거나 버그가 발생했을 때 해결하는 역량이 크게 향상됩니다.
    
---
### 2). 단일 층 구현: `dense()` 함수

하나의 신경망 층(Dense Layer)을 수행하는 함수입니다. 이전 층의 활성화 벡터 `a_in`과 현재 층의 파라미터 `W`, `b`를 입력받아 현재 층의 활성화 벡터 `a_out`을 반환합니다.

#### 표기법 규칙 (Notational Conventions)

- **대문자 `W`**: 선형대수학 관례에 따라 가중치들이 열(Column) 단위로 쌓인 2차원 행렬(Matrix)을 나타냅니다.
    
- **소문자 `b` 및 `w`**: 벡터(Vector) 또는 스칼라(Scalar)를 나타냅니다.
    

#### dense() 함수 내부 동작 순서



``` Python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense(a_in, W, b):
    """
    a_in: 이전 층의 활성화 벡터 (입력)
    W: 현재 층의 가중치 행렬 (shape: 이전 층 뉴런 수 x 현재 층 뉴런 수)
    b: 현재 층의 편향 벡터 (shape: 현재 층 뉴런 수,)
    """
    # 1. 현재 층의 뉴런(유닛) 개수 추출 (W 행렬의 열 개수)
    units = W.shape[1]
    
    # 2. 출력 활성화 벡터 초기화 (0으로 채워진 1D 배열 생성)
    a_out = np.zeros(units)
    
    # 3. 각 뉴런별 연산 수행 (for-loop)
    for j in range(units):
        # W 행렬에서 j번째 열(j번째 뉴런의 가중치 벡터) 슬라이싱 추출
        w = W[:, j]
        
        # 선형 결합: 가중치와 입력의 내적 + 편향
        z = np.dot(w, a_in) + b[j]
        
        # 활성화 함수(시그모이드) 적용
        a_out[j] = sigmoid(z)
        
    return a_out
```

### 3). 다층 신경망의 순전파 연결: sequential() 구현

정의한 `dense()` 함수를 순차적으로 호출하여 전체 신경망의 순전파(Forward Propagation)를 완성합니다.


``` Python

def sequential(x, W1, b1, W2, b2, W3, b3, W4, b4):
    """
    4개 층으로 구성된 신경망의 순전파 연산
    """
    # Layer 1 연산 (입력 x -> a1)
    a1 = dense(x, W1, b1)
    
    # Layer 2 연산 (a1 -> a2)
    a2 = dense(a1, W2, b2)
    
    # Layer 3 연산 (a2 -> a3)
    a3 = dense(a2, W3, b3)
    
    # Layer 4 연산 (a3 -> a4 / 최종 출력)
    a4 = dense(a3, W4, b4)
    
    # 최종 예측값 f(x) 반환
    f_x = a4
    return f_x
```

### 4). 핵심 요약

|**요소**|**역할 및 핵심 내용**|
|---|---|
|**가중치 행렬 $W$**|각 뉴런의 가중치 벡터가 **열(Column) 방향**으로 묶여 있는 2차원 행렬|
|**`W[:, j]` 슬라이싱**|$W$ 행렬에서 **$j$번째 뉴런에 해당하는 가중치 벡터**를 추출하는 파이썬/NumPy 문법|
|**`dense()` 함수**|이전 층의 출력(`a_in`)과 $W, b$를 입력받아 현재 층의 출력을 반환하는 기본 블록|
|**`sequential()` 함수**|`dense()` 함수를 체인처럼 연결하여 최종 예측값 $f(x)$를 도출하는 전체 파이프라인|

---
---
---
# < Vectorization >

## 1. 신경망의 효율적인 구현 : 벡터화
### 1). 벡터화(Vectorization)의 중요성

- **딥러닝 스케일업의 핵심**: 지난 10년간 신경망의 규모를 수백~수천 배 이상 확대할 수 있었던 주요 원인은 벡터화(Vectorization)를 통해 신경망 연산을 극도로 효율화했기 때문입니다.
    
- **하드웨어 최적화**: GPU 및 현대 CPU의 병렬 처리 하드웨어는 대규모 **행렬 곱(Matrix Multiplication)** 연산을 고속으로 처리하는 데 최적화되어 있습니다.
    
---

### 2). 루프(For-loop) 방식 vs 벡터화(Vectorized) 방식 비교

단일 Dense Layer의 순전파 연산을 구현할 때, 기존의 반복문 방식과 행렬 연산을 활용한 벡터화 방식을 비교합니다.

#### (1) 기존 반복문 방식 (dense 구현)

각 뉴런의 가중치와 편향을 `for` 루프로 하나씩 접근하여 계산하는 방식입니다.

``` Python
# 기존 방식: 뉴런 개수만큼 for-loop 반복
def dense_for_loop(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out
```

#### (2) 벡터화(Vectorized) 방식 구현

반복문을 제거하고, 모든 연산을 2차원 행렬(2D Array)의 행렬 곱으로 일괄 처리합니다.


``` Python
import numpy as np

def dense_vectorized(A_in, W, B):
    """
    A_in : 입력 행렬 (2D Array, shape: 1 x n)
    W    : 가중치 행렬 (2D Array, shape: n x units)
    B    : 편향 행렬 (2D Array, shape: 1 x units)
    """
    # 1. 행렬 곱과 편향 더하기 (np.matmul 활용)
    Z = np.matmul(A_in, W) + B
    
    # 2. 활성화 함수(sigmoid)를 행렬 Z의 모든 요소에 적용 (Element-wise)
    A_out = g(Z)
    
    return A_out
```
---
### 3). 주요 차이점 및 특징

- **모든 변수의 2차원 행렬화 (2D Arrays)**:
    
    - 입력 `x` (또는 `A_in`), 가중치 `W`, 편향 `B` (Capital B 사용), 선형결합 `Z`, 출력 `A_out` 모두 **2차원 행렬** 형태(NumPy의 `[ [ ... ] ]` 이중 대괄호)를 갖습니다.
        
- **`np.matmul` 활용**:
    
    - NumPy의 `np.matmul(A_in, W)`을 사용하여 전체 뉴런의 선형 결합을 단 한 번의 행렬 곱 연산으로 처리합니다.
        
- **요소별(Element-wise) 활성화 함수 적용**:
    
    - 행렬 $Z$에 시그모이드 함수 $g(Z)$를 적용하면, 행렬 내의 모든 원소에 개별적으로 함수가 적용되어 $A_{out}$ 행렬이 완성됩니다.

---
---

## 2. Matrix 곱 코드

### 1). NumPy에서의 행렬 연산 및 전치(Transpose)

파이썬 NumPy 라이브러리를 사용해 행렬 전치와 행렬 곱을 수행하는 방법입니다.

- **행렬 전치 (Transpose)**:
        
    ```     Python
    # A의 열(Column)들을 행(Row)으로 눕혀서 AT 생성
    AT = A.T  # 또는 np.transpose(A)
    ```
    
- **행렬 곱셈 (Matrix Multiplication)**:
    
    ```     Python
    # NP.matmul 사용 (권장)
    Z = np.matmul(AT, W)
    
    # 파이썬 @ 연산자 사용 (대안)
    Z = AT @ W
    ```
    
---
### 2). 신경망 순전파(Forward Prop) 벡터화 수식 및 연산 과정

입력 데이터와 가중치, 편향을 모두 2차원 행렬(2D Array)로 구성하여 단 한 줄의 행렬 곱 연산으로 전체 레이어를 계산합니다.

#### 입력 및 파라미터 차원 (Shape)

- **입력 데이터 ($A_{in}$ 또는 $A^T$)**: $1 \times 2$ 행렬 $\rightarrow \begin{bmatrix} 200 & 17 \end{bmatrix}$ (온도 200도, 로스팅 시간 17분)
    
- **가중치 행렬 ($W$)**: $2 \times 3$ 행렬 $\rightarrow$ 각 뉴런의 가중치 $W_1, W_2, W_3$를 열(Column) 단위로 결합
    
- **편향 행렬 ($B$)**: $1 \times 3$ 행렬 $\rightarrow \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix}$
    

#### 연산 순서

1. **선형 결합 ($Z$)**:
    
    $$Z = A_{in} W + B$$
    
    - $A_{in}$과 $W$의 1열 내적 + $b_1 \rightarrow z_1 = 165$
        
    - $A_{in}$과 $W$의 2열 내적 + $b_2 \rightarrow z_2 = -531$
        
    - $A_{in}$과 $W$의 3열 내적 + $b_3 \rightarrow z_3 = 900$
        
    - **결과 $Z$**: $\begin{bmatrix} 165 & -531 & 900 \end{bmatrix}$ ($1 \times 3$ 행렬)
        
2. **활성화 함수 적용 ($A_{out}$)**:
    
    $$A_{out} = g(Z)$$
    
    - 시그모이드 함수 $g(z)$를 행렬 $Z$의 각 원소에 요소별(Element-wise)로 적용합니다.
        
    - $g(165) \approx 1$, $g(-531) \approx 0$, $g(900) \approx 1$
        
    - **최종 출력 $A_{out}$**: $\begin{bmatrix} 1 & 0 & 1 \end{bmatrix}$ ($1 \times 3$ 행렬)
        
---

### 3). 최종 벡터화 Dense Layer 파이썬 코드

반복문(`for-loop`) 없이 단 두 줄의 연산으로 순전파가 완성됩니다.

``` Python
import numpy as np


def dense_vectorized(A_in, W, B):
    """A_in : 입력 행렬 (1 x 2)

    W    : 가중치 행렬 (2 x 3)
    B    : 편향 행렬 (1 x 3)
    """
    # 1. 행렬 곱과 편향 더하기
    Z = np.matmul(A_in, W) + B

    # 2. 요소별 활성화 함수(Sigmoid) 적용
    A_out = g(Z)

    return A_out
```