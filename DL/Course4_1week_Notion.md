

이미지 처리 분야에서 일반 다층 퍼셉트론(Fully Connected Layer)이 가지는 한계를 극복하고, 컴퓨터 비전의 핵심 기술이 된 **CNN의 기본 구성 요소와 동작 원리**를 다룹니다.

## 1. 컴퓨터 비전과 합성곱 연산 (Computer Vision & Convolution Operation)

### **1) 일반 신경망의 한계**

- **입력 차원의 폭발:** 예를 들어 $1000 \times 1000$ 해상도의 RGB 이미지($3$채널)를 입력으로 사용하면, 입력 특성(Feature) 수만 $1000 \times 1000 \times 3 = 3,000,000$개가 됩니다.
    
- 첫 번째 은닉층의 유닛이 1,000개라면 가중치 매트릭스 $W$의 파라미터 개수만 **30억 개**에 달해 오버피팅과 메모리 부족 문제가 발생합니다.
    

### **2) 에지 감지 (Edge Detection) 예시**

이미지에서 수직/수평 테두리를 감지하기 위해 필터(Filter 또는 Kernel)를 이미지 위에서 슬라이딩하며 요소별 곱셈 후 합을 구하는 **합성곱(Convolution) 연산**을 수행합니다.

- **수직 에지 감지 필터 (Vertical Edge Detection Filter):**
    
    $$\begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$
    
- **수평 에지 감지 필터 (Horizontal Edge Detection Filter):**
    
    $$\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$
    
- **소벨 필터(Sobel Filter), 라플라시안 필터(Sobel/Scharr Filter) 등:** 중앙 가중치를 다르게 주어 에지를 더 선명하게 포착합니다.
    
- **딥러닝의 핵심:** 사람이 필터의 숫자를 직접 설계하지 않고, **필터 내의 파라미터 값들을 역전파(Backprop)를 통해 모델이 직접 학습**합니다.
    

## 2. 패딩과 스트라이드 (Padding & Stride)

### **1) 패딩 (Padding)**

$n \times n$ 이미지에 $f \times f$ 필터를 적용하면 출력 크기는 $(n - f + 1) \times (n - f + 1)$로 줄어듭니다.

#### **단점**

1. 레이어를 거칠 때마다 이미지 크기가 계속 축소됩니다.
    
2. 코너/모서리에 위치한 피셀은 필터 연산에 적게 참여하여 외곽 정보가 손실됩니다.
    

#### **해결책 (Padding $p$)**

이미지 테두리에 0을 둘러싸서 크기를 유지합니다.

- **Valid Convolution:** 패딩 없음 ($p = 0$). 출력 크기: $n - f + 1$
    
- **Same Convolution:** 입력과 출력이 동일한 크기를 갖도록 패딩 적용.
    
    $$p = \frac{f - 1}{2} \quad (\text{단, } f\text{는 주로 홀수 사용})$$
    

### **2) 스트라이드 (Stride $s$)**

필터를 이미지 위에서 몇 칸씩 이동시킬지 결정하는 간격입니다.

#### **출력 크기 계산 공식 (입력 $n \times n$, 필터 $f \times f$, 패딩 $p$, 스트라이드 $s$)**

$$\left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor$$

_(소수점이 나올 경우 버림 $\lfloor \cdot \rfloor$ 연산을 적용하여 이미지 바깥으로 필터가 나가지 않도록 처리)_

## 3. 입체 데이터(Volume)에 대한 합성곱

RGB와 같은 3차원 이미지($n \times n \times n_c$)에 합성곱을 적용할 때:

- **채널 수 일치 법칙:** 필터의 채널 수는 **반드시 입력 데이터의 채널 수와 동일**해야 합니다 ($f \times f \times n_c$).
    
- **여러 개의 필터 사용:** $n_c'$ 개의 서로 다른 특징(수직 에지, 수평 에지, 색상 변화 등)을 감지하고 싶다면, $n_c'$ 개의 필터를 적용합니다.
    
- **출력 크기:**
    
    $$\text{Input: } (n \times n \times n_c) \; * \; \text{Filter: } (f \times f \times n_c) \times n_c' \longrightarrow \text{Output: } \left( \frac{n+2p-f}{s}+1 \right) \times \left( \frac{n+2p-f}{s}+1 \right) \times n_c'$$
    

## 4. CNN의 한 레이어 구성 (One Layer of a CNN)

1. **입력:** $A^{[l-1]}$ (크기: $n_H^{[l-1]} \times n_W^{[l-1]} \times n_C^{[l-1]}$)
    
2. **합성곱 연산:** $f^{[l]} \times f^{[l]} \times n_C^{[l-1]}$ 크기의 필터 $n_C^{[l]}$개를 적용.
    
3. **편향(Bias) 및 활성화 함수:** 각 필터 출력에 편향 $b_k$를 더하고 활성화 함수(예: ReLU) 적용.
    
    $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \quad A^{[l]} = g(Z^{[l]})$$
    

### **기호 표기법 정리 (Notation)**

- $f^{[l]}$: 필터 크기
    
- $p^{[l]}$: 패딩 크기
    
- $s^{[l]}$: 스트라이드
    
- $n_C^{[l]}$: 필터의 개수 (출력 채널 수)
    
- **특징 맵(Feature Map) 크기:** $n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}$
    
- **파라미터 개수:** $(f^{[l]} \times f^{[l]} \times n_C^{[l-1]} + 1) \times n_C^{[l]}$ _(+1은 편향 값)_
    

## 5. 풀링 레이어 (Pooling Layers)

특징 맵의 공간적 크기(가로/세로)를 줄여 연산량을 감소시키고, 위치 변화에 대한 불변성(Invariance)을 제공합니다.

- **최대 풀링 (Max Pooling):** 설정된 영역($f \times f$) 내에서 **가장 큰 값**을 선택합니다. (특징의 존재 여부를 강하게 포착)
    
- **평균 풀링 (Average Pooling):** 설정된 영역 내의 **평균값**을 구합니다. (네트워크 후반부 외에는 잘 안 쓰임)
    
- **특징:**
    
    - **학습해야 할 파라미터가 없음** ($W, b$ 없음).
        
    - 채널 수($n_C$)는 유지되고 가로/세로 크기만 축소됩니다.
        
    - 하이퍼파라미터 $f$(크기), $s$(스트라이드)만 지정합니다. (자주 쓰이는 조합: $f=2, s=2$)
        

## 6. CNN 구조의 직관적 이유 (Why Convolutions?)

일반 Fully Connected(FC) 레이어 대비 합성곱 레이어가 우수한 2가지 핵심 이유:

1. **파라미터 공유 (Parameter Sharing):**
    
    이미지의 한 구획(예: 좌상단)에서 에지를 감지하는 데 유용했던 필터는 다른 구획(예: 우하단)에서도 동일하게 유용하게 사용됩니다.
    
2. **희소 연결성 (Sparsity of Connections):**
    
    출력층의 한 픽셀은 입력층의 작은 특정 영역($f \times f$)에만 영향을 받으므로 전체 연결에 비해 입력과 가중치 간의 연결성이 매우 희소해집니다.
    

$\rightarrow$ 이 두 특성 덕분에 **파라미터 수를 획기적으로 줄여 적은 데이터로도 오버피팅 없이 학습**할 수 있습니다.

