

기본 경사하강법(Gradient Descent)의 한계를 극복하고 학습 속도를 비약적으로 향상시키는 **최신 최적화 알고리즘 및 테크닉**을 다룹니다.

## 1. 미니배치 경사하강법 (Mini-batch Gradient Descent)

전체 데이터셋($m$)을 작은 묶음인 미니배치(Mini-batch) $X^{{t}}, Y^{{t}}$로 나누어 학습을 진행하는 기법입니다.

### **배치(Batch) vs 미니배치(Mini-batch) vs 확률적(Stochastic) 비교**

- **배치 경사하강법 (Batch GD):** 미니배치 크기 = $m$
    
    - 전체 데이터를 다 본 후 1번 업데이트. $m$이 매우 크면 1스텝에 오랜 시간 소요.
        
- **확률적 경사하강법 (Stochastic GD):** 미니배치 크기 = $1$
    
    - 데이터 1개마다 업데이트. 벡터화(Vectorization)의 연산 이점을 잃고 진동이 매우 심함.
        
- **미니배치 경사하강법 (Mini-batch GD):** 미니배치 크기 = $1 < \text{size} < m$
    
    - 벡터화 이점을 살리면서도 1개의 epoch 동안 여러 번 파라미터 업데이트 가능.
        

### **미니배치 크기 선택 가이드라인**

- **$m \le 2000$ (소규모 데이터):** 배치 경사하강법(Batch GD) 권장.
    
- **일반적인 미니배치 크기:** **64, 128, 256, 512** (CPU/GPU 메모리 구조상 $2^n$ 크기가 연산 최적화에 유리).
    
- **주의사항:** 미니배치 $X^{{t}}, Y^{{t}}$가 CPU/GPU 메모리에 한 번에 올라가는 크기여야 함.
    

## 2. 지수 가중 이동 평균 (Exponentially Weighted Averages)

최근 데이터에 더 높은 가중치를 주어 유동적인 데이터의 경향성(Trend)을 매끄럽게 다듬는 통계 기법입니다. (Momentum, RMSprop, Adam의 기본 원리)

### **기본 수식**

$$v_t = \beta v_{t-1} + (1 - \beta) \theta_t$$

- $\beta$ **(하이퍼파라미터):** 과거 데이터 영향력 결정. 대략 최근 $\frac{1}{1-\beta}$일간의 데이터를 평균 낸 효과.
    
    - $\beta = 0.9 \approx$ 최근 **10일** 평균 (적절히 매끄러움)
        
    - $\beta = 0.98 \approx$ 최근 **50일** 평균 (더 매끄럽지만 시차/지연 발생)
        
- **편향 보정 (Bias Correction):**
    
    초기값 $v_0 = 0$ 설정 때문에 초기 구간 추정치가 0 방향으로 왜곡(편향)되는 현상을 방지합니다.
    
    $$v_t^{\text{corrected}} = \frac{v_t}{1 - \beta^t}$$
    
    _(t가 커질수록 $1-\beta^t \approx 1$이 되어 초기 단계 이후에는 보정 효과가 자연스럽게 사라짐)_
    

## 3. 모멘텀 경사하강법 (Gradient Descent with Momentum)

경사하강법에 '관성(Momentum)'을 추가하여 최적화 경로를 가속하는 기법입니다.

- **원리:** 수직 방향의 불필요한 진동(Oscillation)은 상쇄시키고, 수평 방향(목표점)으로의 진행 속도를 가속합니다.
    
- **알고리즘 (각 iteration $t$마다):**
    
    $$v_{dW} = \beta v_{dW} + (1 - \beta) dW$$
    
    $$v_{db} = \beta v_{db} + (1 - \beta) db$$
    
    $$W := W - \alpha v_{dW}, \quad b := b - \alpha v_{db}$$
    
- **하이퍼파라미터 추천:** $\beta = 0.9$ (가장 널리 쓰이며 관습적인 기본값)
    

## 4. RMSprop (Root Mean Square Prop)

경사하강법의 진동을 제어하기 위해 **기울기의 제곱 평균**을 나누어주는 기법입니다.

- **원리:** 기울기가 커서 크게 진동하는 축은 학습률을 줄이고, 기울기가 작아 천천히 움직이는 축은 상대적으로 학습률을 유지/가속합니다.
    
- **알고리즘:**
    
    $$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) (dW)^2 \quad (\text{element-wise 제곱})$$
    
    $$S_{db} = \beta_2 S_{db} + (1 - \beta_2) (db)^2$$
    
    $$W := W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}, \quad b := b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon}$$
    
    - $\epsilon$ ($10^{-8}$ 수준): 분모가 0이 되는 Zero Division 에러 방지용 아주 작은 값.
        

## 5. Adam 최적화 알고리즘 (Adam Optimization Algorithm)

**Momentum + RMSprop**을 결합한 알고리즘으로, 현대 딥러닝에서 가장 널리 쓰이고 뛰어난 성능을 보이는 하이퍼파라미터 최적화 알고리즘입니다.

### **알고리즘 단계**

1. **Momentum (1차 모멘트):** $v_{dW} = \beta_1 v_{dW} + (1 - \beta_1) dW$
    
2. **RMSprop (2차 모멘트):** $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) (dW)^2$
    
3. **편향 보정 (Bias Correction):**
    
    $$v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1 - \beta_1^t}, \quad S_{dW}^{\text{corrected}} = \frac{S_{dW}}{1 - \beta_2^t}$$
    
4. **파라미터 업데이트:**
    
    $$W := W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}} + \epsilon}$$
    

### **권장 하이퍼파라미터 기본값**

- $\alpha$: **튜닝 필요** (학습률)
    
- $\beta_1 = 0.9$ (Momentum용)
    
- $\beta_2 = 0.999$ (RMSprop용)
    
- $\epsilon = 10^{-8}$
    

## 6. 학습률 감쇠 (Learning Rate Decay)

학습 초기에는 큰 학습률($\alpha$)로 빠르게 이동하고, 최적점 근처에 도달할수록 학습률을 점차 줄여 미세하게 최적점으로 수렴시키는 기법입니다.

### **주요 감쇠 공식**

- **기본 감쇠 (Standard Decay):**
    
    $$\alpha = \frac{1}{1 + \text{decay\_rate} \times \text{epoch\_num}} \alpha_0$$
    
- **지수 감쇠 (Exponential Decay):** $\alpha = 0.95^{\text{epoch\_num}} \alpha_0$
    
- **기타 방식:** $\alpha = \frac{k}{\sqrt{\text{epoch\_num}}} \alpha_0$, 계단식 감쇠(Discrete Staircase Decay), 수동 감쇠(Manual Decay) 등.
    

## 7. 국소 최적점 문제 (The Problem of Local Optima)

고차원 딥러닝 공간에서의 기울기 0 지점들에 대한 오해와 진실을 설명합니다.

- **안장점 (Saddle Points):** 고차원 공간에서 기울기가 0인 지점은 지역 최적점(Local Optima)이 아니라 대부분 안장점(Saddle Point)입니다. (일부 차원은 극소, 일부 차원은 극대)
    
- **평탄한 지역 (Plateaus):** 기울기가 0에 매우 가까운 평지 구간이 존재하여 경사하강법의 학습 속도를 매우 느리게 만듭니다.
    
- **해결책:** **Momentum, RMSprop, Adam** 기법은 안장점이나 평탄한 지역(Plateaus)의 완만한 경사에서 빠져나오는 데 탁월한 효과를 발휘합니다.
    
