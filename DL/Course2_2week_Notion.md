
- 2주차의 핵심 주제는 "학습 속도 향상(Optimization Algorithms)"입니다. 
- 기본 경사 하강법(Batch Gradient Descent)의 한계를 극복하고 학습 속도를 몇 배 이상 올려주는 기법들을 다룹니다.


---

## 1. 미니배치 경사하강법 (Mini-batch Gradient Descent)

전체 데이터 $m$개를 한 번에 처리하는 **Batch** 방식은 데이터가 수십만~수백만 개일 때 1회 업데이트 조차 너무 오래 걸립니다.

* **동작 원리:** 전체 데이터를 $X^{{1}}, X^{{2}}, \dots, X^{{t}}$처럼 작은 묶음(예: 64, 128, 256, 512개)으로 나눕니다.
* **학습 방식:** 1개의 미니배치($X^{\{t\}}$)를 통과시킬 때마다 즉시 가중치($W, b$)를 업데이트합니다.
* **Epoch 정의:** 전체 데이터셋($m$개)을 1번 완전히 통과하는 과정. (1 Epoch 동안 미니배치 개수만큼 파라미터 업데이트 발생)
* **배치 크기(Batch Size) 선택 팁:**
	* **$m \le 2000$ (소규모):** 그냥 Batch Gradient Descent ($Batch Size = m$) 사용.
	* **$m > 2000$ (대규모):** Mini-batch 사용 ($64, 128, 256, 512$ 등 **$2^n$ 단위** 사용 - GPU 메모리 효율 최적화).

---

## 2. 지수 가중 이동 평균 (Exponentially Weighted Averages)
- 경사하강법(SGD)을 쓸 때 기울기가 튀고 흔들리는 노이즈를 부드럽게 정돈해주기 위해 도입했다.

- 고급 최적화 알고리즘(Momentum, RMSprop)의 수학적 기반이 되는 핵심 개념입니다.

$$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$$

* **$\beta$ (Hyperparameter):** 보통 $0.9$ 사용 (최근 약 $\frac{1}{1-\beta} = 10$일간의 데이터를 평균 내는 효과).
* **개념:** 과거 데이터의 영향을 지수적으로 감소시키며 노이즈를 줄이고 부드러운 추세선(Smooth Curve)을 만드는 기법.
* **편향 보정 (Bias Correction):** 초반 $V_0 = 0$ 설정으로 인해 시작 지점이 0에 치우치는 현상을 막기 위해 $\frac{V_t}{1 - \beta^t}$ 로 보정합니다.

- 이 점화식을 과거 방향으로 계속 풀어 쓰면 이유가 명확해집니다.
$$V_t = (1 - \beta)\theta_t + \beta(1 - \beta)\theta_{t-1} + \beta^2(1 - \beta)\theta_{t-2} + \beta^3(1 - \beta)\theta_{t-3} + \dots$$
- 현재값 $\theta_t$의 반영 비율: $(1 - \beta)$
- 1단계 전 $\theta_{t-1}$의 반영 비율: $\beta(1 - \beta)$
- 2단계 전 $\theta_{t-2}$의 반영 비율: $\beta^2(1 - \beta)$
- $\beta$가 $1$보다 작으므로, **과거로 갈수록 $\beta^k$가 곱해져 가중치가 지수 함수(Exponential) 형태로 급격히 감소**합니다.

- **$\beta = 0.9$일 때**: $\frac{1}{1 - 0.9} = 10$
    
    -  최근 **약 10일간의 데이터**를 주로 반영하여 평균을 냅니다. 과거의 미세한 노이즈가 부드럽게 정돈됩니다.
    
- **$\beta = 0.98$일 때**: $\frac{1}{1 - 0.98} = 50$
    
    -  최근 **약 50일간의 데이터**를 반영합니다. 그래프가 훨씬 더 부드러워지지만, 최근 변화를 빠르게 반영하지 못하고 **지연(Lag)** 현상이 커집니다.




---

## 3. 고급 최적화 알고리즘 (Optimization Algorithms)

### 1). Momentum (관성)
- 기울기의 이동 평균을 구해서 관성(속도)을 붙여줍니다.
- 기존 경사하강법에 **물리학의 관성(가속도)** 개념을 더해, 경사하강법의 좌우 진동을 줄이고 목적지로 빠르게 가속합니다.

$$V_{dW} = \beta V_{dW} + (1 - \beta) dW$$

$$W := W - \alpha V_{dW}$$

* **특징:** 지그재그 방향의 기울기는 서로 상쇄되어 줄어들고, 최적점을 향한 방향의 기울기는 누적되어 속도가 빨라집니다. ($\beta = 0.9$ 권장)




### 2). RMSprop (Root Mean Square Prop)
- 기울기 제곱의 이동 평균을 구해서 보폭(Learning Rate)을 조절합니다.
- 기울기가 가파른 방향(진동이 심한 축)의 업데이트는 줄이고, 기울기가 완만한 방향의 업데이트는 키워 균형을 맞춥니다.

$$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW^2$$

$$W := W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}$$

* **특징:** $dW$가 크면 $S_{dW}$도 커져서 나누는 분모가 커지므로 업데이트 폭이 줄어듭니다. ($\epsilon = 10^{-8}$ 은 0 나누기 방지용)

### 3).Adam (Adaptive Moment Estimation) 
- 위 둘(Momentum + RMSprop)을 합친 뒤, 편향 보정(Bias Correction)까지 적용한 알고리즘입니다.
- **Momentum + RMSprop**을 결합한 알고리즘으로, 현재 딥러닝에서 **가장 널리 쓰이는 표준 Optimizer**입니다.

1. Momentum 계산: $V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW$
2. RMSprop 계산: $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW^2$
3. 편향 보정 후 업데이트:

$$W := W - \alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}} + \epsilon}$$



* **하이퍼파라미터 권장 기본값:**
* $\alpha$: 튜닝 필요
* $\beta_1$ (Momentum용): $0.9$
* $\beta_2$ (RMSprop용): $0.999$
* $\epsilon$: $10^{-8}$



---

## 4. 학습률 감쇠 (Learning Rate Decay)

학습 초반에는 큰 학습률($\alpha$)로 성큼성큼 이동하다가, 최적점에 가까워질수록 $\alpha$를 줄여서 최적점 주변에서 맴돌지 않고 정밀하게 수렴시키는 기법입니다.

$$
\alpha = \frac{1}{1 + \text{decay rate} \times \text{epoch num}} \alpha_0
$$
