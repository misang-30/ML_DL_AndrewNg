

Week 3는 모델 성능을 최대로 끌어올리기 위한 **하이퍼파라미터 탐색 전략**, 학습을 안정화하고 속도를 획기적으로 개선하는 **배치 정규화(Batch Normalization)**, 그리고 다중 클래스 분류(Softmax) 및 딥러닝 프레임워크(TensorFlow)를 다룹니다.

## 1. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

### **1) 하이퍼파라미터 우선순위 (Priority Level)**

- **1순위 (가장 중요):** 학습률 $\alpha$ (Learning Rate)
    
- **2순위:** Momentum 베타 $\beta_1$ (보통 0.9), 은닉 유닛 수 ($n^{[l]}$), 미니배치 크기 (Mini-batch size)
    
- **3순위:** 레이어 개수 ($L$), 학습률 감쇠율 (Learning rate decay)
    
- **기본 고정값:** Adam 최적화의 $\beta_2=0.999$, $\epsilon=10^{-8}$
    

### **2) 탐색 공간 정의 방식 (Grid vs Random Search)**

- **격자 탐색 (Grid Search):** 과거 정적인 모델에서 쓰이던 방식으로, 딥러닝에서는 비효율적입니다.
    
- **무작위 탐색 (Random Search):** 하이퍼파라미터 공간 내에서 무작위 점들을 선택합니다. 어떤 하이퍼파라미터가 성능에 더 지대한 영향을 미치는지 미리 알 수 없기 때문에 훨씬 유용합니다.
    
- **조밀화 탐색 (Coarse to Fine Search):** 무작위 탐색 후 성능이 잘 나오는 특정 영역을 발견하면, 해당 구간을 집중적으로 밀도 있게 다시 탐색합니다.
    

### **3) 적절한 스케일(Scale) 선택 (Log Scale)**

하이퍼파라미터의 범위를 무작위로 선택할 때 선형 스케일(Linear Scale)이 아닌 로그 스케일(Log Scale)을 써야 하는 경우가 있습니다.

- **학습률 $\alpha \in [0.0001, 1]$ 탐색 예시:**
    
    - 선형 스케일로 선택하면 $0.1 \sim 1$ 구간에 데이터의 90%가 몰리게 되어 $0.0001 \sim 0.1$ 구간을 제대로 탐색하지 못함.
        
    - **로그 스케일 적용:** $r \in [-4, 0]$ 범위에서 $r$을 무작위 선택한 뒤 $\alpha = 10^r$ 로 설정 ($10^{-4} \sim 10^0$).
        
- **Momentum $\beta \in [0.9, 0.999]$ 탐색 예시:**
    
    - $1 - \beta \in [0.001, 0.1]$ 로 변환하여 $r \in [-3, -1]$ 범위에서 로그 스케일 탐색 적용.
        

### **4) 하이퍼파라미터 튜닝 접근 방식**

- **판다 방식 (Panda Approach / Babysitting Model):** 리소스(GPU 등)가 부족할 때 사용. 하나의 모델을 돌려보면서 손실 곡선을 지켜보고 학습률이나 파라미터를 수동으로 조금씩 조절해 나감.
    
- **캐비어 방식 (Caviar Approach / Parallel Models):** 리소스가 충분할 때 사용. 여러 하이퍼파라미터 조합을 가진 다양한 모델을 동시에 대량으로 학습시켜 최적의 결과를 골라냄.
    

## 2. 배치 정규화 (Batch Normalization)

입력 데이터 $X$를 정규화했던 것처럼, **은닉층의 활성화 전 값 $Z^{[l]}$ (또는 $A^{[l]}$)을 정규화**하여 다음 레이어로 전달하는 강력한 테크닉입니다.

### **배치 정규화 연산 단계 (미니배치 $B$에 대해)**

1. **평균 계산:** $\mu = \frac{1}{m} \sum_{i} z^{(i)}$
    
2. **분산 계산:** $\sigma^2 = \frac{1}{m} \sum_{i} (z^{(i)} - \mu)^2$
    
3. **정규화:** $z_{\text{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$ (평균 0, 분산 1로 변환)
    
4. **스케일 및 이동 (Re-scaling & Shift):**
    
    $$\tilde{z}^{(i)} = \gamma z_{\text{norm}}^{(i)} + \beta$$
    
    - $\gamma, \beta$: 학습 가능한 파라미터 (네트워크가 필요한 표현력을 스스로 유지하도록 보장).
        
    - 만약 $\gamma = \sqrt{\sigma^2 + \epsilon}$, $\beta = \mu$ 라면 $\tilde{z}^{(i)} = z^{(i)}$로 원상복구 가능.
        

> **주의사항 (Bias $b^{[l]}$의 제거):**
> 
> 배치 정규화 과정에서 평균 $\mu$를 빼기 때문에, 기존의 편향 벡터 $b^{[l]}$의 효과는 상쇄되어 사라집니다. 따라서 BatchNorm을 적용할 때는 $b^{[l]}$을 제거하거나 0으로 설정하고, 대신 **$\beta^{[l]}$** 파라미터가 편향 역할을 대신하도록 합니다.

### **배치 정규화가 잘 작동하는 이유**

- **학습 속도 향상:** 가중치 초기화에 덜 민감해지고 더 높은 학습률($\alpha$)을 사용할 수 있습니다.
    
- **공변량 변화 (Covariate Shift) 완화:** 이전 레이어의 가중치가 바뀜에 따라 다음 레이어의 입력 분포가 계속 흔들리는 현상을 줄여주어, 각 레이어가 다른 레이어와 독립적으로 안정적인 학습을 하도록 돕습니다.
    
- **경미한 정규화 효과 (Regularization Effect):** 각 미니배치의 평균과 분산에 미세한 노이즈가 추가되므로, 드롭아웃처럼 약간의 정규화(과적합 방지) 효과를 줍니다.
    

### **테스트 단계(Inference)에서의 BatchNorm**

테스트 시에는 샘플 1개만 들어올 수도 있으므로 미니배치의 $\mu, \sigma^2$를 직접 구할 수 없습니다.

대신 학습 도중 각 미니배치에서 계산된 $\mu, \sigma^2$의 지수 가중 이동 평균(Exponentially Weighted Average)을 미리 저장해두었다가 테스트 시 사용합니다.

## 3. 다중 클래스 분류 (Softmax Regression)

이진 분류(Sigmoid)를 확장하여 **$C$개의 클래스** 중 하나로 분류하는 방법입니다.

### **Softmax 연산 공식**

출력층 $L$에서 선형 연산 $z^{[L]}$이 주어졌을 때:

1. Temporary activation: $t = e^{z^{[L]}}$ (element-wise)
    
2. 확률화 (Softmax):
    
    $$a^{[L]}_i = \frac{e^{z^{[L]}_i}}{\sum_{j=1}^{C} e^{z^{[L]}_j}}$$
    
    _(모든 출력의 합은 1이 되며, 각 출력을 해당 클래스일 확률로 해석 가능)_
    

### **손실 함수 (Loss Function)**

$$\mathcal{L}(\hat{y}, y) = -\sum_{j=1}^{C} y_j \log \hat{y}_j$$

_(원-핫 인코딩된 정답 클래스의 확률값에만 $-\log$를 취해 손실 계산)_

## 4. 딥러닝 프레임워크 (Deep Learning Frameworks)

경사하강법과 역전파 알고리즘을 매번 직접 작성하는 대신, 효율적인 프레임워크를 사용합니다.

### **대표적 프레임워크 및 선택 기준**

- **주요 프레임워크:** TensorFlow, PyTorch 등
    
- **선택 기준:** 코드 작성의 용이성, 학습/실행 속도, 오픈소스 생태계 및 지속성
    

### **TensorFlow 기본 개념 및 구조**

- **자동 미분 (Auto-Diff):** 순전파(Forward Prop) 연산만 작성해 두면, 프레임워크가 그래프를 기반으로 역전파(Backprop) 및 기울기 계산을 자동으로 수행합니다.
    
- **비용 함수 및 최적화 구현 예시:**
    
    Python
    
    ```
    import tensorflow as tf
    
    # 파라미터 선언
    w = tf.Variable(0.0, dtype=tf.float32)
    # 최적화 알고리즘 설정 (Adam)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    # 손실 함수 정의 및 1스텝 최적화
    def train_step():
        with tf.GradientTape() as tape:
            cost = w**2 - 10*w + 25  # (w - 5)^2
        grads = tape.gradient(cost, [w])
        optimizer.apply_gradients(zip(grads, [w]))
    ```
    

