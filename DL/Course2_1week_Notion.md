

Week 1은 딥러닝 모델의 성능을 향상시키기 위해 데이터를 어떻게 나누고, 과적합(Overfitting)을 방지하며, 기울기 소실/폭주 문제를 해결할지에 대한 **실무 핵심 테크닉**을 다룹니다.

## 1. 머신러닝 데이터셋 구성 (Train / Dev / Test Sets)

머신러닝 개발 프로세스는 **아이디어 $\rightarrow$ 코드 $\rightarrow$ 실험**의 반복(Empirical Process)입니다. 이를 위해 데이터셋을 올바르게 나누는 것이 중요합니다.

- **전통적인 비율 (소규모 데이터, ~10,000개):** 60% / 20% / 20%
    
- **빅데이터 시대의 비율 (100만 개 이상):** **98% / 1% / 1%** 또는 **99.5% / 0.25% / 0.25%**
    
    - _이유:_ 모델을 검증(Dev)하고 평가(Test)하는 데는 10,000개 정도의 데이터만으로도 충분히 통계적 의미가 있기 때문입니다.
        
- **주의사항:** Dev 세트와 Test 세트는 반드시 동일한 분포(Same Distribution)를 가진 데이터에서 추출해야 합니다.
    

## 2. 편향(Bias)과 분산(Variance)

모델의 성능 문제를 진단하는 두 가지 주요 축입니다.

|**구분**|**편향이 높은 상태 (High Bias)**|**분산이 높은 상태 (High Variance)**|
|---|---|---|
|**상태**|**과소적합 (Underfitting)**|**과적합 (Overfitting)**|
|**특징**|Train 오차가 높음|Train 오차는 낮으나 Dev 오차가 훨씬 높음|
|**해결책**|• 더 큰 네트워크(레이어/유닛 추가) 사용<br><br>  <br><br>• 더 오래 학습 (More Iterations)<br><br>  <br><br>• 다른 신경망 구조 찾기|• **더 많은 데이터 수집**<br><br>  <br><br>• **정규화(Regularization) 적용**<br><br>  <br><br>• 다른 신경망 구조 찾기|

## 3. 정규화 (Regularization)

High Variance(과적합)를 해결하기 위해 가중치 $W$가 너무 커지지 않도록 제약을 가하는 방법입니다.

### **L2 정규화 (Weight Decay)**

비용 함수 $J$에 가중치 제곱합 테두리를 추가합니다.

- **비용 함수:**
    
    $$J(W,b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} \Vert{}W^{[l]}\Vert{}_F^2$$
    
    - $\lambda$: 정규화 하이퍼파라미터
        
    - $\Vert{}W^{[l]}\Vert{}_F^2$: 프로베니우스 노름 (Frobenius Norm)
        
- **원리:** $\lambda$가 커지면 $W$ 값이 0에 가까워져, 특정 유닛들의 영향력이 줄어들고 모델이 단순해집니다.
    

### **드롭아웃 (Dropout - Inverted Dropout)**

학습 시 각 레이어의 뉴런을 정해진 확률($1 - \text{keep\_prob}$)로 무작위 삭제(0으로 설정)하는 기법입니다.

- **작동 원리:** 특정 뉴런에만 과도하게 의존하는 것을 방지하여 가중치를 분산시킵니다.
    
- **Inverted Dropout 핵심 코드 구조:**
    
    Python
    
    ```
    d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob  # 마스크 생성
    a3 = np.multiply(a3, d3)  # 드롭아웃 적용
    a3 /= keep_prob  # Scaling (테스트 시 기대값 유지용)
    ```
    
- **주의:** 테스트 단계에서는 드롭아웃을 **절대 적용하지 않습니다.**
    

### **기타 정규화 기법**

- **Data Augmentation (데이터 증강):** 이미지 반전, 회전, 자르기 등으로 데이터 양을 인위적으로 늘림.
    
- **Early Stopping (조기 종료):** Dev set 오차가 증가하기 시작하는 시점에 학습을 중단 (단, 손실 함수 최적화와 과적합 방지를 한 번에 처리하려 한다는 단점 존재).
    

## 4. 학습 세팅 및 최적화 기법 (Setup Optimization)

### **입력 데이터 정규화 (Normalizing Inputs)**

입력 데이터 $X$의 평균을 0으로 맞추고 분산을 1로 정규화합니다.

1. 평균 차감: $\mu = \frac{1}{m} \sum X$ $\rightarrow$ $X := X - \mu$
    
2. 분산 정규화: $\sigma^2 = \frac{1}{m} \sum X^2$ $\rightarrow$ $X := \frac{X}{\sigma}$
    

> **효과:** 비용 함수 $J$의 등고선 모양이 길쭉한 타원형에서 **동심원(원형)**에 가깝게 변하여, 경사하강법이 진동하지 않고 **더 높은 학습률 $\alpha$로 빠르게 최적점에 수렴**할 수 있습니다.

### **기울기 소실 및 폭주 (Vanishing / Exploding Gradients)**

신경망이 매우 깊어지면($L$이 커지면) 가중치 $W$의 곱이 지수적으로 증가하거나($>1$) 감소하여($<1$) 기울기(Gradient)가 0에 가깝게 소실되거나 폭발적으로 커집니다.

- **해결책: 가중치 초기화 (Weight Initialization)**
    
    각 레이어의 입력 유닛 수($n^{[l-1]}$)에 맞게 가중치의 초기 분산을 조절합니다.
    
    - **He 초기화 (ReLU 활성화 함수용):**
        
        $$W^{[l]} = \text{np.random.randn}(\dots) \times \sqrt{\frac{2}{n^{[l-1]}}}$$
        
    - **Xavier 초기화 (tanh 활성화 함수용):**
        
        $$W^{[l]} = \text{np.random.randn}(\dots) \times \sqrt{\frac{1}{n^{[l-1]}}}$$
        

## 5. 기울기 검증 (Gradient Checking)

수학적으로 구한 역전파 미분값($d\theta_{approx}$)과 수치 미분(Numerical Approximation)을 비교하여 **역전파 구현에 버그가 없는지 검증**하는 방법입니다.

- **수치 미분 공식:**
    
    $$f'(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} \quad (\epsilon = 10^{-7})$$
    
- **검증 비율 계산:**
    
    $$\text{Difference} = \frac{\Vert{} d\theta_{approx} - d\theta \Vert{}_2}{\Vert{} d\theta_{approx} \Vert{}_2 + \Vert{} d\theta \Vert{}_2}$$
    
    - 결과가 $10^{-7}$ 이하이면 성공적으로 구현된 것입니다.
        
- **주의사항:**
    
    - 학습(Training) 중에 매번 실행하면 너무 느리므로 **디버깅용으로만 사용**합니다.
        
    - 드롭아웃(Dropout) 적용 시에는 계산이 복잡해지므로 **드롭아웃을 끄고 검증**해야 합니다.
        

### Week 1 요약 한눈에 보기

- **데이터:** Train/Dev/Test 비율을 데이터 크기에 맞게 재조정 ($1M \rightarrow 98/1/1$)
    
- **과적합 해결:** L2 Regularization, Inverted Dropout
    
- **학습 속도 향상:** Input Normalization, He Initialization
    
- **버그 디버깅:** Gradient Checking ($\epsilon = 10^{-7}$)
    
