
Week 1에서 ML 프로젝트의 목표 설정과 평가 지표를 다루었다면, Week 2에서는 **오류 분석(Error Analysis), 데이터 분포가 다른 상황 대처법(Data Mismatch), 전이 학습(Transfer Learning) 및 종단간 딥러닝(End-to-End Deep Learning)** 등 머신러닝 시스템 구축 시 겪는 실전 문제 해결 전략을 다룹니다.

## 1. 오류 분석 (Error Analysis)

모델의 성능을 개선할 때, 무작정 아이디어를 적용하기보다 **어떤 에러를 먼저 수정해야 가장 큰 효과를 볼 수 있는지** 우선순위를 정하는 프로세스입니다.

- **수동 오류 분석 (Manual Error Analysis):**
    
    - Dev 세트에서 모델이 틀린 샘플 중 **약 100~500개**를 무작위로 추출하여 수동으로 검토합니다.
        
    - 표(Ceiling Analysis Table)를 만들어 오류 원인별 비율을 기록합니다.
        
    - **예시 (고양이 분류기):**
        
        - 개(Dog) 그림을 고양이로 착각: 전체 오류의 8%
            
        - 큰 고양이과(사자/호랑이) 착각: 전체 오류의 43%
            
        - 화질이 낮음(Blurry): 전체 오류의 61%
            
        - $\rightarrow$ 개 이미지를 구분하는 알고리즘을 개발해 봤자 개선 가능한 최대 오차는 8%뿐이므로, **화질이 낮거나 큰 고양이과 이미지를 처리하는 데 우선순위**를 두어야 합니다.
            
- **잘못 레이블링된 데이터 (Incorrectly Labeled Data) 처리:**
    
    - **Train set:** 딥러닝은 랜덤 노이즈 레이블에 비교적 강하므로, 오류 비율이 높지 않다면 굳이 일일이 수정할 필요는 없습니다.
        
    - **Dev/Test set:** 오류 분석 표에 "Incorrect Label" 항목을 추가하여 전체 오류에 미치는 영향을 측정합니다. 해당 오류가 Dev set 전체 오차의 상당 부분을 차지한다면 수동으로 수정해야 합니다.
        
- **빠른 초기 시스템 구축 (Build First System Quickly & Iterate):**
    
    - 처음부터 완벽한 시스템을 설계하려 하지 말고, **빠르게 초판(Baseline)을 만든 뒤** 오류 분석을 통해 개선 방향을 결정하는 것이 훨씬 효율적입니다.
        

## 2. 분포가 다른 데이터셋 다루기 (Mismatched Training and Dev/Test Sets)

실무에서는 인터넷에서 수집한 고화질 데이터(량은 많음)와 실제 서비스에서 수집한 저화질 데이터(량은 적지만 중요한 목표)처럼 **분포가 다른 데이터**를 다루어야 할 때가 많습니다.

### **1) 데이터 분할 전략**

- **권장 방법:** 타깃 데이터(실제 서비스 데이터)를 Dev/Test set에 집중 배치합니다.
    
    - 예: 인터넷 데이터 20만 장 + 서비스 데이터 1만 장이 있을 때
        
    - **Train set:** 인터넷 데이터 20만 장 + 서비스 데이터 5,000장
        
    - **Dev / Test set:** 서비스 데이터 각각 2,500장 (동일 분포 유지)
        

### **2) Training-Dev Set을 활용한 오차 진단**

Train set과 Dev set의 데이터 분포가 다르면, Dev set의 오차가 높을 때 **과적합(Variance)** 때문인지 **데이터 불일치(Data Mismatch)** 때문인지 구분하기 어렵습니다. 이를 해결하기 위해 **Training-Dev Set**을 도입합니다.

- **Training-Dev Set:** Train set과 동일한 분포에서 추출되었지만, 모델 학습에는 사용되지 않는 검증용 데이터셋입니다.
    

|**평가 구간**|**오차 명칭**|**원인 및 해결책**|
|---|---|---|
|**Human-level Error $\leftrightarrow$ Train Error**|**Avoidable Bias**|모델 크기 확대, 더 긴 학습|
|**Train Error $\leftrightarrow$ Training-Dev Error**|**Variance**|정규화(Regularization), 데이터 추가 수집|
|**Training-Dev Error $\leftrightarrow$ Dev Error**|**Data Mismatch**|데이터 합성(Data Synthesis), Train set에 Dev 분포 데이터 추가|
|**Dev Error $\leftrightarrow$ Test Error**|**Dev Set Overfitting**|더 큰 Dev set 준비|

### **3) Data Mismatch 해결 방법**

- 수동 오류 분석을 통해 Train set과 Dev set의 차이점을 파악합니다.
    
- **인공 데이터 합성 (Artificial Data Synthesis):** 깨끗한 음성에 자동차 소음을 합성하여 시끄러운 환경의 음성 데이터를 만들어내는 기법입니다. (단, 소음 종류가 단순하면 특정 소음에만 과적합될 위험 존재)
    

## 3. 전이 학습과 다중 작업 학습 (Transfer & Multi-task Learning)

### **1) 전이 학습 (Transfer Learning)**

한 분야(Task A)에서 학습한 지식을 다른 분야(Task B)의 모델에 재사용하는 기법입니다.

- **적용 조건:**
    
    1. Task A와 Task B의 입력 타입이 동일할 때 (예: 둘 다 이미지, 둘 다 음성).
        
    2. Task A의 데이터량이 Task B보다 월등히 많을 때.
        
    3. Task A의 하위 개념(Low-level feature)이 Task B를 학습하는 데 도움이 될 때.
        
- **방식:** ImageNet으로 사전 학습(Pre-training)된 모델의 출력층만 교체한 뒤, 대상 데이터셋으로 미세 조정(Fine-tuning)합니다.
    

### **2) 다중 작업 학습 (Multi-task Learning)**

하나의 신경망이 동시에 여러 가지 작업(Task)을 수행하도록 학습시키는 방식입니다 (예: 자율주행 차에서 보행자, 표지판, 차량을 동시에 감지).

- **적용 조건:**
    
    1. 여러 작업이 저수준 특징(Low-level features)을 공유할 때.
        
    2. 각 작업별 데이터의 양이 비슷할 때.
        
    3. 모든 작업을 다룰 수 있을 만큼 신경망의 크기가 충분히 클 때.
        

## 4. 종단간 딥러닝 (End-to-End Deep Learning)

파이프라인 형태의 여러 단계 처리 과정을 **하나의 거대한 신경망으로 통합**하여 입력($X$)에서 출력($Y$)을 직접 파이프라이닝하는 방식입니다.

- **예시 (음성 인식):**
    
    - _전통적 방식:_ 음성 $X \rightarrow$ 특징 추출 $\rightarrow$ 음소 식별 $\rightarrow$ 단어 형성 $\rightarrow$ 텍스트 $Y$
        
    - _End-to-End 방식:_ 음성 $X \rightarrow$ **[ 하나의 신경망 ]** $\rightarrow$ 텍스트 $Y$
        
- **장점:**
    
    - 데이터 본연의 특징을 스스로 학습하며, 인간의 편향된 사전 지식(Hand-designed components)에 제약받지 않습니다.
        
    - 파이프라인 중간 단계 수동 설계 비용을 절감합니다.
        
- **단점:**
    
    - $X \rightarrow Y$ 직접 매핑을 학습하기 위해 **엄청난 양의 레이블링 데이터**가 필요합니다.
        
    - 복잡한 문제를 해결할 때 데이터가 부족하면, 문제를 작은 단계로 나누어 각각 처리하는 방식(Multi-step approach)이 훨씬 잘 작동합니다.
        
