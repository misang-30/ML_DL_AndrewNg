
Course 4의 마지막 주차에서는 CNN의 대용량 응용 분야인 얼굴 인식(Face Recognition)과 합성 기술인 신경 화풍 변환(Neural Style Transfer)을 배우며 CNN 전 과정을 마무리합니다.

## 1. 얼굴 인식 (Face Recognition)

### **1) Face Verification vs Face Recognition**

- **Face Verification (1:1 매칭):** 입력 이미지와 ID/이름이 주어졌을 때, 본인이 맞는지 검증하는 문제 (쉬움).
    
- **Face Recognition (1:K 매칭):** $K$명의 등록된 사람 DB가 있을 때, 입력 이미지가 그 중 누구인지 식별하는 문제 (어려움, $K$가 커질수록 정확도 요구됨).
    

### **2) One-Shot Learning (원샷 학습)**

- **문제점:** 직원의 사진이 단 1장(또는 몇 장)만 존재할 때, 일반적인 Softmax 분류기는 데이터 부족으로 오버피팅되고 새 직원이 입사할 때마다 전체 네트워크를 재학습해야 합니다.
    
- **해결책 (Similarity Function):** 두 이미지 간의 차이를 계산하는 유사도 함수 $d(x^{(1)}, x^{(2)})$를 학습시킵니다.
    
    - $d(x^{(1)}, x^{(2)}) \le \tau$: 같은 사람으로 판단
        
    - $d(x^{(1)}, x^{(2)}) > \tau$: 다른 사람으로 판단
        

### **3) 샴 네트워크 (Siamese Network)**

동일한 가중치(Parameters)를 공유하는 두 개(또는 세 개)의 동일한 CNN을 병렬 배치하여 입력 이미지를 고차원 특징 벡터(Embedding Vector)로 변환합니다.

- 입력 $x^{(1)}$의 인코딩 결과를 $f(x^{(1)})$, $x^{(2)}$의 인코딩 결과를 $f(x^{(2)})$라 할 때, 두 벡터의 유클리드 거리를 측정합니다:
    
    $$d(x^{(1)}, x^{(2)}) = \vert{}\vert{}f(x^{(1)}) - f(x^{(2)})\vert{}\vert{}_2^2$$
    

## 4) 삼중주 손실 함수 (Triplet Loss)

모델이 같은 사람의 인코딩은 가깝게, 다른 사람의 인코딩은 멀게 만들도록 학습시키는 손실 함수입니다.

- **3개 데이터 구성:**
    
    - **Anchor ($A$):** 기준 이미지
        
    - **Positive ($P$):** Anchor와 같은 사람의 이미지
        
    - **Negative ($N$):** Anchor와 다른 사람의 이미지
        
- **조건식:**
    
    $$\vert{}\vert{}f(A) - f(P)\vert{}\vert{}^2 \le \vert{}\vert{}f(A) - f(N)\vert{}\vert{}^2$$
    
    - _주의점:_ 네트워크가 모든 $f(x) = 0$ 또는 정수 상수로 출력해버리는 편법(Trivial Solution)을 막기 위해 마진 $\alpha$ (Margin)를 도입합니다.
        
        $$\vert{}\vert{}f(A) - f(P)\vert{}\vert{}^2 - \vert{}\vert{}f(A) - f(N)\vert{}\vert{}^2 + \alpha \le 0$$
        
- **Triplet Loss 수식:**
    
    $$\mathcal{L}(A, P, N) = \max \left( 0, \vert{}\vert{}f(A) - f(P)\vert{}\vert{}^2 - \vert{}\vert{}f(A) - f(N)\vert{}\vert{}^2 + \alpha \right)$$
    
- **데이터 선택 팁 (Hard Negative Mining):**
    
    랜덤으로 선택하면 $d(A, P) + \alpha \ll d(A, N)$ 조건을 쉽게 만족하여 학습이 잘 안 됩니다. 학습 효율을 극대화하려면 $d(A, P) \approx d(A, N)$인 까다로운 샘플(Hard Triplets)을 일부러 골라 학습시켜야 합니다.
    

## 2. 신경 화풍 변환 (Neural Style Transfer - NST)

내용 이미지(Content Image, $C$)의 구도에 스타일 이미지(Style Image, $S$)의 예술적 화풍을 입혀 새로운 이미지(Generated Image, $G$)를 생성하는 기법입니다.

### **1) 깊은 CNN 레이어가 학습하는 피처 (Visualizing CNN)**

- **얕은 레이어 (Layer 1~2):** 단순한 에지(Edge), 색상, 선, 질감 등 저수준 특징 포착.
    
- **중간 레이어 (Layer 3~4):** 복잡한 질감, 패턴, 물체의 부분(눈, 바퀴 등) 포착.
    
- **깊은 레이어 (Layer 5~):** 객체 전체 클래스(dog, car) 및 고수준의 추상적 개념 포착.
    

### **2) Cost Function (전체 손실 함수)**

- **특징:** CNN의 가중치 $W, b$를 업데이트하는 것이 아니라, **픽셀 데이터 자체인 $G$를 최적화**합니다 ($G$를 미동 상태에서 그라디언트 데센트로 업데이트).
    
    $$\mathcal{J}(G) = \alpha \cdot \mathcal{J}_{\text{content}}(C, G) + \beta \cdot \mathcal{J}_{\text{style}}(S, G)$$
    

### **3) Content Cost ($\mathcal{J}_{\text{content}}$)**

- 너무 얕지도 너무 깊지도 않은 중간 레이어 $l$을 선택합니다.
    
- 이미지 $C$의 활성화값 $a^{[l](https://gemini.google.com/app/C)}$와 생성 이미지 $G$의 활성화값 $a^{[l](https://gemini.google.com/app/G)}$의 차이(L2 Norm)를 계산합니다:
    
    $$\mathcal{J}_{\text{content}}(C, G) = \frac{1}{2} \vert{}\vert{}a^{[l](C)} - a^{[l](G)}\vert{}\vert{}^2$$
    

### **4) Style Cost ($\mathcal{J}_{\text{style}}$)**

스타일은 특정 레이어 내 서로 다른 채널(Channel) 간의 상관관계(Correlation)로 정의합니다.

- **Gram Matrix ($G^{[l]}$):** $i$번째 채널과 $j$번째 채널이 얼마나 함께 활성화되는지 내적으로 측정합니다.
    
    $$G_{i, j}^{[l]} = \sum_{k=1}^{n_H^{[l]}} \sum_{l=1}^{n_W^{[l]}} a_{i, k, l}^{[l]} \cdot a_{j, k, l}^{[l]}$$
    
- 스타일 이미지 $S$의 그람 매트릭스와 $G$의 그람 매트릭스 차이를 최소화하며, 여러 레이어의 스타일 손실을 합산하여 최종 Style Cost를 완성합니다.
    
