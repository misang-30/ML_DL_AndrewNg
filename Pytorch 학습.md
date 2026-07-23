# < 순서 >
- 1.텐서 다루기
	- 텐서 생성, 모양 변환
	- 텐서 연산 및 인덱싱/슬라이싱
	- GPU 연산 전환 (`.to('cuda')` 또는 `.to('device')`)
	- **목표**: NumPy 배열을 파이토치 텐서로 바꾸고 자유자재로 모양을 조작할 수 있게 되기
	
- 2.핵심 메커니즘 이해
	- **Autograd (자동 미분)**: `requires_grad=True`, `.backward()`의 원리
	- **선형 회귀(Linear Regression)** 구현해보기
	- **학습 3총사**
	    1. 손실 함수 (Loss Function, 예: `nn.MSELoss()`)
	    2. 최적화 알고리즘 (Optimizer, 예: `torch.optim.SGD`, `Adam`)
	    3. 경사 초기화 (`optimizer.zero_grad()`)

- 3.PyTorch답게 코드 작성하기(nn.Module & DataLoader)
    - `nn.Module`을 상속받아 커스텀 모델 클래스 정의하기 (`forward` 함수)
    - `Dataset`과 `DataLoader`를 이용해 데이터 mini-batch 단위로 로드하기
    - **기본 훈련 루프(Training Loop)** 패턴 익히기 (Epoch, Batch, Loss 계산, Weight 업데이트)

- 4.커스텀 데이터셋 구축 및 전처리 (Custom Dataset & Transforms) 

- 5.남이 만든 강력한 모델 가져다 쓰기 (Pretrained Model & Transfer Learning) 

- 6.모델 저장/불러오기 및 데모 서비스 만들기 (Gradio/Streamlit)

- 7.**실습**: MNIST 손글씨 숫자 분류 (다층 인공신경망, MLP 구현)
---
---
# < 기초 >

## 1. 텐서 다루기 

``` Python
import torch

# 1. 텐서 만들기
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x)
print(x.shape)  # 모양 확인 (2, 2)

# 2. 모양 바꾸기 (Reshape) - reshape나 view를 사용합니다.
x_flat = x.view(-1)  # 1차원으로 쫙 펴기 -> [1.0, 2.0, 3.0, 4.0]
print(x_flat.shape)  # (4,)

# 3. GPU로 보내기 (GPU가 사용 가능한 환경이라면)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
```

** 핵심 포인트:**

- 파이토치의 모든 데이터와 모델은 **같은 장치(CPU 또는 GPU)** 위에 있어야 연산이 됩니다.
    
- `.view(-1)`에서 `-1`은 "나머지 차원에 맞춰서 자동으로 개수를 맞춰라"라는 뜻입니다.

---
---
## 2. 자동 미분(Autograd), 경사 하강법

- 파이토치가 강력한 이유는 "내가 계산 과정을 적어두기만 하면 알아서 미분(기울기 계산)을 해준다"는 점 때문입니다.

- 간단한 예시: $y = 2x + 1$ 이라는 공식을 파이토치가 스스로 학습해서 $w=2$, $b=1$을 찾아내게 만들어봅시다.

```Python
import torch
import torch.optim as optim

# 데이터 준비 (x가 1일때 y는 3, x가 2일때 y는 5 ...) -> y = 2x + 1 관계
x_train = torch.tensor([[1.0], [2.0], [3.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0]])

# 가중치(w)와 편향(b) 초기화
# requires_grad=True: "이 변수의 미분값(기울기)을 계속 추적해줘!" 라는 설정
W = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 최적화 도구(Optimizer) 설정: W와 b를 경사하강법(SGD)으로 수정하겠다!
optimizer = optim.SGD([W, b], lr=0.01)

# 학습 루프 (200번 반복)
for epoch in range(200):
    # 1. 예측값 계산 (y_pred = x * W + b)
    hypothesis = x_train * W + b

    # 2. 손실(Loss) 계산: 예측값과 실제값의 차이 (평균 제곱 오차)
    loss = torch.mean((hypothesis - y_train) ** 2)

    # 3. 파이토치 학습 3단 공식 (외우두시면 끝입니다!)
    optimizer.zero_grad()  # 1) 지난번 계산된 기울기 초기화
    loss.backward()        # 2) 미분 계산 (기울기 구하기)
    optimizer.step()       # 3) 구한 기울기로 W, b 업데이트

print(f"학습 후 W: {W.item():.2f}, b: {b.item():.2f}")
# 출력 결과: W: 2.00, b: 1.00 에 매우 가까워집니다!

```
---
---
## 3. PyTorch답게 코드 작성하기 (nn.Module & DataLoader)

- 실제 딥러닝 프로젝트에서는 위처럼 `W`, `b`를 직접 다루지 않고, `nn.Module`과 `DataLoader`라는 도구를 사용해 깔끔하게 작성합니다.

- 아래 구조가 **현업과 연구에서 사용하는 표준 파이토치 코드 패턴**입니다.


### 1). 모델 클래스 만들기 (nn.Module):조립식 신경망.

`nn.Module`을 상속받아 신경망의 구조를 정의합니다.

``` Python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 1개 -> 은닉층 4개 -> 출력 1개의 레이어 정의
        self.linear1 = nn.Linear(1, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        # 데이터가 들어왔을 때 계산되는 순서 (순전파)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

### 2). 데이터 로더 세팅 (DataLoader):배치 단위로 쪼개기.

데이터를 한꺼번에 넣으면 메모리가 터지므로, 묶음(Batch) 단위로 쪼개서 넣습니다.

``` Python
from torch.utils.data import TensorDataset, DataLoader

# Dataset 생성
dataset = TensorDataset(x_train, y_train)

# DataLoader 생성 (데이터를 2개씩 묶고, 셔플해서 가져오기)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### 3).모델, 손실함수, 옵티마이저 선언 : 학습 준비.

``` Python
model = SimpleNet()
criterion = nn.MSELoss()  # 손실함수 (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 성능 좋은 Adam 사용
```

### 4). 최종 실전 학습 루프 (Training Loop):완성형 코드 패턴.

``` Python
epochs = 100

for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        # 1. 예측
        prediction = model(batch_x)

        # 2. 손실 계산
        loss = criterion(prediction, batch_y)

        # 3. 학습 3단 공식
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

---
----
---
# < 플젝 > 
## 4. 커스텀 데이터셋 구축 및 전처리 (Custom Dataset & Transforms)

실제 프로젝트에서는 기본 제공 데이터셋이 아니라 **내 컴퓨터에 있는 이미지나 CSV 파일**을 파이토치로 읽어와야 합니다. 이를 위한 표준 클래스 구조와 전처리 방법입니다.

### 1) 커스텀 Dataset 클래스 기본 틀

`torch.utils.data.Dataset`을 상속받아 딱 **3가지 필수 메서드**(`__init__`, `__len__`, `__getitem__`)를 구현합니다.


``` Python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MyCustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # 1. 파일 경로, 라벨, 변환 기법(Transform)을 받아 보관
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # 2. 데이터 전체 개수 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 3. idx번째 데이터 한 개를 읽어서 텐서로 변환 후 리턴
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image) # 이미지 전처리 적용

        return image, torch.tensor(label)
```

### 2) torchvision.transforms를 활용한 전처리 & 증강(Augmentation)

이미지 크기를 통일하거나, 모델이 잘 학습하도록 이미지를 돌리고 뒤집는 전처리를 적용합니다.



``` Python
from torchvision import transforms

# 학습용 데이터 전처리 (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # 크기 고정 (224x224)
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
    transforms.ToTensor(),                  # PIL 이미지를 파이토치 텐서로 변환 (0~1 범위)
    transforms.Normalize(                   # 이미지 정규화 (표준화)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

> **🎯 4단계 목표:** 내 컴퓨터 폴더에 있는 이미지들을 읽어서 `Dataset`을 만들고, `DataLoader`에 태워 묶음(Batch)으로 뽑아낼 수 있게 되기.

---
## 5. 전이 학습(Transfer Learning)과 기존 모델 활용

직접 모델 구조를 설계하면 성능도 안 나오고 학습 시간도 오래 걸립니다. 이미 수백만 장의 데이터로 학습된 사전 학습 모델(Pretrained Model)을 가져와서 내 문제에 맞게 개조해 사용합니다.

### 1). 사전 학습된 모델 불러오기:torchvision.models 활용.

```Python
import torchvision.models as models
import torch.nn as nn

# 이미 학습된 ResNet-18 모델의 뼈대와 가중치 다운로드
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

### 2). 분류기(Final Layer) 수술하기 : 내 클래스 개수로 변경.

ResNet의 원래 마지막 출력층은 1,000개 클래스용입니다. 이를 내 데이터(예: 개/고양이 2개)에 맞게 교체합니다.


``` Python
# ResNet의 마지막 레이어(fc) 입력 차원 수 확인
num_ftrs = model.fc.in_features

# 마지막 레이어를 내 클래스 개수(2개)에 맞춰 교체
model.fc = nn.Linear(num_ftrs, 2)
```

### 3). 미세 조정(Fine-Tuning) 모드 설정:선택사항.

앞쪽 뼈대는 학습되지 않게 동결(Freeze)시키고, 내가 새로 붙인 마지막 레이어만 학습시킬 수도 있습니다.


``` Python
# 뼈대 가중치 고정 (학습 안 함)
for param in model.parameters():
    param.requires_grad = False

# 새로 만든 fc 레이어는 가중치 학습 가능하도록 설정
for param in model.fc.parameters():
    param.requires_grad = True
```

> **🎯 5단계 목표:** 남이 만든 거대 모델(ResNet 등)을 가져와 내 데이터 분류용으로 10분 만에 수정해서 높은 정확도를 얻기.

---
## 6. 모델 저장/불러오기 & 결과물(데모 웹) 만들기

학습시킨 모델을 파일로 저장해 두고, 외부 앱이나 웹 화면에 붙여서 **진짜 동작하는 서비스**로 완성합니다.

### 1) 모델 가중치 저장 및 불러오기

``` Python

# 1. 학습 완료 후 가중치만 파일로 저장
torch.save(model.state_dict(), 'my_model.pt')

# 2. 나중에 추론(Inference)할 때 불러오기
loaded_model = models.resnet18()
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 2) # 구조 동일하게 세팅
loaded_model.load_state_dict(torch.load('my_model.pt'))     # 가중치 덮어씌우기

# 3. 추론 필수 2개 모드
loaded_model.eval() # 평가 모드 전환
with torch.no_grad(): # 미분 계산 비활성화 (메모리 절약)
    output = loaded_model(test_image_tensor)
```

### 2) Gradio로 5분 만에 웹 UI 만들기

파이썬 라이브러리인 `gradio`를 쓰면 웹 코드(HTML/CSS) 없이도 이미지 드래그&드롭 예측 사이트를 만들 수 있습니다.


``` Python
import gradio as gr

def predict_image(img):
    # img를 받아서 전처리 -> 모델 넣기 -> 결과 반환하는 함수
    tensor_img = train_transform(img).unsqueeze(0) # 배치 차원 추가
    with torch.no_grad():
        out = loaded_model(tensor_img)
        pred = torch.argmax(out, dim=1).item()
    
    classes = ['고양이', '강아지']
    return classes[pred]

# 웹 화면 구성
demo = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="🐶/🐱 개 고양이 분류기"
)

demo.launch() # 웹 서버 실행!
```

> **🎯 6단계 목표:** `.pt` 파일로 내 모델을 보관하고, 브라우저에서 직접 사진을 넣어보고 결과를 확인하는 나만의 데모 완성하기.