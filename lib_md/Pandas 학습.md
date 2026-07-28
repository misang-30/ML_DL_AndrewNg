
- 딥러닝과 데이터 분석 과정에서는 1차원 리스트 슬라이싱을 넘어 2차원 이상의 **Pandas DataFrame**과 **PyTorch/NumPy Tensor**를 자유자재로 다루는 능력이 필수적입니다. 

## 1. Pandas DataFrame 다루기 (엑셀형 데이터 처리)

### 1). 특정 행과 열을 자유롭게 자르기 (loc, iloc)

Pandas에서는 위치 기반의 `iloc`와 이름 기반의 `loc`를 사용하여 엑셀의 영역 선택처럼 데이터를 슬라이싱합니다.



``` Python
import pandas as pd

# 샘플 데이터 생성
df = pd.DataFrame({
    'Feature_A': [10, 20, 30, 40, 50],
    'Feature_B': [1.1, 2.2, 3.3, 4.4, 5.5],
    'Target': [0, 1, 0, 1, 0]
}, index=['r1', 'r2', 'r3', 'r4', 'r5'])

# (1). 인덱스 번호로 자르기 (iloc)
# 0~2번 행과 0~1번 열 선택 (마지막 인덱스는 미포함)
sub_df1 = df.iloc[0:3, 0:2]

# (2). 라벨 이름으로 자르기 (loc)
# 'r1'부터 'r3' 행까지, 'Feature_A'부터 'Feature_B' 열 선택 (마지막 라벨 포함)
sub_df2 = df.loc['r1':'r3', 'Feature_A':'Feature_B']

# (3). 딥러닝 입력값(X)과 정답지(y) 분리하기
# 마지막 열 제외한 모든 열을 X로, 마지막 열을 y로 분리
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

### 2). 데이터 붙이기 (concat, merge)

엑셀에서 표 아래에 데이터를 덧붙이거나(행 결합), 옆에 새로운 열을 붙이는(열 결합) 작업입니다.


``` Python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'C': [100, 200]})

# (1). 아래로 붙이기 (행 결합, 엑셀의 행 추가)
# ignore_index=True로 설정하여 인덱스 재정렬
df_vertical = pd.concat([df1, df2], axis=0, ignore_index=True)

# (2). 옆으로 붙이기 (열 결합, 엑셀의 열 추가)
df_horizontal = pd.concat([df1, df3], axis=1)

# (3). 특정 키 기준으로 합치기 (merge, 엑셀의 VLOOKUP 효과)
left = pd.DataFrame({'id': [1, 2, 3], 'val1': ['a', 'b', 'c']})
right = pd.DataFrame({'id': [2, 3, 4], 'val2': ['x', 'y', 'z']})
df_merged = pd.merge(left, right, on='id', how='inner')
```

## 2. PyTorch Tensor 다루기 (딥러닝 모델 입력용 데이터 처리)

### 1). 차원 슬라이싱 및 형태 변경 (slice, reshape, squeeze, unsqueeze)

딥러닝 모델에 데이터를 넣을 때 형태(Shape)를 맞추는 작업이 매우 중요합니다.


``` Python
import torch

# (3, 4) 크기의 2차원 텐서 생성
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# (1). 특정 부분 슬라이싱 (행: 1번~끝, 열: 2번~3번)
sliced_x = x[1:, 2:4]

# (2). 차원 늘리기 (unsqueeze) 및 줄이기 (squeeze)
# 배치 차원(Batch Dimension) 추가시 자주 사용: (3, 4) -> (1, 3, 4)
x_expanded = x.unsqueeze(0)

# 크기가 1인 차원 제거: (1, 3, 4) -> (3, 4)
x_squeezed = x_expanded.squeeze(0)

# (3). 형태 완전히 바꾸기 (view 또는 reshape)
# (3, 4) -> (6, 2)
x_reshaped = x.reshape(6, 2)
```

### 2). 텐서 합치기 및 분할하기 (cat, stack, split)

여러 데이터 조각을 하나의 배치(Batch)로 모으거나, 결합된 텐서를 나누는 작업입니다.


``` Python
import torch

t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])

# (1). 기존 차원을 따라 붙이기 (torch.cat)
# 행 방향(아래)으로 붙이기 -> 결과 크기: (4, 2)
cat_dim0 = torch.cat([t1, t2], dim=0)

# 열 방향(옆)으로 붙이기 -> 결과 크기: (2, 4)
cat_dim1 = torch.cat([t1, t2], dim=1)

# (2). 새로운 차원을 생성하며 쌓기 (torch.stack)
# 배치 단위로 묶을 때 사용 -> 결과 크기: (2, 2, 2)
stacked = torch.stack([t1, t2], dim=0)

# (3). 텐서 자르기 (torch.split)
# 행 방향으로 1개씩 나누기
splits = torch.split(cat_dim0, split_size_or_sections=1, dim=0)
```

---

## 4. 조건에 맞는 데이터 뽑아내기 (불리언 인덱싱)

엑셀의 '필터' 기능처럼 특정 조건을 만족하는 데이터만 잘라내는 기술입니다.

### 1). Pandas에서 조건 필터링


``` Python
import pandas as pd

df = pd.DataFrame({'age': [15, 25, 35, 45], 'score': [80, 95, 70, 85]})

# (1). 나이가 20세 이상인 데이터만 추출
adults = df[df['age'] >= 20]

# (2). 여러 조건 조합 (&: AND, |: OR)
target = df[(df['age'] >= 20) & (df['score'] >= 80)]
```

### 2). PyTorch에서 조건 마스킹 및 값 변경


```Python
import torch

x = torch.tensor([1, 5, 8, 3, 10])

# (1). 5보다 큰 값만 추출
high_values = x[x > 5]  # tensor([8, 10])

# (2). 조건에 따라 값 교체 (5보다 작은 값은 0으로 변경)
x_masked = torch.where(x < 5, torch.tensor(0), x)  # tensor([0, 5, 8, 0, 10])
```

## 5. 차원 맞추기 (Permute, Transpose)

이미지 데이터나 텍스트 데이터를 다룰 때, 데이터의 형태(Shape) 순서가 맞지 않아 에러가 나는 경우가 많습니다.

### 1). 이미지 데이터 차원 변경


``` Python
import torch

# OpenCV/PIL 이미지 형태: (높이, 너비, 채널) -> (224, 224, 3)
# PyTorch 모델 입력 형태: (채널, 높이, 너비) -> (3, 224, 224)

img_tensor = torch.randn(224, 224, 3)

# (1). 차원 순서 변경 (permute)
img_transformed = img_tensor.permute(2, 0, 1)  # (3, 224, 224)로 변경
```

## 6. 넘파이(NumPy)와 텐서(Tensor)의 호환

데이터 전처리는 Pandas나 NumPy로 하고, 모델 입력은 PyTorch Tensor로 변환하는 흐름이 일반적입니다.

### 1). 변환 스킬

```Python
import numpy as np
import torch

# (1). NumPy 배열 -> PyTorch Tensor
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_array)

# (2). PyTorch Tensor -> NumPy 배열
tensor = torch.tensor([1.0, 2.0, 3.0])
np_from_tensor = tensor.numpy()  # CPU 텐서일 때 작동
```

> 처음부터 모든 문법을 외울 필요는 없습니다. 데이터를 다루다가 **"행/열을 나누고 싶다"** 하면 `iloc`/`cat`을, **"형태(Shape)가 안 맞는다"** 하면 `reshape`/`permute`를 검색해서 활용해 보시는 것을 추천합니다.
