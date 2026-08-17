##  0. 목차

- 2.1 퍼셉트론이란?
- 2.2 단순한 논리 회로 (AND, NAND, OR)
- 2.3 퍼셉트론 구현하기
- 2.4 퍼셉트론의 한계 (XOR 문제)
- 2.5 다층 퍼셉트론 (XOR 구현)
- 2.6 NAND에서 컴퓨터까지
- 2.7 정리

---
## 퍼셉트론 구현


``` python
# theta를 이용한 버전
def AND (x1, x2):
	w1, w2 theta = 0.5, 0.5, 0.7
	tmp = x1 * w1 + x2*w2
	if tmp <= theta:
		return 0
	elif tmp > theta : 
		return 1
	

```

``` python
# bias를 이용한 버전 
def AND (x1, x2):
	x = np.array([x1,x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else : 
		return 1
	
```


``` python
def NAND (x1, x2):
	x = np.array([x1,x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else : 
		return 1
		
def OR (x1, x2):
	x = np.array([x1,x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else : 
		return 1
```


## 퍼셉트론 한계
- 직선으로 XOR 검출 못한다.
- 그래서 비선형을 도입. 다층 퍼센트론으로 해결
-  XOR (x,y) = AND (NAND(x1,x2), OR(x1,x2))

``` python
def XOR (x1, x2 );
	s1 = NAND(x1,x2)
	s2 = OR(x1,x2)
	y = AND(s1,s2)
	return y
	
```