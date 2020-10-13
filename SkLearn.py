import numpy as np
import matplotlib.pyplot as plt

#输入数据
X=np.array([[1,0,1],
	  [1,-1,0],
	  [1,0,-1],
      [1,1,0]])
#标签
Y=np.array([1,1,-1,-1])
#权值初始化,1行3列，取值范围-1到1
W=(np.random.random(3)-0.5)*2
print(W)
#学习率设置
lr=0.11
#计算迭代次数
n=0
#神经网络输出
o=0

def update():
	global X,Y,W,lr,n
	n+=1
	o=np.sign(np.dot(X,W.T))
	W_c=lr*((Y-o.T).dot(X))/int(X.shape[0])
	W=W+W_c

for _ in range(100):
	update() #更新权值
	print(W) #打印当前权值
	print(n) #打印迭代次数
	o=np.sign(np.dot(X,W.T)) #计算当前输出
	if(o==Y.T).all(): #如果实际输出等于期望输出，模型收敛，循环结束
		print('Finished')
		print('epoch:',n)
		break

#正样本
x1=[1,0]
y1=[0,1]
#负样本
x2=[-1,0]
y2=[0,-1]

#计算分界线的斜率以及截距
k=-W[1]/W[2]
d=-W[0]/W[2]
print('k=',k,',d=',d)

xdata=np.linspace(-5,10)
plt.plot(xdata,xdata*k+d,'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()