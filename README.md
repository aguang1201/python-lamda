# python-lamda
	
lambda	匿名函数	
f=lambda x:x*x
f(3)"	"(lambda x:x+x)(3)
return:6"			

map	
将一个函数映射到一个可枚举类型上面.想要输入多个序列，需要支持多个参数的函数，注意的是各序列的长度必须一样，否则报错	
map(lambda x:x+1,[1,2,3])	
map( lambda x: x*x, [y for y in range(10)] )	
map(lambda x:x+x,'abcd')
return['aa','bb','cc','dd']"	
def add(x,y): return x+y
map(add,range(5),range(5))
return [0,2,4,6,8]
map(str,[1,2,3])
return: [""1"",""2"",""3""]

sort	排序	
s = [('a', 3), ('b', 2), ('c', 1)]
sorted(s, key=lambda x:x[1])

filter	
对sequence中的item依次执行function(item)，将执行结果为True的item组成一个List/String/Tuple（取决于sequence的类型）	"I=['foo','bar','far']
filter(lambda x: 'f' in x, I)
return：['foo','far']"	"I=['foo','bar','far']
map(lambda x: x.upper(), filter(lambda x: 'f' in x, I))
return:['FOO','FAR']
filter(lambda x:x%2!=0 and x%3!=0,range(2,25))
return:[5,7,11,13,17,19,23]"	"filter(lambda x:x!='a','abcde')
returen:'bcde'
filter(lambda x : not [x%i for i in range(2,x) if x%i == 0],range(start,stop))

reduce	
对sequence中的item顺序迭代调用function，如果有starting_value，还可以作为初始值调用	"reduce(lambda a,b: a*b,range(1,5))
return:24
reduce(lambda a,b:a+b,range(5),10)
return:10+0+1+2+3+4"			

s.strip(rm)	
s为字符串，rm为要删除的字符序列,当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' '),这里的rm删除序列是只要边（开头或结尾）上的字符在删除序列内，就删除掉。	"123abc'.strip('21')
return:'3abc'
'123abc'.strip('12')
return:'3abc'
'   123'.strip()
return:'123'
'123\n\r'.strip()
return:'123'"			
		"I=[1,2,3,4]
x*x for x in I if (x>3)

array	
只能存储同样地数据类型的数据。它所占的存储空间的大小就是数据的大小。	

var(a,axis=0) 
方差：样本中各数据与样本平均数的差的平方的和的平均数叫做样本方差
a=array([[6, 7, 1, 6], 
       [1, 0, 2, 3], 
       [7, 8, 2, 1]]) 
np.var(a,axis=0) 
array([  6.88888889,  12.66666667,   0.22222222,   4.22222222])

std(a,axis=0) 	标准差:方差平方根
axis=0：在列上求
axis=1：在行上求"	"np.std(a,axis=0) 
array([ 2.62466929,  3.55902608,  0.47140452,  2.05480467]) 

sum(a,axis=0) 	"求和，把所有行加起来，编成一行
或者说，消灭多行，编成一行，在列上求和
np.sum(a,axis= 0 ) 
array([14, 15,  5, 10]) 
>>> np.sum(a,axis= 1 ) 
array([20,  6, 18]) 

mean(a,axis=0)	均值，同上	
np.mean(a,axis=0) 
array([ 4.66666667,  5.        ,  1.66666667,  3.33333333]) 	

random	随机数	">>> b=np.random.randint(0,5,8) 
>>> b 
array([2, 3, 3, 0, 1, 4, 2, 4]) 

unique	保留数组中不同的值，返回两个参数。bincount（）对数组中各个元素出现的次数进行统计，还可以设定相应的 权值	">>> np.unique(b) 
array([0, 1, 2, 3, 4]) "	">>> c,s=np.unique(b,return_index=True) 
>>> c 
array([0, 1, 2, 3, 4]) 
>>> s 
array([3, 4, 0, 1, 5])（元素出现的起始位置） 	

np.max()
np.min() 
np.argmax()
np.argmin()
np.sort()"	最值和排序：最值有np.max(),np.min() 他们都有axis和out（输出）参数, 而通过np.argmax(), np.argmin()可以得到取得最大或最小值时的 下标。排序通过np.sort(), 而np.argsort()得到的是排序后的数据原来位置的下标。					
