# python-lamda skimage
	
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

image = imread(r"C:\Users\Tavish\Desktop\7.jpg")  
show_img(image)  
  
red, yellow =   image.copy(), image.copy()  
red[:,:,(1,2)] = 0  
yellow[:,:,2]=0  
show_images(images=[red,yellow], titles=['Red Intensity','Yellow Intensity'])  
  
from skimage.color import rgb2gray  
gray_image = rgb2gray(image)  
show_images(images=[image,gray_image],titles=["Color","Grayscale"])  
print "Colored image shape:", image.shape  
print "Grayscale image shape:", gray_image.shape  
  
from skimage.filter import threshold_otsu  
thresh = threshold_otsu(gray_image)  
binary = gray_image > thresh  
show_images(images=[gray_image,binary_image,binary],titles=["Grayscale","Otsu Binary"])  
  
from skimage.filter import gaussian_filter  
blurred_image = gaussian_filter(gray_image,sigma=20)  
show_images(images=[gray_image,blurred_image],titles=["Gray Image","20 Sigma Blur"])  

人間の虹彩の写真を導入して、画像の特徴を抽出して、分類します。
画像から特徴を抽出するには、「Convolutional Neural Net」と呼ばれる方法です。
"Convolutional Neural Netは主に下記の二つのフェーズを繰り返すことにより特徴抽出を行います。
Convolutional層
Pooling層"
"これらの処理を何度も繰り返し、特徴を抽出し、得られた全特徴を用いて多層のニューラルネットやSVM
（Support Vector Machine）を用いて予測を行うことになります。
この一連の特徴抽出の過程がConvolutional Neural Netです。"

画像の前処理
画像特徴抽出

1,faster R-CNN tensorflow实现	
2,faster R-CNN训练自己的数据	
3,GPU使用，CUNN编程	
4,分布式训练	
	
batch_normalization位置	每次maxpooling之后，conv_2d或fully_connected之前
线形回归，逻辑回归	损失函数：loss='mean_square'
	"激活函数：中间层：activation='linear'
最后一层：activation='sigmoid'
损失函数有时用：loss='binary_crossentropy'"
分类训练	"fully_connected之后接dropout
前面的激活函数：activation='relu'
分类时也就是最后一层激活函数：activation='softmax'"
tf.layer	
tf.slim	
tf.learn	
deeps	
weights	
Deconvolution	
ethereon/caffe-tensorflow	
wide deep learning	
batch size	batch为32的时候，alex开始收敛，但是googlenet不收敛；提高batch size，googlenet开始收敛
数据的预处理	"1,去均值(zero-center),将输入数据的各个维度都中心化到0
2,归一化(normalize),将特征的幅度变换到统一范围（不一定是0-1之间）
3,主成分分析(PCA)，一种降维的方法,例如把特征从二维分布转到一维分布
4，白化（whitened）其实是指将特征转换成正态分布。比如圆形"
tensorflow Multi-task	
把公司员工的虹膜分为一类，不是公司员工的虹膜分为一类，都输入faster RCNN中训练	
確率	
算法，数据，硬件关于slim
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
http://www.jiqizhixin.com/article/1474

Tensor运算：http://www.jianshu.com/p/00ab12bc357c

caffe tensorflow	"https://my.oschina.net/yilian/blog/672135
http://blog.csdn.net/u012235274/article/details/52593632"
milti labels	"https://github.com/lan2720/cnn-for-captcha
https://github.com/jg8610/multi-task-part-1-notebook/blob/master/Multi-Task%20Learning%20Tensorflow%20Part%201.ipynb"
slice	http://qiita.com/supersaiakujin/items/464cc053418e9a37fa7b#slice
concat	
split	
tile	
pad	
fill	
constant	
random_normal	
truncated_normal	
random_uniform	
四种Cross Entropy算法	http://ms.csdn.net/geek/126833

slim	"https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
http://www.jiqizhixin.com/article/1474"	
slim.nets	https://github.com/tensorflow/models/tree/master/slim/nets	把所有的fc全变成了conv，看来大势已定。输入imagesize=224*224
expand_dims	http://www.jianshu.com/p/00ab12bc357c	插入维
tf.squeeze		减少维
Image PreLoader	tflearn.data_utils.image_preloader (target_path, image_shape, mode='file', normalize=True, grayscale=False, categorical_labels=True, files_extension=None, filter_channel=False)	
Build HDF5 Image Dataset	tflearn.data_utils.build_hdf5_image_dataset (target_path, image_shape, output_path='dataset.h5', mode='file', categorical_labels=True, normalize=True, grayscale=False, files_extension=None, chunks=False)	
load_csv	tflearn.data_utils.load_csv (filepath, target_column=-1, columns_to_ignore=None, has_header=True, categorical_labels=False, n_classes=None)	
	full-connect-{}'.format(i + 1)	

Multi-Task Learning	https://jg8610.github.io/Multi-Task/
tf.nn.l2_loss(Y-tf.matmul(X,W))==tf.pow(tf.add(Y,-tf.matmul(X,W)),2)
