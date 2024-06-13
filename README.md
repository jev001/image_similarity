# 图片相似性

## 图片特征提取

常见判断相似, 可以计算它们的指纹信息,比如md5,hash等 

本文中使用javacv中phash计算,  
PS: javacv为opencv的套壳, javacv是javacpp中的一部分, javacpp将常见的需要c/c++库和java进行绑定

```java
    public static byte[] pHash(String img_path) {
        Mat imread = imread(img_path);
        // 计算 pHash
        Mat pHash = new Mat();
        opencv_img_hash.pHash(imread, pHash);
        // 将 pHash 转换为 Base64 编码的字符串
        // phash 是一个 64位的值 使用byte数组(单个byte表示8位) 数组大小8 就可以表示 完整phash值
        byte[] hashBytes = new byte[(int) pHash.total() * pHash.channels()];
        pHash.data().get(hashBytes);
        return hashBytes;
    }
```

## 相似性算法

相似算法有多重, 欧式距, 曼哈顿距, 杰卡德距, 余弦距, 汉明距离等, 当前使用汉明距离, 该算法核心在于计算两个hash值之间的距离.也即 64位中有多少个差异值.

```java
  public Integer distance(byte[] u, byte[] v) {
        int distance = 0;
        for (int i = 0; i < u.length; i++) {
            byte xor = (byte) (u[i] ^ v[i]);
            distance += Integer.bitCount(xor & 0xFF); // 计算字节中1的数量
        }
        // 输出汉明距离
        return distance;
    }
```

## 相似搜索

本文中使用Hnsw算法作为 用作向量查询.
该算法核心在于,利用汉明算法作为 节点的距离, 将最近的节点放在一层

```java
 // 构建索引
        System.out.println("=====>构建索引====>");
        System.out.println("=================>hnsw算法 dimensions 表示数据维度,按照phash是64位的概念应当使用64维, 只是现在使用了byte数组表示 所以使用8维");
        System.out.println("=================>hnsw算法 withM 表示数据相邻节点数(双向链表),用于构建数据多维数据下每一层节点相关,用于搜索TopK有效");
        System.out.println("=================>hnsw算法 maxItemCount 表示该hnsw最大支持的节点个数");
        HnswIndex<String, byte[], Image, Integer> index = HnswIndex.newBuilder(8, new HammingDistanceFunction(), 100 * 1000).withM(16).build();
        URL origin = openCVImageSimilarity.getClass().getResource("images/origin/");
        URL other = openCVImageSimilarity.getClass().getResource("images/other/");
        
        
        List<Path> origin_path_files = Files.list(Paths.get(origin.getPath())).collect(Collectors.toList());
        
        List<Path> other_path_files = Files.list(Paths.get(other.getPath())).collect(Collectors.toList());
        
        List<Path> allPath = new ArrayList<>();
        allPath.addAll(origin_path_files);
        allPath.addAll(other_path_files);
        
        for (Path path : allPath) {
            byte[] bytes = pHash(path.toString());
            index.add(new Image(path.getFileName().toString(), bytes));
        }

```





## 备注:

图片的相似同样可以使用hash判断, 但此hash值为计算机视觉中的hash 包含 pHash、dHash 和 aHash 三种常见的图像指纹算法的特征提取方法如下:

1. pHash (Perceptual Hash)
>pHash 是基于离散余弦变换 (DCT) 的图像指纹算法
主要步骤包括:
将图像缩放到 32x32 的尺寸
计算 DCT 系数
取 DCT 系数的低频部分
将低频部分量化为 0 和 1 构成 64 位二进制指纹
2. dHash (Difference Hash)
>dHash 是基于图像像素差值的图像指纹算法
主要步骤包括:
将图像缩放到 (w+1)x(h) 的尺寸
计算每个像素与右侧像素的差值
将差值量化为 0 和 1 构成 64 位二进制指纹

3. aHash (Average Hash)
>aHash 是基于图像像素平均值的图像指纹算法
主要步骤包括:
将图像缩放到 8x8 的尺寸
计算每个像素的灰度值
计算所有像素灰度值的平均值
将每个像素与平均值的比较结果量化为 0 和 1 构成 64 位二进制指纹
这三种算法都可以用于有效的图像相似性比较和重复图像检测。它们各有优缺点,适用于不同的场景:


PS: 图片指纹和图片特征有区别

>图像特征提取是一个广泛的领域,有许多不同的方法和技术。下面列举一些常见的图像特征提取方法:

>基于像素的特征提取:
>>颜色直方图
>>纹理特征（如 LBP、GLCM 等）
>>边缘特征（如 Canny、Sobel 等）
>
>基于关键点的特征提取:
>>SIFT（尺度不变特征变换）
>>SURF（加速稳健特征）
>>ORB（优化的 FAST 和 BRIEF）
>
>基于深度学习的特征提取:
>>使用预训练的卷积神经网络（如 VGG、ResNet、Inception 等）提取特征
>>自定义训练的卷积神经网络提取特征
>
>其他特征提取方法:
>>Haar 特征
>>傅里叶变换特征
>>直方图of梯度（HOG）
>>这些特征提取方法各有优缺点,适用于不同的场景和任务。例如,基于像素的方法简单易实现,但对噪声和变形敏感;基于关键点的方法对图像变换和部分遮挡较为鲁棒,但需要进行关键点检测和描述;深度学习方法可以自动学习到更加丰富的特征,但需要大量的训练数据和计算资源。  
>
> 
>具体选择哪种方法,需要根据具体的应用场景、数据特点和计算资源等因素进行权衡和选择。
> 
> 
>图像指纹算法和图像特征提取算法虽然都是用于处理图像的技术,但它们之间有一些明显的区别:   
>>目标不同:  
>>图像指纹算法的目标是生成一个简洁、紧凑的图像指纹或哈希值,用于图像相似性比较和重复图像检测。  
>>图像特征提取算法的目标是提取图像中的有意义特征,如颜色、纹理、形状等,用于图像识别、分类等任务。
> 
>>算法原理不同:  
>>图像指纹算法通常基于计算图像的统计量、频域特征等,生成稳健的二进制指纹。  
>>图像特征提取算法则更关注从图像中提取有语义的特征向量,如 SIFT、HOG 等。
> 
>>输出形式不同:  
>>图像指纹算法输出一个紧凑的二进制数字,如 64 位或 128 位。  
>>图像特征提取算法输出一个特征向量,维度取决于所提取的特征数量。
> 
>>应用场景不同:  
>>图像指纹算法主要用于图像相似性比较、重复图像检测等。  
>>图像特征提取算法则广泛应用于图像分类、检测、检索等任务中。  
>>总的来说,图像指纹算法注重生成一种紧凑、稳健的图像识别码,而图像特征提取算法关注从图像中提取有意义的特征表示。两种技术各有特点,可以根据具体应用场景和需求选择合适的方法。  



## 参考链接

javacv https://github.com/bytedeco/javacv  
javacpp项目 https://bytedeco.org/  

opencv https://opencv.org/  
phash算法opencv实现 https://github.com/opencv/opencv_contrib/blob/4.x/modules/img_hash/src/phash.cpp  
图片相似算法 https://zhuanlan.zhihu.com/p/88696520  
java实现PHash https://chatgpt.com/share/f122cd73-c04d-4111-9be4-764c245cf039  


opencv计算汉明距离实现  
https://docs.opencv.org/3.4/d2/de8/group__core__array.html  
https://docs.opencv.org/3.4/d3/d59/structcv_1_1Hamming.html  


hnsw向量数据库  
hnsw算法论文 https://arxiv.org/abs/1603.09320  
hnsw-java实现 https://github.com/jelmerk/hnswlib  
向量数据库对比(Milvus等核心算法中含有hnsw) https://www.modb.pro/db/516016  
ES向量搜索(es7.x 需要安装插件) https://cloud.tencent.com/document/product/845/98224  
ES向量插件 https://github.com/alexklibisz/elastiknn
ES向量插件 https://github.com/opendistro-for-elasticsearch/k-NN
ANNS:HNSW算法详解 https://songlinlife.github.io/2022/%E6%95%B0%E6%8D%AE%E5%BA%93/Efficient-and-robust-approximate-nearest-neighbor-search-using-Hierarchical-Navigable-Small-World-graphs/  
