import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;
import org.bytedeco.opencv.global.opencv_img_hash;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_img_hash.AverageHash;
import org.bytedeco.opencv.opencv_img_hash.PHash;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class ImageSimilarity {
    
    private static void printDetail(String image_path) {
        byte[] pHash = pHash(image_path);
        System.out.println("phash byte:  " + Arrays.toString(pHash));
        String s = Base64.getEncoder().encodeToString(pHash);
        System.out.println("phash base64:  " + s);
        
        
        byte[] averageHash = averageHash(image_path);
        System.out.println("averageHash byte:  " + Arrays.toString(averageHash));
        String s1 = Base64.getEncoder().encodeToString(averageHash);
        System.out.println("averageHash base64:  " + s1);
    }
    
    private static byte[] averageHash(String imagePath) {
        Mat imread = imread(imagePath);
        // 计算 averageHash
        Mat averageHash = new Mat();
        opencv_img_hash.averageHash(imread, averageHash);
        // 将 averageHash 转换为 Base64 编码的字符串
        byte[] hashBytes = new byte[(int) averageHash.total() * averageHash.channels()];
        averageHash.data().get(hashBytes);
        return hashBytes;
    }
    
    /**
     * <a href="https://github.com/opencv/opencv_contrib/blob/4.x/modules/img_hash/src/phash.cpp">opencv_contrib下使用phash算法 phash.cpp</a>
     * <pre>
     *
     * virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr) CV_OVERRIDE
     *     {
     *         cv::Mat const input = inputArr.getMat();
     *         CV_Assert(input.type() == CV_8UC4 ||
     *                   input.type() == CV_8UC3 ||
     *                   input.type() == CV_8U);
     *
     *         cv::resize(input, resizeImg, cv::Size(32,32), 0, 0, INTER_LINEAR_EXACT);
     *         if(input.channels() > 1)
     *             cv::cvtColor(resizeImg, grayImg, COLOR_BGR2GRAY);
     *         else
     *             grayImg = resizeImg;
     *
     *         grayImg.convertTo(grayFImg, CV_32F);
     *         cv::dct(grayFImg, dctImg);
     *         dctImg(cv::Rect(0, 0, 8, 8)).copyTo(topLeftDCT);
     *         topLeftDCT.at<float>(0, 0) = 0;
     *         float const imgMean = static_cast<float>(cv::mean(topLeftDCT)[0]);
     *
     *         cv::compare(topLeftDCT, imgMean, bitsImg, CMP_GT);
     *         bitsImg /= 255;
     *         outputArr.create(1, 8, CV_8U);
     *         cv::Mat hash = outputArr.getMat();
     *         uchar *hash_ptr = hash.ptr<uchar>(0);
     *         uchar const *bits_ptr = bitsImg.ptr<uchar>(0);
     *         std::bitset<8> bits;
     *         for(size_t i = 0, j = 0; i != bitsImg.total(); ++j)
     *         {
     *             for(size_t k = 0; k != 8; ++k)
     *             {
     *                 //avoid warning C4800, casting do not work
     *                 bits[k] = bits_ptr[i++] != 0;
     *             }
     *             hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
     *         }
     *     }
     *
     * </pre>
     *
     * @param img_path
     * @return
     */
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
    
    public static double pHash_compare(String img_path1, String img_path2) {
        Mat imread1 = imread(img_path1);
        Mat imread2 = imread(img_path2);
        // 计算 pHash
        Mat pHash1 = new Mat();
        opencv_img_hash.pHash(imread1, pHash1);
        Mat pHash2 = new Mat();
        opencv_img_hash.pHash(imread2, pHash2);
        
        return PHash.create().compare(pHash1, pHash2);
    }
    
    public static double average_compare(String img_path1, String img_path2) {
        Mat imread1 = imread(img_path1);
        Mat imread2 = imread(img_path2);
        
        // 计算 pHash
        Mat averageHash1 = new Mat();
        opencv_img_hash.averageHash(imread1, averageHash1);
        Mat averageHash2 = new Mat();
        opencv_img_hash.averageHash(imread2, averageHash2);
        // 计算 pHash
        return AverageHash.create().compare(averageHash1, averageHash2);
    }
    
    public static byte[] avgHash(String img_path) {
        Mat imread = imread(img_path);
        // 计算 pHash
        Mat pHash = new Mat();
        opencv_img_hash.averageHash(imread, pHash);
        // 将 pHash 转换为 Base64 编码的字符串
        byte[] hashBytes = new byte[(int) pHash.total() * pHash.channels()];
        pHash.data().get(hashBytes);
        return hashBytes;
    }
    
    public static void main(String[] args) throws IOException, NoSuchFieldException, IllegalAccessException {
        // "/Users/jonah/Documents/output2/22.png" 原图
        // "/Users/jonah/Documents/WechatIMG2021.jpg" 当前图片 通过微信传播1次图
        // "/Users/jonah/Documents/WechatIMG2022.jpg" 当前图片 针对原图选择性裁剪
        
        ImageSimilarity openCVImageSimilarity = new ImageSimilarity();
        String img_1 = openCVImageSimilarity.getClass().getResource("images/origin/22.png").getPath();
        String img_2 = openCVImageSimilarity.getClass().getResource("images/other/WechatIMG2021.jpg").getPath();
        String img_3 = openCVImageSimilarity.getClass().getResource("images/other/WechatIMG2022.jpg").getPath();
        
        // 打印输出 phash 和 age_hash结果
        printDetail(img_1);
        printDetail(img_2);
        printDetail(img_3);
        
        
        // phash对比
        System.out.println(pHash_compare(img_1, img_2));
        System.out.println(pHash_compare(img_1, img_3));
        System.out.println(pHash_compare(img_2, img_3));
        
        // phash对比
        System.out.println(pHash_compare2(img_1, img_2));
        System.out.println(pHash_compare2(img_1, img_3));
        System.out.println(pHash_compare2(img_2, img_3));
        
        // age_hash对比
        System.out.println(average_compare(img_1, img_2));
        System.out.println(average_compare(img_1, img_3));
        System.out.println(average_compare(img_2, img_3));
        // age_hash对比
        System.out.println(average_compare2(img_1, img_2));
        System.out.println(average_compare2(img_1, img_3));
        System.out.println(average_compare2(img_2, img_3));


//        phash byte:  [124, 124, -125, -125, -20, 124, 30, 30]
//        phash base64:  fHyDg+x8Hh4=
//                               averageHash byte:  [-1, -5, -31, -31, -21, -1, -1, -1]
//        averageHash base64:  //vh4ev///8=
//        phash byte:  [124, 124, -125, -125, -20, 124, 30, 30]
//        phash base64:  fHyDg+x8Hh4=
//                               averageHash byte:  [-1, -5, -31, -31, -21, -1, -1, -1]
//        averageHash base64:  //vh4ev///8=
//        phash byte:  [-4, 35, 3, -36, 19, -98, -29, -68]
//        phash base64:  /CMD3BOe47w=
//                                averageHash byte:  [-1, -2, -1, -11, -15, -7, -15, -11]
//        averageHash base64:  //7/9fH58fU=
        
        byte[] decode1 = Base64.getDecoder().decode("fHyDg+x8Hh4=");
        byte[] decode2 = Base64.getDecoder().decode("fHyDg+x8Hh4=");
        byte[] decode3 = Base64.getDecoder().decode("/CMD3BOe47w=");
        
        
        // 汉明距离
        HammingDistanceFunction distanceFunction = new HammingDistanceFunction();
//        System.out.println(distanceFunction.distance(decode1, decode2));
//        System.out.println(distanceFunction.distance(decode1, decode3));
//        System.out.println(distanceFunction.distance(decode2, decode3));
        
        
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
        Image image1 = new Image(img_1, decode1);
        Image image2 = new Image(img_2, decode2);
        Image image3 = new Image(img_3, decode3);

//
        index.add(image1);
        index.add(image1);
        index.add(image1);
        
        
        System.out.println("===========>进行搜索");
        List<SearchResult<Image, Integer>> nearest = index.findNearest(decode1, 3);
        System.out.println("===========>进行搜索完成");
        
        for (SearchResult<Image, Integer> result : nearest) {
            // 距离越近则表示 越相似
            Integer distance = result.distance();
            Image item = result.item();
            System.out.println(item.id() + ":" + distance);
        }
        index.save(new File("hnsw.dat"));
        
    }
    
    
    private static double average_compare2(String img1, String img2) {
        byte[] averageHash1 = averageHash(img1);
        byte[] averageHash2 = averageHash(img2);
        return calculateHammingDistance(averageHash1, averageHash2);
    }
    
    private static double pHash_compare2(String img1, String img2) {
        byte[] pHash1 = pHash(img1);
        byte[] pHash2 = pHash(img2);
        return calculateHammingDistance(pHash1, pHash2);
    }
    
    // 计算两个 pHash 的汉明距离
    public static double calculateHammingDistance(byte[] hash1, byte[] hash2) {
        if (hash1.length != hash2.length) {
            throw new IllegalArgumentException("Hashes must be of the same length");
        }
        
        double distance = 0;
        for (int i = 0; i < hash1.length; i++) {
            byte xor = (byte) (hash1[i] ^ hash2[i]);
            distance += Integer.bitCount(xor & 0xFF); // 计算字节中1的数量
        }
        return distance;
    }
}
