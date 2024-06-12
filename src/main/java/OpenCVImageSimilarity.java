import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;
import org.bytedeco.opencv.global.opencv_img_hash;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_img_hash.AverageHash;
import org.bytedeco.opencv.opencv_img_hash.PHash;

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

public class OpenCVImageSimilarity {
    
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
    
    public static byte[] pHash(String img_path) {
        Mat imread = imread(img_path);
        // 计算 pHash
        Mat pHash = new Mat();
        opencv_img_hash.pHash(imread, pHash);
        // 将 pHash 转换为 Base64 编码的字符串
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
    
    public static void main(String[] args) throws IOException {
        // "/Users/jonah/Documents/output2/22.png" 原图
        // "/Users/jonah/Documents/WechatIMG2021.jpg" 当前图片 通过微信传播1次图
        // "/Users/jonah/Documents/WechatIMG2022.jpg" 当前图片 针对原图选择性裁剪
        
        OpenCVImageSimilarity openCVImageSimilarity = new OpenCVImageSimilarity();
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
        System.out.println(distanceFunction.distance(decode1,decode2));;
        System.out.println(distanceFunction.distance(decode1,decode3));;
        System.out.println(distanceFunction.distance(decode2,decode3));;
        
        
        // 构建索引
        System.out.println("=====>构建索引====>");
        HnswIndex<String, byte[], Image, Double> index = HnswIndex.newBuilder(8, new HammingDistanceFunction(), 1000).withM(10).build();
        
        URL origin = openCVImageSimilarity.getClass().getResource("images/origin/");
        URL other = openCVImageSimilarity.getClass().getResource("images/other/");
        List<Path> origin_path_files = Files.list(Paths.get(origin.getPath())).collect(Collectors.toList());
        
        List<Path> other_path_files = Files.list(Paths.get(other.getPath())).collect(Collectors.toList());
        
        List<Path> allPath  =new ArrayList<>();
        allPath.addAll(origin_path_files);
        allPath.addAll(other_path_files);
        
        
        for (Path path : allPath) {
            byte[] bytes = pHash(path.toString());
            index.add(new Image(path.getFileName().toString(),bytes));
        }
        
//
//        index.add(word1);
//        index.add(word2);
//        index.add(word3);
        
        
        List<SearchResult<Image, Double>> nearest = index.findNeighbors("20.png", 3);
        
        for (SearchResult<Image, Double> wordDoubleSearchResult : nearest) {
            Double distance = wordDoubleSearchResult.distance();
            Image item = wordDoubleSearchResult.item();
            System.out.println(item.id()+":"+distance);
        }
        

////
////
//        HnswIndex<String, float[], Word, Float> index = HnswIndex
//                                                                .newBuilder(10,DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE, 10)
//                                                                .withM(10)
//                                                                .build();
//
//        index.addAll(words);
//
//        List<SearchResult<Word, Float>> nearest = index.findNeighbors("king", 10);
//
//        for (SearchResult<Word, Float> result : nearest) {
//            System.out.println(result.item().id() + " " + result.getDistance());
//        }
        
//        Object dimensions
//                ;
//        HnswIndex<String, float[], Word, Float> index = HnswIndex
//                                                                .newBuilder(DistanceFunctions.FLOAT_COSINE_DISTANCE, dimensions, words.size())
//                                                                .withM(10)
//                                                                .build();
//
//        // 创建 HNSW 索引
//        HnswIndex.newBuilder()
//        HnswIndex<float[], Integer> options = HnswIndex.newBuilder(DistanceFunctions.DOUBLE_COSINE_DISTANCE,(a, b) -> euclideanDistance(a, b))
//                                                      .withDistanceFunction()
//                                                      .build();
//
//        HnswIndexer<float[], Integer> indexer = new HnswIndexer<>(options);
//
//        // 图片 ID 和 phash 值的双向映射
//        BiMap<Integer, float[]> idToPhashMap = HashBiMap.create();
//
//        // 添加数据到 HNSW 索引
//        addDataToIndex(indexer, idToPhashMap);
//
//        // 构建索引
//        HnswIndex<float[], Integer> index = indexer.buildIndex();
//
//        // 进行最近邻搜索
//        float[] queryPhash = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
//        List<HnswIndex.PayloadAndDistance<Integer>> results = index.searchKnn(queryPhash, 2);
//
//        // 输出结果
//        for (HnswIndex.PayloadAndDistance<Integer> result : results) {
//            int imageId = result.getPayload();
//            float[] phash = idToPhashMap.inverse().get(imageId);
//            System.out.println("Nearest neighbor image ID: " + imageId + ", phash: " + phash);
//        }
//
    
    
        
    }
    
    private static double average_compare2(String img1, String img2) {
        byte[] averageHash1 = averageHash(img1);
        byte[] averageHash2 = averageHash(img2);
        return calculateHammingDistance(averageHash1,averageHash2);
    }
    
    private static double pHash_compare2(String img1, String img2) {
        byte[] pHash1 = pHash(img1);
        byte[] pHash2 = pHash(img2);
        return calculateHammingDistance(pHash1,pHash2);
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
