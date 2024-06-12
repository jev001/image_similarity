//import org.bytedeco.javacpp.FloatPointer;
//import org.bytedeco.javacpp.IntPointer;
//import org.bytedeco.javacpp.PointerPointer;
//import org.bytedeco.opencv.opencv_core.Mat;
//import org.bytedeco.opencv.opencv_core.Scalar;
//import org.bytedeco.opencv.opencv_core.Size;
//import org.bytedeco.opencv.opencv_img_hash.PHash;
//
//import static org.bytedeco.opencv.global.opencv_core.*;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//
//public class OpenCVImageSimilarity2 {
//
//    static {
////        System.load("/Users/jonah/Sources/yunji/marketing/untitled/src/main/resources/libopencv_java490.dylib");
////        URL url = ClassLoader.getSystemResource("lib/opencv/opencv_java455.dll");
////        URL url = ClassLoader.getSystemResource("src/main/resources/libopencv_java490.dylib");
////        System.load("/Users/jonah/Sources/yunji/marketing/untitled/target/classes/libopencv_java470.dylib");
//    }
//
//
//    public static void main(String[] args) {
//        System.out.printf(System.getProperty("java.library.path") + "\n");
//        // 加载OpenCV库
////        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
////        System.loadLibrary("opencv_java470");
//        // 读取两张图像。准备比对的图片
////        Mat image1 = imread("/Users/jonah/Documents/output2/22.png");
////        Mat image2 = imread("/Users/jonah/Documents/output2/22.png");
//        Mat image1 = imread("/Users/jonah/Desktop/2.png");
//        Mat image2 = imread("/Users/jonah/Documents/output2/24.png");
//        Mat image4 = imread("/Users/jonah/Documents/output2/25.png");
//        Mat image5 = imread("/Users/jonah/Documents/output2/26.png");
//
//
//        // 将图片处理成一样大
//        resize(image1, image1, image2.size());
//        resize(image2, image2, image1.size());
//
//        // 计算均方差（MSE）
//        double mse = calculateMSE(image1, image2);
//        System.out.println("均方差（MSE）: " + mse);
//
//        // 计算结构相似性指数（SSIM）
//        double ssim = calculateSSIM(image1, image2);
//        System.out.println("结构相似性指数（SSIM）: " + ssim);
//
//        // 计算峰值信噪比（PSNR）
//        double psnr = calculatePSNR(image1, image2);
//        System.out.println("峰值信噪比（PSNR）: " + psnr);
//
//        // 计算直方图
//        final double similarity = calculateHistogram(image1, image2);
//        System.out.println("图片相似度(直方图): " + similarity);
//
//        // 计算归一化交叉相关（NCC）
//        double ncc = calculateDHashSimilarity(image1, image2);
//        System.out.println("归一化交叉相关（NCC）: " + ncc);
//
//        // 计算图像的 pHash 值
//
////        PHash pHash = new PHash(image1.getPointer());
////        PHash pHash2 = new PHash(image2.getPointer());
//
//        Mat hashMat1 = new Mat(1, 8, CV_64FC1);
//        Mat hashMat2 = new Mat(1, 8, CV_64FC1);
//
//
//
//
//        PHash pHash = PHash.create();
//        pHash.compute(image1, hashMat1);
//        pHash.compute(image2, hashMat2);
//        long aLong1 = hashMat1.data().getDouble();
//        long aLong2 = hashMat2.data().getLong();
//        // 比较两个 PHash 值
//        double compare = pHash.compare(hashMat1, hashMat2);
//        System.out.println("Hamming distance: " + compare);
////
////        // 比较两个图像的相似度
////        double distance = hammingDistance(pHash1, pHash2);
////        System.out.println("PHash（NCC）: " + distance);
//
//
//    }
//
//    // 计算均方差（MSE）
//    private static double calculateHistogram(Mat image1, Mat image2) {
//        // 计算直方图
//        Mat hist1 = calculateHistogram(image1);
//
//        Mat hist2 = calculateHistogram(image2);
//
//        // 计算相似度
//        final double similarity = compareHist(hist1, hist2, 1);
//        return similarity;
//    }
//
//
//    // 计算均方差（MSE）
//    private static double calculateMSE(Mat image1, Mat image2) {
//        Mat diff = new Mat();
//        absdiff(image1, image2, diff);
//        Mat squaredDiff = new Mat();
//        multiply(diff, diff, squaredDiff);
//        Scalar mseScalar = mean(squaredDiff);
//        return mseScalar.get();
////        return mseScalar.val[0];
//    }
//
//    // 计算结构相似性指数（SSIM）
//    private static double calculateSSIM(Mat image1, Mat image2) {
//        Mat image1Gray = new Mat();
//        Mat image2Gray = new Mat();
//        cvtColor(image1, image1Gray, COLOR_BGR2GRAY);
//        cvtColor(image2, image2Gray, COLOR_BGR2GRAY);
//        Mat ssimMat = new Mat();
//        matchTemplate(image1Gray, image2Gray, ssimMat, CV_COMP_CORREL);
//        Scalar ssimScalar = mean(ssimMat);
//        return ssimScalar.get();
////        return ssimScalar.val[0];
//    }
//
//    // 计算峰值信噪比（PSNR）
//    private static double calculatePSNR(Mat image1, Mat image2) {
//        Mat diff = new Mat();
//        absdiff(image1, image2, diff);
//        Mat squaredDiff = new Mat();
//        multiply(diff, diff, squaredDiff);
//        Scalar mseScalar = mean(squaredDiff);
//        double mse = mseScalar.get();
////        double mse = mseScalar.val[0];
//        double psnr = 10.0 * Math.log10(255.0 * 255.0 / mse);
//        return psnr;
//    }
//
//    // 计算归一化交叉相关（NCC）
////    private static double calculateNCC(Mat image1, Mat image2) {
////        Mat image1Gray = new Mat();
////        Mat image2Gray = new Mat();
////        cvtColor(image1, image1Gray, Imgproc.COLOR_BGR2GRAY);
////        cvtColor(image2, image2Gray, Imgproc.COLOR_BGR2GRAY);
////        MatOfInt histSize = new MatOfInt(256);
////        MatOfFloat ranges = new MatOfFloat(0, 256);
////        Mat hist1 = new Mat();
////        Mat hist2 = new Mat();
////
////        normalize(hist1, hist1, 0, 1, NORM_MINMAX);
////        normalize(hist2, hist2, 0, 1, NORM_MINMAX);
////        double ncc = compareHist(hist1, hist2, CV_COMP_CORREL);
////        return ncc;
////    }
//
//    private static Mat calculateHistogram(Mat image) {
//        final int[] channels = new int[]{0,1,2};
//        final Mat mask = new Mat();
//        final Mat hist = new Mat();
//        final int[] histSize = new int[]{16,16,16};
//        final float[] histRange = new float[]{0f, 255f};
//        IntPointer intPtrChannels = new IntPointer(channels);
//        IntPointer intPtrHistSize = new IntPointer(histSize);
//        final PointerPointer<FloatPointer> ptrPtrHistRange = new PointerPointer<>(histRange, histRange, histRange);
//        calcHist(image, 1, intPtrChannels, mask, hist, 3, intPtrHistSize, ptrPtrHistRange, true, false);
//        return hist;
//    }
//
//    public static long calculateDHash(Mat image) {
//        Mat resizedImage = new Mat();
//        int width = image.cols();
//        int height = image.rows();
//        double aspectRatio = (double)width / height;
////        Size newSize = null;
////        if (aspectRatio > 1) {
////            newSize = new Size(8, (int)(8 / aspectRatio));
////        } else {
////            newSize = new Size((int)(8 * aspectRatio), 8);
////        }
//        Size newSize = new Size(8, 8);
//        resize(image, resizedImage, newSize, 0, 0, INTER_AREA);
//        long hash = 0;
//        for (int row = 0; row < 8; row++) {
//            for (int col = 0; col < 8; col++) {
//                double leftPixel = resizedImage.ptr(row, col).getDouble();
//                double rightPixel = resizedImage.ptr(row, col + 1).getDouble();
//                hash = (hash << 1) | (leftPixel < rightPixel ? 1 : 0);
//            }
//        }
//        return hash;
//    }
//
//    public static double calculateDHashSimilarity(Mat img1, Mat img2) {
//        long hash1 = calculateDHash(img1);
//        long hash2 = calculateDHash(img2);
//        int hammingDistance = Long.bitCount(hash1 ^ hash2);
//        double similarity = hammingDistance / 64.0;
//        return similarity;
//    }
//
//    private static PHash calculatePHash(Mat image) {
//        // 图像预处理
//        // 图像预处理
//        return new PHash(image.getPointer());
//    }
//
//    private static double hammingDistance(long h1, long h2) {
//        long x = h1 ^ h2;
//        int setBits = 0;
//        while (x != 0) {
//            setBits += (int) (x & 1);
//            x >>= 1;
//        }
//        return setBits;
//    }
//}
