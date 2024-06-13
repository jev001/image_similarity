import com.github.jelmerk.knn.DistanceFunction;

public class ImageExtHammingDistanceFunction implements DistanceFunction<ImageExt, Integer> {
    @Override
    public Integer distance(ImageExt x, ImageExt y) {
        int distance = 0;
        byte[] u = x.getVector();
        byte[] v = y.getVector();
        for (int i = 0; i < u.length; i++) {
            byte xor = (byte) (u[i] ^ v[i]);
            distance += Integer.bitCount(xor & 0xFF); // 计算字节中1的数量
        }
        // 输出汉明距离
        System.out.printf("x.itemId=%s,y.itemId=%s, Hamming distance: %s%n", x.id(),y.id(),distance);
        return distance;
    }
    
}