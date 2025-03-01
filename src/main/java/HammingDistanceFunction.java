import com.github.jelmerk.knn.DistanceFunction;

public class HammingDistanceFunction implements DistanceFunction<byte[], Integer> {
    @Override
    public Integer distance(byte[] u, byte[] v) {
        int distance = 0;
        for (int i = 0; i < u.length; i++) {
            byte xor = (byte) (u[i] ^ v[i]);
            distance += Integer.bitCount(xor & 0xFF); // 计算字节中1的数量
        }
        // 输出汉明距离
        return distance;
    }
    
}