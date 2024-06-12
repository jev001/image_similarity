import com.github.jelmerk.knn.DistanceFunction;

public class HammingDistanceFunction implements DistanceFunction<byte[], Double> {
    @Override
    public Double distance(byte[] u, byte[] v) {
        double distance = 0;
        for (int i = 0; i < u.length; i++) {
            byte xor = (byte) (u[i] ^ v[i]);
            distance += Integer.bitCount(xor & 0xFF); // 计算字节中1的数量
        }
        return distance;
    }
    
}