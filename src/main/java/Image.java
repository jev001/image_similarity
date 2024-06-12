import com.github.jelmerk.knn.Item;

import java.util.Arrays;

public class Image implements Item<String, byte[]> {
    
    private static final long serialVersionUID = 1L;
    
    private final String id;
    private final byte[] vector;
    
    public Image(String id, byte[] vector) {
        this.id = id;
        this.vector = vector;
    }
    
    @Override
    public String id() {
        return id;
    }
    
    @Override
    public byte[] vector() {
        return vector;
    }
    
    @Override
    public int dimensions() {
        return vector.length;
    }
    
    @Override
    public String toString() {
        return "Word{" +
                       "id='" + id + '\'' +
                       ", vector=" + Arrays.toString(vector) +
                       '}';
    }
}