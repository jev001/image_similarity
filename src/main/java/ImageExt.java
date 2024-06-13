import com.github.jelmerk.knn.Item;

import java.util.Arrays;

public class ImageExt implements Item<String, ImageExt> {
    
    private static final long serialVersionUID = 1L;
    
    private final String id;
    private final byte[] vector;
    
    public ImageExt(String id, byte[] vector) {
        this.id = id;
        this.vector = vector;
    }
    
    public byte[] getVector() {
        return vector;
    }
    
    @Override
    public String id() {
        return id;
    }
    
    @Override
    public ImageExt vector() {
        return this;
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