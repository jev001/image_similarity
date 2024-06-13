package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.Item;
import org.eclipse.collections.api.iterator.MutableIntIterator;
import org.eclipse.collections.api.list.primitive.MutableIntList;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class HnswVisual<TId, TVector, TItem extends Item<TId, TVector>, TDistance> {
    
    private final HnswIndex<TId, TVector, TItem, TDistance> index;
    
    public HnswVisual(HnswIndex<TId, TVector, TItem, TDistance> index) {
        this.index = index;
    }
    
    public void show() throws NoSuchFieldException, IllegalAccessException {
        AtomicReferenceArray<HnswIndex.Node<TItem>> nodes = index.getNodes();
        
        
        // 入口节点
        HnswIndex.Node<TItem> entryPoint = index.getEntryPoint();
        Set<Integer> objects = new HashSet<>();
        extracted(entryPoint, nodes,objects);
        
        HnswIndex.Node<TItem> entryPointCopy = entryPoint;
        
        HnswIndex.Node<TItem> currObj = entryPointCopy;
        
        System.out.println("======>输出节点信息");
        
    }
    
    private static <TId, TVector, TItem extends Item<TId, TVector>> void extracted(HnswIndex.Node<TItem> entryPoint, AtomicReferenceArray<HnswIndex.Node<TItem>> nodes, Set<Integer> objects) {
        for(int level = entryPoint.maxLevel(); level>0; level--){
            // 从高纬度遍历
            MutableIntList connection = entryPoint.connections[level];
            // 每个层级下有不同的节点
            MutableIntIterator mutableIntIterator = connection.intIterator();
            while (mutableIntIterator.hasNext()){
                // 获取当前节点的节点信息
                int nodeId = mutableIntIterator.next();
                HnswIndex.Node<TItem> tItemNode = nodes.get(nodeId);
                // 排除当前节点
                if(objects.add(tItemNode.id)){
                    System.out.println("item.id:"+tItemNode.item.id());
                    extracted(tItemNode,nodes, objects);
                }
            }
        }
    }
    
    private boolean lt(TDistance x, TDistance y) {
        return index.getMaxValueDistanceComparator().compare(x, y) < 0;
    }
    
    public void traverse() {
        HnswIndex.Node<TItem> entryPoint = index.getEntryPoint();
        AtomicReferenceArray<HnswIndex.Node<TItem>> nodes = index.getNodes();
        Set<String>  already = new HashSet<>();
        traverseLevel(entryPoint, 0,nodes,already);
    }
    
    
    private void traverseLevel(HnswIndex.Node<TItem> node, int currentLevel, AtomicReferenceArray<HnswIndex.Node<TItem>> nodes, Set<String> already) {
        if(!already.add(""+node.id)){
            return;
        }
        System.out.println("Visiting node: " + node.id +",currentLevel:"+currentLevel+ ", item: " + node.item.id());
        for (int layer = 0; layer < node.connections.length; layer++) {
            // 查找当前层级
            MutableIntList connection = node.connections[layer];
            MutableIntIterator mutableIntIterator = connection.intIterator();
            while (mutableIntIterator.hasNext()){
                // 查找当前层级的邻居
                int next = mutableIntIterator.next();
                HnswIndex.Node<TItem> tItemNode = nodes.get(next);
                traverseLevel(tItemNode,layer,nodes,already);
            }
            
        }
        
    }
    
    
}
