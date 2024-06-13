package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.*;
import com.github.jelmerk.knn.util.*;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.map.primitive.MutableObjectIntMap;
import org.eclipse.collections.api.map.primitive.MutableObjectLongMap;
import org.eclipse.collections.api.tuple.primitive.ObjectIntPair;
import org.eclipse.collections.api.tuple.primitive.ObjectLongPair;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectLongHashMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.*;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm.
 *
 * @param <TId>       Type of the external identifier of an item
 * @param <TVector>   Type of the vector to perform distance calculation on
 * @param <TItem>     Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 * @see <a href="https://arxiv.org/abs/1603.09320">
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance> {
    
    private static final byte VERSION_1 = 0x01;
    
    private static final long serialVersionUID = 1L;
    
    private static final int NO_NODE_ID = -1;
    
    // 距离公式, 使用哪一个公式的问题
    private DistanceFunction<TVector, TDistance> distanceFunction;
    // 距离对比器
    private Comparator<TDistance> distanceComparator;
    private MaxValueComparator<TDistance> maxValueDistanceComparator;
    // 多少维度的数据 重要,多维数据是固定的, 是一个数组, 数组大小是不变的
    private int dimensions;
    // 当前集合中有多少总量处理
    private int maxItemCount;
    // 当前节点最多有多少个邻居---->多少个双向节点
    private int m;
    private int maxM;
    private int maxM0;
    private double levelLambda;
    private int ef;
    private int efConstruction;
    // 是否可以删除
    private boolean removeEnabled;
    
    private int nodeCount;
    
    private volatile Node<TItem> entryPoint;
    
    private AtomicReferenceArray<Node<TItem>> nodes;
    private MutableObjectIntMap<TId> lookup;
    private MutableObjectLongMap<TId> deletedItemVersions;
    // 对于不同id操作需要进行锁定
    private Map<TId, Object> locks;
    // 和保存与存储有关系 和实际操作无关
    private ObjectSerializer<TId> itemIdSerializer;
    private ObjectSerializer<TItem> itemSerializer;
    
    // 全局锁,暂时没发现问题
    private ReentrantLock globalLock;
    
    // 位连接池
    private GenericObjectPool<ArrayBitSet> visitedBitSetPool;
    
    private ArrayBitSet excludedCandidates;
    
    // hnsw是模糊模式, exactView是精确模式
    private ExactView exactView;
    
    private HnswIndex(RefinedBuilder<TId, TVector, TItem, TDistance> builder) {
        
        this.dimensions = builder.dimensions;
        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.maxValueDistanceComparator = new MaxValueComparator<>(this.distanceComparator);
        
        this.m = builder.m;
        this.maxM = builder.m;
        this.maxM0 = builder.m * 2;
        // 计算不同层级之间的连接率, 那么 m就不是邻居的个数 而是层级个数？
        this.levelLambda = 1 / Math.log(this.m);
        this.efConstruction = Math.max(builder.efConstruction, m);
        this.ef = builder.ef;
        this.removeEnabled = builder.removeEnabled;
        
        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);
        
        this.lookup = new ObjectIntHashMap<>();
        this.deletedItemVersions = new ObjectLongHashMap<>();
        this.locks = new HashMap<>();
        
        this.itemIdSerializer = builder.itemIdSerializer;
        this.itemSerializer = builder.itemSerializer;
        
        this.globalLock = new ReentrantLock();
        
        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());
        
        this.excludedCandidates = new ArrayBitSet(this.maxItemCount);
        
        this.exactView = new ExactView();
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        globalLock.lock();
        try {
            return lookup.size();
        } finally {
            globalLock.unlock();
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        globalLock.lock();
        try {
            int nodeId = lookup.getIfAbsent(id, NO_NODE_ID);
            
            if (nodeId == NO_NODE_ID) {
                return Optional.empty();
            } else {
                return Optional.of(nodes.get(nodeId).item);
            }
        } finally {
            globalLock.unlock();
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<TItem> items() {
        globalLock.lock();
        try {
            List<TItem> results = new ArrayList<>(size());
            
            Iterator<TItem> iter = new ItemIterator();
            
            while(iter.hasNext()) {
                results.add(iter.next());
            }
            
            return results;
        } finally {
            globalLock.unlock();
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {
        
        if (!removeEnabled) {
            return false;
        }
        
        globalLock.lock();
        
        try {
            int internalNodeId = lookup.getIfAbsent(id, NO_NODE_ID);
            
            if (internalNodeId == NO_NODE_ID) {
                return false;
            }
            
            Node<TItem> node = nodes.get(internalNodeId);
            
            if (version < node.item.version()) {
                return false;
            }
            
            node.deleted = true;
            
            lookup.remove(id);
            
            deletedItemVersions.put(id, version);
            
            return true;
        } finally {
            globalLock.unlock();
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public boolean add(TItem item) {
        // 当前数据维度和配置的维度不符合
        if (item.dimensions() != dimensions) {
            throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
        }
        // 分配层级 获取不同层级连通率
        int randomLevel = assignLevel(item.id(), this.levelLambda);
        // 每一个层级的个数为什么需要层级+1?
        IntArrayList[] connections = new IntArrayList[randomLevel + 1];
        
        for (int level = 0; level <= randomLevel; level++) {
            int levelM = randomLevel == 0 ? maxM0 : maxM;
            connections[level] = new IntArrayList(levelM);
        }
        
        globalLock.lock();
        
        try {
            // 查询当前是否已经存在节点?
            int existingNodeId = lookup.getIfAbsent(item.id(), NO_NODE_ID);
            
            
            // 如果索引数据中已经存在 节点, 则更新时间
            if (existingNodeId != NO_NODE_ID) {
                // 非删除模式下 不会更新历史信息
                if (!removeEnabled) {
                    return false;
                }
                // 查看已有的节点id
                Node<TItem> node = nodes.get(existingNodeId);
                
                if (item.version() < node.item.version()) {
                    return false;
                }
                
                if (Objects.deepEquals(node.item.vector(), item.vector())) {
                    node.item = item;
                    return true;
                } else {
                    remove(item.id(), item.version());
                }
                
            } else if (item.version() < deletedItemVersions.getIfAbsent(item.id(), -1)) {
                return false;
            }
            // 当前节点总数超过配置数据大小, 所以需要谨慎添加节点个数
            
            if (nodeCount >= this.maxItemCount) {
                throw new SizeLimitExceededException("The number of elements exceeds the specified limit.");
            }
            
            // 节点数据+1
            int newNodeId = nodeCount++;
            
            synchronized (excludedCandidates) {
                excludedCandidates.add(newNodeId);
            }
            
            // 创建一个新的节点信息, 塞入不是很懂
            Node<TItem> newNode = new Node<>(newNodeId, connections, item, false);
            
            // 节点库中增加一个, 节点库是内部维护的, 维护是按照自增id进行的.理论上10W个数据有10W的id
            nodes.set(newNodeId, newNode);
            // 这个是按照数据的id 查找内部节点的id
            lookup.put(item.id(), newNodeId);
            // 删除节点中去除该节点, 因为删除节点表示需要被删除掉
            deletedItemVersions.remove(item.id());
            
            // 增加当前数据节点的 锁
            Object lock = locks.computeIfAbsent(item.id(), k -> new Object());
            
            // 进入节点 第一次进入这个值会是空值====>该数据是 高纬度下的第一个节点.
            // PS 在HSNW 算法中, 数据是从高纬度往低纬度增加的, 也就是第一个数据一定是在高纬度中的。
            Node<TItem> entryPointCopy = entryPoint;
            
            try {
                synchronized (lock) {
                    synchronized (newNode) {
                        
                        if (entryPoint != null && randomLevel <= entryPoint.maxLevel()) {
                            globalLock.unlock();
                        }
                        
                        Node<TItem> currObj = entryPointCopy;
                        if (currObj != null) {
                            //如果当前节点的维度 初始节点的维度小  则需要查找到当前节点的层级维度
                            if (newNode.maxLevel() < entryPointCopy.maxLevel()) {
                                
                                // 计算当前节点和该入口节点之间的距离
                                TDistance curDist = distanceFunction.distance(item.vector(), currObj.item.vector());
                                
                                // 从高纬度向低纬度遍历 寻找和当前节点最近节点的 然后再往下一层就遍历, 这样做这可找到 所有维度中 和需要添加的节点最近的节点
                                for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > newNode.maxLevel(); activeLevel--) {
                                    
                                    boolean changed = true;
                                    
                                    while (changed) {
                                        changed = false;
                                        
                                        synchronized (currObj) {
                                            MutableIntList candidateConnections = currObj.connections[activeLevel];
                                            
                                            for (int i = 0; i < candidateConnections.size(); i++) {
                                                
                                                int candidateId = candidateConnections.get(i);
                                                
                                                Node<TItem> candidateNode = nodes.get(candidateId);
                                                
                                                TDistance candidateDistance = distanceFunction.distance(
                                                        item.vector(),
                                                        candidateNode.item.vector()
                                                );
                                                
                                                // 该节点距离如果比已知的小则选择该节点作为 锚点
                                                if (lt(candidateDistance, curDist)) {
                                                    curDist = candidateDistance;
                                                    currObj = candidateNode;
                                                    changed = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // 从当前节点可供选择的数据中搜索是否有节点可以插入
                            for (int level = Math.min(randomLevel, entryPointCopy.maxLevel()); level >= 0; level--) {
                                // 从高纬度 向下查找最近的节点
                                PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                                        searchBaseLayer(currObj, item.vector(), efConstruction, level);
                                
                                if (entryPointCopy.deleted) {
                                    TDistance distance = distanceFunction.distance(item.vector(), entryPointCopy.item.vector());
                                    topCandidates.add(new NodeIdAndDistance<>(entryPointCopy.id, distance, maxValueDistanceComparator));
                                    
                                    if (topCandidates.size() > efConstruction) {
                                        topCandidates.poll();
                                    }
                                }
                                
                                
                                mutuallyConnectNewElement(newNode, topCandidates, level);
                                
                            }
                        }
                        
                        // zoom out to the highest level
                        if (entryPoint == null || newNode.maxLevel() > entryPointCopy.maxLevel()) {
                            // this is thread safe because we get the global lock when we add a level
                            this.entryPoint = newNode;
                        }
                        
                        return true;
                    }
                }
            } finally {
                synchronized (excludedCandidates) {
                    excludedCandidates.remove(newNodeId);
                }
            }
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }
    
    private void mutuallyConnectNewElement(Node<TItem> newNode,
                                           PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates,
                                           int level) {
        
        int bestN = level == 0 ? this.maxM0 : this.maxM;
        
        int newNodeId = newNode.id;
        TVector newItemVector = newNode.item.vector();
        MutableIntList newItemConnections = newNode.connections[level];
        
        getNeighborsByHeuristic2(topCandidates, m);
        
        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId;
            
            synchronized (excludedCandidates) {
                if (excludedCandidates.contains(selectedNeighbourId)) {
                    continue;
                }
            }
            
            newItemConnections.add(selectedNeighbourId);
            
            Node<TItem> neighbourNode = nodes.get(selectedNeighbourId);
            
            synchronized (neighbourNode) {
                
                TVector neighbourVector = neighbourNode.item.vector();
                
                MutableIntList neighbourConnectionsAtLevel = neighbourNode.connections[level];
                
                if (neighbourConnectionsAtLevel.size() < bestN) {
                    neighbourConnectionsAtLevel.add(newNodeId);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    
                    TDistance dMax = distanceFunction.distance(
                            newItemVector,
                            neighbourNode.item.vector()
                    );
                    
                    Comparator<NodeIdAndDistance<TDistance>> comparator = Comparator
                                                                                  .<NodeIdAndDistance<TDistance>>naturalOrder().reversed();
                    
                    PriorityQueue<NodeIdAndDistance<TDistance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new NodeIdAndDistance<>(newNodeId, dMax, maxValueDistanceComparator));
                    
                    neighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(
                                neighbourVector,
                                nodes.get(id).item.vector()
                        );
                        
                        candidates.add(new NodeIdAndDistance<>(id, dist, maxValueDistanceComparator));
                    });
                    
                    getNeighborsByHeuristic2(candidates, bestN);
                    
                    neighbourConnectionsAtLevel.clear();
                    
                    while (!candidates.isEmpty()) {
                        neighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }
        }
    }
    
    private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates, int m) {
        
        if (topCandidates.size() < m) {
            return;
        }
        
        PriorityQueue<NodeIdAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
        List<NodeIdAndDistance<TDistance>> returnList = new ArrayList<>();
        
        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }
        
        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }
            
            NodeIdAndDistance<TDistance> currentPair = queueClosest.poll();
            
            TDistance distToQuery = currentPair.distance;
            
            boolean good = true;
            for (NodeIdAndDistance<TDistance> secondPair : returnList) {
                
                TDistance curdist = distanceFunction.distance(
                        nodes.get(secondPair.nodeId).item.vector(),
                        nodes.get(currentPair.nodeId).item.vector()
                );
                
                if (lt(curdist, distToQuery)) {
                    good = false;
                    break;
                }
                
            }
            if (good) {
                returnList.add(currentPair);
            }
        }
        
        topCandidates.addAll(returnList);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {
        
        if (entryPoint == null) {
            return Collections.emptyList();
        }
        
        Node<TItem> entryPointCopy = entryPoint;
        
        Node<TItem> currObj = entryPointCopy;
        
        TDistance curDist = distanceFunction.distance(destination, currObj.item.vector());
        
        for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > 0; activeLevel--) {
            
            boolean changed = true;
            
            while (changed) {
                changed = false;
                
                synchronized (currObj) {
                    MutableIntList candidateConnections = currObj.connections[activeLevel];
                    
                    for (int i = 0; i < candidateConnections.size(); i++) {
                        
                        int candidateId = candidateConnections.get(i);
                        
                        TDistance candidateDistance = distanceFunction.distance(
                                destination,
                                nodes.get(candidateId).item.vector()
                        );
                        if (lt(candidateDistance, curDist)) {
                            curDist = candidateDistance;
                            currObj = nodes.get(candidateId);
                            changed = true;
                        }
                    }
                }
                
            }
        }
        
        PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = searchBaseLayer(
                currObj, destination, Math.max(ef, k), 0);
        
        while (topCandidates.size() > k) {
            topCandidates.poll();
        }
        
        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            NodeIdAndDistance<TDistance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(nodes.get(pair.nodeId).item, pair.distance, maxValueDistanceComparator));
        }
        
        return results;
    }
    
    /**
     * Changes the maximum capacity of the index.
     * @param newSize new size of the index
     */
    public void resize(int newSize) {
        globalLock.lock();
        try {
            this.maxItemCount = newSize;
            
            this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                    Runtime.getRuntime().availableProcessors());
            
            AtomicReferenceArray<Node<TItem>> newNodes = new AtomicReferenceArray<>(newSize);
            for(int i = 0; i < this.nodes.length(); i++) {
                newNodes.set(i, this.nodes.get(i));
            }
            this.nodes = newNodes;
            
            this.excludedCandidates = new ArrayBitSet(this.excludedCandidates, newSize);
        } finally {
            globalLock.unlock();
        }
    }
    
    private PriorityQueue<NodeIdAndDistance<TDistance>> searchBaseLayer(
            Node<TItem> entryPointNode, TVector destination, int k, int layer) {
        
        ArrayBitSet visitedBitSet = visitedBitSetPool.borrowObject();
        
        try {
            PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                    new PriorityQueue<>(Comparator.<NodeIdAndDistance<TDistance>>naturalOrder().reversed());
            PriorityQueue<NodeIdAndDistance<TDistance>> candidateSet = new PriorityQueue<>();
            
            TDistance lowerBound;
            
            if (!entryPointNode.deleted) {
                TDistance distance = distanceFunction.distance(destination, entryPointNode.item.vector());
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance, maxValueDistanceComparator);
                
                topCandidates.add(pair);
                lowerBound = distance;
                candidateSet.add(pair);
                
            } else {
                lowerBound = MaxValueComparator.maxValue();
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, lowerBound, maxValueDistanceComparator);
                candidateSet.add(pair);
            }
            
            visitedBitSet.add(entryPointNode.id);
            
            while (!candidateSet.isEmpty()) {
                
                NodeIdAndDistance<TDistance> currentPair = candidateSet.poll();
                
                if (gt(currentPair.distance, lowerBound)) {
                    break;
                }
                
                Node<TItem> node = nodes.get(currentPair.nodeId);
                
                synchronized (node) {
                    
                    MutableIntList candidates = node.connections[layer];
                    
                    for (int i = 0; i < candidates.size(); i++) {
                        
                        int candidateId = candidates.get(i);
                        
                        if (!visitedBitSet.contains(candidateId)) {
                            
                            visitedBitSet.add(candidateId);
                            
                            Node<TItem> candidateNode = nodes.get(candidateId);
                            
                            TDistance candidateDistance = distanceFunction.distance(destination,
                                    candidateNode.item.vector());
                            
                            if (topCandidates.size() < k || gt(lowerBound, candidateDistance)) {
                                
                                NodeIdAndDistance<TDistance> candidatePair =
                                        new NodeIdAndDistance<>(candidateId, candidateDistance, maxValueDistanceComparator);
                                
                                candidateSet.add(candidatePair);
                                
                                if (!candidateNode.deleted) {
                                    topCandidates.add(candidatePair);
                                }
                                
                                if (topCandidates.size() > k) {
                                    topCandidates.poll();
                                }
                                
                                if (!topCandidates.isEmpty()) {
                                    lowerBound = topCandidates.peek().distance;
                                }
                            }
                        }
                    }
                    
                }
            }
            
            return topCandidates;
        } finally {
            visitedBitSet.clear();
            visitedBitSetPool.returnObject(visitedBitSet);
        }
    }
    
    /**
     * Creates a read only view on top of this index that uses pairwise comparision when doing distance search. And as
     * such can be used as a baseline for assessing the precision of the index.
     * Searches will be really slow but give the correct result every time.
     *
     * @return read only view on top of this index that uses pairwise comparision when doing distance search
     */
    public Index<TId, TVector, TItem, TDistance> asExactIndex() {
        return exactView;
    }
    
    /**
     * Returns the dimensionality of the items stored in this index.
     *
     * @return the dimensionality of the items stored in this index
     */
    public int getDimensions() {
        return dimensions;
    }
    
    /**
     * Returns the number of bi-directional links created for every new element during construction.
     *
     * @return the number of bi-directional links created for every new element during construction
     */
    public int getM() {
        return m;
    }
    
    /**
     * The size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @return The size of the dynamic list for the nearest neighbors
     */
    public int getEf() {
        return ef;
    }
    
    /**
     * Set the size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @param ef The size of the dynamic list for the nearest neighbors
     */
    public void setEf(int ef) {
        this.ef = ef;
    }
    
    /**
     * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
     *
     * @return the parameter has the same meaning as ef, but controls the index time / index precision
     */
    public int getEfConstruction() {
        return efConstruction;
    }
    
    /**
     * Returns the distance function.
     *
     * @return the distance function
     */
    public DistanceFunction<TVector, TDistance> getDistanceFunction() {
        return distanceFunction;
    }
    
    
    /**
     * Returns the comparator used to compare distances.
     *
     * @return the comparator used to compare distance
     */
    public Comparator<TDistance> getDistanceComparator() {
        return distanceComparator;
    }
    
    /**
     * Returns if removes are enabled.
     *
     * @return true if removes are enabled for this index.
     */
    public boolean isRemoveEnabled() {
        return removeEnabled;
    }
    
    /**
     * Returns the maximum number of items the index can hold.
     *
     * @return the maximum number of items the index can hold
     */
    public int getMaxItemCount() {
        return maxItemCount;
    }
    
    /**
     * Returns the serializer used to serialize item id's when saving the index.
     *
     * @return the serializer used to serialize item id's when saving the index
     */
    public ObjectSerializer<TId> getItemIdSerializer() {
        return itemIdSerializer;
    }
    
    /**
     * Returns the serializer used to serialize items when saving the index.
     *
     * @return the serializer used to serialize items when saving the index
     */
    public ObjectSerializer<TItem> getItemSerializer() {
        return itemSerializer;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }
    
    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.writeByte(VERSION_1);
        oos.writeInt(dimensions);
        oos.writeObject(distanceFunction);
        oos.writeObject(distanceComparator);
        oos.writeObject(itemIdSerializer);
        oos.writeObject(itemSerializer);
        oos.writeInt(maxItemCount);
        oos.writeInt(m);
        oos.writeInt(maxM);
        oos.writeInt(maxM0);
        oos.writeDouble(levelLambda);
        oos.writeInt(ef);
        oos.writeInt(efConstruction);
        oos.writeBoolean(removeEnabled);
        oos.writeInt(nodeCount);
        writeMutableObjectIntMap(oos, lookup);
        writeMutableObjectLongMap(oos, deletedItemVersions);
        writeNodesArray(oos, nodes);
        oos.writeInt(entryPoint == null ? -1 : entryPoint.id);
    }
    
    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        @SuppressWarnings("unused") byte version = ois.readByte(); // for coping with future incompatible serialization
        this.dimensions = ois.readInt();
        this.distanceFunction = (DistanceFunction<TVector, TDistance>) ois.readObject();
        this.distanceComparator = (Comparator<TDistance>) ois.readObject();
        this.maxValueDistanceComparator = new MaxValueComparator<>(distanceComparator);
        this.itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
        this.itemSerializer = (ObjectSerializer<TItem>) ois.readObject();
        
        this.maxItemCount = ois.readInt();
        this.m = ois.readInt();
        this.maxM = ois.readInt();
        this.maxM0 = ois.readInt();
        this.levelLambda = ois.readDouble();
        this.ef = ois.readInt();
        this.efConstruction = ois.readInt();
        this.removeEnabled = ois.readBoolean();
        this.nodeCount = ois.readInt();
        this.lookup = readMutableObjectIntMap(ois, itemIdSerializer);
        this.deletedItemVersions = readMutableObjectLongMap(ois, itemIdSerializer);
        this.nodes = readNodesArray(ois, itemSerializer, maxM0, maxM);
        
        int entrypointNodeId = ois.readInt();
        this.entryPoint = entrypointNodeId == -1 ? null : nodes.get(entrypointNodeId);
        
        this.globalLock = new ReentrantLock();
        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());
        this.excludedCandidates = new ArrayBitSet(this.maxItemCount);
        this.locks = new HashMap<>();
        this.exactView = new ExactView();
    }
    
    private void writeMutableObjectIntMap(ObjectOutputStream oos, MutableObjectIntMap<TId> map) throws IOException {
        oos.writeInt(map.size());
        
        for (ObjectIntPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeInt(pair.getTwo());
        }
    }
    
    private void writeMutableObjectLongMap(ObjectOutputStream oos, MutableObjectLongMap<TId> map) throws IOException {
        oos.writeInt(map.size());
        
        for (ObjectLongPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeLong(pair.getTwo());
        }
    }
    
    private void writeNodesArray(ObjectOutputStream oos, AtomicReferenceArray<Node<TItem>> nodes) throws IOException {
        oos.writeInt(nodes.length());
        for (int i = 0; i < nodes.length(); i++) {
            writeNode(oos, nodes.get(i));
        }
    }
    
    private void writeNode(ObjectOutputStream oos, Node<TItem> node) throws IOException {
        if (node == null) {
            oos.writeInt(-1);
        } else {
            oos.writeInt(node.id);
            oos.writeInt(node.connections.length);
            
            for (MutableIntList connections : node.connections) {
                oos.writeInt(connections.size());
                for (int j = 0; j < connections.size(); j++) {
                    oos.writeInt(connections.get(j));
                }
            }
            itemSerializer.write(node.item, oos);
            oos.writeBoolean(node.deleted);
        }
    }
    
    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file        File to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(File file)
            throws IOException {
        return load(new FileInputStream(file));
    }
    
    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file        File to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(File file, ClassLoader classLoader)
            throws IOException {
        return load(new FileInputStream(file), classLoader);
    }
    
    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path        Path to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(Path path)
            throws IOException {
        return load(Files.newInputStream(path));
    }
    
    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path        Path to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     *
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(Path path, ClassLoader classLoader)
            throws IOException {
        return load(Files.newInputStream(path), classLoader);
    }
    
    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException              in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream)
            throws IOException {
        return load(inputStream, Thread.currentThread().getContextClassLoader());
    }
    
    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param classLoader the classloader to use
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException              in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream, ClassLoader classLoader)
            throws IOException {
        
        try(ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
            return (HnswIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }
    
    private static IntArrayList readIntArrayList(ObjectInputStream ois, int initialSize) throws IOException {
        int size = ois.readInt();
        
        IntArrayList list = new IntArrayList(initialSize);
        
        for (int j = 0; j < size; j++) {
            list.add(ois.readInt());
        }
        
        return list;
    }
    
    private static <TItem> Node<TItem> readNode(ObjectInputStream ois,
                                                ObjectSerializer<TItem> itemSerializer,
                                                int maxM0,
                                                int maxM) throws IOException, ClassNotFoundException {
        
        int id = ois.readInt();
        
        if (id == -1) {
            return null;
        } else {
            int connectionsSize = ois.readInt();
            
            MutableIntList[] connections = new MutableIntList[connectionsSize];
            
            for (int i = 0; i < connectionsSize; i++) {
                int levelM = i == 0 ? maxM0 : maxM;
                connections[i] = readIntArrayList(ois, levelM);
            }
            
            TItem item = itemSerializer.read(ois);
            
            boolean deleted = ois.readBoolean();
            
            return new Node<>(id, connections, item, deleted);
        }
    }
    
    private static <TItem> AtomicReferenceArray<Node<TItem>> readNodesArray(ObjectInputStream ois,
                                                                            ObjectSerializer<TItem> itemSerializer,
                                                                            int maxM0,
                                                                            int maxM)
            throws IOException, ClassNotFoundException {
        
        int size = ois.readInt();
        AtomicReferenceArray<Node<TItem>> nodes = new AtomicReferenceArray<>(size);
        
        for (int i = 0; i < nodes.length(); i++) {
            nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM));
        }
        
        return nodes;
    }
    
    private static <TId> MutableObjectIntMap<TId> readMutableObjectIntMap(ObjectInputStream ois,
                                                                          ObjectSerializer<TId> itemIdSerializer)
            throws IOException, ClassNotFoundException {
        
        int size = ois.readInt();
        
        MutableObjectIntMap<TId> map = new ObjectIntHashMap<>(size);
        
        for (int i = 0; i < size; i++) {
            TId key = itemIdSerializer.read(ois);
            int value = ois.readInt();
            
            map.put(key, value);
        }
        return map;
    }
    
    private static <TId> MutableObjectLongMap<TId> readMutableObjectLongMap(ObjectInputStream ois,
                                                                            ObjectSerializer<TId> itemIdSerializer)
            throws IOException, ClassNotFoundException {
        
        int size = ois.readInt();
        
        MutableObjectLongMap<TId> map = new ObjectLongHashMap<>(size);
        
        for (int i = 0; i < size; i++) {
            TId key = itemIdSerializer.read(ois);
            long value = ois.readLong();
            
            map.put(key, value);
        }
        return map;
    }
    
    /**
     * Start the process of building a new HNSW index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance extends Comparable<TDistance>> Builder<TVector, TDistance> newBuilder(
            int dimensions,
            DistanceFunction<TVector, TDistance> distanceFunction,
            int maxItemCount) {
        
        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();
        
        return new Builder<>(dimensions, distanceFunction, distanceComparator, maxItemCount);
    }
    
    /**
     * Start the process of building a new HNSW index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param distanceComparator used to compare distances
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance> Builder<TVector, TDistance> newBuilder(
            int dimensions,
            DistanceFunction<TVector, TDistance> distanceFunction,
            Comparator<TDistance> distanceComparator,
            int maxItemCount) {
        
        return new Builder<>(dimensions, distanceFunction, distanceComparator, maxItemCount);
    }
    
    /**
     * 这个Java方法assignLevel的主要目的是为图中的节点分配一个层级（level），这是在构建诸如Hierarchical Navigable Small World (HNSW)等高效近似最近邻搜索数据结构时的一个关键步骤。具体来说，它的功能和逻辑可以这样理解：
     * 利用外部ID稳定性: 方法基于传入的TId value（一个外部ID）来决定节点的层级，这样的设计有助于使得图的构建过程更加稳定。外部ID到层级的映射通过哈希运算实现，减少了相似ID节点在层级分配上的随机性，参考了HNSW库的一个讨论(#28)。
     * 哈希值处理: 首先，计算value的哈希码，并将其分割成四个字节，这有助于均匀分布哈希值并准备进行下一步的随机化处理。
     * Murmur3哈希: 使用Murmur3哈希算法进一步处理这四个字节，得到一个新的哈希值。Murmur3是一种非加密、高性能的哈希函数，常用于各种哈希相关应用中，因为它能提供良好的分布特性。
     * 归一化与计算随机数: 将Murmur3得到的哈希值除以Integer.MAX_VALUE并取绝对值得到一个在[0, 1)范围内的随机数。这样做是为了将大的整数映射到一个易于处理的概率区间。
     * 计算层级: 利用计算出的随机数和输入参数lambda，通过公式-Math.log(random) * lambda计算出一个值。这个公式基于泊松分布或指数分布的思想，-Math.log(random)相当于求得一个服从指数分布的随机数，乘以lambda则是调整这个分布的尺度参数，影响层级的分布宽度。最终结果转换为整数作为节点的层级。
     * 综上所述，assignLevel方法通过一系列精心设计的步骤，将每个节点的ID映射到一个层级上，这一过程旨在优化图结构，促进高效的近似最近邻搜索，特别是在高维空间数据的索引构建场景中。
     * @param value
     * @param lambda
     * @return
     */
    private int assignLevel(TId value, double lambda) {
        
        // by relying on the external id to come up with the level, the graph construction should be a lot mor stable
        // see : https://github.com/nmslib/hnswlib/issues/28
        
        int hashCode = value.hashCode();
        
        byte[] bytes = new byte[]{
                (byte) (hashCode >> 24),
                (byte) (hashCode >> 16),
                (byte) (hashCode >> 8),
                (byte) hashCode
        };
        
        double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE);
        
        double r = -Math.log(random) * lambda;
        return (int) r;
    }
    
    private boolean lt(TDistance x, TDistance y) {
        return maxValueDistanceComparator.compare(x, y) < 0;
    }
    
    private boolean gt(TDistance x, TDistance y) {
        return maxValueDistanceComparator.compare(x, y) > 0;
    }
    
    class ExactView implements Index<TId, TVector, TItem, TDistance> {
        
        private static final long serialVersionUID = 1L;
        
        @Override
        public int size() {
            return HnswIndex.this.size();
        }
        
        @Override
        public Optional<TItem> get(TId tId) {
            return HnswIndex.this.get(tId);
        }
        
        
        @Override
        public Collection<TItem> items() {
            return HnswIndex.this.items();
        }
        
        @Override
        public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {
            
            Comparator<SearchResult<TItem, TDistance>> comparator = Comparator
                                                                            .<SearchResult<TItem, TDistance>>naturalOrder()
                                                                            .reversed();
            
            PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);
            
            for (int i = 0; i < nodeCount; i++) {
                Node<TItem> node = nodes.get(i);
                if (node == null || node.deleted) {
                    continue;
                }
                
                TDistance distance = distanceFunction.distance(node.item.vector(), vector);
                
                SearchResult<TItem, TDistance> searchResult = new SearchResult<>(node.item, distance, maxValueDistanceComparator);
                queue.add(searchResult);
                
                if (queue.size() > k) {
                    queue.poll();
                }
            }
            
            List<SearchResult<TItem, TDistance>> results = new ArrayList<>(queue.size());
            
            SearchResult<TItem, TDistance> result;
            while ((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
                results.add(0, result);
            }
            
            return results;
        }
        
        @Override
        public boolean add(TItem item) {
            return HnswIndex.this.add(item);
        }
        
        @Override
        public boolean remove(TId id, long version) {
            return HnswIndex.this.remove(id, version);
        }
        
        @Override
        public void save(OutputStream out) throws IOException {
            HnswIndex.this.save(out);
        }
        
        @Override
        public void save(File file) throws IOException {
            HnswIndex.this.save(file);
        }
        
        @Override
        public void save(Path path) throws IOException {
            HnswIndex.this.save(path);
        }
        
        @Override
        public void addAll(Collection<TItem> items) throws InterruptedException {
            HnswIndex.this.addAll(items);
        }
        
        @Override
        public void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
            HnswIndex.this.addAll(items, listener);
        }
        
        @Override
        public void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval) throws InterruptedException {
            HnswIndex.this.addAll(items, numThreads, listener, progressUpdateInterval);
        }
    }
    
    class ItemIterator implements Iterator<TItem> {
        
        private int done = 0;
        private int index = 0;
        
        @Override
        public boolean hasNext() {
            return done < HnswIndex.this.size();
        }
        
        @Override
        public TItem next() {
            Node<TItem> node;
            
            do {
                node = HnswIndex.this.nodes.get(index++);
            } while(node == null || node.deleted);
            
            done++;
            
            return node.item;
        }
    }
    
    static class Node<TItem> implements Serializable {
        
        private static final long serialVersionUID = 1L;
        
        final int id;
        
        final MutableIntList[] connections;
        
        volatile TItem item;
        
        volatile boolean deleted;
        
        Node(int id, MutableIntList[] connections, TItem item, boolean deleted) {
            this.id = id;
            this.connections = connections;
            this.item = item;
            this.deleted = deleted;
        }
        
        int maxLevel() {
            return this.connections.length - 1;
        }
    }
    
    static class NodeIdAndDistance<TDistance> implements Comparable<NodeIdAndDistance<TDistance>> {
        
        final int nodeId;
        final TDistance distance;
        final Comparator<TDistance> distanceComparator;
        
        NodeIdAndDistance(int nodeId, TDistance distance, Comparator<TDistance> distanceComparator) {
            this.nodeId = nodeId;
            this.distance = distance;
            this.distanceComparator = distanceComparator;
        }
        
        @Override
        public int compareTo(NodeIdAndDistance<TDistance> o) {
            return distanceComparator.compare(distance, o.distance);
        }
    }
    
    
    static class MaxValueComparator<TDistance> implements Comparator<TDistance>, Serializable  {
        
        private static final long serialVersionUID = 1L;
        
        private final Comparator<TDistance> delegate;
        
        MaxValueComparator(Comparator<TDistance> delegate) {
            this.delegate = delegate;
        }
        
        @Override
        public int compare(TDistance o1, TDistance o2) {
            return o1 == null ? o2 == null ? 0 : 1
                           : o2 == null ? -1 : delegate.compare(o1, o2);
        }
        
        static <TDistance> TDistance maxValue() {
            return null;
        }
    }
    
    /**
     * Base class for HNSW index builders.
     *
     * @param <TBuilder> Concrete class that extends from this builder
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of items stored in the index
     */
    public static abstract class BuilderBase<TBuilder extends BuilderBase<TBuilder, TVector, TDistance>, TVector, TDistance> {
        
        public static final int DEFAULT_M = 10;
        public static final int DEFAULT_EF = 10;
        public static final int DEFAULT_EF_CONSTRUCTION = 200;
        public static final boolean DEFAULT_REMOVE_ENABLED = false;
        
        int dimensions;
        DistanceFunction<TVector, TDistance> distanceFunction;
        Comparator<TDistance> distanceComparator;
        
        int maxItemCount;
        
        int m = DEFAULT_M;
        int ef = DEFAULT_EF;
        int efConstruction = DEFAULT_EF_CONSTRUCTION;
        boolean removeEnabled = DEFAULT_REMOVE_ENABLED;
        
        BuilderBase(int dimensions,
                    DistanceFunction<TVector, TDistance> distanceFunction,
                    Comparator<TDistance> distanceComparator,
                    int maxItemCount) {
            
            this.dimensions = dimensions;
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
            this.maxItemCount = maxItemCount;
        }
        
        abstract TBuilder self();
        
        /**
         * Sets the number of bi-directional links created for every new element during construction. Reasonable range
         * for m is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall,
         * while low m work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter
         * also determines the algorithm's memory consumption.
         * As an example for d = 4 random vectors optimal m for search is somewhere around 6, while for high dimensional
         * datasets (word embeddings, good face descriptors), higher M are required (e.g. m = 48, 64) for optimal
         * performance at high recall. The range mM = 12-48 is ok for the most of the use cases. When m is changed one
         * has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly estimated by
         * assuming that m  efConstruction is a constant.
         *
         * @param m the number of bi-directional links created for every new element during construction
         * @return the builder.
         */
        public TBuilder withM(int m) {
            this.m = m;
            return self();
        }
        
        /**
         * `
         * The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
         * leads to longer construction, but better index quality. At some point, increasing efConstruction does not
         * improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
         * a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
         * there is room for improvement.
         *
         * @param efConstruction controls the index time / index precision
         * @return the builder
         */
        public TBuilder withEfConstruction(int efConstruction) {
            this.efConstruction = efConstruction;
            return self();
        }
        
        /**
         * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
         * accurate but slower search. The value ef of can be anything between k and the size of the dataset.
         *
         * @param ef size of the dynamic list for the nearest neighbors
         * @return the builder
         */
        public TBuilder withEf(int ef) {
            this.ef = ef;
            return self();
        }
        
        /**
         * Call to enable support for the experimental remove operation. Indices that support removes will consume more
         * memory.
         *
         * @return the builder
         */
        public TBuilder withRemoveEnabled() {
            this.removeEnabled = true;
            return self();
        }
    }
    
    
    /**
     * Builder for initializing an {@link HnswIndex} instance.
     *
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder<TVector, TDistance> extends BuilderBase<Builder<TVector, TDistance>, TVector, TDistance> {
        
        /**
         * Constructs a new {@link Builder} instance.
         *
         * @param dimensions the dimensionality of the vectors stored in the index
         * @param distanceFunction the distance function
         * @param maxItemCount     the maximum number of elements in the index
         */
        Builder(int dimensions,
                DistanceFunction<TVector, TDistance> distanceFunction,
                Comparator<TDistance> distanceComparator,
                int maxItemCount) {
            
            super(dimensions, distanceFunction, distanceComparator, maxItemCount);
        }
        
        @Override
        Builder<TVector, TDistance> self() {
            return this;
        }
        
        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer   serializes the
         * @param <TId>            Type of the external identifier of an item
         * @param <TItem>          implementation of the Item interface
         * @return the builder
         */
        public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
            return new RefinedBuilder<>(dimensions, distanceFunction, distanceComparator, maxItemCount, m, ef, efConstruction,
                    removeEnabled, itemIdSerializer, itemSerializer);
        }
        
        /**
         * Build the index that uses java object serializers to store the items when reading and writing the index.
         *
         * @param <TId>   Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the hnsw index instance
         */
        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            ObjectSerializer<TId> itemIdSerializer = new JavaObjectSerializer<>();
            ObjectSerializer<TItem> itemSerializer = new JavaObjectSerializer<>();
            
            return withCustomSerializers(itemIdSerializer, itemSerializer)
                           .build();
        }
        
    }
    
    /**
     * Extension of {@link Builder} that has knows what type of item is going to be stored in the index.
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class RefinedBuilder<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            extends BuilderBase<RefinedBuilder<TId, TVector, TItem, TDistance>, TVector, TDistance> {
        
        private ObjectSerializer<TId> itemIdSerializer;
        private ObjectSerializer<TItem> itemSerializer;
        
        RefinedBuilder(int dimensions,
                       DistanceFunction<TVector, TDistance> distanceFunction,
                       Comparator<TDistance> distanceComparator,
                       int maxItemCount,
                       int m,
                       int ef,
                       int efConstruction,
                       boolean removeEnabled,
                       ObjectSerializer<TId> itemIdSerializer,
                       ObjectSerializer<TItem> itemSerializer) {
            
            super(dimensions, distanceFunction, distanceComparator, maxItemCount);
            
            this.m = m;
            this.ef = ef;
            this.efConstruction = efConstruction;
            this.removeEnabled = removeEnabled;
            
            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;
        }
        
        @Override
        RefinedBuilder<TId, TVector, TItem, TDistance> self() {
            return this;
        }
        
        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer   serializes the
         * @return the builder
         */
        public RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(
                ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
            
            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;
            
            return this;
        }
        
        /**
         * Build the index.
         *
         * @return the hnsw index instance
         */
        public HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<>(this);
        }
        
    }
    
    public void setDistanceFunction(DistanceFunction<TVector, TDistance> distanceFunction) {
        this.distanceFunction = distanceFunction;
    }
    
    public void setDistanceComparator(Comparator<TDistance> distanceComparator) {
        this.distanceComparator = distanceComparator;
    }
    
    public MaxValueComparator<TDistance> getMaxValueDistanceComparator() {
        return maxValueDistanceComparator;
    }
    
    public void setMaxValueDistanceComparator(MaxValueComparator<TDistance> maxValueDistanceComparator) {
        this.maxValueDistanceComparator = maxValueDistanceComparator;
    }
    
    public void setDimensions(int dimensions) {
        this.dimensions = dimensions;
    }
    
    public void setMaxItemCount(int maxItemCount) {
        this.maxItemCount = maxItemCount;
    }
    
    public void setM(int m) {
        this.m = m;
    }
    
    public int getMaxM() {
        return maxM;
    }
    
    public void setMaxM(int maxM) {
        this.maxM = maxM;
    }
    
    public int getMaxM0() {
        return maxM0;
    }
    
    public void setMaxM0(int maxM0) {
        this.maxM0 = maxM0;
    }
    
    public double getLevelLambda() {
        return levelLambda;
    }
    
    public void setLevelLambda(double levelLambda) {
        this.levelLambda = levelLambda;
    }
    
    public void setEfConstruction(int efConstruction) {
        this.efConstruction = efConstruction;
    }
    
    public void setRemoveEnabled(boolean removeEnabled) {
        this.removeEnabled = removeEnabled;
    }
    
    public int getNodeCount() {
        return nodeCount;
    }
    
    public void setNodeCount(int nodeCount) {
        this.nodeCount = nodeCount;
    }
    
    public Node<TItem> getEntryPoint() {
        return entryPoint;
    }
    
    public void setEntryPoint(Node<TItem> entryPoint) {
        this.entryPoint = entryPoint;
    }
    
    public AtomicReferenceArray<Node<TItem>> getNodes() {
        return nodes;
    }
    
    public void setNodes(AtomicReferenceArray<Node<TItem>> nodes) {
        this.nodes = nodes;
    }
    
    public MutableObjectIntMap<TId> getLookup() {
        return lookup;
    }
    
    public void setLookup(MutableObjectIntMap<TId> lookup) {
        this.lookup = lookup;
    }
    
    public MutableObjectLongMap<TId> getDeletedItemVersions() {
        return deletedItemVersions;
    }
    
    public void setDeletedItemVersions(MutableObjectLongMap<TId> deletedItemVersions) {
        this.deletedItemVersions = deletedItemVersions;
    }
    
    public Map<TId, Object> getLocks() {
        return locks;
    }
    
    public void setLocks(Map<TId, Object> locks) {
        this.locks = locks;
    }
    
    public void setItemIdSerializer(ObjectSerializer<TId> itemIdSerializer) {
        this.itemIdSerializer = itemIdSerializer;
    }
    
    public void setItemSerializer(ObjectSerializer<TItem> itemSerializer) {
        this.itemSerializer = itemSerializer;
    }
    
    public ReentrantLock getGlobalLock() {
        return globalLock;
    }
    
    public void setGlobalLock(ReentrantLock globalLock) {
        this.globalLock = globalLock;
    }
    
    public GenericObjectPool<ArrayBitSet> getVisitedBitSetPool() {
        return visitedBitSetPool;
    }
    
    public void setVisitedBitSetPool(GenericObjectPool<ArrayBitSet> visitedBitSetPool) {
        this.visitedBitSetPool = visitedBitSetPool;
    }
    
    public ArrayBitSet getExcludedCandidates() {
        return excludedCandidates;
    }
    
    public void setExcludedCandidates(ArrayBitSet excludedCandidates) {
        this.excludedCandidates = excludedCandidates;
    }
    
    public ExactView getExactView() {
        return exactView;
    }
    
    public void setExactView(ExactView exactView) {
        this.exactView = exactView;
    }
}
