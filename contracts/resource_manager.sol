// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract ResourceManager is Ownable, ReentrancyGuard {
    struct NodeResources {
        uint256 computePower;    // GFLOPS
        uint256 storage;         // GB
        uint256 bandwidth;       // Mbps
        uint256 lastUpdateTime;
        uint256 taskCount;
        uint256 totalWorkload;
    }

    mapping(address => NodeResources) public nodeResources;
    uint256 public totalComputePower;
    uint256 public totalStorage;
    uint256 public totalBandwidth;
    uint256 public constant REBALANCE_THRESHOLD = 20; // 20% imbalance triggers reallocation

    uint256 public minRebalanceThreshold = 20;
    uint256 public maxRebalanceThreshold = 40;
    mapping(address => uint256) public reputationScores;
    mapping(address => uint256) public lastActivityTime;

    // Network performance metrics
    uint256 public totalWorkload;
    uint256 public totalCapacity;
    uint256 public rebalanceWindow = 1 hours;
    uint256 public lastRebalanceTime;
    uint256 public targetUtilization = 70; // 70% target network utilization
    bool public autoAdjustEnabled = true;

    // Thresholds and limits
    uint256 public constant MIN_COMPUTE = 1000; // Minimum 1000 GFLOPS
    uint256 public constant MIN_STORAGE = 100;  // Minimum 100 GB
    uint256 public constant MIN_BANDWIDTH = 100; // Minimum 100 Mbps
    uint256 public constant MAX_TASKS_PER_NODE = 1000;

    struct Task {
        uint256 id;
        uint256 workload;
        address assignedNode;
        bool completed;
    }

    Task[] public taskQueue;
    mapping(address => uint256[]) public nodeTasks;

    event ResourcesUpdated(address indexed node, uint256 compute, uint256 storage, uint256 bandwidth);
    event WorkloadAssigned(address indexed node, uint256 workload);
    event TaskQueued(uint256 indexed taskId, uint256 workload);
    event TaskAssigned(uint256 indexed taskId, address indexed node);
    event ThresholdsUpdated(uint256 minThreshold, uint256 maxThreshold);
    event AutoAdjustToggled(bool enabled);
    event AnomalyDetected(address indexed node, string reason);
    event NetworkMetricsUpdated(uint256 totalWorkload, uint256 totalCapacity, uint256 utilization);

    modifier validResourceLevels(uint256 compute, uint256 storage, uint256 bandwidth) {
        require(compute >= MIN_COMPUTE, "Compute power too low");
        require(storage >= MIN_STORAGE, "Storage capacity too low");
        require(bandwidth >= MIN_BANDWIDTH, "Bandwidth too low");
        _;
    }

    function updateResources(uint256 compute, uint256 storage, uint256 bandwidth) 
        external 
        nonReentrant 
        validResourceLevels(compute, storage, bandwidth) 
    {
        NodeResources storage resources = nodeResources[msg.sender];

        totalComputePower = totalComputePower + compute - resources.computePower;
        totalStorage = totalStorage + storage - resources.storage;
        totalBandwidth = totalBandwidth + bandwidth - resources.bandwidth;

        resources.computePower = compute;
        resources.storage = storage;
        resources.bandwidth = bandwidth;
        resources.lastUpdateTime = block.timestamp;

        emit ResourcesUpdated(msg.sender, compute, storage, bandwidth);

        updateNetworkMetrics();
        if (autoAdjustEnabled && shouldRebalance()) {
            adjustRebalanceThreshold();
        }
    }

    function calculateFairShare(address node) public view returns (uint256) {
        NodeResources storage resources = nodeResources[node];
        uint256 nodeCapacity = (resources.computePower * 4 + resources.storage + resources.bandwidth) / 6;
        uint256 totalCapacity = (totalComputePower * 4 + totalStorage + totalBandwidth) / 6;
        return (nodeCapacity * 100) / totalCapacity;
    }

    function needsRebalancing(address node) public view returns (bool) {
        uint256 fairShare = calculateFairShare(node);
        uint256 currentShare = (nodeResources[node].totalWorkload * 100) / 
                              (nodeResources[node].taskCount > 0 ? nodeResources[node].taskCount : 1);
        uint256 imbalance = fairShare > currentShare ? 
                           fairShare - currentShare : 
                           currentShare - fairShare;
        return imbalance > REBALANCE_THRESHOLD;
    }

    function adjustRebalanceThreshold() internal {
        uint256 networkLoad = (totalWorkload * 100) / totalCapacity;
        if (networkLoad > 80) {
            minRebalanceThreshold = 30; // More tolerance when busy
        } else if (networkLoad < 30) {
            minRebalanceThreshold = 10; // Stricter when idle
        }
    }

    function shouldRebalance() internal view returns (bool) {
        return block.timestamp >= lastRebalanceTime + rebalanceWindow;
    }

    function updateNetworkMetrics() internal {
        totalWorkload = 0;
        totalCapacity = 0;

        for (uint i = 0; i < taskQueue.length; i++) {
            if (!taskQueue[i].completed) {
                totalWorkload += taskQueue[i].workload;
            }
        }

        uint256 utilization = (totalWorkload * 100) / (totalCapacity > 0 ? totalCapacity : 1);
        emit NetworkMetricsUpdated(totalWorkload, totalCapacity, utilization);
    }

    function setAutoAdjust(bool enabled) external onlyOwner {
        autoAdjustEnabled = enabled;
        emit AutoAdjustToggled(enabled);
    }

    function setRebalanceWindow(uint256 newWindow) external onlyOwner {
        require(newWindow >= 15 minutes && newWindow <= 24 hours, "Invalid window");
        rebalanceWindow = newWindow;
    }

    function setTargetUtilization(uint256 newTarget) external onlyOwner {
        require(newTarget > 0 && newTarget <= 90, "Invalid target");
        targetUtilization = newTarget;
    }

    function queueTask(uint256 workload) external returns (uint256) {
        uint256 taskId = taskQueue.length;
        taskQueue.push(Task({
            id: taskId,
            workload: workload,
            assignedNode: address(0),
            completed: false
        }));

        emit TaskQueued(taskId, workload);
        _assignQueuedTasks();
        return taskId;
    }

    function updateReputation(address node) internal {
        uint256 timeSinceActivity = block.timestamp - lastActivityTime[node];
        // Decay reputation by 1% per day of inactivity
        if (timeSinceActivity > 1 days) {
            uint256 decayFactor = timeSinceActivity / 1 days;
            reputationScores[node] = reputationScores[node] * 
                (100 - decayFactor) / 100;
        }
    }

    function detectAnomaly(address node) internal view returns (bool) {
        NodeResources storage resources = nodeResources[node];

        // Check for sudden spikes in resource claims
        bool resourceSpike = resources.computePower > totalComputePower * 40 / 100 || 
                           resources.storage > totalStorage * 40 / 100 ||
                           resources.bandwidth > totalBandwidth * 40 / 100;

        if (resourceSpike) {
            emit AnomalyDetected(node, "Resource spike detected");
            return true;
        }

        // Check workload distribution
        uint256 avgWorkloadPerTask = resources.totalWorkload / 
            (resources.taskCount > 0 ? resources.taskCount : 1);

        bool workloadAnomaly = avgWorkloadPerTask > getTotalAverageWorkload() * 150 / 100 ||
                              avgWorkloadPerTask < getTotalAverageWorkload() * 50 / 100;

        if (workloadAnomaly) {
            emit AnomalyDetected(node, "Workload distribution anomaly");
            return true;
        }

        return false;
    }
}
