const mongoose = require('mongoose');

const trainingSessionSchema = new mongoose.Schema({
  roundId: {
    type: Number,
    required: true
  },
  participants: [{
    nodeId: String,
    contribution: Number,
    accuracy: Number,
    rewardAmount: String,
    modelVersion: String,
    computationTime: Number,
    resourceUsage: {
      cpu: Number,
      memory: Number,
      bandwidth: Number
    },
    validationMetrics: {
      loss: Number,
      accuracy: Number,
      f1Score: Number
    },
    hardwareStats: {
      gpuCount: Number,
      gpuType: String,
      tpmVersion: String,
      bandwidth: Number,
      uptime: Number,
      regionLatency: Number
    },
    rewardMultipliers: {
      hardware: Number,  // Up to 2.5x for high-end GPUs
      uptime: Number,    // Up to 1.5x for 99.9%+ uptime
      bandwidth: Number, // Up to 1.3x for 1Gbps+
      latency: Number,   // Up to 1.2x for <50ms latency
      commitment: Number // Up to 1.2x for 6month+ stake
    },
    fraudMetrics: {
      updateDeviation: Number,     // How much model update deviates from mean
      consistencyScore: Number,    // Consistency of updates over time
      hardwareValidations: Number, // Number of successful P2P validations
      byzantineScore: Number,      // Probability of being Byzantine
      lastVerifiedHardware: Date,  // Last hardware verification timestamp
      verifiedBy: [{              // List of nodes that verified hardware
        nodeId: String,
        timestamp: Date,
        signature: String
      }]
    }
  }],
  globalModelHash: String,
  startTime: Date,
  endTime: Date,
  status: {
    type: String,
    enum: ['pending', 'active', 'completed', 'failed'],
    default: 'pending'
  },
  metrics: {
    averageAccuracy: Number,
    participationRate: Number,
    consensusAchieved: Boolean,
    convergenceRate: Number,
    modelSize: Number,
    aggregationTime: Number,
    globalLoss: Number,
    byzantineNodesDetected: Number,
    fraudulentUpdatesRejected: Number,
    averageUpdateDeviation: Number,
    hardwareVerificationRate: Number
  },
  hyperparameters: {
    learningRate: Number,
    batchSize: Number,
    epochs: Number,
    optimizerConfig: Object
  },
  aggregationMethod: {
    type: String,
    enum: ['fedAvg', 'fedProx', 'fedMA'],
    default: 'fedAvg'
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('TrainingSession', trainingSessionSchema);
