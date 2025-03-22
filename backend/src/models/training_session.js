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

// Indexes for frequent queries
trainingSessionSchema.index({ roundId: 1, status: 1 });
trainingSessionSchema.index({ startTime: -1 });
trainingSessionSchema.index({ 'participants.nodeId': 1 });
trainingSessionSchema.index({ 'metrics.averageAccuracy': -1 });

// Instance methods
trainingSessionSchema.methods.isActive = function() {
  return this.status === 'active';
};

trainingSessionSchema.methods.addParticipant = async function(nodeData) {
  try {
    if (!nodeData.nodeId) {
      throw new Error('NodeId is required');
    }
    if (this.status !== 'pending' && this.status !== 'active') {
      throw new Error('Cannot add participant to non-active session');
    }
    this.participants.push(nodeData);
    return this.save();
  } catch (err) {
    logger.error(`Failed to add participant: ${err.message}`);
    throw err;
  }
};

trainingSessionSchema.methods.updateMetrics = async function(newMetrics) {
  this.metrics = { ...this.metrics, ...newMetrics };
  return this.save();
};

trainingSessionSchema.methods.verifyHardwareStats = async function(nodeId, verifierNodeId, signature) {
  const participant = this.participants.find(p => p.nodeId === nodeId);
  if (!participant) throw new Error('Participant not found');

  participant.fraudMetrics.hardwareValidations++;
  participant.fraudMetrics.verifiedBy.push({
    nodeId: verifierNodeId,
    timestamp: new Date(),
    signature
  });

  return this.save();
};

// Validation middleware
trainingSessionSchema.pre('save', async function(next) {
  if (this.isModified('status') && this.status === 'completed') {
    this.endTime = new Date();
  }
  
  // Validate hardware multipliers
  this.participants.forEach(p => {
    if (p.rewardMultipliers.hardware > 2.5) p.rewardMultipliers.hardware = 2.5;
    if (p.rewardMultipliers.uptime > 1.5) p.rewardMultipliers.uptime = 1.5;
    if (p.rewardMultipliers.bandwidth > 1.3) p.rewardMultipliers.bandwidth = 1.3;
    if (p.rewardMultipliers.latency > 1.2) p.rewardMultipliers.latency = 1.2;
    if (p.rewardMultipliers.commitment > 1.2) p.rewardMultipliers.commitment = 1.2;
  });

  next();
});

// Add cleanup hook
trainingSessionSchema.post('save', async function(doc) {
  if (doc.status === 'completed') {
    // Cleanup old data
    await doc.model('TrainingSession').deleteMany({
      endTime: { $lt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }, // 30 days
      status: 'completed'
    });
  }
});

// Static methods
trainingSessionSchema.statics.findActive = function() {
  return this.find({ status: 'active' });
};

trainingSessionSchema.statics.findByNode = function(nodeId) {
  return this.find({ 'participants.nodeId': nodeId }).sort({ startTime: -1 });
};

trainingSessionSchema.statics.getParticipantStats = async function(nodeId) {
  return this.aggregate([
    { $match: { 'participants.nodeId': nodeId } },
    { $unwind: '$participants' },
    { $match: { 'participants.nodeId': nodeId } },
    { $group: {
      _id: null,
      totalSessions: { $sum: 1 },
      averageAccuracy: { $avg: '$participants.accuracy' },
      totalRewards: { $sum: { $toDouble: '$participants.rewardAmount' } },
      averageComputeTime: { $avg: '$participants.computationTime' }
    }}
  ]);
};

// Add virtual for total participants
trainingSessionSchema.virtual('participantCount').get(function() {
  return this.participants.length;
});

module.exports = mongoose.model('TrainingSession', trainingSessionSchema);
