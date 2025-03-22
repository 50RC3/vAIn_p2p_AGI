const mongoose = require('mongoose');

const nodeSchema = new mongoose.Schema({
    nodeId: {
        type: String,
        required: [true, 'NodeId is required'],
        unique: true,
        trim: true,
        minlength: [8, 'NodeId must be at least 8 characters'],
        maxlength: [64, 'NodeId cannot exceed 64 characters']
    },
    status: {
        type: String,
        enum: {
            values: ['active', 'inactive', 'penalized'],
            message: '{VALUE} is not a valid status'
        },
        default: 'inactive',
        index: true
    },
    stake: {
        type: String,
        default: '0',
        validate: {
            validator: function(v) {
                return /^\d+$/.test(v);
            },
            message: 'Stake must be a valid number string'
        }
    },
    performance: {
        accuracy: {
            type: Number,
            min: [0, 'Accuracy cannot be negative'],
            max: [1, 'Accuracy cannot exceed 1']
        },
        latency: {
            type: Number,
            min: [0, 'Latency cannot be negative']
        },
        lastUpdate: {
            type: Date,
            default: Date.now
        }
    },
    metrics: {
        cpu: {
            type: Number,
            min: [0, 'CPU usage cannot be negative'],
            max: [100, 'CPU usage cannot exceed 100%']
        },
        memory: {
            type: Number,
            min: [0, 'Memory usage cannot be negative'],
            max: [100, 'Memory usage cannot exceed 100%']
        },
        bandwidth: {
            type: Number,
            min: [0, 'Bandwidth cannot be negative']
        },
        lastReported: {
            type: Date,
            default: Date.now
        }
    },
    lastHeartbeat: {
        type: Date,
        default: Date.now,
        index: true
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Indexes
nodeSchema.index({ 'performance.accuracy': -1 });
nodeSchema.index({ 'performance.latency': 1 });
nodeSchema.index({ createdAt: 1 });

// Instance methods
nodeSchema.methods.isActive = function() {
    const HEARTBEAT_TIMEOUT = 5 * 60 * 1000; // 5 minutes
    return this.status === 'active' && 
           (Date.now() - this.lastHeartbeat) < HEARTBEAT_TIMEOUT;
};

nodeSchema.methods.updateMetrics = async function(metrics) {
    // Validate metrics
    if (!metrics || typeof metrics !== 'object') {
        throw new Error('Invalid metrics object');
    }

    // Ensure values are within valid ranges
    if (metrics.cpu && (metrics.cpu < 0 || metrics.cpu > 100)) {
        throw new Error('CPU usage must be between 0-100%');
    }
    if (metrics.memory && (metrics.memory < 0 || metrics.memory > 100)) {
        throw new Error('Memory usage must be between 0-100%');
    }
    if (metrics.bandwidth && metrics.bandwidth < 0) {
        throw new Error('Bandwidth cannot be negative');
    }

    this.metrics = {
        ...metrics,
        lastReported: Date.now()
    };
    this.lastHeartbeat = Date.now();
    
    // Emit metrics update event
    this.emit('metricsUpdated', this.metrics);
    
    return this.save();
};

// Add health check method
nodeSchema.methods.checkHealth = function() {
    const health = {
        status: this.isActive() ? 'healthy' : 'unhealthy',
        lastHeartbeat: this.lastHeartbeat,
        metrics: this.metrics,
        stake: this.stake,
        warnings: []
    };

    // Check for warning conditions
    if (this.metrics.cpu > 90) {
        health.warnings.push('High CPU usage');
    }
    if (this.metrics.memory > 90) {
        health.warnings.push('High memory usage');
    }
    if (Date.now() - this.metrics.lastReported > 5 * 60 * 1000) {
        health.warnings.push('Stale metrics');
    }

    return health;
};

// Virtuals
nodeSchema.virtual('isStaked').get(function() {
    return BigInt(this.stake) > BigInt(0);
});

// Pre-save middleware
nodeSchema.pre('save', function(next) {
    if (this.isModified('metrics')) {
        this.lastHeartbeat = Date.now();
    }
    next();
});

// Static methods
nodeSchema.statics.findActive = function() {
    const HEARTBEAT_TIMEOUT = 5 * 60 * 1000; // 5 minutes
    return this.find({
        status: 'active',
        lastHeartbeat: { $gt: new Date(Date.now() - HEARTBEAT_TIMEOUT) }
    });
};

const Node = mongoose.model('Node', nodeSchema);
module.exports = Node;
