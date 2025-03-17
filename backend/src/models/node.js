const mongoose = require('mongoose');

const nodeSchema = new mongoose.Schema({
    nodeId: {
        type: String,
        required: true,
        unique: true
    },
    status: {
        type: String,
        enum: ['active', 'inactive', 'penalized'],
        default: 'inactive'
    },
    stake: {
        type: String,
        default: '0'
    },
    performance: {
        accuracy: Number,
        latency: Number,
        lastUpdate: Date
    },
    metrics: {
        cpu: Number,
        memory: Number,
        bandwidth: Number,
        lastReported: Date
    }
});

module.exports = mongoose.model('Node', nodeSchema);
