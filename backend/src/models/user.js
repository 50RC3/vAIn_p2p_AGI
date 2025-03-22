const mongoose = require('mongoose');
const { ethers } = require('ethers');

const userSchema = new mongoose.Schema({
    address: {
        type: String,
        required: [true, 'Ethereum address is required'],
        unique: true,
        validate: {
            validator: (v) => ethers.utils.isAddress(v),
            message: 'Invalid Ethereum address'
        }
    },
    reputation: {
        type: Number,
        default: 0
    },
    totalStaked: {
        type: String,
        default: '0'
    },
    lastActive: {
        type: Date,
        default: Date.now
    }
}, {
    timestamps: true
});

// Indexes
userSchema.index({ reputation: -1 });
userSchema.index({ lastActive: -1 });
userSchema.index({ 'totalStaked': -1 });

// Instance methods
userSchema.methods.updateReputation = async function(change) {
    this.reputation = Math.max(0, this.reputation + change);
    this.lastActive = new Date();
    return this.save();
};

userSchema.methods.getTier = function() {
    if (this.reputation >= 1000) return 3;
    if (this.reputation >= 500) return 2;
    return 1;
};

// Virtuals
userSchema.virtual('isActive').get(function() {
    return Date.now() - this.lastActive < 24 * 60 * 60 * 1000; // 24 hours
});

// Static methods
userSchema.statics.findTopStakers = function(limit = 10) {
    return this.find()
        .sort({ totalStaked: -1 })
        .limit(limit);
};

// Middleware
userSchema.pre('save', function(next) {
    if (this.isModified('totalStaked')) {
        // Validate staked amount is numeric string
        if (!/^\d+$/.test(this.totalStaked)) {
            next(new Error('Total staked must be a numeric string'));
            return;
        }
    }
    next();
});

module.exports = mongoose.model('User', userSchema);
