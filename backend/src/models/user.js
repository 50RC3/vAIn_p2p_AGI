const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    address: {
        type: String,
        required: true,
        unique: true
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
});

module.exports = mongoose.model('User', userSchema);
