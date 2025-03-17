const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// API Routes
app.use('/api/auth', require('./src/api/auth'));
app.use('/api/staking', require('./src/api/staking'));

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
