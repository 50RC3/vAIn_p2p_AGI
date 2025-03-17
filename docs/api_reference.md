# vAIn API Reference

## REST API Endpoints

### Authentication
- POST `/api/auth/login`
  - Body: `{ "address": string, "signature": string }`
  - Returns: `{ "token": string }`

### Staking
- POST `/api/staking/stake`
  - Body: `{ "amount": string, "address": string }`
  - Returns: `{ "success": boolean, "txHash": string }`

### Training
- GET `/api/training/status`
  - Returns: `{ "currentRound": number, "accuracy": number, "participation": number }`

## WebSocket Events

### Node Communication
- `node:discovery` - Broadcast node presence
- `node:metrics` - Share node performance metrics
- `model:update` - Share model updates

## Smart Contract Methods

### vAInToken
- `stake(uint256 amount)`
- `withdraw(uint256 amount)`
- `getRewards()`

### Governance
- `propose(bytes[] calldata, string memory description)`
- `castVote(uint256 proposalId, uint8 support)`
