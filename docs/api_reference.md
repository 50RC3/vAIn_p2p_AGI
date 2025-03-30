# vAIn API Reference - Development Build

> Note: This API is under active development by me (Vincent). Breaking changes should be expected.
> Current Version: 0.2.1
> Last Updated: 2024-01-25

## REST API Endpoints

### Currently Implemented
- Basic authentication with ECDSA ✓
- Simple staking operations ✓
- Basic training status queries ✓
- Node health monitoring ✓
- Resource metrics tracking ✓
- Network health monitoring ✓

### Rate Limiting
Current implementation:
- Auth endpoints: 30 requests/minute
- General endpoints: 10 requests/minute
- Websocket connections: 1 per node
- Metrics endpoints: 60 requests/minute
- TODO: Dynamic rate limiting
- TODO: Reputation-based limits

### Authentication
POST `/api/auth/login`
```json
{
  "address": string,  // Ethereum address
  "signature": string // ECDSA signature
}
```
Note: Currently using basic ECDSA verification. Will be enhanced with hardware attestation in future versions.

### Core Endpoints
- POST `/api/staking/stake` ✓
- GET `/api/training/status` ✓
- POST `/api/training/control` (60% implemented)
- GET `/api/node/health` ✓
- POST `/api/node/resources` ✓
- POST `/api/model/update` (Basic implementation)
- GET `/api/metrics/cluster` (Basic implementation)

### Additional Core Endpoints
- POST `/api/node/authorize` ✓
- GET `/api/node/network-stats` ✓
- POST `/api/memory/share` ✓
- GET `/api/memory/usage` ✓
- POST `/api/cognitive/state` ✓
- GET `/api/cognitive/metrics` ✓
- POST `/api/model/register` ✓
- GET `/api/model/state` ✓

### Node Management
- GET `/api/node/list` ✓
- GET `/api/node/stats` ✓
- POST `/api/node/register` ✓
- PUT `/api/node/update` ✓

### Training Control
- POST `/api/training/start` ✓
- POST `/api/training/stop` ✓
- GET `/api/training/metrics` ✓
- GET `/api/training/participants` ✓

### Memory Management 
- GET `/api/memory/status` ✓
- POST `/api/memory/optimize` ✓
- GET `/api/memory/metrics` ✓

### Cache Management
- POST `/api/cache/invalidate` ✓
- GET `/api/cache/stats` ✓
- POST `/api/cache/optimize` (70% complete)
- GET `/api/cache/metrics` ✓

### Metrics & Monitoring
- GET `/api/metrics/system` ✓
  - Returns current system resource metrics (CPU, memory, disk)
- GET `/api/metrics/network` ✓
  - Returns network health metrics (peers, latency, bandwidth)
- GET `/api/metrics/training` (70% complete)
  - Returns model training performance metrics
- GET `/api/metrics/history?type=system&duration=24h` ✓
  - Returns historical metrics for specified type and duration
- POST `/api/metrics/alerts/configure` ✓
  - Configure alert thresholds and notification settings
- GET `/api/metrics/alerts/status` ✓
  - Get current alert status and recent notifications

## WebSocket Events

Currently working:
- `node:discovery` ✓
- `node:metrics` ✓
- `model:update` (Basic implementation)
- `health:status` ✓

### Additional WebSocket Events
- `training:progress` ✓
- `memory:status` ✓
- `node:validation` ✓
- `metrics:update` ✓
- `cache:invalidate` ✓
- `memory:shared` ✓
- `cognitive:update` ✓
- `model:registered` ✓
- `validation:failed` ✓

### Metrics WebSocket Events
- `metrics:system:update` ✓
  - Real-time system metrics updates
- `metrics:network:update` ✓
  - Real-time network metrics updates
- `metrics:alert` ✓
  - Real-time alert notifications
- `metrics:threshold:exceeded` ✓
  - Notification when metrics exceed configured thresholds

TODO:
- Robust error handling
- Advanced monitoring
- Full metrics system
- Training coordination
- P2P message routing

## Smart Contract Methods

Basic implementation is working but needs expansion. See `contracts/` directory for current state.
Full documentation coming in v0.3.0.
