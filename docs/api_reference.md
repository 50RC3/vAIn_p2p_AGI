# vAIn API Reference - Development Build

> Note: This API is under active development by me (Vincent). Breaking changes should be expected.
> Current Version: 0.2.1
> Last Updated: [current date]

## REST API Endpoints

### Currently Implemented
- Basic authentication with ECDSA ✓
- Simple staking operations ✓
- Basic training status queries ✓
- Node health monitoring ✓

### Rate Limiting
Current implementation:
- Auth endpoints: 30 requests/minute
- General endpoints: 10 requests/minute
- Websocket connections: 1 per node
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
- POST `/api/training/control` (Partially implemented)
- GET `/api/node/health` ✓
- POST `/api/node/resources` ✓

## WebSocket Events

Currently working:
- `node:discovery` ✓
- `node:metrics` ✓
- `model:update` (Basic implementation)
- `health:status` ✓

TODO:
- Robust error handling
- Advanced monitoring
- Full metrics system
- Training coordination
- P2P message routing

## Smart Contract Methods

Basic implementation is working but needs expansion. See `contracts/` directory for current state.
Full documentation coming in v0.3.0.
