# Copilot Coding Agent Instructions for vAIn P2P AGI

## Big Picture Architecture
- This project is a decentralized, peer-to-peer AGI system with federated learning, blockchain governance, and secure resource sharing.
- Major components:
  - `ai_core/`: Federated learning, model coordination, evaluation, and memory management.
  - `network/`: Peer discovery, DHT, consensus (PBFT), node communication, and monitoring.
  - `security/`: Encryption, authentication, firewall rules.
  - `contracts/`: Smart contracts for governance, staking, reputation, and tokenomics.
  - `backend/` & `frontend/`: API server and React-based UI for user/node interaction.
  - `training/`, `models/`, `memory/`, `data/`: AI model training, storage, and data handling.

## Developer Workflows
- **Python:**
  - Install dependencies: `pip install -r requirements.txt`
  - Start system: `python start.py` (supports flags for minimal, non-interactive, auto-install, etc.)
  - Run tests: `pytest tests/`
  - Validate core modules: `python test_critical_imports.py`
- **Node.js:**
  - Install: `npm install` in `backend/` and `frontend/`
  - Start backend: `node backend/server.js`
  - Start frontend: `npm run start` in `frontend/`
- **Smart Contracts:**
  - Deploy: `npx hardhat run scripts/deploy.js`
  - Test: `npx hardhat test`
- **Docker:**
  - Build: `docker build -t vain .`
  - Compose: `docker-compose up`

## Project-Specific Patterns & Conventions
- **Consensus:** PBFT is implemented in `network/consensus/pbft.py` and integrated via `network/pbft_consensus.py`.
- **DHT:** Node discovery and routing via `network/dht.py` and `network/dht_network.py`.
- **Security:** All node communication uses TLS 1.3+ and Ed25519 keys. Hardware attestation is supported.
- **Reputation:** Node reputation and tiering are managed in `network/reputation.py` and smart contracts.
- **Federated Learning:** Model updates are compressed and privacy-preserving; see `training/federated_training.py` and `ai_core/`.
- **Extensibility:** All coordinator and manager classes (e.g., `SystemCoordinator`, `ModuleRegistry`, `LearningCoordinator`) are designed for easy stubbing and extension.
- **Testing:** Use `test_critical_imports.py` to validate all core imports and integration.

## Integration Points & Data Flows
- **Node Startup:** `start.py` initializes all subsystems, validates dependencies, and launches network/UI.
- **Cross-Component Communication:**
  - Python modules communicate via async interfaces and shared managers.
  - Backend and frontend communicate via REST APIs (`backend/src/api/`).
  - Blockchain contracts interact with backend via web3.js and hardhat scripts.
- **External Dependencies:**
  - Python: numpy, torch, cryptography, websockets, tqdm, web3, etc.
  - Node.js: express, web3, hardhat, react, etc.
  - Docker for containerization.

## Examples & Key Files
- `start.py`: Main entry point, shows how all systems are initialized and integrated.
- `ai_core/system_coordinator.py`: Orchestrates all AI and network modules.
- `network/pbft_consensus.py`, `network/dht_network.py`: Consensus and discovery integration.
- `contracts/`: Smart contract source for governance and reputation.
- `test_critical_imports.py`: Validates that all core modules are importable and functional.

## Conventions
- All new modules should be designed for extension and scalability (see `LearningCoordinator` pattern). Only stub if scaling is not feasible at the moment.
- All modules must be functionally integrated and interconnected; isolated modules are not permitted. Ensure every new module communicates with at least one other system component (e.g., via async managers, shared registries, or network protocols).
- Use async/await for all network and AI coordination logic.
- Configuration is managed via `config/` and `.env` files; override via CLI flags as needed.
- Logs and metrics are stored in `logs/` and `monitoring/`.

---

_If any section is unclear or missing, please provide feedback so the instructions can be improved for future AI agents._
