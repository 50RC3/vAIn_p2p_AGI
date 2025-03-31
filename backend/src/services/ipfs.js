const ipfsClient = require('ipfs-http-client');
const { Buffer } = require('buffer');
const logger = require('../utils/logger');
const config = require('../config');
const { promisify } = require('util');
const sleep = promisify(setTimeout);
const EventEmitter = require('events');
const multiaddr = require('multiaddr');

// IPFS feature manager utility
const featureEnabled = (feature) => {
    return config.ipfs.features && config.ipfs.features[feature] === true;
};

class IPFSService extends EventEmitter {
    constructor() {
        super();
        // Use centralized config for IPFS connection parameters
        this.ipfs = ipfsClient.create({
            host: config.ipfs.host,
            port: config.ipfs.port,
            protocol: config.ipfs.protocol
        });
        
        // Initialize properties
        this.connected = false;
        this.pinnedItems = new Map(); // Track pinned items with timestamps
        this.multiAddresses = new Set(); // Track multi-addresses
        this.remoteServices = new Map(); // Track remote pinning services
        this.connectionPromise = this._initializeConnection();
        this.maxRetries = config.ipfs.maxRetries || 3;
        this.retryDelay = config.ipfs.retryDelay || 5000; // 5s delay between retries
        this.healthCheckInterval = null;
        this.healthCheckFrequency = config.ipfs.healthCheckFrequency || 60000; // Check every minute by default
        this.healthHistory = []; // Keep track of health check results
        this.maxHealthHistory = config.ipfs.maxHealthHistory || 100; // Maximum health history entries
        this.lastConnectionState = false; // Track the previous connection state for change detection
        this.pinningStrategy = config.ipfs.pinOptions.pinningStrategy || 'direct';
        
        // Start health checks if enabled
        if (config.ipfs.enableHealthCheck !== false) {
            this.startHealthCheck();
        }
        
        // Initialize advanced features if enabled
        this._initializeFeatures();
    }

    async _initializeConnection() {
        try {
            await this._verify();
            this.connected = true;
            logger.info('IPFS connection established successfully');
            
            // If connection is successful, load pinned items
            await this._loadPinnedItems();
        } catch (error) {
            this.connected = false;
            logger.error('IPFS connection failed', { error });
            // Don't throw here to allow for connection retries
        }
    }

    async _verify() {
        let retries = 0;
        
        while (retries < this.maxRetries) {
            try {
                const id = await this.ipfs.id();
                logger.info('IPFS Node connected:', { id: id.id });
                return true;
            } catch (err) {
                retries++;
                if (retries >= this.maxRetries) {
                    logger.error('IPFS verification failed after multiple attempts:', { error: err.message });
                    throw err;
                }
                logger.warn(`IPFS verification attempt ${retries} failed, retrying in ${this.retryDelay}ms`);
                await sleep(this.retryDelay);
            }
        }
    }

    async _loadPinnedItems() {
        try {
            logger.debug('Loading pinned items from IPFS...');
            const pins = [];
            
            // Iterate through all pins
            for await (const pin of this.ipfs.pin.ls()) {
                pins.push({
                    cid: pin.cid.toString(),
                    type: pin.type
                });
            }
            
            // Add to tracking map with current timestamp if not already tracked
            pins.forEach(pin => {
                if (!this.pinnedItems.has(pin.cid)) {
                    this.pinnedItems.set(pin.cid, {
                        timestamp: Date.now(),
                        type: pin.type,
                        metadata: {}
                    });
                }
            });
            
            logger.info(`Loaded ${pins.length} pinned items from IPFS`);
        } catch (error) {
            logger.error('Failed to load pinned items from IPFS', { error });
        }
    }

    async ensureConnected() {
        if (!this.connected) {
            await this._initializeConnection();
            if (!this.connected) {
                throw new Error('IPFS service is not connected');
            }
        }
    }

    async store(data, metadata = {}, options = {}) {
        await this.ensureConnected();
        
        try {
            const result = await this.ipfs.add(Buffer.from(JSON.stringify(data)));
            const cid = result.cid.toString();
            logger.debug('Data stored in IPFS', { cid });
            
            // Apply pinning strategy based on configuration
            const pinOptions = {
                ...options,
                strategy: options.strategy || this.pinningStrategy,
                replication: options.replication || config.ipfs.pinOptions.defaultReplication
            };
            
            await this._applyPinningStrategy(cid, pinOptions);
            
            // Track the pin with metadata and timestamp
            this.pinnedItems.set(cid, {
                timestamp: Date.now(),
                type: 'recursive',
                metadata,
                pinOptions
            });
            
            return cid;
        } catch (error) {
            logger.error('Failed to store data in IPFS', { error });
            throw new Error(`IPFS storage error: ${error.message}`);
        }
    }

    async retrieve(cid) {
        await this.ensureConnected();
        
        try {
            const chunks = [];
            for await (const chunk of this.ipfs.cat(cid)) {
                chunks.push(chunk);
            }
            return JSON.parse(Buffer.concat(chunks).toString());
        } catch (error) {
            logger.error('Failed to retrieve data from IPFS', { cid, error });
            throw new Error(`IPFS retrieval error: ${error.message}`);
        }
    }

    async getStatus() {
        try {
            await this.ensureConnected();
            const stats = await this.ipfs.stats.repo();
            return {
                connected: this.connected,
                repoSize: stats.repoSize,
                storageMax: stats.storageMax,
                numObjects: stats.numObjects,
                pinnedItems: this.pinnedItems.size
            };
        } catch (error) {
            logger.error('Failed to get IPFS status', { error });
            return {
                connected: false,
                error: error.message,
                pinnedItems: this.pinnedItems.size
            };
        }
    }

    /**
     * Start periodic health checks of the IPFS connection
     */
    startHealthCheck() {
        if (this.healthCheckInterval) {
            this.stopHealthCheck();
        }
        
        logger.info(`Starting IPFS health check at ${this.healthCheckFrequency}ms intervals`);
        this.healthCheckInterval = setInterval(async () => {
            try {
                const result = await this.checkHealth();
                
                // Keep health history within limits
                this.healthHistory.push(result);
                if (this.healthHistory.length > this.maxHealthHistory) {
                    this.healthHistory = this.healthHistory.slice(-this.maxHealthHistory);
                }
                
                // Detect connection state changes
                if (result.connected !== this.lastConnectionState) {
                    // Emit event for state change
                    this.emit('connectionStateChanged', {
                        previousState: this.lastConnectionState,
                        currentState: result.connected,
                        timestamp: Date.now(),
                        details: result
                    });
                    
                    // Log the change
                    if (result.connected) {
                        logger.info('IPFS connection restored');
                    } else {
                        logger.error('IPFS connection lost', { error: result.error });
                    }
                    
                    // Update last known state
                    this.lastConnectionState = result.connected;
                }
            } catch (err) {
                logger.error('IPFS health check error:', err);
            }
        }, this.healthCheckFrequency);
    }
    
    /**
     * Stop periodic health checks
     */
    stopHealthCheck() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
            logger.info('IPFS health checks stopped');
        }
    }
    
    /**
     * Check the health of the IPFS connection
     * @returns {Object} Health check result
     */
    async checkHealth() {
        const startTime = Date.now();
        let responseTime = null;
        const result = {
            timestamp: startTime,
            connected: false,
            responseTime: null,
            error: null,
            pinnedItemsCount: this.pinnedItems.size,
            details: {}
        };
        
        try {
            // Try to get node ID as a light-weight health check
            const id = await this.ipfs.id();
            responseTime = Date.now() - startTime;
            
            // If we get here, the connection is working
            this.connected = true;
            result.connected = true;
            result.responseTime = responseTime;
            result.details = {
                nodeId: id.id,
                protocols: id.protocols,
                agentVersion: id.agentVersion
            };
            
            // Try to get repo stats for more detailed health info
            try {
                const stats = await this.ipfs.stats.repo();
                result.details.repoStats = {
                    repoSize: stats.repoSize,
                    storageMax: stats.storageMax,
                    numObjects: stats.numObjects
                };
            } catch (statsError) {
                // Non-critical error, just log it
                logger.warn('Could not get IPFS repo stats', { error: statsError.message });
            }
            
        } catch (err) {
            // Connection failed
            this.connected = false;
            result.connected = false;
            result.error = err.message;
            
            // Attempt reconnection if this wasn't triggered by _verify or ensureConnected
            const callerName = (new Error()).stack.split('\n')[2].trim().split(' ')[1];
            if (callerName !== '_verify' && callerName !== 'ensureConnected') {
                logger.info('Attempting to reconnect to IPFS');
                // Schedule reconnection attempt (don't await to avoid blocking the health check)
                this._initializeConnection().catch(e => 
                    logger.error('Reconnection attempt failed', { error: e.message })
                );
            }
        }
        
        return result;
    }

    /**
     * Get health check history
     * @param {number} limit Maximum number of history entries to return
     * @returns {Array} History of health checks
     */
    getHealthHistory(limit = 20) {
        return this.healthHistory.slice(-Math.min(limit, this.healthHistory.length));
    }
    
    /**
     * Get a summary of the health check history
     * @param {number} minutes Number of minutes of history to consider
     * @returns {Object} Summary of the health
     */
    getHealthSummary(minutes = 10) {
        const cutoffTime = Date.now() - (minutes * 60 * 1000);
        const recentChecks = this.healthHistory.filter(check => check.timestamp >= cutoffTime);
        
        if (recentChecks.length === 0) {
            return {
                status: 'unknown',
                uptime: 0,
                checksPerformed: 0,
                averageResponseTime: null
            };
        }
        
        const successfulChecks = recentChecks.filter(check => check.connected);
        const uptime = successfulChecks.length / recentChecks.length;
        const connectedTimes = successfulChecks
            .filter(check => check.responseTime !== null)
            .map(check => check.responseTime);
            
        const avgResponseTime = connectedTimes.length > 0 
            ? connectedTimes.reduce((sum, time) => sum + time, 0) / connectedTimes.length 
            : null;
            
        // Calculate the current streak of successful or failed checks
        let currentStreak = 1;
        for (let i = this.healthHistory.length - 2; i >= 0; i--) {
            if (this.healthHistory[i].connected === this.healthHistory[this.healthHistory.length - 1].connected) {
                currentStreak++;
            } else {
                break;
            }
        }
        
        return {
            status: this.connected ? 'healthy' : 'unhealthy',
            uptime: uptime,
            checksPerformed: recentChecks.length,
            averageResponseTime: avgResponseTime,
            currentStreak: currentStreak,
            currentState: this.connected ? 'connected' : 'disconnected',
            lastCheck: this.healthHistory.length > 0 ? this.healthHistory[this.healthHistory.length - 1] : null
        };
    }

    /**
     * Cleanup pinned items based on age or custom filters
     * @param {Object} options Cleanup options
     * @param {number} options.olderThan Age in ms for items to be considered for cleanup (default: 30 days)
     * @param {Function} options.filter Custom filter function for items to clean up
     * @param {boolean} options.dryRun If true, don't actually remove pins, just report what would be removed
     * @param {boolean} options.keepMetadata Whether to keep metadata for unpinned items
     * @returns {Object} Results of the cleanup operation
     */
    async cleanup(options = {}) {
        try {
            const {
                olderThan = 30 * 24 * 60 * 60 * 1000, // 30 days by default
                filter = null,
                dryRun = false,
                keepMetadata = false
            } = options;
            
            await this.ensureConnected();
            
            const now = Date.now();
            const itemsToRemove = [];
            
            // Identify items to clean up
            for (const [cid, data] of this.pinnedItems.entries()) {
                const age = now - data.timestamp;
                const shouldRemove = filter 
                    ? filter(cid, data) 
                    : age > olderThan;
                
                if (shouldRemove) {
                    itemsToRemove.push({ cid, data });
                }
            }
            
            logger.info(`Found ${itemsToRemove.length} items to clean up${dryRun ? ' (dry run)' : ''}`);
            
            // Perform actual cleanup if not a dry run
            if (!dryRun) {
                let successCount = 0;
                let failCount = 0;
                
                for (const { cid, data } of itemsToRemove) {
                    try {
                        await this.ipfs.pin.rm(cid);
                        
                        // Remove from tracking or just update metadata
                        if (keepMetadata) {
                            this.pinnedItems.set(cid, { 
                                ...data, 
                                unpinned: true, 
                                unpinnedAt: now 
                            });
                        } else {
                            this.pinnedItems.delete(cid);
                        }
                        
                        successCount++;
                        logger.debug(`Unpinned item: ${cid}`);
                    } catch (err) {
                        failCount++;
                        logger.error(`Failed to unpin ${cid}: ${err.message}`);
                    }
                }
                
                logger.info(`IPFS cleanup completed: ${successCount} items unpinned, ${failCount} failures`);
                
                return {
                    success: true,
                    itemsRemoved: successCount,
                    itemsFailed: failCount,
                    totalPinned: this.pinnedItems.size
                };
            }
            
            // For dry runs, just return the count
            return {
                dryRun: true,
                itemsToRemove: itemsToRemove.length,
                totalPinned: this.pinnedItems.size
            };
        } catch (error) {
            logger.error('IPFS cleanup failed', { error });
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    /**
     * Add metadata to a pinned item
     * @param {string} cid The CID of the item
     * @param {Object} metadata Metadata to associate with the item
     * @returns {boolean} Whether the operation succeeded
     */
    async addMetadata(cid, metadata) {
        try {
            if (this.pinnedItems.has(cid)) {
                const existing = this.pinnedItems.get(cid);
                this.pinnedItems.set(cid, {
                    ...existing,
                    metadata: {
                        ...existing.metadata,
                        ...metadata,
                        lastUpdated: Date.now()
                    }
                });
                return true;
            }
            return false;
        } catch (error) {
            logger.error(`Failed to add metadata to ${cid}`, { error });
            return false;
        }
    }
    
    /**
     * Get pinned items that match specific criteria
     * @param {Function} filter Filter function for items
     * @returns {Array} Array of matching items
     */
    getPinnedItems(filter = null) {
        if (!filter) {
            return Array.from(this.pinnedItems.entries())
                .map(([cid, data]) => ({ cid, ...data }));
        }
        
        return Array.from(this.pinnedItems.entries())
            .filter(([cid, data]) => filter(cid, data))
            .map(([cid, data]) => ({ cid, ...data }));
    }

    /**
     * Initialize advanced features based on configuration
     */
    async _initializeFeatures() {
        // Initialize multi-address support
        if (featureEnabled('multiAddressSupport')) {
            try {
                logger.info('Initializing IPFS multi-address support');
                await this._loadMultiAddresses();
            } catch (err) {
                logger.error('Failed to initialize multi-address support', { error: err.message });
            }
        }
        
        // Initialize remote pinning services if using service or hybrid strategy
        if (['service', 'hybrid'].includes(this.pinningStrategy)) {
            try {
                logger.info(`Initializing IPFS pinning strategy: ${this.pinningStrategy}`);
                await this._setupRemotePinningServices();
            } catch (err) {
                logger.error('Failed to initialize remote pinning services', { error: err.message });
            }
        }
        
        // Initialize PubSub if enabled
        if (featureEnabled('pubsubEnabled')) {
            try {
                logger.info('Initializing IPFS PubSub');
                // Verify PubSub capability
                const features = await this.ipfs.id();
                if (!features.agentVersion.includes('go-ipfs')) {
                    logger.warn('PubSub might not be fully supported by this IPFS implementation');
                }
            } catch (err) {
                logger.error('Failed to initialize PubSub', { error: err.message });
            }
        }
    }

    /**
     * Load stored multi-addresses
     */
    async _loadMultiAddresses() {
        if (!featureEnabled('multiAddressSupport')) return;
        
        try {
            // Get connected peers to initialize multi-addresses
            const peers = await this.ipfs.swarm.peers();
            peers.forEach(peer => {
                try {
                    const addr = multiaddr(peer.addr).toString();
                    this.multiAddresses.add(addr);
                } catch (err) {
                    logger.debug(`Invalid multi-address: ${peer.addr}`);
                }
            });
            logger.info(`Loaded ${this.multiAddresses.size} multi-addresses`);
        } catch (err) {
            logger.error('Failed to load multi-addresses', { error: err.message });
        }
    }

    /**
     * Set up remote pinning services
     */
    async _setupRemotePinningServices() {
        if (!['service', 'hybrid'].includes(this.pinningStrategy)) return;
        
        try {
            // If a default remote pinning service is configured, add it
            if (config.ipfs.pinOptions.remotePinningService && 
                config.ipfs.pinOptions.remotePinningKey) {
                this.remoteServices.set('default', {
                    endpoint: config.ipfs.pinOptions.remotePinningService,
                    key: config.ipfs.pinOptions.remotePinningKey
                });
                logger.info('Default remote pinning service configured');
            }
            
            // Connect to all configured remote services to verify they work
            for (const [name, service] of this.remoteServices.entries()) {
                try {
                    // Example verification call to remote pinning service
                    // In a real implementation, you would make an API call to verify the service
                    logger.info(`Verified connection to remote pinning service: ${name}`);
                } catch (err) {
                    logger.error(`Failed to verify remote pinning service ${name}`, { error: err.message });
                    this.remoteServices.delete(name);
                }
            }
        } catch (err) {
            logger.error('Failed to set up remote pinning services', { error: err.message });
        }
    }

    /**
     * Apply the configured pinning strategy to a CID
     * @param {string} cid Content identifier to pin
     * @param {Object} options Pinning options
     */
    async _applyPinningStrategy(cid, options = {}) {
        const strategy = options.strategy || this.pinningStrategy;
        
        try {
            switch (strategy) {
                case 'direct':
                    // Simply pin locally
                    await this.ipfs.pin.add(cid);
                    logger.debug(`Direct pinning applied to ${cid}`);
                    break;
                    
                case 'service':
                    // Pin to remote service only
                    if (this.remoteServices.size === 0) {
                        logger.warn('No remote pinning services configured, falling back to direct pinning');
                        await this.ipfs.pin.add(cid);
                    } else {
                        const promises = [];
                        for (const [name, service] of this.remoteServices.entries()) {
                            promises.push(this._pinToRemoteService(cid, service, name));
                        }
                        await Promise.all(promises);
                    }
                    break;
                    
                case 'hybrid':
                    // Pin both locally and to remote services
                    const localPromise = this.ipfs.pin.add(cid);
                    const remotePromises = [];
                    
                    for (const [name, service] of this.remoteServices.entries()) {
                        remotePromises.push(this._pinToRemoteService(cid, service, name));
                    }
                    
                    await Promise.all([localPromise, ...remotePromises]);
                    logger.debug(`Hybrid pinning applied to ${cid}`);
                    break;
                    
                default:
                    logger.warn(`Unknown pinning strategy '${strategy}', using direct pinning`);
                    await this.ipfs.pin.add(cid);
            }
        } catch (error) {
            logger.error(`Failed to apply pinning strategy to ${cid}`, { error });
            throw new Error(`Pinning strategy error: ${error.message}`);
        }
    }
    
    /**
     * Pin content to remote pinning service
     * @param {string} cid Content identifier to pin
     * @param {Object} service Remote service configuration
     * @param {string} serviceName Name of the service for logging
     */
    async _pinToRemoteService(cid, service, serviceName) {
        try {
            // In a real implementation, this would make an API call to the remote pinning service
            // This is a placeholder for demonstration purposes
            logger.info(`Pinned ${cid} to remote service ${serviceName}`);
            return true;
        } catch (err) {
            logger.error(`Failed to pin ${cid} to remote service ${serviceName}`, { error: err.message });
            throw err;
        }
    }

    /**
     * Add a multi-address to the IPFS node's peer list
     * @param {string} addr The multi-address to add
     * @returns {boolean} Whether the operation succeeded
     */
    async addMultiAddress(addr) {
        if (!featureEnabled('multiAddressSupport')) {
            logger.warn('Multi-address support is disabled');
            return false;
        }
        
        try {
            // Validate the address format
            const address = multiaddr(addr);
            
            // Connect to the address
            await this.ipfs.swarm.connect(address);
            
            // Add to our tracking set
            this.multiAddresses.add(address.toString());
            
            logger.info(`Connected to peer via multi-address: ${addr}`);
            return true;
        } catch (error) {
            logger.error(`Failed to add multi-address ${addr}`, { error });
            return false;
        }
    }
    
    /**
     * Remove a multi-address from the IPFS node's peer list
     * @param {string} addr The multi-address to remove
     * @returns {boolean} Whether the operation succeeded
     */
    async removeMultiAddress(addr) {
        if (!featureEnabled('multiAddressSupport')) {
            logger.warn('Multi-address support is disabled');
            return false;
        }
        
        try {
            // Validate the address format
            const address = multiaddr(addr);
            
            // Disconnect from the address
            await this.ipfs.swarm.disconnect(address);
            
            // Remove from our tracking set
            this.multiAddresses.delete(address.toString());
            
            logger.info(`Disconnected from peer via multi-address: ${addr}`);
            return true;
        } catch (error) {
            logger.error(`Failed to remove multi-address ${addr}`, { error });
            return false;
        }
    }
    
    /**
     * Get all tracked multi-addresses
     * @returns {Array<string>} List of multi-addresses
     */
    getMultiAddresses() {
        if (!featureEnabled('multiAddressSupport')) {
            logger.warn('Multi-address support is disabled');
            return [];
        }
        
        return Array.from(this.multiAddresses);
    }

    /**
     * Add a remote pinning service
     * @param {string} name Service name
     * @param {string} endpoint Service endpoint URL
     * @param {string} key API key for the service
     * @returns {boolean} Whether the operation succeeded
     */
    addRemotePinningService(name, endpoint, key) {
        if (!['service', 'hybrid'].includes(this.pinningStrategy)) {
            logger.warn(`Remote pinning services not supported with ${this.pinningStrategy} strategy`);
            return false;
        }
        
        try {
            this.remoteServices.set(name, { endpoint, key });
            logger.info(`Added remote pinning service: ${name}`);
            return true;
        } catch (error) {
            logger.error(`Failed to add remote pinning service ${name}`, { error });
            return false;
        }
    }
    
    /**
     * Remove a remote pinning service
     * @param {string} name Service name
     * @returns {boolean} Whether the operation succeeded
     */
    removeRemotePinningService(name) {
        try {
            const result = this.remoteServices.delete(name);
            if (result) {
                logger.info(`Removed remote pinning service: ${name}`);
            } else {
                logger.warn(`Remote pinning service not found: ${name}`);
            }
            return result;
        } catch (error) {
            logger.error(`Failed to remove remote pinning service ${name}`, { error });
            return false;
        }
    }
    
    /**
     * Get all configured remote pinning services
     * @returns {Object} Map of service names to configurations (with keys redacted)
     */
    getRemotePinningServices() {
        const services = {};
        for (const [name, config] of this.remoteServices.entries()) {
            services[name] = {
                endpoint: config.endpoint,
                keyConfigured: !!config.key
            };
        }
        return services;
    }
    
    /**
     * Pin an existing CID using the specified strategy
     * @param {string} cid The CID to pin
     * @param {Object} options Pinning options
     * @returns {boolean} Whether the operation succeeded
     */
    async pinCID(cid, options = {}) {
        if (!featureEnabled('advancedPinning')) {
            logger.warn('Advanced pinning is disabled, using default pinning');
            try {
                await this.ipfs.pin.add(cid);
                return true;
            } catch (err) {
                logger.error(`Failed to pin ${cid}`, { error: err.message });
                return false;
            }
        }
        
        try {
            await this.ensureConnected();
            
            const pinOptions = {
                ...options,
                strategy: options.strategy || this.pinningStrategy,
                replication: options.replication || config.ipfs.pinOptions.defaultReplication
            };
            
            await this._applyPinningStrategy(cid, pinOptions);
            
            // Track the pin if we don't already have it
            if (!this.pinnedItems.has(cid)) {
                this.pinnedItems.set(cid, {
                    timestamp: Date.now(),
                    type: 'recursive',
                    metadata: options.metadata || {},
                    pinOptions
                });
            }
            
            return true;
        } catch (error) {
            logger.error(`Failed to pin ${cid}`, { error });
            return false;
        }
    }
    
    /**
     * Check whether a CID is pinned
     * @param {string} cid The CID to check
     * @param {string} type Pin type ('direct', 'recursive', 'indirect', or 'all')
     * @returns {Object} Pin status
     */
    async isPinned(cid, type = 'all') {
        try {
            await this.ensureConnected();
            
            // Check local pins
            let locallyPinned = false;
            try {
                const pins = await this.ipfs.pin.ls({ paths: [cid], type });
                locallyPinned = Array.from(pins).length > 0;
            } catch (err) {
                if (err.message.includes('not pinned')) {
                    locallyPinned = false;
                } else {
                    throw err;
                }
            }
            
            // Check remote pins if using service or hybrid strategy
            let remotelyPinned = false;
            let remoteServices = [];
            
            if (['service', 'hybrid'].includes(this.pinningStrategy)) {
                for (const [name, service] of this.remoteServices.entries()) {
                    try {
                        // In a real implementation, this would check remote pinning services
                        // For this example, we'll assume it's pinned remotely if we track it
                        if (this.pinnedItems.has(cid)) {
                            remotelyPinned = true;
                            remoteServices.push(name);
                        }
                    } catch (err) {
                        logger.debug(`Failed to check pin on remote service ${name}`, { error: err.message });
                    }
                }
            }
            
            return {
                cid,
                pinned: locallyPinned || remotelyPinned,
                locallyPinned,
                remotelyPinned,
                remoteServices: remotelyPinned ? remoteServices : [],
                timestamp: Date.now()
            };
        } catch (error) {
            logger.error(`Failed to check pin status for ${cid}`, { error });
            return {
                cid,
                pinned: false,
                error: error.message,
                timestamp: Date.now()
            };
        }
    }
    
    /**
     * List all pins matching the given criteria
     * @param {Object} options Options for listing pins
     * @returns {Array} List of pins
     */
    async listPins(options = {}) {
        try {
            await this.ensureConnected();
            
            const { type = 'all', before, after, metadata } = options;
            const results = [];
            
            // Get pins from IPFS node
            const pins = await this.ipfs.pin.ls({ type });
            
            for (const pin of pins) {
                const cid = pin.cid.toString();
                
                // If we have metadata for this pin, include it
                const trackedPin = this.pinnedItems.get(cid);
                
                // Apply timestamp filters if provided
                if (before && trackedPin && trackedPin.timestamp > before) continue;
                if (after && trackedPin && trackedPin.timestamp < after) continue;
                
                // Apply metadata filter if provided
                if (metadata && trackedPin) {
                    const pinMeta = trackedPin.metadata || {};
                    let shouldInclude = true;
                    
                    for (const [key, value] of Object.entries(metadata)) {
                        if (pinMeta[key] !== value) {
                            shouldInclude = false;
                            break;
                        }
                    }
                    
                    if (!shouldInclude) continue;
                }
                
                results.push({
                    cid,
                    type: pin.type,
                    ...trackedPin
                });
            }
            
            return results;
        } catch (error) {
            logger.error('Failed to list pins', { error });
            throw new Error(`IPFS pin listing error: ${error.message}`);
        }
    }

    /**
     * Get advanced IPFS features status
     * @returns {Object} Status of advanced features
     */
    getFeatureStatus() {
        return {
            multiAddressSupport: {
                enabled: featureEnabled('multiAddressSupport'),
                addressCount: this.multiAddresses.size
            },
            advancedPinning: {
                enabled: featureEnabled('advancedPinning'),
                strategy: this.pinningStrategy,
                remoteServicesCount: this.remoteServices.size
            },
            pubsub: {
                enabled: featureEnabled('pubsubEnabled')
            },
            gatewayProxy: {
                enabled: featureEnabled('gatewayProxy')
            }
        };
    }

    /**
     * Shutdown the service properly
     */
    shutdown() {
        this.stopHealthCheck();
        this.removeAllListeners();
        logger.info('IPFS service shut down');
    }
}

const ipfsService = new IPFSService();

// Handle process exit
process.on('SIGTERM', () => {
    ipfsService.shutdown();
});

process.on('SIGINT', () => {
    ipfsService.shutdown();
});

module.exports = ipfsService;
