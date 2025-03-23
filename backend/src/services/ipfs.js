const ipfsClient = require('ipfs-http-client');
const { Buffer } = require('buffer');

class IPFSService {
    constructor() {
        this.ipfs = ipfsClient.create({
            host: process.env.IPFS_HOST || 'localhost',
            port: process.env.IPFS_PORT || '5001',
            protocol: process.env.IPFS_PROTOCOL || 'http'
        });
        this._verify();
    }

    async _verify() {
        try {
            const id = await this.ipfs.id();
            console.log('IPFS Node connected:', id.id);
        } catch (err) {
            console.error('IPFS connection failed:', err);
            throw err;
        }
    }

    async store(data) {
        const result = await this.ipfs.add(Buffer.from(JSON.stringify(data)));
        return result.cid.toString();
    }

    async retrieve(cid) {
        const chunks = [];
        for await (const chunk of this.ipfs.cat(cid)) {
            chunks.push(chunk);
        }
        return JSON.parse(Buffer.concat(chunks).toString());
    }

    async getStatus() {
        try {
            const id = await this.ipfs.id();
            const repo = await this.ipfs.repo.stat();
            
            return {
                connected: true,
                nodeId: id.id,
                storageUsed: repo.repoSize,
                storageMax: repo.storageMax,
                numObjects: repo.numObjects
            };
        } catch (err) {
            return {
                connected: false,
                error: err.message
            };
        }
    }

    async cleanup() {
        try {
            await this.ipfs.repo.gc();
            return true;
        } catch (err) {
            console.error('IPFS cleanup failed:', err);
            return false;
        }
    }
}

module.exports = new IPFSService();
