const request = require('supertest');
const { expect } = require('chai');
const app = require('../../backend/server');

describe('Backend API Tests', () => {
    describe('Authentication', () => {
        it('should authenticate with valid credentials', async () => {
            const res = await request(app)
                .post('/api/auth/login')
                .send({
                    address: '0x1234...5678',
                    signature: 'validSignature'
                });
            expect(res.status).to.equal(200);
            expect(res.body).to.have.property('token');
        });
    });

    describe('Staking', () => {
        it('should accept valid stake amount', async () => {
            const res = await request(app)
                .post('/api/staking/stake')
                .send({
                    amount: '1000000000000000000', // 1 VAIN
                    address: '0x1234...5678'
                });
            expect(res.status).to.equal(200);
            expect(res.body.success).to.be.true;
        });
    });
});
