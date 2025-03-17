const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ModelAudit", function() {
    let modelAudit;
    let owner;
    let validator;

    beforeEach(async function() {
        const ModelAudit = await ethers.getContractFactory("ModelAudit");
        [owner, validator] = await ethers.getSigners();
        modelAudit = await ModelAudit.deploy(ethers.utils.parseEther("1000"));
        await modelAudit.deployed();
    });

    describe("Input Validation", function() {
        it("should reject empty model hash", async function() {
            await expect(
                modelAudit.submitValidation("", true, 50)
            ).to.be.revertedWith("Model hash cannot be empty");
        });

        it("should reject invalid scores", async function() {
            await expect(
                modelAudit.submitValidation("hash123", true, 101)
            ).to.be.revertedWith("Score must be between 0 and 100");
        });

        it("should reject model hash that's too long", async function() {
            const longHash = "a".repeat(65);
            await expect(
                modelAudit.submitValidation(longHash, true, 50)
            ).to.be.revertedWith("Model hash too long");
        });

        it("should validate stake requirements", async function() {
            await expect(
                modelAudit.connect(validator).submitValidation("hash123", true, 50)
            ).to.be.revertedWith("Insufficient stake");
        });
    });

    describe("State Changes", function() {
        beforeEach(async function() {
            // Setup validator with minimum stake
            await modelAudit.connect(validator).stake({ 
                value: ethers.utils.parseEther("1000") 
            });
        });

        it("should properly store validation", async function() {
            await modelAudit.connect(validator).submitValidation("hash123", true, 75);
            const validation = await modelAudit.modelValidations("hash123", 0);
            
            expect(validation.modelHash).to.equal("hash123");
            expect(validation.approved).to.equal(true);
            expect(validation.score).to.equal(75);
            expect(validation.validator).to.equal(validator.address);
        });

        it("should emit ModelValidated event", async function() {
            await expect(
                modelAudit.connect(validator).submitValidation("hash123", true, 75)
            ).to.emit(modelAudit, "ModelValidated")
              .withArgs("hash123", validator.address, true, 75);
        });
    });
});
