const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("vAInToken", function() {
    let token;
    let owner;
    let addr1;
    let addr2;

    beforeEach(async function() {
        const Token = await ethers.getContractFactory("vAInToken");
        [owner, addr1, addr2] = await ethers.getSigners();
        token = await Token.deploy();
        await token.deployed();
    });

    it("Should assign total supply to owner", async function() {
        const ownerBalance = await token.balanceOf(owner.address);
        expect(await token.totalSupply()).to.equal(ownerBalance);
    });
});
