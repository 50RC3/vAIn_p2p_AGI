const { execSync } = require('child_process');
const path = require('path');

console.log('Installing vAIn dependencies...');

try {
    // Install Python dependencies
    console.log('\nInstalling Python dependencies...');
    execSync('python -m pip install -r requirements.txt', { stdio: 'inherit' });

    // Install root project dependencies
    console.log('\nInstalling Node.js root dependencies...');
    execSync('npm install', { stdio: 'inherit' });

    // Install backend dependencies
    console.log('\nInstalling backend dependencies...');
    process.chdir(path.join(__dirname, 'backend'));
    execSync('npm install', { stdio: 'inherit' });

    console.log('\nInstallation complete! You can now run "npm run dev" to start the development server.');
} catch (error) {
    console.error('Installation failed:', error.message);
    process.exit(1);
}
