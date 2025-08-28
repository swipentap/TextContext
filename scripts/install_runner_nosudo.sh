#!/bin/bash

# GitHub Actions Runner Installation Script (No Sudo)
# This script installs an additional GitHub Actions runner without sudo service

set -e

# Configuration
RUNNER_VERSION="2.328.0"
REPO_URL="https://github.com/swipentap/TextContext"
RUNNER_TOKEN="AYPADNHSORI6NUTOX2YE5A3IWCCBK"
RUNNER_USER="jaal"
RUNNER_HOST="10.11.2.6"
RUNNER_NAME="mindmodel-runner-$(date +%s)"  # Unique name with timestamp

echo "🚀 Installing additional GitHub Actions Runner on $RUNNER_USER@$RUNNER_HOST"
echo "📝 Runner name: $RUNNER_NAME"

# Create the runner directory and download the runner
ssh $RUNNER_USER@$RUNNER_HOST << EOF
set -e

echo "📁 Creating runner directory..."
mkdir -p ~/actions-runner-$RUNNER_NAME
cd ~/actions-runner-$RUNNER_NAME

echo "📥 Downloading runner package..."
curl -o actions-runner-linux-x64-2.328.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.328.0/actions-runner-linux-x64-2.328.0.tar.gz

echo "🔍 Validating hash..."
echo "01066fad3a2893e63e6ca880ae3a1fad5bf9329d60e77ee15f2b97c148c3cd4e  actions-runner-linux-x64-2.328.0.tar.gz" | shasum -a 256 -c

echo "📦 Extracting installer..."
tar xzf ./actions-runner-linux-x64-2.328.0.tar.gz

echo "⚙️ Configuring runner..."
./config.sh --url https://github.com/swipentap/TextContext --token AYPADNHSORI6NUTOX2YE5A3IWCCBK --name $RUNNER_NAME --unattended --replace

echo "✅ Additional runner installation completed!"
echo "📋 Runner details:"
echo "  - Name: $RUNNER_NAME"
echo "  - Directory: ~/actions-runner-$RUNNER_NAME"
echo "  - Status: Configured (not running as service)"

echo ""
echo "🚀 To start the runner manually:"
echo "  cd ~/actions-runner-$RUNNER_NAME && ./run.sh"
echo ""
echo "📋 To run in background:"
echo "  cd ~/actions-runner-$RUNNER_NAME && nohup ./run.sh > runner.log 2>&1 &"
echo ""
echo "📋 To check if running:"
echo "  ps aux | grep actions-runner"
EOF

echo "🎉 Additional GitHub Actions Runner successfully installed!"
echo ""
echo "📋 Runner Information:"
echo "  - Name: $RUNNER_NAME"
echo "  - Directory: ~/actions-runner-$RUNNER_NAME"
echo "  - Status: Configured (manual start required)"
echo ""
echo "📋 Management commands:"
echo "1. Start runner: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && ./run.sh'"
echo "2. Start in background: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && nohup ./run.sh > runner.log 2>&1 &'"
echo "3. Check if running: ssh $RUNNER_USER@$RUNNER_HOST 'ps aux | grep actions-runner'"
echo "4. Stop runner: ssh $RUNNER_USER@$RUNNER_HOST 'pkill -f actions-runner'"
echo "5. Remove runner: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && ./config.sh remove --unattended'"
