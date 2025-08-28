#!/bin/bash

# GitHub Actions Runner Installation Script
# This script installs an additional GitHub Actions runner on a remote server

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

# Prompt for sudo password once
echo -n "Enter sudo password for $RUNNER_USER@$RUNNER_HOST: "
read -s SUDO_PASSWORD
echo

# Create the runner directory and download the runner
ssh $RUNNER_USER@$RUNNER_HOST << EOF
set -e

# Store sudo password for use in the session
SUDO_PASS='$SUDO_PASSWORD'

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

echo "🔧 Installing as service..."
echo "\$SUDO_PASS" | sudo -S ./svc.sh install $RUNNER_USER

echo "🚀 Starting service..."
echo "\$SUDO_PASS" | sudo -S ./svc.sh start

echo "✅ Additional runner installation completed!"
echo "📊 Runner status:"
./svc.sh status

echo "📋 Runner details:"
echo "  - Name: $RUNNER_NAME"
echo "  - Directory: ~/actions-runner-$RUNNER_NAME"
echo "  - Service: actions.runner.$RUNNER_NAME"
EOF

echo "🎉 Additional GitHub Actions Runner successfully installed and started!"
echo ""
echo "📋 Runner Information:"
echo "  - Name: $RUNNER_NAME"
echo "  - Directory: ~/actions-runner-$RUNNER_NAME"
echo "  - Service: actions.runner.$RUNNER_NAME"
echo ""
echo "📋 Management commands:"
echo "1. Check runner status: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && ./svc.sh status'"
echo "2. Stop service: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && sudo ./svc.sh stop'"
echo "3. Start service: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && sudo ./svc.sh start'"
echo "4. Uninstall: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && sudo ./svc.sh uninstall'"
echo "5. Remove runner: ssh $RUNNER_USER@$RUNNER_HOST 'cd ~/actions-runner-$RUNNER_NAME && ./config.sh remove --unattended'"
