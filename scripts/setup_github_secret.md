# Setting up GitHub Secret for Sudo Password

To deploy MindModel successfully, you need to set up a GitHub secret for the sudo password.

## Steps to Add the Secret

1. **Go to your GitHub repository**
   - Navigate to: https://github.com/swipentap/TextContext

2. **Go to Settings**
   - Click on the "Settings" tab in your repository

3. **Go to Secrets and Variables**
   - In the left sidebar, click on "Secrets and variables"
   - Then click on "Actions"

4. **Add New Repository Secret**
   - Click the "New repository secret" button
   - Name: `SUDO_PASSWORD`
   - Value: Enter your sudo password for the server `jaal@10.11.2.6`

5. **Save the Secret**
   - Click "Add secret"

## Alternative: Using GitHub CLI

If you have GitHub CLI installed, you can also add the secret via command line:

```bash
# Install GitHub CLI if you don't have it
# brew install gh (on macOS)
# or download from: https://cli.github.com/

# Login to GitHub
gh auth login

# Add the secret
gh secret set SUDO_PASSWORD --repo swipentap/TextContext
# It will prompt you to enter the password securely
```

## Security Notes

- The secret is encrypted and only accessible during workflow runs
- The password is never logged or displayed in the workflow output
- Only repository owners and collaborators with admin access can manage secrets

## Testing the Secret

After adding the secret, you can test it by:

1. Making a small change to any file
2. Committing and pushing the change
3. The workflow will automatically trigger and use the secret

## Troubleshooting

If the deployment still fails:

1. **Check the secret name**: Make sure it's exactly `SUDO_PASSWORD` (case sensitive)
2. **Verify the password**: Ensure the password is correct for the `jaal` user on `10.11.2.6`
3. **Check permissions**: Make sure the `jaal` user has sudo privileges
4. **Check workflow logs**: Look at the GitHub Actions logs for specific error messages

## Manual Deployment (Alternative)

If you prefer not to use GitHub secrets, you can deploy manually:

```bash
# SSH to the server
ssh jaal@10.11.2.6

# Clone the repository
git clone https://github.com/swipentap/TextContext.git
cd TextContext

# Run deployment manually
./scripts/deploy_docker.sh
# It will prompt for the sudo password
```
