#!/bin/bash

# Export specific ENV variables to /etc/rp_environment
echo "Exporting environment variables..."
printenv | grep -E '^RUNPOD_|^PATH=|^_=' | sed 's/^\(.*\)=\(.*\)$/export \1="\2"/' >> /etc/rp_environment
conda init
# this needs to come after conda init
echo 'source /etc/rp_environment' >> ~/.bashrc

add_keys_to_authorized() {
    local key_value=$1

    # Create the ~/.ssh directory and set permissions
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh

    # Create the authorized_keys file if it doesn't exist
    touch ~/.ssh/authorized_keys

    # Initialize an empty key variable
    local key=""

    # Read the key variable word by word
    for word in $key_value; do
        # Check if the word looks like the start of a key
        if [[ $word == ssh-* ]]; then
            # If there's a key being built, add it to the authorized_keys file
            if [[ -n $key ]]; then
                echo $key >> ~/.ssh/authorized_keys
            fi
            # Start a new key
            key=$word
        else
            # Append the word to the current key
            key="$key $word"
        fi
    done

    # Add the last key to the authorized_keys file
    if [[ -n $key ]]; then
        echo $key >> ~/.ssh/authorized_keys
    fi

    # Set the correct permissions
    chmod 600 ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
}

if [[ $PUBLIC_KEY ]]; then
    # runpod
    add_keys_to_authorized "$PUBLIC_KEY"
    # Start the SSH service in the background
    service ssh start
elif [[ $SSH_KEY ]]; then
    # latitude.sh
    add_keys_to_authorized "$SSH_KEY"
    # Start the SSH service in the background
    service ssh start
else
    echo "No PUBLIC_KEY or SSH_KEY environment variable provided, not starting openSSH daemon"
fi

# Check if JUPYTER_PASSWORD is set and not empty
if [ -n "$JUPYTER_PASSWORD" ]; then
    # Set JUPYTER_TOKEN to the value of JUPYTER_PASSWORD
    export JUPYTER_TOKEN="$JUPYTER_PASSWORD"
fi

if [ "$JUPYTER_DISABLE" != "1" ]; then
    # Run Jupyter Lab in the background
    jupyter lab --port=8888 --ip=* --allow-root --ServerApp.allow_origin=* &
fi

if [ ! -d "/workspace/data/axolotl-artifacts" ]; then
    mkdir -p /workspace/data/axolotl-artifacts
fi
if [ ! -L "/workspace/axolotl/outputs" ]; then
    ln -sf /workspace/data/axolotl-artifacts /workspace/axolotl/outputs
fi

# Execute the passed arguments (CMD)
exec "$@"
