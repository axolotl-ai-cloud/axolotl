#!/bin/bash

echo $PUBLIC_KEY >> ~/.ssh/authorized_keys
chmod 700 -R ~/.ssh

# Start the SSH service in the background
service ssh start

# Execute the passed arguments (CMD)
exec "$@"
