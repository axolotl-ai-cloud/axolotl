#!/bin/bash

# Export specific ENV variables to /etc/rp_environment
echo "Exporting environment variables..."
printenv | grep -E '^RUNPOD_|^PATH=|^_=' | sed 's/^\(.*\)=\(.*\)$/export \1="\2"/' >> /etc/rp_environment
echo 'source /etc/rp_environment' >> ~/.bashrc

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo $PUBLIC_KEY >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
    # Start the SSH service in the background
    service ssh start
else
    echo "No PUBLIC_KEY ENV variable provided, not starting openSSH daemon"
fi

# Execute the passed arguments (CMD)
exec "$@"
