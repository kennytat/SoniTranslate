#!/bin/bash
set -e

MODEL_DIR="/app/model"

# Check if model directory exists and is not empty
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model directory found and not empty, skipping download."
else
    echo "Model directory is missing or empty, downloading models..."

    shareid="${MODEL_SHAREID}"
    password="${MODEL_PASSWORD}"

    # Prompt from terminal if env vars are not set
    if [ -z "$shareid" ]; then
        read -p "Enter your shareid: " shareid
    fi
    if [ -z "$password" ]; then
        read -sp "Enter your password: " password
        echo
    fi

    if [ -n "$shareid" ] && [ -n "$password" ]; then
        mkdir -p "$MODEL_DIR"
        curl -o /tmp/model.zip --retry 3 --retry-all-errors \
            -u "${shareid}:${password}" \
            -H "X-Requested-With: XMLHttpRequest" \
            "https://vgm.cloud/public.php/webdav/"
        unzip /tmp/model.zip -d /app
        rm -f /tmp/model.zip
        echo "Model download complete."
    else
        echo "ERROR: shareid or password is empty. Cannot download models."
        exit 1
    fi
fi

# Start cron for cleanup jobs
service cron start

# Execute the command passed to the container
exec "$@"
