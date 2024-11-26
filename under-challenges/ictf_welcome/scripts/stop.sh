#!/bin/bash
IMAGE="ictf_welcome"

# Get the container IDs of all running containers based on the specified image
CONTAINER_IDS=$(docker ps -q --filter ancestor=$IMAGE)

# Check if any containers were found
if [ -z "$CONTAINER_IDS" ]; then
    echo "No running containers found for image '$IMAGE'"
    exit 0
fi

# Stop each container
for CONTAINER_ID in $CONTAINER_IDS
do
    echo "Stopping container $CONTAINER_ID..."
    docker stop $CONTAINER_ID
done

echo "All containers associated with the image '$IMAGE' have been stopped."
