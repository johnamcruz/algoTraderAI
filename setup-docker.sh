# 1. Build the Docker image
docker-compose build

# 2. Run the bot in detached mode (use -f for follow logs)
docker-compose up -d

# 3. View the logs to check for successful connection/authentication
docker-compose logs -f algo-bot

# 4. Stop the container
docker-compose down