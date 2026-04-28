#!/bin/bash
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

prompt() {
    osascript -e "Tell application \"System Events\" to display dialog \"$1\" default answer \"$2\"" \
              -e "text returned of result" 2>/dev/null
}

echo "AlgoTraderAI Setup"
echo "=================="

mkdir -p "$APP_DIR/configs" "$APP_DIR/logs"

# Let the user pick which template to start from
TEMPLATE=$(osascript -e 'button returned of (display dialog "Which strategy do you want to run?" buttons {"MES (CISD-OTE v7)", "MNQ (SuperTrend)", "MES (VWAP)"} default button "MES (CISD-OTE v7)")')

case "$TEMPLATE" in
    "MES (CISD-OTE v7)") SRC="combine-mes" ;;
    "MNQ (SuperTrend)")  SRC="practice-mnq" ;;
    "MES (VWAP)")        SRC="practice-mes" ;;
esac

cp "$APP_DIR/configs.example/$SRC.yaml" "$APP_DIR/configs/bot.yaml"

# Credentials
USERNAME=$(prompt "Enter your TopstepX username:" "")
APIKEY=$(prompt "Enter your TopstepX API key:" "")
ACCOUNT=$(prompt "Enter your TopstepX account ID:" "")

FILE="$APP_DIR/configs/bot.yaml"
sed -i '' "s/^account:.*/account:    \"$ACCOUNT\"/"  "$FILE"
sed -i '' "s/^username:.*/username:   \"$USERNAME\"/" "$FILE"
sed -i '' "s/^apikey:.*/apikey:     \"$APIKEY\"/"    "$FILE"

# Check Docker is installed
if ! command -v docker &>/dev/null; then
    osascript -e 'display dialog "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop and re-run setup." buttons {"OK"} default button "OK"'
    exit 1
fi

cd "$APP_DIR"

echo "Pulling latest AlgoTraderAI image..."
docker compose pull

docker compose up -d

osascript -e 'display dialog "Bot is running! Use Docker Desktop to monitor it." buttons {"OK"} default button "OK"'
echo "Done. Logs are in: $APP_DIR/logs/"
