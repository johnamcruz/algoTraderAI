#!/bin/bash
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

prompt() {
    osascript -e "Tell application \"System Events\" to display dialog \"$1\" default answer \"$2\"" \
              -e "text returned of result" 2>/dev/null
}

echo "AlgoTraderAI Setup"
echo "=================="

# Copy templates into configs/ (gitignored, safe for credentials)
mkdir -p "$APP_DIR/configs"
for cfg in combine-mes practice-mnq practice-mes; do
    cp "$APP_DIR/configs.example/$cfg.yaml" "$APP_DIR/configs/$cfg.yaml"
done

# Credentials (shared across all bots)
USERNAME=$(prompt "Enter your TopstepX username:" "")
APIKEY=$(prompt "Enter your TopstepX API key:" "")

# Accounts
COMBINE_ACCOUNT=$(prompt "Enter your COMBINE account ID:" "")
PRACTICE_ACCOUNT=$(prompt "Enter your PRACTICE account ID:" "")

# Write credentials into combine config
FILE="$APP_DIR/configs/combine-mes.yaml"
sed -i '' "s/^account:.*/account:    \"$COMBINE_ACCOUNT\"/" "$FILE"
sed -i '' "s/^username:.*/username:   \"$USERNAME\"/"       "$FILE"
sed -i '' "s/^apikey:.*/apikey:     \"$APIKEY\"/"           "$FILE"

# Write credentials into practice configs
for cfg in practice-mnq practice-mes; do
    FILE="$APP_DIR/configs/$cfg.yaml"
    sed -i '' "s/^account:.*/account:    \"$PRACTICE_ACCOUNT\"/" "$FILE"
    sed -i '' "s/^username:.*/username:   \"$USERNAME\"/"        "$FILE"
    sed -i '' "s/^apikey:.*/apikey:     \"$APIKEY\"/"            "$FILE"
done

# Check Docker is installed
if ! command -v docker &>/dev/null; then
    osascript -e 'display dialog "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop and re-run setup." buttons {"OK"} default button "OK"'
    exit 1
fi

cd "$APP_DIR"

echo "Pulling latest AlgoTraderAI image..."
docker compose pull

echo ""
CHOICE=$(osascript -e 'button returned of (display dialog "Which bots do you want to run?" buttons {"All 3", "Combine only", "Cancel"} default button "All 3")')

case "$CHOICE" in
    "All 3")
        docker compose up -d
        ;;
    "Combine only")
        docker compose up -d combine-mes
        ;;
    *)
        echo "Cancelled."
        exit 0
        ;;
esac

osascript -e 'display dialog "Bots are running! Use Docker Desktop to monitor them." buttons {"OK"} default button "OK"'
echo "Done. Logs are in: $APP_DIR/logs/"
