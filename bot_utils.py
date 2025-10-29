import logging
import requests


# =========================================================
# CONSTANTS
# =========================================================

MARKET_HUB = "https://rtc.topstepx.com/hubs/market"
BASE_URL = "https://api.topstepx.com/api"

# This map is ESSENTIAL because naming is not consistent.
MICRO_TO_MINI_MAP = {
    # Indices
    "MNQ": "NQ",  # Micro E-mini Nasdaq-100
    "MES": "ES",  # Micro E-mini S&P 500
    "MYM": "YM",  # Micro E-mini Dow
    "M2K": "RTY", # Micro E-mini Russell 2000

    # Metals
    "MGC": "GC",  # Micro Gold
    "SIL": "SI",  # Micro Silver (Note: Parent is SI, not SIZ)
    "MHG": "HG",  # Micro Copper

    # Energy
    "MCL": "CL",  # Micro WTI Crude Oil
    "MNG": "NG",  # Micro Henry Hub Natural Gas

    # Crypto
    "MBT": "BTC", # Micro Bitcoin
    "MET": "ETH", # Micro Ether

    # Micro FX (Maps to their E-Micro parent, e.g., M6E -> 6E)
    "M6A": "6A",
    "M6B": "6B",
    "M6E": "6E",
    # Note: 'E7' is already an E-mini, not a micro.
}

# =========================================================
# LOGGING SETUP
# =========================================================
def setup_logging(level=logging.INFO, log_file=None):
    """Configures basic logging, prioritizing file if specified."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [] # Start with no handlers

    if log_file:
        print(f"Logging configured to file: {log_file}") # Still print confirmation to console
        handlers.append(logging.FileHandler(log_file))
    else:
        # Fallback to console only if no log file is given
        print("Logging configured to console.")
        handlers.append(logging.StreamHandler())

    # If no handlers were added (e.g., log_file was empty string), add console handler
    if not handlers:
         handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True) # Use force=True to allow reconfiguration
    logging.info("--- Log Start ---") # Add a marker for new log session

# =========================================================
# AUTHENTICATION
# =========================================================
def authenticate(base_url, username, api_key):
    """Authenticates and returns a JWT token."""
    auth_url = f"{base_url}/Auth/loginKey"
    payload = {"userName": username, "apiKey": api_key}
    try:
        logging.info("🔐 Authenticating...")
        logging.info(payload)     
        response = requests.post(auth_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('success') and data.get('token'):
            logging.info("✅ Authentication successful!")
            return data['token']
        else:
            logging.error(f"❌ Authentication failed: {data.get('errorMessage', 'Unknown error')}")
            return None
    except Exception as e:
        logging.exception(f"❌ Authentication error: {e}")
        return None

# =========================================================
# Parse Future Symbol
# =========================================================
def parse_future_symbol(contract_name):    
    """
    Parses the base future symbol from a contract name (e.g., "MNQZ5")
    and dynamically maps known Micro contracts to their parent symbol
    (e.g., "NQ", "ES", "GC").
    """
    if not contract_name:
        return None

    # 1. Get the abbreviated core part (e.g., "MNQZ5")
    name_field = contract_name.upper().split('.')[-1]

    # 2. Find the split point between the symbol and the expiry code.
    # We find the index of the *first digit* (the year).
    first_digit_index = -1
    for i, char in enumerate(name_field):
        if char.isdigit():
            first_digit_index = i
            break

    if first_digit_index == -1:
        # No digit found, assume it's just the symbol (e.g., "BTC")
        base_symbol_with_month = name_field
    else:
        # We have the part before the year (e.g., "NQZ", "MGCZ", "6BZ")
        base_symbol_with_month = name_field[:first_digit_index]

    # 3. Strip the month code (the last letter)
    month_codes = "FGHJKMNQUVXZ"
    base_symbol = base_symbol_with_month

    if base_symbol and base_symbol[-1] in month_codes:
        base_symbol = base_symbol[:-1] # "NQZ" -> "NQ", "MGCZ" -> "MGC"

    # 4. Apply the Micro-to-Mini mapping
    if base_symbol in MICRO_TO_MINI_MAP:
        return MICRO_TO_MINI_MAP[base_symbol]

    # 5. Return the parsed base symbol if not in the map
    return base_symbol