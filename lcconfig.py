import json

CONFIG_FILE = "lcconfig.json"

# Load existing config
def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# Save updated config
def save_config(new_data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(new_data, f, indent=4)

# Load data on import
config = load_config()

# Expose variables
username = config["username"]
password = config["password"]
host = config["host"]
port = config['port']
uri = config['uri']
db = config['DB']
