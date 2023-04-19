import pandas as pd
import requests
from web3 import Web3
from collections import defaultdict
import datetime
import os

# Add the function definition at the beginning of the script
def train_and_evaluate_models(data):
    # Import necessary libraries
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error

# Replace the URL with the one you got from Infura
infura_url = os.environ.get('INFURA_URL')
w3 = Web3(Web3.HTTPProvider(infura_url))

etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')

# Now you can use w3 object to interact with the Ethereum blockchain
print("is_connected:", w3.provider.is_connected())

# List of user Ethereum addresses
user_addresses = ['0xe9c53645DA478509BdBC09e8A13bc6194CbD9Db0']

# ERC20 token contract addresses and ABI (Application Binary Interface)
token_contracts = {
    'TokenWrapped Ether (WETH)': {
        'address': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'abi': 
        [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [
                    {
                        "internalType": "uint8",
                        "name": "",
                        "type": "uint8"
                    }
                ],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {
                        "internalType": "address",
                        "name": "_owner",
                        "type": "address"
                    }
                ],
                "name": "balanceOf",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "balance",
                        "type": "uint256"
                    }
                ],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            }
        ]
    }
}

# Initialize data collection variables
wallet_balances = []
transaction_counts = []
token_diversity = []
token_balances = defaultdict(list)
token_wrapped_ether_WETH = []  # Modified variable name
smart_contract_interactions = []
token_ownership_over_time = []
total_spent_over_time = []
num_transactions_over_time = []
num_staking_interactions_over_time = []
total_staking_time = []

# Fetch wallet balances, transaction counts, and token balances
for address in user_addresses:
    balance = w3.eth.get_balance(address)
    balance_ether = w3.from_wei(balance, 'ether')
    wallet_balances.append(balance_ether)

    transaction_count = w3.eth.get_transaction_count(address)
    transaction_counts.append(transaction_count)

unique_tokens = 0
for token_name, token_data in token_contracts.items():
    token_contract = w3.eth.contract(address=token_data['address'], abi=token_data['abi'])
    latest_block = w3.eth.block_number
    token_balance = token_contract.functions.balanceOf(address).call(block_identifier=latest_block)
    if token_balance > 0:
        unique_tokens += 1
        token_balances[token_name].append(token_balance)
token_diversity.append(unique_tokens)

# Analyze transactions
for address in user_addresses:
    # Initialize per-address data collection variables
    address_token_ownership_over_time = defaultdict(list)
    address_total_spent = 0
    address_num_transactions = 0
    address_num_staking_interactions = 0
    address_total_staking_time = 0

# Get transaction history using Etherscan API
url = f'https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={etherscan_api_key}'
response = requests.get(url)
if response.status_code == 200:
        json_data = response.json()
        if json_data['message'] == 'OK':
            transaction_history = json_data['result']
        else:
            print("Error fetching transaction history from Etherscan API")
            transaction_history = []
else:
        print("Error fetching transaction history from Etherscan API")
        transaction_history = []

for tx in transaction_history:
    tx_value = w3.from_wei(int(tx['value']), 'ether')
    address_total_spent += tx_value
    address_num_transactions += 1

    if tx['to'] in token_contracts.values():
        # Token transfer event
        address_num_staking_interactions += 1
        staking_start_time = datetime.datetime.fromtimestamp(tx['timestamp'])
        staking_end_time = datetime.datetime.fromtimestamp(tx['next_timestamp'])
        address_total_staking_time += (staking_end_time - staking_start_time).total_seconds()

    # Update token balances
    for token_name, token_data in token_contracts.items():
        token_contract = w3.eth.contract(address=token_data['address'], abi=token_data['abi'])
        latest_block = w3.eth.block_number
        token_balance = token_contract.functions.balanceOf(address).call(block_identifier=latest_block)
        address_token_ownership_over_time[token_name].append(token_balance)

token_ownership_over_time.append(address_token_ownership_over_time)
total_spent_over_time.append(address_total_spent)
num_transactions_over_time.append(address_num_transactions)
num_staking_interactions_over_time.append(address_num_staking_interactions)
total_staking_time.append(address_total_staking_time)

# Create a DataFrame to store the collected data
data = {
'address': user_addresses,
'wallet_balance': wallet_balances,
'transaction_count': transaction_counts,
'unique_tokens': token_diversity,
'token_balances': token_balances,
'TokenWrapped Ether (WETH)': token_balances,
'token_ownership_over_time': token_ownership_over_time,
'total_spent_over_time': total_spent_over_time,
'num_transactions_over_time': num_transactions_over_time,
'num_staking_interactions_over_time': num_staking_interactions_over_time,
'total_staking_time': total_staking_time
}

# Create a DataFrame to store the collected data
data = {
    'address': pd.Series(user_addresses),
    'wallet_balance': pd.Series(wallet_balances),
    'transaction_count': pd.Series(transaction_counts),
    'unique_tokens': pd.Series(token_diversity),
    'token_balances': pd.Series(token_balances),
    'TokenWrapped Ether (WETH)': pd.Series(token_balances['TokenWrapped Ether (WETH)']),
    'token_ownership_over_time': pd.Series(token_ownership_over_time),
    'total_spent_over_time': pd.Series(total_spent_over_time),
    'num_transactions_over_time': pd.Series(num_transactions_over_time),
    'num_staking_interactions_over_time': pd.Series(num_staking_interactions_over_time),
    'total_staking_time': pd.Series(total_staking_time)
}

df = pd.DataFrame(data)

# Display the DataFrame
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "left")
print(df)

# Save the DataFrame to a CSV file
df.to_csv('ethereum_data.csv', index=False)

# Call the function to train and evaluate the models
train_and_evaluate_models(df)