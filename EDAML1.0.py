import pandas as pd
import requests
from web3 import Web3
from collections import defaultdict
import datetime
import os 
import numpy as np
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from typing import Tuple

infura_url = os.environ['INFURA_URL']
w3 = Web3(Web3.HTTPProvider(infura_url))

etherscan_api_key = os.environ['ETHERSCAN_API_KEY']

# Uses w3 object to interact with the Ethereum blockchain
print("is_connected:", w3.provider.is_connected())

# List of user Ethereum addresses
user_addresses = ['0xe9c53645DA478509BdBC09e8A13bc6194CbD9Db0']

# ERC20 token contract addresses and ABI (Application Binary Interface)
token_contracts = {
    'TokenWrapped Ether (WETH)': {
        'address': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'abi': [
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
token_wrapped_ether_WETH = []  
smart_contract_interactions = []
token_ownership_over_time = []
total_spent_over_time = []
num_transactions_over_time = []
num_staking_interactions_over_time = []
total_staking_time = []

# Get transaction history using Etherscan API
infura_url = os.environ['INFURA_URL']
w3 = Web3(Web3.HTTPProvider(infura_url))
etherscan_api_key = os.environ['ETHERSCAN_API_KEY']
etherscan_base_url = "https://api.etherscan.io/api"
endpoint = "?module=account&action=txlist&address=" + str(0xe9c53645DA478509BdBC09e8A13bc6194CbD9Db0) + "&startblock=0&endblock=99999999&sort=asc&apikey=" + etherscan_api_key

complete_url = etherscan_base_url + endpoint
response = requests.get(complete_url)

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

# Get transaction history using Etherscan API
infura_url = os.environ['INFURA_URL']
w3 = Web3(Web3.HTTPProvider(infura_url))
etherscan_api_key = os.environ['ETHERSCAN_API_KEY']
etherscan_base_url = "https://api.etherscan.io/api"
endpoint = "?module=account&action=txlist&address=" + address + "&startblock=0&endblock=99999999&sort=asc&apikey=" + etherscan_api_key

complete_url = etherscan_base_url + endpoint
response = requests.get(complete_url)

# Transaction Analysis
for address in user_addresses:
    address_token_ownership_over_time = defaultdict(list)
    address_total_spent = 0
    address_num_transactions = 0
    address_num_staking_interactions = 0
    address_total_staking_time = 0

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

# Get transaction history using Etherscan API
infura_url = os.environ['INFURA_URL']
w3 = Web3(Web3.HTTPProvider(infura_url))
etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')
etherscan_base_url = "https://api.etherscan.io/api"
endpoint = "?module=account&action=txlist&address=" + address + "&startblock=0&endblock=99999999&sort=asc&apikey=" + etherscan_api_key

complete_url = etherscan_base_url + endpoint
response = requests.get(complete_url)

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
            address_num_staking_interactions += 1
            staking_start_time = datetime.datetime.fromtimestamp(tx['timestamp'])
            staking_end_time = datetime.datetime.fromtimestamp(tx['next_timestamp'])
            address_total_staking_time += (staking_end_time - staking_start_time).total_seconds()

        for token_name, token_data in token_contracts.items():
            token_contract = w3.eth.contract(address=token_data['address'], abi=token_data['abi'])
            latest_block = w3.eth.block_number
            token_balance = token_contract.functions.balanceOf(address).call(block_identifier=latest_block - 10)
            address_token_ownership_over_time[token_name].append(token_balance)

token_ownership_over_time.append(address_token_ownership_over_time)
total_spent_over_time.append(address_total_spent)
num_transactions_over_time.append(address_num_transactions)
num_staking_interactions_over_time.append(address_num_staking_interactions)
total_staking_time.append(address_total_staking_time)

# Creates a DataFrame to store the collected data
data = {
'address': user_addresses,
'wallet_balance': wallet_balances,
'transaction_count': transaction_counts,
'unique_tokens': token_diversity,
'token_balances': token_balances,
'TokenWrapped Ether (WETH)': token_balances,
'token_ownership_over_time': pd.Series(token_ownership_over_time),
'total_spent_over_time': pd.Series(total_spent_over_time),
'num_transactions_over_time': pd.Series(num_transactions_over_time),
'num_staking_interactions_over_time': pd.Series(num_staking_interactions_over_time),
'total_staking_time': pd.Series(total_staking_time)
}

# Creates a DataFrame to store the collected data
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

# Displays the DataFrame
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "left")
print(df)

# Saves the DataFrame to a CSV file
df.to_csv('ethereum_data.csv', index=False)

def train_and_evaluate_models(data: pd.DataFrame) -> None:

    # Loads the dataset
    data_df = pd.read_csv('ethereum_data.csv')

    # Preprocesses the data
    X = data_df.iloc[:, :1].values
    y = data_df.iloc[:, -1].values

    # Splits the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calls the function to train and evaluate the models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train_scaled, y_train)
    nn = Sequential()
    nn.add(Dense(units=16, activation='relu', input_dim=X_train_scaled.shape[1]))
    nn.add(Dense(units=8, activation='relu'))
    nn.add(Dense(units=1, activation='sigmoid'))

    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)

    def ensemble_predict(X):
        rf_prediction = rf.predict(X)
        svm_prediction = svm.predict(X)
        xgb_prediction = xgb.predict(X)
        nn_prediction = (nn.predict(X) > 0.5).astype(int).flatten()

        combined_prediction = np.round((rf_prediction + svm_prediction + xgb_prediction + nn_prediction) / 4).astype(int)

        return combined_prediction

    ensemble_predictions = ensemble_predict(X_test_scaled)
    accuracy = np.mean(ensemble_predictions == y_test)

    print("Ensemble model accuracy: {:.4f}".format(accuracy))