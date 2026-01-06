from config import GAMMA

def returns(rewards):
    returns = []
    G = 0 

    for reward in reversed(rewards):
        G = reward + GAMMA * G
        returns.insert(0, G)
    
    return returns