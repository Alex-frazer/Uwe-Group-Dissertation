# Define the membership functions for each emotion
def low_joy(x):
    return max(0, min((1 - x) / 0.2, 1)) if 0 <= x <= 1 else 0

def medium_joy(x):
    return max(0, min((x - 0.2) / 0.3, (0.8 - x) / 0.3, 1)) if 0 <= x <= 1 else 0

def high_joy(x):
    return max(0, min((x - 0.7) / 0.3, 1)) if 0 <= x <= 1 else 0

def low_trust(x):
    return max(0, min((1 - x) / 0.3, 1)) if 0 <= x <= 1 else 0

def medium_trust(x):
    return max(0, min((x - 0.3) / 0.4, (0.7 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_trust(x):
    return max(0, min((x - 0.7) / 0.3, 1)) if 0 <= x <= 1 else 0

def low_fear(x):
    return max(0, min((1 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def medium_fear(x):
    return max(0, min((x - 0.4) / 0.4, (0.6 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_fear(x):
    return max(0, min((x - 0.6) / 0.4, 1)) if 0 <= x <= 1 else 0

def low_surprise(x):
    return max(0, min((1 - x) / 0.3, 1)) if 0 <= x <= 1 else 0

def medium_surprise(x):
    return max(0, min((x - 0.3) / 0.4, (0.7 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_surprise(x):
    return max(0, min((x - 0.7) / 0.3, 1)) if 0 <= x <= 1 else 0

def low_sadness(x):
    return max(0, min((1 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def medium_sadness(x):
    return max(0, min((x - 0.4) / 0.4, (0.6 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_sadness(x):
    return max(0, min((x - 0.6) / 0.4, 1)) if 0 <= x <= 1 else 0

def low_disgust(x):
    return max(0, min((1 - x) / 0.3, 1)) if 0 <= x <= 1 else 0

def medium_disgust(x):
    return max(0, min((x - 0.3) / 0.4, (0.7 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_disgust(x):
    return max(0, min((x - 0.7) / 0.3, 1)) if 0 <= x <= 1 else 0

def low_anger(x):
    return max(0, min((1 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def medium_anger(x):
    return max(0, min((x - 0.4) / 0.4, (0.6 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_anger(x):
    return max(0, min((x - 0.6) / 0.4, 1)) if 0 <= x <= 1 else 0

def low_anticipation(x):
    return max(0, min((1 - x) / 0.3, 1)) if 0 <= x <= 1 else 0

def medium_anticipation(x):
    return max(0, min((x - 0.3) / 0.4, (0.7 - x) / 0.4, 1)) if 0 <= x <= 1 else 0

def high_anticipation(x):
    return max(0, min((x - 0.7) / 0.3, 1)) if 0 <= x <= 1 else 0

# Example function to get fuzzy membership values
def get_membership_values(score):
    return {
        'Joy': {
            'Low': low_joy(score),
            'Medium': medium_joy(score),
            'High': high_joy(score)
        },
        'Trust': {
            'Low': low_trust(score),
            'Medium': medium_trust(score),
            'High': high_trust(score)
        },
        'Fear': {
            'Low': low_fear(score),
            'Medium': medium_fear(score),
            'High': high_fear(score)
        },
        'Surprise': {
            'Low': low_surprise(score),
            'Medium': medium_surprise(score),
            'High': high_surprise(score)
        },
        'Sadness': {
            'Low': low_sadness(score),
            'Medium': medium_sadness(score),
            'High': high_sadness(score)
        },
        'Disgust': {
            'Low': low_disgust(score),
            'Medium': medium_disgust(score),
            'High': high_disgust(score)
        },
        'Anger': {
            'Low': low_anger(score),
            'Medium': medium_anger(score),
            'High': high_anger(score)
        },
        'Anticipation': {
            'Low': low_anticipation(score),
            'Medium': medium_anticipation(score),
            'High': high_anticipation(score)
        }
    }

# Example usage
if __name__ == "__main__":
    sentiment_score = 0.5  # Example sentiment score
    memberships = get_membership_values(sentiment_score)
    
    for emotion, levels in memberships.items():
        print(f"{emotion}:")
        for level, value in levels.items():
            print(f"  {level}: {value:.2f}")
            
            # Example usage
if __name__ == "__main__":
    sentiment_score = 0.632  # Example sentiment score
    memberships = get_membership_values(sentiment_score)
    
    for emotion, levels in memberships.items():
        print(f"{emotion}:")
        for level, value in levels.items():
            print(f"  {level}: {value:.2f}")