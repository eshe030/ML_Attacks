import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Simulated XOR Arbiter PUF
def xor_arbiter_puf(challenge):
    return challenge[0] ^ challenge[1]

# Generate simulated PUF responses
def generate_puf_responses(challenge_pairs):
    responses = []
    for challenge in challenge_pairs:
        response = xor_arbiter_puf(challenge)
        responses.append(response)
    return np.array(responses)

# Create training data
num_samples = 1000
challenge_pairs = np.random.randint(0, 2, size=(num_samples, 2))
responses = generate_puf_responses(challenge_pairs)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(challenge_pairs, responses, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Attack simulation
predicted_responses = model.predict(X_test)
attack_accuracy = np.mean(predicted_responses == y_test)
print("Attack Accuracy:", attack_accuracy)
