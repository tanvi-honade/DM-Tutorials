import joblib
from preprocess import clean_text

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_sentiment(text):
    """
    Predict sentiment of the given text.
    """

    # Clean the input
    cleaned_text = clean_text(text)

    # Convert to TF-IDF
    vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector)[0]

    return prediction


if __name__ == "__main__":

    while True:

        print("\n==============================")
        user_input = input("Enter Election Text (or type exit): ")

        if user_input.lower() == "exit":
            print("Program Closed.")
            break

        sentiment = predict_sentiment(user_input)

        print("------------------------------")
        print("Predicted Sentiment :", sentiment)
        print("==============================")