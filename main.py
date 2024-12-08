import random
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

from aRUn.basic import MainGo
from config.variables import text_name

app = Flask(__name__)
CORS(app)
mian = MainGo()
emails_df = pd.read_csv("./dataset/web_mails.csv")


# Define the API route
@app.route('/post_emails', methods=['POST'])
def classify_email():
    try:
        # Get email content from the request
        data = request.json
        email_content = data.get('content', '')#supposed to be list of str

        if not email_content:
            return jsonify({"error": "Email content is required"}), 400
        mian.input_email(email_content)
        mian.process()
        prediction = mian.evaluate()
        spam_idx = [idx_ for idx_, value_ in enumerate(prediction) if value_ == "spam"]
        # ham_idx = [idx_ for idx_, value_ in enumerate(prediction) if value_ == "ham"]
        return jsonify({"spams": spam_idx})


    except Exception as e:
        return jsonify({"error": str(e)}), 500


# GET route to retrieve stored emails
@app.route('/get_emails', methods=['GET'])
def get_emails():
    try:
        # Get the number of emails to return from the query parameter
        # n = random.randint(1, min(6, len(emails_df)//2))
        n = 10

        sampled_emails = emails_df.sample(n=n)
        sampled_emails.drop(columns="Label", inplace=True)
        sampled_emails = sampled_emails.to_numpy().reshape(-1)
        sampled_emails = sampled_emails.tolist()
        #mian.input_email(sampled_emails[text_name])
        #mian.process()
        #label_guess = mian.evaluate()
        #sampled_emails["prediction"] = label_guess

        return jsonify({"Message":sampled_emails})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
