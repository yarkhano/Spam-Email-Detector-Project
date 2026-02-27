# import required libraries
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI(
    title="Spam Email Detector API",
    description="Created an API for spam detection project",
    version="1.0.0"
)

# Initialize variables as None so they can be accessed globally
model = None
vectorizer = None

# load the model and vectorizer
try:
    with open("spam_mail_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("spam_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
        print("Model and vectorizer loaded successfully")

except FileNotFoundError as e:
    print(f"Error loading model: {e}")


# Building API for Model

# Creating class, to make sure the input is as expected(string).
class input_message(BaseModel):
    message: str


# Creating class, to make sure the output is as expected(string).
class output_message(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Shield AI</title>
        <style>
            :root {
                --primary: #4f46e5;
                --primary-hover: #4338ca;
                --bg: #f3f4f6;
                --card-bg: #ffffff;
                --text-main: #1f2937;
            }
            body {
                font-family: 'Segoe UI', Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background: var(--card-bg);
                padding: 2.5rem;
                border-radius: 12px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 500px;
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 1rem;
                margin-top: 1rem;
                margin-bottom: 1rem;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                resize: none;
                font-family: inherit;
                box-sizing: border-box;
            }
            textarea:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
            }
            button {
                width: 100%;
                padding: 0.75rem;
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: bold;
                cursor: pointer;
            }
            button:hover { background-color: var(--primary-hover); }
            #result {
                margin-top: 1.5rem;
                padding: 1rem;
                border-radius: 8px;
                font-weight: bold;
                font-size: 1.2rem;
                display: none;
            }
            .spam { background-color: #fee2e2; color: #991b1b; }
            .safe { background-color: #dcfce7; color: #166534; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Spam Shield AI</h1>
            <p>Paste an email or message below to test the AI model.</p>

            <textarea id="emailInput" placeholder="Paste message text here..."></textarea>
            <button onclick="checkSpam()">Analyze Message</button>

            <div id="result"></div>

            <div style="margin-top: 1.5rem; font-size: 0.9rem;">
                <a href="/docs" style="color: var(--primary); text-decoration: none;">⚙️ View API Developer Docs</a>
            </div>
        </div>

        <script>
            async function checkSpam() {
                const text = document.getElementById("emailInput").value;
                const resultBox = document.getElementById("result");

                if (!text.trim()) {
                    alert("Please enter some text!");
                    return;
                }

                resultBox.style.display = "block";
                resultBox.className = "";
                resultBox.innerText = "Analyzing...";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ message: text })
                    });

                    const data = await response.json();

                    if (data.message === "Spam") {
                        resultBox.className = "spam";
                        resultBox.innerText = "🚨 WARNING: This is SPAM";
                    } else {
                        resultBox.className = "safe";
                        resultBox.innerText = "✅ SAFE: No spam detected";
                    }
                } catch (error) {
                    resultBox.innerText = "❌ Error connecting to server.";
                }
            }
        </script>
    </body>
    </html>
    """


# This request will accept the input message from the user
@app.post("/predict", response_model=output_message)
async def predict_spam(input_data: input_message):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="The 'Brain' or 'Translator' files are missing!")

    raw_text = input_data.message  # .message came from above pydantic validation

    vectorized_text = vectorizer.transform([raw_text])
    prediction = model.predict(vectorized_text)

    label = "Spam" if prediction[0] == 1 else "Safe"

    return {"message": label}


# Running the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)