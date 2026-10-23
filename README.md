# 🚗 Vehicle Price Prediction System | AutoValuate AI

### AI-Powered Vehicle Valuation and Price Prediction

An interactive Machine Learning web application designed to estimate vehicle market prices using important vehicle specifications, performance characteristics, and ownership history.

**AutoValuate AI** provides a modern vehicle valuation experience that allows users to enter vehicle details, obtain a predicted price, explore estimated valuation metrics, and view prices in different currencies.

<p align="center">
  <a href="https://vehicle-price-prediction-system.vercel.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Visit%20Application-brightgreen?style=for-the-badge&logo=vercel" alt="Live Demo">
  </a>
  <a href="https://github.com/Siddhesh030/Vehicle_Price_Prediction_System">
    <img src="https://img.shields.io/badge/GitHub-View%20Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

## 🌐 Live Demo

**Try the application here:**

### 👉 [Vehicle Price Prediction System — Live Application](https://vehicle-price-prediction-system.vercel.app/)

Explore the interactive vehicle valuation interface, enter vehicle specifications, and calculate an estimated vehicle price.

**GitHub Repository:** [Siddhesh030/Vehicle_Price_Prediction_System](https://github.com/Siddhesh030/Vehicle_Price_Prediction_System)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Vehicle Information and Input Parameters](#-vehicle-information-and-input-parameters)
- [How the System Works](#-how-the-system-works)
- [Machine Learning Approach](#-machine-learning-approach)
- [Technology Stack](#-technology-stack)
- [Application Architecture](#-application-architecture)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application Locally](#-running-the-application-locally)
- [Prediction Workflow](#-prediction-workflow)
- [Deployment](#-deployment)
- [Use Cases](#-use-cases)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 📖 Project Overview

The **Vehicle Price Prediction System** is a Machine Learning-powered web application that estimates the market value of a vehicle based on its specifications, condition-related information, and usage history.

Vehicle valuation can be challenging because the price of a vehicle depends on multiple factors, including its manufacturer, model, manufacturing year, mileage, engine capacity, horsepower, transmission, fuel type, and ownership history.

This project combines a Python-based prediction backend with an interactive web interface to make vehicle valuation more accessible.

The application uses a trained Random Forest model stored in `RFA_model.pkl` to generate predictions when the model is successfully loaded. The Flask application processes vehicle information and returns the estimated price to the interface.

The application also includes a currency selection feature, allowing the displayed valuation to be converted into supported currencies.

### 🎯 Main Goal

To develop an accessible, interactive, and data-driven vehicle valuation application that demonstrates the practical use of Machine Learning in an automotive pricing scenario.

---

## ❓ Problem Statement

Estimating the market value of a vehicle can be difficult for buyers, sellers, and automotive professionals.

Traditional valuation methods may depend on manual comparisons, individual judgment, or limited market information. These approaches can produce inconsistent estimates and require additional research.

A vehicle's value is influenced by several characteristics, such as:

- Manufacturer and model
- Manufacturing year
- Engine capacity
- Mileage
- Horsepower and torque
- Fuel type
- Transmission type
- Accident history
- Service history
- Vehicle body type
- Drivetrain
- Fuel efficiency
- Number of previous owners
- Location

The objective of this project is to use these vehicle attributes as inputs to a Machine Learning-based prediction system to generate an estimated market value.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. **Machine Learning Integration:** Integrate a trained Random Forest model into a functional web application.

2. **Interactive User Interface:** Provide a convenient interface for entering vehicle specifications and viewing valuation results.

3. **Multi-Feature Prediction:** Use multiple vehicle characteristics to support price estimation.

4. **Backend Integration:** Implement a Flask application to process prediction requests and return results.

5. **Currency Conversion:** Support displaying estimated prices in multiple currencies.

6. **Valuation Insights:** Present price-related indicators and vehicle valuation information through the application's interface.

7. **Web Deployment:** Make the application accessible through a publicly hosted web interface.

8. **Practical Application:** Demonstrate how Machine Learning can be applied to a real-world automotive pricing problem.

---

## ✨ Key Features

### 🤖 1. Machine Learning-Based Price Prediction

The application integrates a saved Random Forest model for predicting vehicle prices from supplied vehicle specifications.

When the trained model is available and loads successfully, the backend uses it to calculate a prediction.

### 🚘 2. Comprehensive Vehicle Information

Users can provide multiple vehicle characteristics to support the valuation process.

These include vehicle identification, technical specifications, usage information, and condition-related attributes.

### 📊 3. Interactive Valuation Dashboard

The application presents an interactive valuation interface with vehicle input fields and a dedicated results area.

Users can enter vehicle information and initiate the valuation process through the prediction button.

### 💱 4. Multiple Currency Options

The application includes support for several currencies, including:

- USD — US Dollar
- INR — Indian Rupee
- EUR — Euro
- GBP — British Pound
- AED — United Arab Emirates Dirham
- CAD — Canadian Dollar

The backend converts the predicted base price using the currency rates configured in the application.

**Note:** The configured conversion rates should not be treated as live exchange rates unless they are connected to a live exchange-rate service.

### 📉 5. Vehicle Valuation Indicators

The interface includes valuation-related indicators, such as:

- Estimated low and high price ranges
- Estimated vehicle value
- Depreciation information
- Projected residual value

These interface indicators should be interpreted according to their actual calculation logic and should not automatically be considered statistically validated model outputs.

### 🎨 6. Modern Interface

The application is presented under the name AutoValuate AI and provides a dedicated vehicle technical profile form and valuation results panel.

The interface includes selectable vehicle categories, numeric input fields, currency selection, and a valuation action.

### ⚡ 7. Flask Backend

Flask handles the application's HTTP requests, serves the interface, processes prediction inputs, and returns prediction results as JSON.

### ☁️ 8. Web Deployment

The application is configured for deployment on Vercel, making it accessible through the live application link.

### 🧩 9. Model Loading and Fallback Logic

The backend attempts to load the saved model from `RFA_model.pkl`.

The prediction route also contains a fallback calculation when the model is unavailable. Predictions produced by this fallback are heuristic estimates rather than outputs from the trained Random Forest model.

---

## 🚙 Vehicle Information and Input Parameters

The application accepts vehicle attributes that describe the vehicle's technical specifications, usage, and condition.

| Parameter | Description |
|---|---|
| Make | Vehicle manufacturer or brand |
| Model | Vehicle model |
| Year | Manufacturing year |
| Fuel Type | Petrol, diesel, hybrid, or electric |
| Transmission | Automatic, manual, or semi-automatic |
| Engine Size | Engine capacity in liters |
| Mileage | Distance traveled by the vehicle |
| Horsepower | Engine power |
| Torque | Engine torque |
| Owners | Number of previous owners |
| Accident History | Reported accident condition |
| Service History | Vehicle maintenance history |
| Color | Vehicle exterior color |
| Body Type | Vehicle category, such as SUV, sedan, or hatchback |
| Drivetrain | AWD, FWD, RWD, or 4WD |
| Fuel Efficiency | Fuel economy in kilometers per liter |
| Location | General location category |

The available choices for categorical fields are defined in the application code.

The prediction results depend on the model's training data, the feature representation expected by the model, and the quality of the supplied inputs.

---

## ⚙️ How the System Works

The application's prediction workflow can be summarized in the following stages.

### Step 1: Enter Vehicle Details

The user opens the application and supplies the requested vehicle information.

Examples include manufacturer, model, manufacturing year, mileage, engine size, horsepower, and transmission type.

### Step 2: Submit the Valuation Request

The user initiates the prediction through the application's valuation button.

The interface sends the supplied vehicle information to the Flask backend.

### Step 3: Process Input Features

The backend receives the request and prepares the vehicle attributes in the feature order expected by the prediction model.

The `encode_inputs()` function is responsible for preparing the model input representation.

### Step 4: Generate the Prediction

When the saved model is loaded, the backend calls its prediction method to estimate the vehicle price.

If the model is unavailable, the application contains a fallback calculation based on selected numeric vehicle attributes.

The two prediction paths are different and should not be interpreted as equivalent.

### Step 5: Convert the Currency

The application uses its configured currency conversion rates to calculate the displayed value in the selected currency.

### Step 6: Return and Display the Result

The backend returns the prediction information in a JSON response, and the interface updates the displayed valuation.

### Workflow Summary

```text
          User Opens Application
                    |
                    v
          Enters Vehicle Details
                    |
                    v
          Submits Valuation Form
                    |
                    v
           Flask Backend (app.py)
                    |
                    v
           Prepare Input Features
                    |
                    v
         Is Trained Model Available?
                    |
             +------+------+
             |             |
            Yes            No
             |             |
             v             v
      Random Forest    Fallback Price
        Prediction      Calculation
             |             |
             +------+------+
                    |
                    v
          Apply Currency Conversion
                    |
                    v
           Return JSON Response
                    |
                    v
          Display Estimated Value
```

---

## 🧠 Machine Learning Approach

### Random Forest Model

The repository contains a serialized model named `RFA_model.pkl`, which the application loads using Python's pickle functionality.

The code uses a Random Forest-style model for vehicle price prediction when the model loads successfully.

**Random Forest** is a supervised Machine Learning technique that combines predictions from multiple decision trees. It can model nonlinear relationships between input features and a numerical target.

In a vehicle pricing application, this type of model can learn relationships between vehicle specifications and historical prices when trained on an appropriate dataset.

### Input Features

The application's feature preparation function defines 17 input features:

1. Make
2. Model
3. Year
4. Fuel Type
5. Transmission
6. Engine Size
7. Mileage
8. Horsepower
9. Torque
10. Owners
11. Accident History
12. Service History
13. Color
14. Body Type
15. Drivetrain
16. Fuel Efficiency
17. Location

The feature order and representation must remain consistent with the format used to train the saved model.

### Prediction Process

The application follows this general prediction process:

1. Collect vehicle attributes.
2. Prepare the input data in the expected feature order.
3. Pass the prepared input to the loaded model.
4. Obtain the predicted price.
5. Convert the result into the selected currency.
6. Return the result to the user interface.

### Model Evaluation

The repository does not currently provide verified model evaluation metrics in its documented application files.

For a complete Machine Learning evaluation, the model should be assessed on held-out test data using suitable regression metrics.

Recommended metrics include:

| Metric | Purpose |
|---|---|
| MAE (Mean Absolute Error) | Measures the average absolute difference between actual and predicted prices |
| RMSE (Root Mean Squared Error) | Gives greater weight to larger prediction errors |
| R² Score | Measures the model's fit relative to a baseline based on the target mean |
| MAPE (Mean Absolute Percentage Error) | Measures percentage prediction error where the target values make it appropriate |

These metrics should be calculated from actual evaluation results before adding numerical performance claims to the README.

### Important Model Note

A successfully generated prediction does not, by itself, establish the model's accuracy.

The quality of predictions depends on the training dataset, preprocessing, feature encoding, model evaluation, and whether the input vehicles resemble the examples used during training.

---

## 🛠️ Technology Stack

The following technologies are present in the project or its deployment configuration.

| Technology | Purpose |
|---|---|
| Python | Core application programming language |
| Flask | Web framework and prediction backend |
| NumPy | Numerical computing |
| Pandas | Data manipulation dependency |
| Scikit-learn | Machine Learning dependency |
| Pickle | Loading the serialized model |
| HTML | Web interface structure |
| CSS | Interface styling |
| JavaScript | Client-side interactions |
| JSON | Communication between the interface and backend |
| Vercel | Web application deployment |
| Git | Version control |
| GitHub | Source code hosting |

The Python dependencies are listed in `requirements.txt`.

The interface is embedded in the Flask application using a template string rather than being maintained as a separate frontend directory in the current repository.

---

## 🏗️ Application Architecture

The project uses a compact application architecture that combines the web interface, backend logic, and serialized Machine Learning model.

### 1. Presentation Layer

The interface is defined within `app.py` and rendered by Flask.

It allows users to enter vehicle details, select a currency, and view valuation information.

### 2. Application Layer

Flask manages HTTP requests and application routes.

The primary routes include:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Displays the vehicle valuation interface |
| `/predict` | POST | Processes a vehicle prediction request |

### 3. Machine Learning Layer

The model is stored in `RFA_model.pkl`.

The backend loads this artifact and calls the model's prediction method when available.

### 4. Currency Conversion Layer

The backend maintains a mapping of supported currencies and configured conversion rates.

It uses the selected currency to convert the predicted base price for display.

### 5. Deployment Layer

The repository contains `vercel.json`, which provides deployment configuration for Vercel.

The application also exposes `app_handler` as an alias of the Flask application for its deployment setup.

---

## 📁 Project Structure

The current repository has the following core files:

```text
Vehicle_Price_Prediction_System/
│
├── app.py
│   ├── Flask application
│   ├── Embedded HTML, CSS, and JavaScript interface
│   ├── Vehicle input configuration
│   ├── Model loading and input preparation
│   ├── Prediction route
│   └── Currency conversion logic
│
├── RFA_model.pkl
│   └── Serialized Machine Learning model
│
├── requirements.txt
│   └── Python dependency definitions
│
├── vercel.json
│   └── Vercel deployment configuration
│
└── README.md
    └── Project documentation
```

**File responsibilities:**

- **`app.py`** — Main application file containing the Flask backend, embedded interface, vehicle input definitions, prediction workflow, and currency conversion.
- **`RFA_model.pkl`** — Serialized model artifact used to generate predictions when it can be loaded successfully.
- **`requirements.txt`** — Lists the Python packages required by the application.
- **`vercel.json`** — Contains deployment configuration for Vercel.
- **`README.md`** — Documents the project, its functionality, setup instructions, and architecture.

---

## 💻 Installation and Setup

Follow these instructions to run the project locally.

### Prerequisites

Install the following tools before proceeding:

- Python
- pip
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Siddhesh030/Vehicle_Price_Prediction_System.git
```

Move into the project directory:

```bash
cd Vehicle_Price_Prediction_System
```

### Step 2: Create a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows — Command Prompt:**

```bash
venv\Scripts\activate
```

**Windows — PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project's requirements include Flask, NumPy, Pandas, Scikit-learn, and Gunicorn.

### Step 5: Verify the Model File

Ensure that `RFA_model.pkl` is present in the project directory.

The application attempts to load this file when it starts.

If the model is missing or cannot be loaded, the application may use its fallback pricing calculation instead of the trained model.

---

## ▶️ Running the Application Locally

Once the dependencies have been installed, start the Flask application:

```bash
python app.py
```

The application is configured to run on port `5000`.

Open the following address in your browser:

**http://127.0.0.1:5000**

You should see the vehicle valuation interface.

### Testing the Prediction Endpoint

The prediction route is:

```text
POST /predict
```

It expects a JSON request containing vehicle attributes.

For example, the request can contain fields such as:

```json
{
  "Make": "Toyota",
  "Model": "Camry",
  "Year": 2022,
  "Fuel_Type": "Petrol",
  "Transmission": "Automatic",
  "Engine_Size": 2.5,
  "Mileage": 25000,
  "Horsepower": 203,
  "Torque": 250,
  "Owners": 1,
  "Accident_History": "None",
  "Service_History": "Full",
  "Color": "Pearl White",
  "Body_Type": "Sedan",
  "Drivetrain": "FWD",
  "Fuel_Efficiency": 14.2,
  "Location": "Urban",
  "Currency": "USD"
}
```

This is an illustrative request using fields defined by the application. The example values are not a verified test case, and the resulting price is not specified because it depends on the model artifact and execution environment.

The endpoint returns a JSON response containing a success indicator and price information when the request is processed successfully.

**Note:** Use the application's interface or a suitable HTTP client to submit the request as JSON. The request must follow the feature format expected by the backend.

---

## 🔄 Prediction Workflow

The application uses the following general workflow to estimate a vehicle's price:

### Input Collection

Vehicle information is collected from the user through the web interface.

### Feature Preparation

The backend organizes the input fields into the expected feature sequence.

### Model Inference

If the saved model is available, the application invokes the model's prediction function.

### Currency Conversion

The predicted base price is converted using the configured rate for the selected currency.

### Result Presentation

The backend returns a JSON response, and the interface displays the resulting valuation.

### Fallback Behavior

When the trained model cannot be loaded, the prediction endpoint contains a heuristic calculation based on horsepower, manufacturing year, engine size, and mileage.

This fallback is not a substitute for the trained model and should be clearly distinguished from genuine Machine Learning predictions.

---

## ☁️ Deployment

The application is configured for deployment on Vercel.

### Live Deployment

Access the deployed application here:

**[https://vehicle-price-prediction-system.vercel.app/](https://vehicle-price-prediction-system.vercel.app/)**

### Deployment Configuration

The repository includes a `vercel.json` configuration file and a Flask application entry point.

The deployment setup is intended to make the application available through Vercel's hosting infrastructure.

### Deploy Your Own Copy

To deploy your own version:

1. Fork or clone the repository.
2. Review the Python dependencies.
3. Verify that the saved model artifact is available.
4. Review the Vercel configuration.
5. Import the repository into your Vercel account.
6. Configure the appropriate Python runtime and application entry point for the deployment environment.
7. Deploy the application.
8. Open the deployment URL and test the valuation workflow.

Deployment compatibility depends on the current Vercel runtime, dependency versions, and model artifact.

**Important:** Verify that the deployed application successfully loads `RFA_model.pkl`. A working interface does not necessarily mean the trained model is available in the deployed environment.

---

## 🎯 Use Cases

The Vehicle Price Prediction System demonstrates several practical applications of Machine Learning in automotive valuation.

### 🚗 Used Vehicle Buyers

Buyers can explore estimated vehicle values to help inform their purchasing decisions.

### 💼 Vehicle Sellers

Sellers can use price estimates as a starting point when considering a vehicle's potential market value.

### 🏢 Automotive Businesses

Dealerships and automotive service providers could use a validated version of this system as an initial valuation aid.

### 📊 Data Science Learning

Students and developers can study how a trained regression model is integrated into a web application.

### 💻 Machine Learning Portfolio

The project demonstrates practical experience with Python, Flask, model serialization, prediction workflows, and deployment.

**Disclaimer:** The application produces estimates and should not be treated as a certified appraisal or guaranteed market price.

---

## ⚠️ Limitations

Several considerations are important when using or extending this project.

- **Prediction accuracy:** Actual performance should be established using documented evaluation metrics and representative test data.
- **Dataset dependency:** The model's reliability depends on the quality, coverage, and relevance of its training data.
- **Currency conversion:** Configured exchange rates may not reflect current market rates.
- **Model availability:** The application may fall back to a heuristic formula when the saved model cannot be loaded.
- **Market differences:** Vehicle prices can vary substantially by country, region, taxes, demand, and market conditions.
- **Vehicle condition:** User-provided information may not capture every factor that affects the actual selling price.
- **Valuation ranges:** Displayed ranges and additional indicators should be validated against their underlying calculation logic before being treated as statistically reliable.
- **Serialized model compatibility:** The saved model may require compatible Python and Scikit-learn versions.
- **Security:** Pickle files should only be loaded from trusted sources because deserializing an untrusted pickle file can execute malicious code.

---

## 🚀 Future Enhancements

The following improvements could make the application more reliable, maintainable, and useful.

### 1. Model Performance Dashboard

Add an evaluation page displaying verified MAE, RMSE, and R² scores calculated on held-out test data.

### 2. Improved Model Training Pipeline

Create a reproducible training workflow that includes data cleaning, preprocessing, feature engineering, model training, and evaluation.

### 3. Real-Time Exchange Rates

Integrate a trusted exchange-rate API to retrieve current conversion rates instead of relying on fixed values.

### 4. Market-Based Valuation

Incorporate region-specific market information to improve price estimates for different vehicle markets.

### 5. Prediction Confidence and Uncertainty

Provide a statistically justified prediction interval where supported by an appropriate uncertainty estimation method.

### 6. Model Monitoring

Track prediction errors and monitor changes in vehicle market characteristics over time.

### 7. Prediction History

Allow users to save, review, and compare previous vehicle valuation results.

### 8. Enhanced Input Validation

Add stronger validation for vehicle year, mileage, engine capacity, horsepower, ownership count, and other input fields.

### 9. Automated Testing

Introduce unit tests for input preparation, prediction behavior, currency conversion, error handling, and backend routes.

### 10. Improved Model Deployment

Add model versioning and verify the availability and compatibility of model artifacts during application startup.

### 11. Documentation and Visualizations

Include screenshots, a model evaluation report, and a documented example showing the prediction workflow.

### 12. Better Separation of Components

Separate the Flask routes, model inference, configuration, and interface into dedicated modules to improve maintainability.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test the application locally.
5. Commit your changes with a descriptive message.
6. Open a pull request explaining your improvements.

For example:

```bash
git checkout -b feature/improve-validation
git add .
git commit -m "Improve vehicle input validation"
git push origin feature/improve-validation
```

Before submitting changes, verify that the prediction workflow still functions and that the saved model is compatible with the updated code.

---

## 👨‍💻 Author

**Siddhesh**

GitHub: [@Siddhesh030](https://github.com/Siddhesh030)

Project Repository: [Vehicle Price Prediction System](https://github.com/Siddhesh030/Vehicle_Price_Prediction_System)

Live Application: [AutoValuate AI](https://vehicle-price-prediction-system.vercel.app/)

---

## 📜 License

A license file was not identified in the repository's current top-level file listing.

Before distributing or reusing this project, consider adding an appropriate `LICENSE` file and confirming the licensing terms of the model and any third-party resources.

---

## ⭐ Support

If you find this project useful for learning about Machine Learning, Python web applications, and vehicle price prediction, consider giving the repository a star.

Your feedback and suggestions can help improve the project.

**Thank you for exploring the Vehicle Price Prediction System!** 🚗
