import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import plotly.express as px
# ------------------------- Config -------------------------
st.set_page_config(page_title="Car Insight App", layout="wide")

# ------------------------- Background Image -------------------------
def get_img_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_img_base64("landing.png")  # replace if needed

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6, p, label, .stTextInput > label, .stSelectbox > label, .stSlider > label {{
        color: white !important;
        text-shadow: 1px 1px 5px black;
    }}
    .main-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 90vh;
        text-align: center;
    }}
    .main-container h1 {{
        font-size: 52px;
        font-weight: bold;
        text-shadow: 3px 3px 10px black;
        margin-bottom: 50px;
    }}
    .stButton>button {{
        background-color: #ffffff20;
        color: white;
        font-weight: bold;
        border: 2px solid white;
        border-radius: 10px;
        padding: 15px 25px;
        font-size: 16px;
        text-shadow: 1px 1px 2px black;
        transition: all 0.2s ease-in-out;
        white-space: nowrap;
    }}
    /* Force plot containers to stay small and centered */
    .element-container:has(.stPlotlyChart), 
    .element-container:has(.stImage), 
    .element-container:has(.stPyplot), 
    .element-container:has(.stAltairChart), 
    .element-container:has(.stVegaLiteChart) {{
        max-width: 900px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    .stButton>button:hover {{
        background-color: #ffffff40;
        transform: scale(1.05);
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------- Login Helpers -------------------------
def load_users():
    return pd.read_csv("users.csv")

def save_user(username, password):
    users = load_users()
    if username in users['username'].values:
        return False
    new_user = pd.DataFrame({'username': [username], 'password': [password]})
    new_user.to_csv("users.csv", mode='a', header=False, index=False)
    return True

def authenticate(username, password):
    users = load_users()
    user_row = users[users['username'] == username]
    if not user_row.empty and user_row.iloc[0]['password'] == password:
        return True
    return False

# ------------------------- Session Setup -------------------------
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ------------------------- Load & Prep Data -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Car_Data.csv")
    df.dropna(inplace=True)
    df['Car Age'] = 2025 - df['Year']
    df['Sales Efficiency'] = df['Units Sold'] / df['Price (INR)']
    df['Price Category'] = pd.cut(df['Price (INR)'],
                                  bins=[0, 500000, 1000000, np.inf],
                                  labels=['Budget', 'Mid-Range', 'Premium'])
    df['Fuel_Code'] = LabelEncoder().fit_transform(df['Fuel Type'])
    df['Trans_Code'] = LabelEncoder().fit_transform(df['Transmission'])
    return df

df = load_data()

# ------------------------- Login/Sign Up (Home Page Only) -------------------------
if st.session_state.page == "home":
    with st.container():
        col_login, _, col_app = st.columns([2, 4, 3])
        with col_app:
            if not st.session_state.logged_in:
                # Application Title (centered)
                st.markdown("""
                    <h1 style="text-align: center; font-size: 52px; color: white; font-weight: bold; text-shadow: 3px 3px 10px black;">Caralyze: Car Sales Analysis</h1>
                """, unsafe_allow_html=True)

                # Login or Sign Up Section
                st.subheader("🔑 User Authentication")
                tabs = st.tabs(["🔐 Login", "🆕 Sign Up"])
                with tabs[0]:
                    uname = st.text_input("Username", key="login_user")
                    pwd = st.text_input("Password", type="password", key="login_pass")
                    if st.button("Login"):
                        if authenticate(uname, pwd):
                            st.session_state.logged_in = True
                            st.session_state.username = uname
                            st.success(f"✅ Welcome, {uname}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                with tabs[1]:
                    new_user = st.text_input("New Username", key="signup_user")
                    new_pass = st.text_input("New Password", type="password", key="signup_pass")
                    if st.button("Sign Up"):
                        if save_user(new_user, new_pass):
                            st.success("Account created! Please login.")
                        else:
                            st.error("Username already exists.")
            else:
                # Use columns for layout: Left column for "Logout" button, center for "Logged in as"
                col1, col2 = st.columns([1, 4])  # Adjust column size to give more space for the username
                with col1:
                    if st.button("Logout", key='logout'):  # "Logout" button is now at the leftmost side
                        st.session_state.logged_in = False
                        st.session_state.page = "home"
                        st.rerun()

                with col2:
                    st.markdown(f"<p style='text-align:center;'>👤 Logged in as <b>{st.session_state.username}</b></p>", unsafe_allow_html=True)

    # --- If logged in, show home content
    if st.session_state.logged_in:
        st.markdown("""
            <div class="main-container">
                <h1>🚗 Caralyze: Car Sales Analysis</h1>
                <h3 style="margin-top:-20px;">From budget to brand — we help you choose the best deal!</h3>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns([0.5, 1.5, 1.5, 1.5, 1])
        with col2:
            if st.button("💰 Explore by Budget"):
                st.session_state.page = "budget"
        with col3:
            if st.button("⚖️ Compare Cars"):
                st.session_state.page = "compare"
        with col4:
            if st.button("📈 Sales Trend"):
                st.session_state.page = "trend"
        with col5:
            if st.button("💸 Price Predictor"):
                st.session_state.page = "predict_price"


# ------------------------- Shared Model Setup -------------------------
features = ['Mileage (km/l)', 'Engine (cc)', 'Seater', 'Fuel_Code', 'Trans_Code']
X = df[features]
y = df['Price Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ------------------------- Price Prediction Page -------------------------
if st.session_state.page == "predict_price":
    if not st.session_state.logged_in:
        st.warning("🚫 You must be logged in to access this page.")
        st.stop()

    st.title("💸 Predict Car Price")
    
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Select Car Brand", df['Brand'].unique())
        model = st.selectbox("Select Car Model", df[df['Brand'] == brand]['Model'].unique())
    with col2:
        fuel_type = st.selectbox("Fuel Type", df['Fuel Type'].unique())
        transmission = st.selectbox("Transmission Type", df['Transmission'].unique())
    
    # New Year Input
    year = st.number_input("Select Year for Prediction", min_value=2023, max_value=2030, value=2025)

    selected_car = df[(df['Brand'] == brand) & (df['Model'] == model) & 
                      (df['Fuel Type'] == fuel_type) & (df['Transmission'] == transmission)]
    
    if not selected_car.empty:
        selected_car = selected_car.iloc[0]

        # Encode categorical features
        fuel_encoder = LabelEncoder().fit(df['Fuel Type'])
        trans_encoder = LabelEncoder().fit(df['Transmission'])
        fuel_code = fuel_encoder.transform([fuel_type])[0]
        trans_code = trans_encoder.transform([transmission])[0]

        # Prepare input data including the year
        input_data = pd.DataFrame([[selected_car['Mileage (km/l)'], selected_car['Engine (cc)'], 
                                    selected_car['Seater'], fuel_code, trans_code, year]], 
                                   columns=features + ['Year'])

        if st.button("🔮 Predict Price"):
            # Model and scaler
            scaler = StandardScaler()
            # Include 'Year' in the features for scaling
            X_reg = scaler.fit_transform(df[features + ['Year']])
            y_reg = df['Price (INR)']

            lr_model = LinearRegression()
            lr_model.fit(X_reg, y_reg)

            # Prepare input data for prediction
            input_scaled = scaler.transform(input_data)
            predicted_price = lr_model.predict(input_scaled)[0]

            col_pred, col_plot = st.columns([1, 2])
            with col_pred:
                st.success(f"Predicted Price for {year}: ₹{predicted_price:,.2f}")

            with col_plot:
                # Predict prices for all models in the selected brand for the specified year
                brand_models = df[df['Brand'] == brand]
                predicted_prices = []
                model_names = []

                for _, row in brand_models.iterrows():
                    model_input = pd.DataFrame([[
                        row['Mileage (km/l)'],
                        row['Engine (cc)'],
                        row['Seater'],
                        fuel_encoder.transform([row['Fuel Type']])[0],
                        trans_encoder.transform([row['Transmission']])[0],
                        year  # Include the year for prediction
                    ]], columns=features + ['Year'])
                    
                    scaled_input = scaler.transform(model_input)
                    price = lr_model.predict(scaled_input)[0]
                    predicted_prices.append(price)
                    model_names.append(row['Model'])

                # Create dataframe for visualization
                comparison_df = pd.DataFrame({
                    'Model': model_names,
                    'Predicted Price': predicted_prices
                })

                # Highlight selected model
                comparison_df['Color'] = ['Selected' if m == model else 'Other' for m in comparison_df['Model']]

                # Plot using Plotly
                fig = px.bar(
                    comparison_df.sort_values('Predicted Price'),
                    x='Model',
                    y='Predicted Price',
                    color='Color',
                    color_discrete_map={'Selected': 'red', 'Other': 'blue'},
                    title=f"Predicted Prices for {brand} Models in {year}",
                    labels={'Predicted Price': 'Price (INR)', 'Model': 'Car Model'}
                )

                # Format Y-axis to show full numbers (no 'M')
                fig.update_layout(
                    yaxis_tickformat=',',
                    yaxis_title='Predicted Price (INR)',
                    xaxis_title='Car Model'
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("❗ No matching car found.")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"





# ---------- Budget Clustering Page ----------
if st.session_state.page == "budget":
    st.title("💰 Explore Cars by Budget Using Clustering")

    st.subheader("🎯 Refine Your Filters")
    min_price, max_price = int(df['Price (INR)'].min()), int(df['Price (INR)'].max())
    budget = st.slider("Select Your Budget Range (INR)", min_price, max_price, (300000, 1500000))

    brands = st.multiselect("Choose Brand(s)", sorted(df['Brand'].unique()))
    if brands:
        models = st.multiselect("Choose Model(s)", sorted(df[df['Brand'].isin(brands)]['Model'].unique()))
    else:
        models = st.multiselect("Choose Model(s)", sorted(df['Model'].unique()))

    fuels = st.multiselect("Choose Fuel Type(s)", sorted(df['Fuel Type'].unique()))
    transmissions = st.multiselect("Choose Transmission(s)", sorted(df['Transmission'].unique()))

    filtered_df = df.copy()
    filtered_df = filtered_df[ 
        (filtered_df['Price (INR)'] >= budget[0]) & (filtered_df['Price (INR)'] <= budget[1])
    ]
    if brands:
        filtered_df = filtered_df[filtered_df['Brand'].isin(brands)]
    if models:
        filtered_df = filtered_df[filtered_df['Model'].isin(models)]
    if fuels:
        filtered_df = filtered_df[filtered_df['Fuel Type'].isin(fuels)]
    if transmissions:
        filtered_df = filtered_df[filtered_df['Transmission'].isin(transmissions)]

    # Remove clustering visualization
    cluster_df = filtered_df[['Price (INR)', 'Mileage (km/l)', 'Engine (cc)', 'Seater']]
    if not cluster_df.empty:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)
        clustering = DBSCAN(eps=1.5, min_samples=3).fit(X_scaled)
        filtered_df['Cluster'] = clustering.labels_

        st.success(f"✅ Found {len(filtered_df)} cars matching your filters!")
        # Display the DataFrame without the index
        st.dataframe(filtered_df[['Model', 'Brand', 'Price (INR)', 'Mileage (km/l)', 'Engine (cc)', 'Seater', 'Units Sold']].reset_index(drop=True), use_container_width=True)

        # Removed the following plot section:
        # st.subheader("🔍 Clustering Visualization")
        # fig, ax = plt.subplots()
        # sns.scatterplot(data=filtered_df, x='Mileage (km/l)', y='Price (INR)', hue='Cluster', palette='Set2', ax=ax)
        # st.pyplot(fig)
    else:
        st.warning("❗ No cars found matching your filters.")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"

# ------------------------- Compare Cars Page -------------------------
if st.session_state.page == "compare":
    if not st.session_state.logged_in:
        st.warning("🚫 You must be logged in to access this page.")
        st.stop()

    st.title("⚖️ Compare Cars")
    col1, col2 = st.columns(2)
    with col1:
        brand1 = st.selectbox("Brand (Car 1)", df['Brand'].unique())
        model1 = st.selectbox("Model (Car 1)", df[df['Brand'] == brand1]['Model'].unique())
    with col2:
        brand2 = st.selectbox("Brand (Car 2)", df['Brand'].unique(), key='brand2')
        model2 = st.selectbox("Model (Car 2)", df[df['Brand'] == brand2]['Model'].unique(), key='model2')

    if st.button("🔍 Compare Now"):
        car1 = df[(df['Brand'] == brand1) & (df['Model'] == model1)].iloc[0]
        car2 = df[(df['Brand'] == brand2) & (df['Model'] == model2)].iloc[0]

        input1 = pd.DataFrame([[car1[feat] for feat in features]], columns=features)
        input2 = pd.DataFrame([[car2[feat] for feat in features]], columns=features)

        # Using CART algorithm for prediction
        clf = DecisionTreeClassifier(random_state=42)  # Initialize the CART model
        clf.fit(X_train, y_train)  # Train the model on the training data
        pred1, pred2 = clf.predict(input1)[0], clf.predict(input2)[0]

        st.subheader("🔮 Prediction Result")
        if pred1 == pred2:
            st.info(f"Both cars are **{pred1}** category.")
        else:
            better = model1 if ['Budget', 'Mid-Range', 'Premium'].index(pred1) > ['Budget', 'Mid-Range', 'Premium'].index(pred2) else model2
            st.success(f"✅ **{better}** has a better category prediction.")

        st.subheader("📊 Units Sold Comparison")
        comp_df = pd.DataFrame({
            'Model': [model1, model2],
            'Units Sold': [car1['Units Sold'], car2['Units Sold']],
            'Brand': [brand1, brand2]
        })
        fig, ax = plt.subplots(figsize=(4, 2.5))  # Even smaller figure size
        sns.barplot(data=comp_df, x='Model', y='Units Sold', hue='Brand', ax=ax, width=0.4)  # Thinner bars
        ax.set_title("Units Sold Comparison", color='white', fontsize=10)  # White title
        ax.set_ylabel("Units Sold", color='white', fontsize=10)  # White y-axis label
        ax.set_xlabel("Car Model", color='white', fontsize=10)  # White x-axis label
        ax.tick_params(axis='x', colors='white', labelsize=10)  # White x-axis ticks
        ax.tick_params(axis='y', colors='white', labelsize=10)  # White y-axis ticks
        ax.set_facecolor('none') # Transparent axes background
        fig.patch.set_alpha(0) # Make the plot background transparent

        # Adjust legend
        ax.legend(fontsize='small', loc='upper left', frameon=False, labelcolor='white')  # Smaller legend, top right, no frame, white text

        # Adjust layout to prevent labels from overlapping
        plt.tight_layout(pad=0.5)

        st.markdown("<div style='max-width: 300px; margin: auto;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔙 Back to Home", key='back2'):
        st.session_state.page = "home"

# ------------------------- Sales Trend Page -------------------------
if st.session_state.page == "trend":
    if not st.session_state.logged_in:
        st.warning("🚫 You must be logged in to access this page.")
        st.stop()

    st.title("📈 Sales Trend by Brand")
    selected_brands = st.multiselect("Select Brand(s)", sorted(df['Brand'].unique()))
    if selected_brands:
        trend_df = df[df['Brand'].isin(selected_brands)]
        sales_by_year = trend_df.groupby(['Year', 'Brand'])['Units Sold'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=sales_by_year, x='Year', y='Units Sold', hue='Brand', marker="o", ax=ax)
        ax.set_title("Yearly Units Sold by Brand", color='white', fontsize=10)  # White title
        ax.grid(True)
        ax.set_ylabel("Units Sold", color='white', fontsize=10)  # White y-axis label
        ax.set_xlabel("Car Model", color='white', fontsize=10)  # White x-axis label
        ax.tick_params(axis='x', colors='white', labelsize=10)  # White x-axis ticks
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.set_facecolor('none') # Transparent axes background
        fig.patch.set_alpha(0)
        st.pyplot(fig)
    else:
        st.info("Please select at least one brand.")

    if st.button("🔙 Back to Home", key='back3'):
        st.session_state.page = "home"



