import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Ocean Microplastic Early Warning System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Ocean Microplastic Early Warning System")

st.write(
    "AI-based educational prototype for predicting "
    "ocean microplastic pollution risk."
)


# =====================================================
# LOAD DATA
# =====================================================

file_path = Path(__file__).parent / "microplastic_location_dataset.csv"

try:
    df = pd.read_csv(file_path)
except:
    st.error("❌ microplastic_location_dataset.csv not found")
    st.stop()


# Clean column names
df.columns = df.columns.str.strip().str.lower()


# =====================================================
# REQUIRED COLUMNS
# =====================================================

required = [
    "location",
    "latitude",
    "longitude",
    "temperature",
    "ph",
    "turbidity",
    "salinity",
    "microplastic",
    "risk"
]

for col in required:
    if col not in df.columns:
        st.error(f"❌ Column missing: {col}")
        st.stop()


# =====================================================
# AI MODEL
# =====================================================

features = [
    "temperature",
    "ph",
    "turbidity",
    "salinity"
]

X = df[features]
y = df["risk"]

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)


# =====================================================
# DASHBOARD
# =====================================================

st.subheader("📊 Pollution Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📍 Locations", len(df))

with c2:
    st.metric(
        "🔴 High Risk",
        len(
            df[
                df["risk"].astype(str).str.lower() == "high"
            ]
        )
    )

with c3:
    st.metric(
        "🟠 Medium Risk",
        len(
            df[
                df["risk"].astype(str).str.lower() == "medium"
            ]
        )
    )

with c4:
    st.metric(
        "🟢 Low Risk",
        len(
            df[
                df["risk"].astype(str).str.lower() == "low"
            ]
        )
    )


# =====================================================
# SESSION STATE
# =====================================================

if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = None

if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = None


# =====================================================
# MAP
# =====================================================

st.divider()

st.subheader("🗺️ Select Any Ocean Location")

st.info(
    "👆 Red/green marker மட்டும் click செய்ய வேண்டியதில்லை. "
    "Map-ல empty ocean area-விலும் click செய்யலாம்."
)


# Create map
ocean_map = folium.Map(
    location=[10.0, 78.0],
    zoom_start=4,
    tiles="OpenStreetMap"
)


# =====================================================
# EXISTING COLOURED MARKERS
# =====================================================

for _, row in df.iterrows():

    risk = str(row["risk"]).strip().lower()

    if risk == "high":
        color = "red"

    elif risk == "medium":
        color = "orange"

    else:
        color = "green"

    popup = f"""
    <b>🌊 {row['location']}</b><br>
    Risk: {row['risk']}<br>
    Microplastic: {row['microplastic']}<br>
    Temperature: {row['temperature']} °C<br>
    pH: {row['ph']}<br>
    Turbidity: {row['turbidity']} NTU<br>
    Salinity: {row['salinity']} PSU
    """

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=9,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(
            popup,
            max_width=300
        )
    ).add_to(ocean_map)


# =====================================================
# ⭐ NEW SELECTED LOCATION BLUE MARKER
# =====================================================

if (
    st.session_state.selected_lat is not None
    and
    st.session_state.selected_lon is not None
):

    folium.Marker(
        location=[
            st.session_state.selected_lat,
            st.session_state.selected_lon
        ],
        tooltip="📍 Your Selected Location",
        popup=(
            "📍 Selected Location<br>"
            f"Latitude: {st.session_state.selected_lat:.4f}<br>"
            f"Longitude: {st.session_state.selected_lon:.4f}"
        ),
        icon=folium.Icon(
            color="blue",
            icon="map-marker"
        )
    ).add_to(ocean_map)


# =====================================================
# MAP CLICK COORDINATES
# =====================================================

folium.LatLngPopup().add_to(ocean_map)


# =====================================================
# SHOW MAP
# =====================================================

map_data = st_folium(
    ocean_map,
    width=1200,
    height=600,
    returned_objects=[
        "last_clicked"
    ]
)


# =====================================================
# ⭐ GET EMPTY MAP CLICK
# =====================================================

if map_data is not None:

    clicked = map_data.get("last_clicked")

    if clicked is not None:

        lat = clicked.get("lat")
        lon = clicked.get("lng")

        if lat is not None and lon is not None:

            # Save clicked coordinates
            st.session_state.selected_lat = lat
            st.session_state.selected_lon = lon


# =====================================================
# SELECTED LOCATION RESULT
# =====================================================

if (
    st.session_state.selected_lat is not None
    and
    st.session_state.selected_lon is not None
):

    lat = st.session_state.selected_lat
    lon = st.session_state.selected_lon

    st.divider()

    st.header("📍 Selected Ocean Location")


    # =================================================
    # COORDINATES
    # =================================================

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            "✅ New location selected!"
        )

        st.write(
            "You selected this location directly "
            "from the map."
        )


    with col2:

        st.subheader("📌 Coordinates")

        st.metric(
            "Latitude",
            f"{lat:.4f}"
        )

        st.metric(
            "Longitude",
            f"{lon:.4f}"
        )


    # =================================================
    # FIND NEAREST DATASET LOCATION
    # =================================================

    distance = (
        (df["latitude"] - lat) ** 2
        +
        (df["longitude"] - lon) ** 2
    )

    nearest_index = distance.idxmin()

    nearest = df.loc[nearest_index]


    st.info(
        f"📊 Nearest dataset location: "
        f"**{nearest['location']}**"
    )


    # =================================================
    # WATER PARAMETERS
    # =================================================

    st.subheader("🌊 Water Quality")

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.metric(
            "🌡️ Temperature",
            f"{nearest['temperature']} °C"
        )

    with w2:
        st.metric(
            "🧪 pH",
            nearest["ph"]
        )

    with w3:
        st.metric(
            "💧 Turbidity",
            f"{nearest['turbidity']} NTU"
        )

    with w4:
        st.metric(
            "🌊 Salinity",
            f"{nearest['salinity']} PSU"
        )


    # =================================================
    # AI PREDICTION
    # =================================================

    st.divider()

    st.subheader("🤖 AI Pollution Prediction")

    sample = pd.DataFrame([
        {
            "temperature": nearest["temperature"],
            "ph": nearest["ph"],
            "turbidity": nearest["turbidity"],
            "salinity": nearest["salinity"]
        }
    ])

    prediction = model.predict(sample)[0]

    prediction_text = str(prediction).lower()


    if prediction_text == "high":

        st.error(
            "🚨 HIGH RISK — Early Warning!"
        )

    elif prediction_text == "medium":

        st.warning(
            "⚠️ MEDIUM RISK — Monitoring Recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — No Immediate Warning."
        )


    # =================================================
    # PROBABILITY
    # =================================================

    try:

        probabilities = model.predict_proba(sample)[0]

        probability_dict = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        probability = (
            probability_dict[prediction] * 100
        )

        st.metric(
            "🎯 Prediction Probability",
            f"{probability:.1f}%"
        )

    except:
        pass


    # =================================================
    # MICROPLASTIC
    # =================================================

    st.metric(
        "🧴 Microplastic Level",
        nearest["microplastic"]
    )


    # =================================================
    # WATER QUALITY CHART
    # =================================================

    st.divider()

    st.subheader("📈 Water Quality Parameters")

    chart_data = pd.DataFrame({
        "Parameter": [
            "Temperature",
            "pH",
            "Turbidity",
            "Salinity"
        ],
        "Value": [
            nearest["temperature"],
            nearest["ph"],
            nearest["turbidity"],
            nearest["salinity"]
        ]
    })

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        chart_data["Parameter"],
        chart_data["Value"]
    )

    ax.set_title(
        "Water Quality Parameters"
    )

    ax.set_ylabel("Value")

    st.pyplot(fig)


# =====================================================
# RISK DISTRIBUTION
# =====================================================

st.divider()

st.subheader("📊 Risk Distribution")

risk_count = df["risk"].value_counts()

fig2, ax2 = plt.subplots(
    figsize=(8, 4)
)

ax2.bar(
    risk_count.index.astype(str),
    risk_count.values
)

ax2.set_xlabel("Risk")
ax2.set_ylabel("Locations")

ax2.set_title(
    "Microplastic Risk Distribution"
)

st.pyplot(fig2)


# =====================================================
# DATASET
# =====================================================

with st.expander("📋 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =====================================================
# NOTE
# =====================================================

st.divider()

st.caption(
    "⚠️ This is an educational prototype. "
    "For a newly selected location, the prediction "
    "uses the nearest available dataset location."
)