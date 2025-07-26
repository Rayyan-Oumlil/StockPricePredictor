import streamlit as st
import json
import os

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")

def save_alerts(alerts):
    """Save alerts to a JSON file"""
    with open("alerts_config.json", "w") as f:
        json.dump({"alerts": alerts}, f, indent=2)

def load_alerts():
    """Load alerts from JSON file"""
    if os.path.exists("alerts_config.json"):
        try:
            with open("alerts_config.json", "r") as f:
                data = json.load(f)
                return data.get("alerts", [])
        except:
            return []
    return []

# Load existing alerts
existing_alerts = load_alerts()

# Initialize session state with existing data
if 'alertes' not in st.session_state:
    st.session_state.alertes = existing_alerts

st.header("🔔 Alerts Management")
st.info("Set up price threshold alerts for your predictions.")

# Alert form
with st.form("add_alert_form"):
    ticker = st.text_input("Ticker", value="AAPL")
    alert_type = st.selectbox("Alert when price is:", ["Above threshold", "Below threshold"])
    seuil = st.number_input("Price threshold", min_value=0.0, value=200.0)
    submitted = st.form_submit_button("Add alert")
    if submitted and ticker:
        alert_type_short = "above" if alert_type == "Above threshold" else "below"
        st.session_state.alertes.append({
            "ticker": ticker.upper(), 
            "seuil": seuil, 
            "type": alert_type_short
        })
        # Save to file
        save_alerts(st.session_state.alertes)
        st.success(f"Alert added for {ticker.upper()} {alert_type_short} {seuil}")

# List of alerts
st.subheader("Configured Alerts")
if st.session_state.alertes:
    for i, alert in enumerate(st.session_state.alertes):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            alert_symbol = "≥" if alert['type'] == "above" else "≤"
            st.write(f"**{alert['ticker']}** {alert_symbol} ${alert['seuil']:.2f}")
        with col2:
            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                st.session_state.alertes.pop(i)
                # Save to file
                save_alerts(st.session_state.alertes)
                st.rerun()
    
    st.divider()
    
    # Clear all alerts button
    if st.button("🗑️ Clear All Alerts", type="secondary"):
        st.session_state.alertes = []
        # Save to file
        save_alerts([])
        st.rerun()
else:
    st.write("No alerts configured.") 