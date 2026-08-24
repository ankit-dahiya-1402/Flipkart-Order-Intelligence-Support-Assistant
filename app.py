"""Streamlit demo UI: three tabs, one per component, calling the Python
modules directly (no API layer)."""
import os

import streamlit as st

import data_utils
import return_risk
import catalog_model
import assistant

st.set_page_config(page_title="Flipkart Order Intelligence & Support Assistant", layout="wide")
st.title("Flipkart Order Intelligence & Support Assistant")
st.caption(
    "Student capstone demo. Order data is SYNTHETIC (data/generate_orders.py). "
    "Product images are the public Fashion-MNIST dataset used as a stand-in "
    "for real Flipkart catalog images (data/build_catalog.py) — not real "
    "Flipkart data."
)

tab1, tab2, tab3 = st.tabs(["Return Risk", "Catalog Check", "Support Chat"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Order Return-Risk Prediction")
    orders = data_utils.load_orders()
    order_id = st.selectbox("Pick a sample order", orders["order_id"].head(50).tolist())
    order = data_utils.get_order(int(order_id))
    st.json({k: v for k, v in order.items() if k != "returned"})

    if st.button("Predict return risk"):
        features = {k: v for k, v in order.items() if k not in ("order_id", "customer_id", "returned")}
        result = return_risk.predict(features)
        st.metric("Risk bucket", result["risk_bucket"], f"{result['probability']:.0%} probability")
        st.write("Top reasons:")
        for reason in result["top_reasons"]:
            st.write(f"- {reason}")
        st.caption(f"Decision threshold used: {result['threshold_used']}")

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Catalog Image / Category Check")
    st.caption("Images are sample Fashion-MNIST product photos, not real Flipkart listings.")
    catalog = data_utils.load_catalog()
    product_id = st.selectbox("Pick a sample product", catalog["product_id"].tolist())
    item = data_utils.get_catalog_item(product_id)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(os.path.join(os.path.dirname(__file__), item["image_path"]), width=150)
    with col2:
        st.write(f"Declared category: **{item['declared_category']}**")
        if st.button("Check category"):
            result = catalog_model.check_category(item["image_path"], item["declared_category"])
            st.write(f"Predicted class: **{result['predicted_class']}** ({result['confidence']:.0%} confidence)")
            st.write(f"Predicted category group: **{result['predicted_category']}**")
            if result["category_match"] is False:
                st.error(f"Mismatch: {result['issue']}")
            elif result["issue"]:
                st.warning(result["issue"])
            else:
                st.success("Category matches the image.")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("AI Support Assistant")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("No ANTHROPIC_API_KEY set — using deterministic fallback responses instead of live LLM generation.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    user_msg = st.chat_input("Ask about policy, an order's return risk (e.g. 'return risk for order 12'), or a product (e.g. 'check P0003')")
    if user_msg:
        st.session_state.history.append(("user", user_msg))
        st.chat_message("user").write(user_msg)
        result = assistant.chat(user_msg)
        reply = result["reply"]
        if result["escalate_to_human"]:
            reply += f"\n\n_(Flagged for human escalation: {result['escalation_reason']})_"
        st.session_state.history.append(("assistant", reply))
        st.chat_message("assistant").write(reply)
