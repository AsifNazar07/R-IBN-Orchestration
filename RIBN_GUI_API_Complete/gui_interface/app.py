import streamlit as st
import json
import requests

st.set_page_config(layout="wide")
st.title("R-IBN Framework")
st.caption("Research Platform for Intent-Based Network Orchestration")

col1, col2, col3 = st.columns([1.3, 1.5, 1.2])

with col1:
    st.header("Intent Configuration")
    source_node = st.selectbox("Source Node", ["Seoul (Core Network)", "Busan", "Pangyo"])
    service_type = st.selectbox("Service Type", ["8K Video Streaming", "HD Video", "4K Video"])
    qos = st.selectbox("QoS Priority", ["High Bandwidth", "Low Latency"])
    destination = st.selectbox("Destination Zone", ["Core Network", "Edge", "Access"])

    if st.button("Process Intent"):
        llm_response = requests.post("http://localhost:8000/process_intent", json={
            "source": source_node,
            "service": service_type,
            "qos": qos,
            "destination": destination
        })
        policy = llm_response.json()

        orchestration_response = requests.post("http://localhost:8000/run_orchestration", json=policy)
        orchestration_result = orchestration_response.json()

        st.subheader("LLM Translation Output")
        st.code(json.dumps(policy, indent=2), language='json')

        st.subheader("KNN Conflict Analysis")
        st.metric("Similarity Score", policy["similarity_score"])
        st.metric("Conflicts Found", policy["conflicts_found"])
        st.metric("Confidence", f"{policy['confidence']}%")

        with col2:
            st.header("Orchestration Dashboard")
            st.subheader("Computed Optimal Path")
            st.success(" → ".join(policy['orchestration_path']))

            st.subheader("Performance Metrics")
            st.metric("Hop Count", orchestration_result['hop_count'])
            st.metric("End-to-End Latency", orchestration_result['latency'])
            st.metric("Reliability", orchestration_result['reliability'])
            st.metric("Response Time", orchestration_result['response_time'])

            st.subheader("Network Topology Visualization")
            st.info("[Topology graph rendering placeholder]")
            st.subheader("Export Configuration")
            st.button("Export JSON")
            st.button("Export YAML")

        with col3:
            st.header("System Monitoring")
            st.subheader("Active Services")
            st.success(f"{service_type} - Active")
            st.error("4K Video Service - Conflict")

            st.subheader("System Alerts")
            for alert in orchestration_result['alerts']:
                if "conflict" in alert.lower():
                    st.error(alert)
                elif "warning" in alert.lower():
                    st.warning(alert)
                else:
                    st.success(alert)

            st.subheader("System Performance")
            st.metric("Success Rate", orchestration_result['success_rate'])
            st.metric("Avg Latency", orchestration_result['avg_latency'])
            st.metric("CPU Usage", orchestration_result['cpu_usage'])
            st.metric("Memory", orchestration_result['memory'])
