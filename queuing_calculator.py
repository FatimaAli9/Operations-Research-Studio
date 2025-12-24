import streamlit as st
import math
import time

# ------------------ CUSTOM CSS ------------------
def local_css():
    st.markdown("""
        <style>
        div[data-baseweb="select"] > div { border: 1px solid #d1d5db !important; border-radius: 8px !important; background-color: white !important; }
        div[data-baseweb="select"] input { caret-color: white !important; border: none !important; outline: white !important; box-shadow: none !important; }
        div[role="button"] { padding-left: 10px !important; }
        [data-testid="stMetricValue"] { font-size: 22px; color: #1f77b4; }
        .result-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; border-left: 5px solid #1f77b4; margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)

# ------------------ HELPERS ------------------
def unit_to_min(v, u):
    if "hour" in u: return v / 60
    if "sec" in u: return v * 60
    return v

def rate_mean_block(label, key):
    mode = st.radio(f"{label} Input Type", ["Rate", "Mean"], horizontal=True, key=f"{key}_mode")
    unit = st.selectbox(f"{label} Unit", ["per min", "per hour", "per sec"], key=f"{key}_unit")
    if mode == "Rate":
        val = st.number_input(f"{label} Rate Value", value=1.0, key=f"{key}_val")
        return unit_to_min(val, unit.replace("per ", "")), val
    val = st.number_input(f"{label} Mean Value", value=1.0, key=f"{key}_val")
    mean_min = val * 60 if "hour" in unit else val / 60 if "sec" in unit else val
    converted_rate = 1 / mean_min if mean_min != 0 else 0
    return converted_rate, val

def g_distribution_block(prefix):
    dist = st.selectbox(f"{prefix} Distribution", ["Uniform", "Normal", "Gamma"], key=f"{prefix}_dist")
    if dist == "Uniform":
        a = st.number_input(f"{prefix} Min Value", value=1.0, key=f"{prefix}_min")
        b = st.number_input(f"{prefix} Max Value", value=3.0, key=f"{prefix}_max")
        mean = (a + b) / 2
        var = (b - a) ** 2 / 12
        rate = 1 / mean if mean != 0 else 0
        return {"mean": mean, "var": var, "rate": rate, "type": "Uniform", "a": a, "b": b}
    rate_val, raw_val = rate_mean_block(prefix.capitalize(), prefix)
    var_mode = st.radio(f"{prefix.capitalize()} Variation Type", ["Standard Deviation", "Variance"], horizontal=True, key=f"{prefix}_var_mode")
    v_input = st.number_input(f"Enter {var_mode}", value=1.0, key=f"{prefix}_var_input")
    if var_mode == "Standard Deviation":
        sd = v_input
        var = sd ** 2
    else:
        var = v_input
        sd = math.sqrt(max(0, var))
    mean = 1 / rate_val if rate_val != 0 else 0
    return {"mean": mean, "var": var, "rate": rate_val, "type": dist, "sd": sd, "raw_val": raw_val, "var_mode": var_mode, "v_input": v_input}

# ------------------ SEPARATE MODEL FUNCTIONS ------------------

def compute_mm1(lmbd, mu):
    if mu <= lmbd: return {"error": "System unstable (λ ≥ μ)."}
    rho = lmbd / mu
    Lq = (rho**2) / (1 - rho)
    Wq = Lq / lmbd
    W = Wq + (1 / mu)
    L = lmbd * W
    return {"ρ": rho, "P0": 1 - rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq, "Ca2": 1.0, "Cs2": 1.0}

def compute_mms(lmbd, mu, s):
    rho = lmbd / (s * mu)
    if rho >= 1: return {"error": "System unstable (ρ ≥ 1)."}
    a = lmbd / mu
    P0 = 1 / (sum(a**n / math.factorial(n) for n in range(s)) + a**s / (math.factorial(s) * (1 - rho)))
    Lq = (P0 * (a**s) * rho) / (math.factorial(s) * ((1 - rho)**2))
    Wq = Lq / lmbd
    W = Wq + (1 / mu)
    L = lmbd * W
    return {"ρ": rho, "P0": P0, "Lq": Lq, "L": L, "Wq": Wq, "W": W, "Ca2": 1.0, "Cs2": 1.0}

def compute_mg1(lmbd, ms, vs):
    mu = 1 / ms if ms != 0 else 0
    if mu <= lmbd: return {"error": "System unstable (λ ≥ μ)."}
    rho = lmbd / mu
    Lq = ( (lmbd**2 * vs) + (rho**2) ) / (2 * (1 - rho))
    Wq = Lq / lmbd
    W = Wq + ms
    L = lmbd * W
    return {"ρ": rho, "Lq": Lq, "L": L, "Wq": Wq, "W": W, "P0": 1 - rho, "Ca2": 1.0, "Cs2": vs * (mu**2)}

def compute_mgs(lmbd, ms, vs, s):
    mu = 1 / ms if ms != 0 else 0
    base = compute_mms(lmbd, mu, s)
    if "error" in base: return base
    Cs2 = vs * (mu**2)
    Wq = base["Wq"] * (1 + Cs2) / 2
    return {"ρ": base["ρ"], "Lq": lmbd * Wq, "L": lmbd * (Wq + ms), "Wq": Wq, "W": Wq + ms, "P0": base["P0"], "Ca2": 1.0, "Cs2": Cs2}

def compute_gg1(lmbd, ma, va, ms, vs):
    mu = 1 / ms if ms != 0 else 0
    if mu <= lmbd: return {"error": "System unstable (λ ≥ μ)."}
    rho = lmbd / mu
    Ca2 = va / (1 / lmbd)**2
    Cs2 = vs / (1 / mu)**2
    Lq = (rho**2 * (1 + Cs2) * (Ca2 + rho**2 * Cs2)) / (2 * (1 - rho) * (1 + rho**2 * Cs2))
    Wq = Lq / lmbd
    W = Wq + ms
    L = lmbd * W
    return {"ρ": rho, "Lq": Lq, "L": L, "Wq": Wq, "W": W, "P0": 1 - rho, "Ca2": Ca2, "Cs2": Cs2}

def compute_ggs(lmbd, ma, va, ms, vs, s):
    mu = 1 / ms if ms != 0 else 0
    base = compute_mms(lmbd, mu, s)
    if "error" in base: return base
    Ca2 = va * (lmbd**2)
    Cs2 = vs * (mu**2)
    Wq = base["Wq"] * ((Ca2 + Cs2) / 2)
    return {"ρ": base["ρ"], "Lq": lmbd * Wq, "L": lmbd * (Wq + ms), "Wq": Wq, "W": Wq + ms, "P0": base["P0"], "Ca2": Ca2, "Cs2": Cs2}

# ------------------ DISPLAY ------------------
def display_results(res):
    if "error" in res:
        st.error(f"⚠️ {res['error']}")
        return
    
    st.write("---")
    st.header("📊 Performance Dashboard")
    
    # Using a container for the result card
    with st.container():
        #st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Row 1: Key Metrics
        st.subheader("🚀 System Efficiency")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Utilization (ρ)", f"{res['ρ']*100:.1f}%")
        with c2:
            st.metric("Idle Probability (P₀)", f"{res.get('P0', 0)*100:.1f}%", help="Probablity of having zero customers in the system")
        with c3:
            st.metric("Avg in System (Ls)", f"{res['L']:.2f}", help="Average number of customers in the system")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Queue & Time
        st.subheader("⏱️ Queue & Wait Times")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("Avg in Queue (Lq)", f"{res['Lq']:.2f}",  help="Average number of customers waiting in the queue")
        with c5:
            st.metric("Wait in System (Ws)", f"{res['W']:.2f} m",  help="Average waiting time of customers in the system")
        with c6:
            st.metric("Wait in Queue (Wq)", f"{res['Wq']:.2f} m",  help="Average waiting time of customers in the queue")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 3: Variations
        st.subheader("📉 Variation Factors")
        c7, c8 = st.columns(2)
        with c7:
            st.info(f"**Arrival CV² (Ca²):** {res.get('Ca2', 0):.4f}")
        with c8:
            st.info(f"**Service CV² (Cs²):** {res.get('Cs2', 0):.4f}")
            
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ UI ------------------
def queuing_calculator_ui():
    local_css()
    #st.title("Queuing Theory Calculator")
    model = st.selectbox("Select Model", ["M/M/1", "M/M/s", "M/G/1", "M/G/s", "G/G/1", "G/G/s"])

    s = 1
    if model.endswith("s"):
        s = st.number_input("Servers (s)", value=2)

    st.markdown("### Arrival")
    if model.startswith("G"):
        arr_data = g_distribution_block("arr")
        lmbd, ma, va = arr_data['rate'], arr_data['mean'], arr_data['var']
    else:
        lmbd, raw_arr = rate_mean_block("Arrival", "arr")
        arr_data = {"type": "Exponential", "raw_val": raw_arr}
        ma, va = (1/lmbd if lmbd!=0 else 0), 0

    st.divider()
    st.markdown("### Service")
    if "G" in model:
        ser_data = g_distribution_block("ser")
        mu, ms, vs = ser_data['rate'], ser_data['mean'], ser_data['var']
    else:
        mu, raw_ser = rate_mean_block("Service", "ser")
        ser_data = {"type": "Exponential", "raw_val": raw_ser}
        ms, vs = (1/mu if mu!=0 else 0), 0

    if st.button("Calculate", type="primary"):
        # --- Loading Bar ---
        progress_text = "Calculating... Please wait."
        my_bar = st.progress(0, text=progress_text)
        for pct in range(100):
            time.sleep(0.01)
            my_bar.progress(pct + 1, text=progress_text)
        my_bar.empty() 

        errors = []
        if model.endswith("s") and s <= 0: errors.append("Servers must be 1 or more.")
        if arr_data["type"] == "Uniform":
            if arr_data["a"] < 0 or arr_data["b"] < 0: errors.append("Arrival: Values cannot be negative.")
            if arr_data["b"] <= arr_data["a"]: errors.append("Arrival: Max must be > Min.")
        else:
            if arr_data["raw_val"] <= 0: errors.append("Arrival: Mean/Rate must be > 0.")
            if "v_input" in arr_data and arr_data["v_input"] < 0: errors.append(f"Arrival: {arr_data['var_mode']} cannot be negative.")

        if ser_data["type"] == "Uniform":
            if ser_data["a"] < 0 or ser_data["b"] < 0: errors.append("Service: Values cannot be negative.")
            if ser_data["b"] <= ser_data["a"]: errors.append("Service: Max must be > Min.")
        else:
            if ser_data["raw_val"] <= 0: errors.append("Service: Mean/Rate must be > 0.")
            if "v_input" in ser_data and ser_data["v_input"] < 0: errors.append(f"Service: {ser_data['var_mode']} cannot be negative.")

        if errors:
            for e in errors: st.error(f"Invalid Input: {e}")
        else:
            if model == "M/M/1": display_results(compute_mm1(lmbd, mu))
            elif model == "M/M/s": display_results(compute_mms(lmbd, mu, s))
            elif model == "M/G/1": display_results(compute_mg1(lmbd, ms, vs))
            elif model == "M/G/s": display_results(compute_mgs(lmbd, ms, vs, s))
            elif model == "G/G/1": display_results(compute_gg1(lmbd, ma, va, ms, vs))
            elif model == "G/G/s": display_results(compute_ggs(lmbd, ma, va, ms, vs, s))

if __name__ == "__main__":
    queuing_calculator_ui()