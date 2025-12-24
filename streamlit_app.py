import streamlit as st
import math, random, pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from queuing_calculator import queuing_calculator_ui

# ---------- POISSON PROBABILITY FUNCTION ----------
def poisson_probs(lam):
    probs, cum = [], []
    i, total = 0, 0.0
    
    while True:
        p = math.exp(-lam) * (lam ** i) / math.factorial(i)
        total += p
        
        rounded_p = round(p, 5)
        rounded_cp = round(total, 5)
        
        if rounded_cp >= 0.99999:
            probs.append(rounded_p)
            cum.append(1.00000)
            break
            
        probs.append(rounded_p)
        cum.append(rounded_cp)
        i += 1
        
        # Safe break
        if i > 50: break 
        
    return probs, cum

# ---------- Utilization and queue length Graphs ----------
def get_time_series_data(df, max_time, num_servers, gantt):
    times = sorted(list(set(df['Arrival Time'].tolist() + df['Start Time'].tolist() + df['End Time'].tolist() + [0, max_time])))
    queue_length = []
    server_status = {s_id: [] for s_id in range(1, num_servers + 1)}

    for t in times:
        waiting = df[(df['Arrival Time'] <= t) & (df['Start Time'] > t)].shape[0]
        queue_length.append(waiting)
        
        for s_id in range(1, num_servers + 1):
            is_busy = 0
            for cust_segments in gantt:
                for seg in cust_segments:
                    if seg['server'] == s_id and seg['start'] <= t < seg['end']:
                        is_busy = 1
                        break
                if is_busy: break
            server_status[s_id].append(is_busy)
        
    return times, queue_length, server_status

def validate_inputs(lmbd, mu=None, sigma=None, a=None, b=None, s=1):
    errors = []

    # Arrival rate
    if lmbd is None or lmbd <= 0:
        errors.append("λ (Arrival Rate) cannot be zero or negative.")

    # Servers
    if s <= 0:
        errors.append("Number of servers (s) must be at least 1.")

    # MM models
    if st.session_state.model in ["MM1", "MMS"]:
        if mu is None or mu <= 0:
            errors.append("μ (Service Rate) cannot be zero or negative.")

    # MG models
    if st.session_state.model in ["MG1", "MGS"]:

        if st.session_state.service_dist == "normal":
            if mu is None or mu <= 0:
                errors.append("μ (Mean Service Time) cannot be zero or negative.")
            if sigma is None or sigma <= 0:
                errors.append("σ (Standard Deviation) cannot be zero or negative.")

        if st.session_state.service_dist == "uniform":
            if a is None or a <= 0:
                errors.append("a (Minimum Service Time) cannot be zero or negative.")
            if b is None or b <= 0:
                errors.append("b (Maximum Service Time) cannot be zero or negative.")
            if b <= a:
                errors.append("b (Maximum Service Time) must be greater than a (Minimum Service Time).")

    return errors

# ---------- SIMULATION FUNCTION ----------
def generate_simulation(lmbd, mu, s, sigma=None, a=None, b=None, with_priority=False, preemption=False):

    probs, cum = poisson_probs(lmbd)
    n = len(cum)

    inter_arrival = [0]
    arrivals = [0]
    for _ in range(1, n):
        r = random.random()
        idx = next((i for i, c in enumerate(cum) if r < c), len(cum) - 1)
        inter_arrival.append(idx)
        arrivals.append(arrivals[-1] + idx)

    original_service = []

    for _ in range(n):

        # ---------- MM MODELS ----------
        if st.session_state.model in ["MM1", "MMS"]:
            stime = -math.log(random.random()) * mu

        # ---------- MG MODELS ----------
        else:
            r1 = random.random()
            r2 = random.random()

            # Uniform Distribution
            if st.session_state.service_dist == "uniform":
                stime = a + (b - a) * r1

            # Normal Distribution 
            else:
                stime = mu + sigma * math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)

        # Safety
        original_service.append(max(1, int(round(stime))))


    remaining = original_service.copy()
    priority = [random.randint(1, 3) for _ in range(n)] if with_priority else [0]*n
    
    servers = [{"cust": None, "end": 0} for _ in range(s)]
    gantt = [[] for _ in range(n)]
    waiting = []
    
    current_time = 0
   
    event_times = sorted(list(set(arrivals)))

    while current_time < max(arrivals) + sum(original_service) + 10:
        # 1. Job end then free the server
        for j in range(s):
            if servers[j]["cust"] is not None and servers[j]["end"] <= current_time:
                servers[j]["cust"] = None

        # 2. put new arrivals in waiting
        for i in range(n):
            if arrivals[i] == current_time:
                waiting.append(i)

        # 3. Priority Sort
        if with_priority:
            waiting.sort(key=lambda x: (priority[x], arrivals[x]))
        else:
            waiting.sort(key=lambda x: arrivals[x])

        # 4. Preemption Logic
        if with_priority and preemption and waiting:
            for j in range(s):
                curr_c = servers[j]["cust"]
                if curr_c is not None:
                    if priority[waiting[0]] < priority[curr_c]:
                        # Preempt 
                        gantt[curr_c][-1]["end"] = current_time
                        remaining[curr_c] = servers[j]["end"] - current_time
                        waiting.append(curr_c)
                        servers[j]["cust"] = None
                        # Sort again
                        waiting.sort(key=lambda x: (priority[x], arrivals[x]))

        # 5. Give job to free servers (WITHOUT unnecessary switching)
        for j in range(s):
            if servers[j]["cust"] is None and waiting:
                c = waiting.pop(0)
                gantt[c].append({
                    "start": current_time,
                    "end": current_time + remaining[c],
                    "server": j + 1
                })
                servers[j] = {"cust": c, "end": current_time + remaining[c]}

        # Stop condition
        if not waiting and all(srv["cust"] is None for srv in servers) and current_time >= max(arrivals):
            break
            
        # Event-based skipping
        next_arrival = min([a for a in arrivals if a > current_time], default=float('inf'))
        next_finish = min([srv["end"] for srv in servers if srv["cust"] is not None], default=float('inf'))
        current_time = min(next_arrival, next_finish)
        if current_time == float('inf'): break

    # Metrics calculation 
    start_times = [min(seg["start"] for seg in gantt[i]) for i in range(n)]
    end_times = [max(seg["end"] for seg in gantt[i]) for i in range(n)]
    tat = [end_times[i] - arrivals[i] for i in range(n)]
    wt = [max(0, tat[i] - original_service[i]) for i in range(n)]
    rt = [start_times[i] - arrivals[i] for i in range(n)]
    server_assigned = [gantt[i][0]["server"] for i in range(n)]

    df = pd.DataFrame({
        "     ID": range(1, n+1), "Cum. Prob.": cum, "C.P Lookup": [0.0] + cum[:-1],
        "Inter Arrival": inter_arrival, "Arrival Time": arrivals,
        "Service Time": original_service, "Priority": priority if with_priority else "–",
        "Start Time": start_times, "End Time": end_times,
        "Waiting Time": wt, "Turnaround Time": tat, "Response Time": rt,
        "Server": server_assigned
    })

    return df, {"Avg Waiting": sum(wt)/n, "Avg Turnaround": sum(tat)/n}, gantt

# ---------- STREAMLIT UI STYLING ----------
st.set_page_config(page_title="Simulation System", layout="centered")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #F3E5F5 0%, #E0F7FA 100%);
}
div[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F3E5F5 0%, #E0F7FA 100%);
}
h1 {color: #6A1B9A; text-align: center; font-size: 55px; text-shadow: 1px 1px 3px rgba(0,0,0,0.08);}
h2, h3, h4 {color: #00897B;}
.stButton > button {
    background: linear-gradient(90deg, #BA68C8, #4DB6AC);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-size: 17px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 3px 7px rgba(0,0,0,0.15);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #BA68C8, #4DB6AC);
    transform: scale(1.05);
    box-shadow: 0 5px 10px rgba(0,0,0,0.2);
}
div[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 15px;
    padding: 12px;
    box-shadow: 0 0 12px rgba(0,0,0,0.1);
}
input, select, textarea {
    background-color: white !important;
    color: #333 !important;
    border: 1px solid #BDBDBD !important;
    border-radius: 8px !important;
    padding: 0.4em !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
[data-testid="stMetricValue"] {
    color: #6A1B9A;
    font-size: 30px;
}
[data-testid="stMetricLabel"] {
    color: #00897B;
    font-weight: bold;
}
/* Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Hero Section */
.hero-section {
    padding: 50px 20px;
    text-align: center;
    background: white;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 40px;
}

    /* Modern Cards */
.option-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e1e4e8;
    transition: all 0.3s ease;
    text-align: center;
    min-height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.option-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    border-color: #6A1B9A;
}

.icon-circle {
    width: 80px;
    height: 80px;
    background: #f3e5f5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 20px auto;
    font-size: 40px;
}

.card-title {
    color: #6A1B9A;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 15px;
}

.card-text {
    color: #586069;
    font-size: 16px;
    line-height: 1.5;
    margin-bottom: 25px;
}
.footer-card {
    text-align: center;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 15px;
    border: 1px solid #e1e4e8;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
.footer-icon {
    font-size: 30px;
    margin-bottom: 10px;
}
.footer-text {
    font-weight: bold;
    color: #00897B;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "start"
if "model" not in st.session_state:
    st.session_state.model = None
if "priority" not in st.session_state:
    st.session_state.priority = None
if "preemption" not in st.session_state:
    st.session_state.preemption = False
if "mode" not in st.session_state:
    st.session_state.mode = None   # simulator / calculator

if "service_dist" not in st.session_state:
    st.session_state.service_dist = None  # normal / uniform

# ---------- PAGE LOGIC ----------
# 1. Start
if st.session_state.page == "start":
    # --- HERO SECTION ---
    st.markdown("""
        <div class="hero-section">
            <h1 style='color: #6A1B9A; font-size: 48px; margin-bottom: 10px;'>Operations Research Studio</h1>
            <p style='color: #586069; font-size: 20px;'>Advanced Discrete Event Simulation & Queuing Theory Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN OPTIONS ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="option-card">
                <div>
                    <div class="icon-circle">🧪</div>
                    <div class="card-title">Simulator</div>
                    <p class="card-text">
                        Perform <b>Discrete Event Simulation (DES)</b> to visualize real-world scenarios. 
                        Track customer paths, analyze server bottlenecks, and generate Gantt charts with dynamic priority handling.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Simulation Engine", use_container_width=True, key="sim_btn"):
            st.session_state.mode = "simulator"
            st.session_state.page = "model"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="option-card">
                <div>
                    <div class="icon-circle">📊</div>
                    <div class="card-title">Queuing Calculator</div>
                    <p class="card-text">
                        Compute steady-state performance metrics</b>. 
                        Get instant results for Utilization (ρ), Expected Wait Time (Wq), and System Length (L) using pure mathematical models.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Queuing Tools", use_container_width=True, key="calc_btn"):
            st.session_state.mode = "calculator"
            st.session_state.page = "calculator"
            st.rerun()

    # --- FEATURE FOOTER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Using a container and columns to structure the footer
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
            <div class="footer-card">
                <div class="footer-icon">📚</div>
                <div class="footer-text">Support for<br>Multiple Models</div>
            </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
            <div class="footer-card">
                <div class="footer-icon">📈</div>
                <div class="footer-text">Advanced<br>Distributions</div>
            </div>
        """, unsafe_allow_html=True)

    with feat_col3:
        st.markdown("""
            <div class="footer-card">
                <div class="footer-icon">⚡</div>
                <div class="footer-text">Priority<br>Execution</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    feat_col1, feat_col2, feat_col3 = st.columns(3)

# 2. Model Selection
elif st.session_state.page == "model":
    st.markdown("<h3 style='text-align:center;'>Select Model Type</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        if st.button("📘 M/M/1", use_container_width=True):
            st.session_state.model = "MM1"
            st.session_state.page = "priority"
            st.rerun()

    with col2:
        if st.button("📗 M/M/s", use_container_width=True):
            st.session_state.model = "MMS"
            st.session_state.page = "priority"
            st.rerun()

    with col3:
        if st.button("📙 M/G/1", use_container_width=True):
            st.session_state.model = "MG1"
            st.session_state.page = "distribution"
            st.rerun()

    with col4:
        if st.button("📕 M/G/s", use_container_width=True):
            st.session_state.model = "MGS"
            st.session_state.page = "distribution"
            st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.page = "start"
        st.rerun()


#--------------DISTRIBUTION SELECTION--------------
elif st.session_state.page == "distribution":
    st.markdown("<h3 style='text-align:center;'>Select Service Time Distribution</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔔 Normal Distribution", use_container_width=True):
            st.session_state.service_dist = "normal"
            st.session_state.page = "priority"
            st.rerun()

    with col2:
        if st.button("📐 Uniform Distribution", use_container_width=True):
            st.session_state.service_dist = "uniform"
            st.session_state.page = "priority"
            st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.page = "model"
        st.rerun()


# 3. Priority Selection
elif st.session_state.page == "priority":
    st.markdown("<h3 style='text-align:center;'>Select Simulation Type</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧮 Without Priority", use_container_width=True):
            st.session_state.priority = False
            st.session_state.preemption = False
            st.session_state.page = "run"
            st.rerun()
    with col2:
        if st.button("⚙️ With Priority", use_container_width=True):
            st.session_state.priority = True
            st.session_state.page = "priority_options"
            st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back", use_container_width=True):
        if st.session_state.model in ["MG1", "MGS"]:
            st.session_state.page = "distribution"
        else:
            st.session_state.page = "model"
        st.rerun()


# 3.5 Priority Options
elif st.session_state.page == "priority_options":
    st.markdown("<h3 style='text-align:center;'>Select Priority Handling</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏳ Without Preemption", use_container_width=True):
            st.session_state.preemption = False
            st.session_state.page = "run"
            st.rerun()
    with col2:
        if st.button("⚡ With Preemption", use_container_width=True):
            st.session_state.preemption = True
            st.session_state.page = "run"
            st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.page = "priority"
        st.rerun()

# 4. Run Simulation
elif st.session_state.page == "run":

    st.markdown(
        "<h3>Enter Simulation Parameters</h3>",
        unsafe_allow_html=True
    )

    with st.container():

        # ---------- Inputs ----------
        lmbd = st.number_input(
            "λ (Arrival Rate)",
            #min_value=0.1,
            value=2.0,
            step=0.1
        )

        # ---- SERVICE DISTRIBUTION LOGIC ----
        if st.session_state.model in ["MG1", "MGS"]:

            if st.session_state.service_dist == "normal":
                mu = st.number_input(
                    "μ (Service Rate)",
                    #min_value=0.1,
                    value=3.0,
                    step=0.1
                )
                sigma = st.number_input(
                    "σ (Standard Deviation)",
                    #min_value=0.1,
                    value=1.0,
                    step=0.1
                )

            elif st.session_state.service_dist == "uniform":
                a = st.number_input(
                    "a (Minimum Service Time)",
                    #min_value=0.1,
                    value=1.0
                )
                b = st.number_input(
                    "b (Maximum Service Time)",
                    value=4.0
                )

        else:
            mu = st.number_input(
                "μ (Service Rate)",
                #min_value=0.1,
                value=3.0,
                step=0.1
            )   

        # ---------- Servers ----------
        if st.session_state.model in ["MM1", "MG1"]:
            s = 1
            st.info("📘 Single Server Model → Servers (s) = 1")
        else:
            s = st.number_input(
                "Number of Servers (s)",
                min_value=1,
                value=2,
                step=1
            )

        # ---------- Run Simulation ----------
        if st.button("▶️ Run Simulation"):

            # ---------- INPUT VALIDATION ----------
            errors = validate_inputs(
                lmbd=lmbd,
                mu=mu if 'mu' in locals() else None,
                sigma=sigma if 'sigma' in locals() else None,
                a=a if 'a' in locals() else None,
                b=b if 'b' in locals() else None,
                s=s
            )

            if errors:
                st.error("❌ Invalid Inputs:")
                for e in errors:
                    st.write(f"• {e}")
                    st.stop()  

            # ---------- RHO CALCULATION ----------
            if st.session_state.model in ["MM1", "MMS"]:
                rho = lmbd / (mu * s)

            elif st.session_state.service_dist == "uniform":
                mu = (a + b) / 2
                rho = lmbd / (s * mu)

            else:  # normal
                rho = lmbd / (mu * s)

            if rho > 1:
                st.error(
                    f"❌ Simulation does not execute as ρ = {rho:.3f} > 1"
                )
            else:
                st.success(
                    f"✅ Simulation executed successfully — ρ = {rho:.3f}"
                )

                df, summary, gantt = generate_simulation(
                    lmbd,
                    mu,
                    s,
                    sigma=sigma if st.session_state.service_dist == "normal" else None,
                    a=a if st.session_state.service_dist == "uniform" else None,
                    b=b if st.session_state.service_dist == "uniform" else None,
                    with_priority=st.session_state.priority,
                    preemption=st.session_state.preemption
                )

                # ---------- Table ----------
                st.dataframe(
                    df.style.format({
                        "Cum. Prob.": "{:.5f}",
                        "C.P Lookup": "{:.5f}"
                    }),
                    use_container_width=True
                )

                # ---------- Time Series Calculations ----------
                max_sim_time = df["End Time"].max()
                # Pass 's' and 'gantt' to the new function
                times, q_t, server_status_dict = get_time_series_data(df, max_sim_time, s, gantt)

                # ---------- STATISTICAL SUMMARY ----------
                st.markdown("### 📈 Simulation Summary Metrics")

                # ---- 1. Basic Metrics ----
                avg_wt = df["Waiting Time"].mean()
                avg_tat = df["Turnaround Time"].mean()
                avg_rt = df["Response Time"].mean()
                total_sim_time = df["End Time"].max()

                # ---- 2. Individual Server Utilization ----
                server_busy_times = {}

                for s_id, b_t_list in server_status_dict.items():
                    busy_duration = 0
                    for i in range(len(times) - 1):
                        if b_t_list[i] == 1:
                            busy_duration += (times[i + 1] - times[i])
                    server_busy_times[s_id] = busy_duration

                # ---- 3. Overall Utilization ----
                total_busy_time = sum(server_busy_times.values())
                overall_utilization = total_busy_time / (total_sim_time * s)
                idle_factor = (1 - overall_utilization) * 100

                # ---------- UI Metrics ----------
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                with m_col1:
                    st.metric(
                        "Avg Waiting Time (Wq)",
                        f"{avg_wt:.2f}"
                    )

                with m_col2:
                    st.metric(
                        "Avg Turnaround (W)",
                        f"{avg_tat:.2f}"
                    )

                with m_col3:
                    st.metric("Avg Response (RT)", f"{avg_rt:.2f}")

                with m_col4:
                    st.metric(
                        "Total Customers",
                        len(df)
                    )

                # ---------- Per Server Utilization ----------
                st.markdown("#### 🖥️ Server Utilization Details")

                u_cols = st.columns(s if s <= 4 else 4)

                for idx, (s_id, b_time) in enumerate(server_busy_times.items()):
                    s_util = b_time / total_sim_time
                    with u_cols[idx % 4]:
                        st.metric(
                            f"Server {s_id} Util",
                            f"{s_util:.2%}"
                        )

                # ---------- Final Factors ----------
                f_col1, f_col2 = st.columns(2)

                with f_col1:
                    st.metric(
                        "Overall Utilization Factor",
                        f"{overall_utilization:.2%}"
                    )

                with f_col2:
                    st.metric(
                        "System Idle Factor",
                        f"{idle_factor:.2f}%"
                    )

                st.markdown("---")

                # ---------- Q(t) Graph (Optimized) ----------
                st.subheader("📊 Queue Length Over Time $Q(t)$")
                if times:
                    fig_q, ax_q = plt.subplots(figsize=(15, 4))
                    
                    # Step plot with 'post' for correct staircase effect
                    ax_q.step(times, q_t, where="post", color='#863F93', linewidth=1.5)
                    
                    ax_q.fill_between(times, q_t, step="post", facecolor="none", 
                                      edgecolor="#863F93", hatch='xxx', alpha=0.4)
                    
                    ax_q.set_xlabel("t (Time)")
                    ax_q.set_ylabel("Q(t)")

                    ax_q.set_xticks(times) 
                    plt.xticks(rotation=45, fontsize=8) 
                    
                    # Ensure y-axis shows only whole numbers (0, 1, 2, 3...)
                    max_q = int(max(q_t)) if q_t else 0
                    ax_q.set_yticks(range(max_q + 2))
                    
                    # Clean look like the screenshots
                    ax_q.spines['top'].set_visible(False)
                    ax_q.spines['right'].set_visible(False)
                    ax_q.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_q)

                # ---------- Individual B(t) Graphs for Each Server ----------
                st.subheader("💡 Individual Server Utilization $B(t)$")
                
                for s_id, b_t_individual in server_status_dict.items():
                    st.write(f"**Server {s_id} Status**")
                    fig_b, ax_b = plt.subplots(figsize=(12, 2))
                    
                    # Black & White style
                    ax_b.step(times, b_t_individual, where="post", color='#900E40', linewidth=1.5)
                    ax_b.fill_between(times, b_t_individual, step="post", facecolor="none", 
                                      edgecolor="#900E40", hatch='////', alpha=0.5)
                    
                    ax_b.set_ylim(-0.1, 1.2)
                    ax_b.set_yticks([0, 1])
                    ax_b.set_ylabel(f"B{s_id}(t)")

                    ax_b.set_xticks(times)
                    plt.xticks(rotation=45, fontsize=8)
                    
                    # Clean look
                    ax_b.spines['top'].set_visible(False)
                    ax_b.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_b)

                # ---------- Utilization ----------
                fig, ax = plt.subplots()
                ax.bar(["Busy", "Idle"], [rho, max(0, 1 - rho)])
                ax.set_ylabel("Fraction of Time")
                ax.set_title("🧩 Server Utilization Overview")
                st.pyplot(fig)


                # ---------- Gantt Chart ----------
                st.subheader("🧩 Server-wise Gantt Charts")
                colors = ["#BA68C8","#FF8A65","#FFD54F","#4DB6AC","#64B5F6","#A1887F"]

                max_sim_time = max(seg["end"] for segments in gantt for seg in segments)
                server_timelines = {srv: [] for srv in range(1, s+1)}

                for cust_id, segments in enumerate(gantt, start=1):
                    for seg in segments:
                        server_timelines[seg["server"]].append({
                            "cust": f"C{cust_id}",
                            "start": seg["start"],
                            "end": seg["end"],
                            "color": colors[cust_id % len(colors)]
                        })

                for srv_id, tasks in server_timelines.items():
                    tasks.sort(key=lambda x: x["start"])
                    merged_tasks = []
                    if tasks:
                        current_task = tasks[0].copy()
                        for next_task in tasks[1:]:
                            if next_task["cust"] == current_task["cust"] and next_task["start"] == current_task["end"]:
                                current_task["end"] = next_task["end"]
                            else:
                                merged_tasks.append(current_task)
                                current_task = next_task.copy()
                        merged_tasks.append(current_task)

                    filled_tasks = []
                    clean_ticks = [0]
                    last_end = 0
                    for task in merged_tasks:
                        if task["start"] > last_end:
                            filled_tasks.append({"cust":"Idle","start":last_end,"end":task["start"],"color":"#FFCDD2","hatch":"///"})
                        filled_tasks.append(task)
                        clean_ticks += [task["start"], task["end"]]
                        last_end = task["end"]
                    if last_end < max_sim_time:
                        filled_tasks.append({"cust":"Idle","start":last_end,"end":max_sim_time,"color":"#FFCDD2","hatch":"///"})
                        clean_ticks.append(max_sim_time)

                    fig, ax = plt.subplots(figsize=(25,2.5))
                    for t in filled_tasks:
                        duration = t["end"] - t["start"]
                        ax.barh(0,duration,left=t["start"],color=t["color"],edgecolor="black",linewidth=0.5,hatch=t.get("hatch",""))
                        if duration>0.3:
                            ax.text(t["start"]+duration/2,0,t["cust"],ha="center",va="center",fontsize=11,fontweight="bold",color="white" if t["cust"]!="Idle" else "#B71C1C")
                    ax.set_title(f"Server {srv_id}",loc="left",fontweight="bold")
                    ax.set_yticks([])
                    ax.set_xlabel("Time")
                    ax.set_xlim(0,max_sim_time)
                    ax.set_xticks(sorted(set(clean_ticks)))
                    plt.xticks(rotation=45)
                    for spine in ["top","right","left"]: ax.spines[spine].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

        # ---------- Back ----------
        if st.button("🏠 Back to Start"):
            st.session_state.page = "start"
            st.rerun()

# 5. Queuing Calculator
elif st.session_state.page == "calculator":

    st.markdown(
        "<h1 style='text-align:center;'>📊 Queuing Calculator</h1>",
        unsafe_allow_html=True
    )

    queuing_calculator_ui()

    st.markdown("---")
    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.page = "start"
        st.rerun()

