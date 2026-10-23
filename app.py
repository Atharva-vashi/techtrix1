import io
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="iPDS stock control room", page_icon="🌾", layout="wide")
TODAY = date(2026, 10, 23)          # change to date.today() for live use
KG_PER_BENEFICIARY = 5
COMMODITIES = ["Rice", "Wheat"]


# ---------------------------------------------------------------- data (dummy)
@st.cache_data
def make_data():
    np.random.seed(42)
    godowns = pd.DataFrame({"godown_id": ["G1", "G2", "G3", "G4"],
                            "name": ["Godown North", "Godown South", "Godown East", "Godown West"]})
    rows, n = [], 1
    for g in godowns.godown_id:
        for c in COMMODITIES:
            for _ in range(np.random.randint(3, 6)):
                age = int(np.random.randint(10, 170))
                rows.append({"batch_id": f"B{n:03d}", "godown_id": g, "commodity": c,
                             "received_date": TODAY - timedelta(days=age),
                             "quantity_kg": int(np.random.randint(2, 9)) * 1000,
                             "shelf_life_days": 180 if c == "Rice" else 150})
                n += 1
    stock = pd.DataFrame(rows)
    stock["expiry_date"] = stock.apply(lambda r: r.received_date + timedelta(days=int(r.shelf_life_days)), axis=1)
    stock["days_to_expiry"] = stock.expiry_date.apply(lambda d: (d - TODAY).days)
    stock["age_days"] = stock.received_date.apply(lambda d: (TODAY - d).days)

    n_shops = 30
    shops = pd.DataFrame({"shop_id": [f"FPS{i:02d}" for i in range(1, n_shops + 1)],
                          "beneficiaries": np.random.randint(300, 1200, n_shops)})
    shops["entitled_Rice"] = (shops.beneficiaries * KG_PER_BENEFICIARY * 0.6).astype(int)
    shops["entitled_Wheat"] = (shops.beneficiaries * KG_PER_BENEFICIARY * 0.4).astype(int)
    dist = pd.DataFrame(np.random.randint(5, 120, size=(4, n_shops)),
                        index=godowns.godown_id, columns=shops.shop_id)
    return godowns, stock, shops, dist


@st.cache_data
def make_forecast(_shops_key):
    _, _, shops, _ = make_data()
    N = 36
    months = pd.period_range(end="2026-10", periods=N, freq="M")
    rng = np.random.default_rng(7)
    rows = []
    for sh in shops.itertuples():
        growth, uptake = rng.uniform(0.001, 0.006), rng.uniform(0.82, 0.97)
        for c, share, amp in [("Rice", 0.6, 0.08), ("Wheat", 0.4, 0.12)]:
            for t, m in enumerate(months):
                bene = sh.beneficiaries / (1 + growth) ** (N - 1 - t)
                seas = 1 + amp * np.sin(2 * np.pi * (m.month - 8) / 12)
                rows.append({"shop_id": sh.shop_id, "commodity": c, "month": m, "beneficiaries": sh.beneficiaries,
                             "qty_kg": bene * KG_PER_BENEFICIARY * share * uptake * seas * rng.normal(1, 0.04)})
    hist = pd.DataFrame(rows).sort_values(["shop_id", "commodity", "month"]).reset_index(drop=True)
    g = hist.groupby(["shop_id", "commodity"]).qty_kg
    for k in (1, 2, 3, 12):
        hist[f"lag{k}"] = g.shift(k)
    hist["roll3"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    hist["month_sin"] = np.sin(2 * np.pi * hist.month.dt.month / 12)
    hist["month_cos"] = np.cos(2 * np.pi * hist.month.dt.month / 12)
    hist["is_wheat"] = (hist.commodity == "Wheat").astype(int)
    F = ["lag1", "lag2", "lag3", "lag12", "roll3", "month_sin", "month_cos", "beneficiaries", "is_wheat"]
    data = hist.dropna(subset=F)
    train, test = data[data.month < months[-3]], data[data.month >= months[-3]]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(train[F], train.qty_kg)
    p = model.predict(test[F])
    acc = {"model_err": float(np.mean(np.abs(p - test.qty_kg) / test.qty_kg) * 100),
           "naive_err": float(np.mean(np.abs(test.lag1 - test.qty_kg) / test.qty_kg) * 100)}
    model.fit(data[F], data.qty_kg)
    nxt = months[-1] + 1
    fr = []
    for (s, c), gg in hist.groupby(["shop_id", "commodity"]):
        q = gg.qty_kg.values
        fr.append({"shop_id": s, "commodity": c, "beneficiaries": gg.beneficiaries.iloc[0], "lag1": q[-1],
                   "lag2": q[-2], "lag3": q[-3], "lag12": q[-12], "roll3": q[-3:].mean(),
                   "month_sin": np.sin(2 * np.pi * nxt.month / 12), "month_cos": np.cos(2 * np.pi * nxt.month / 12),
                   "is_wheat": int(c == "Wheat")})
    fc = pd.DataFrame(fr)
    fc["forecast_kg"] = model.predict(fc[F])
    return hist[["shop_id", "commodity", "month", "qty_kg"]], fc[["shop_id", "commodity", "forecast_kg"]], acc, str(nxt)


# ---------------------------------------------------------------- planning
@st.cache_data
def make_plan(window, buffer, cost_km, urg_pen):
    godowns, stock, shops, dist = make_data()
    _, fc, acc, nxt = make_forecast(1)
    stock = stock.sort_values(["godown_id", "commodity", "received_date"]).reset_index(drop=True)
    stock["near_expiry"] = stock.days_to_expiry <= window
    demand = {c: fc[fc.commodity == c].set_index("shop_id").forecast_kg.mul(1 + buffer).round().reindex(shops.shop_id)
              for c in COMMODITIES}

    def allocate(c):
        gs = stock[stock.commodity == c].groupby("godown_id").quantity_kg.sum()
        gu = stock[(stock.commodity == c) & stock.near_expiry].groupby("godown_id").quantity_kg.sum() \
            .reindex(gs.index, fill_value=0)
        d = demand[c]
        G, S = list(gs.index), list(d.index)
        nG, nS = len(G), len(S)
        nx = nG * nS
        n = nx + nS + nG
        cost = np.zeros(n)
        for gi, g in enumerate(G):
            for si, s in enumerate(S):
                cost[gi * nS + si] = dist.loc[g, s] * cost_km
        cost[nx:nx + nS], cost[nx + nS:] = 50.0, urg_pen
        Aeq, beq = np.zeros((nS, n)), d.values.astype(float)
        for si in range(nS):
            for gi in range(nG):
                Aeq[si, gi * nS + si] = 1
            Aeq[si, nx + si] = 1
        Aub, bub = np.zeros((2 * nG, n)), np.zeros(2 * nG)
        for gi, g in enumerate(G):
            Aub[gi, gi * nS:(gi + 1) * nS] = 1
            bub[gi] = gs[g]
            Aub[nG + gi, gi * nS:(gi + 1) * nS] = -1
            Aub[nG + gi, nx + nS + gi] = -1
            bub[nG + gi] = -gu[g]
        res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        out = pd.DataFrame([(g, s, res.x[gi * nS + si]) for gi, g in enumerate(G) for si, s in enumerate(S)],
                           columns=["godown_id", "shop_id", "qty_kg"])
        out = out[out.qty_kg > 0.5].copy()
        out["commodity"] = c
        return out

    plan = pd.concat([allocate(c) for c in COMMODITIES], ignore_index=True)
    plan["qty_kg"] = plan.qty_kg.round()

    rem = stock.set_index("batch_id").quantity_kg.to_dict()
    ledger = []
    for (g, c), grp in plan.groupby(["godown_id", "commodity"]):
        batches = stock[(stock.godown_id == g) & (stock.commodity == c)].sort_values("received_date").batch_id
        for r in grp.itertuples():
            need = r.qty_kg
            for b in batches:
                take = min(need, rem[b])
                if take > 0:
                    ledger.append({"godown_id": g, "commodity": c, "shop_id": r.shop_id, "batch_id": b, "qty_kg": take})
                    rem[b] -= take
                    need -= take
                if need <= 0:
                    break
    ledger = pd.DataFrame(ledger)
    stock["left_kg"] = stock.batch_id.map(rem)
    stock["issued_kg"] = stock.quantity_kg - stock.left_kg

    # naive comparison: nearest godown first, random batch order
    avail = stock.groupby(["godown_id", "commodity"]).quantity_kg.sum().to_dict()
    nv = []
    for s in shops.shop_id:
        for c in COMMODITIES:
            need = demand[c][s]
            for g in dist[s].sort_values().index:
                take = min(need, avail.get((g, c), 0))
                if take > 0:
                    nv.append({"godown_id": g, "shop_id": s, "commodity": c, "qty_kg": take})
                    avail[(g, c)] -= take
                    need -= take
                if need <= 0:
                    break
    nv = pd.DataFrame(nv)
    nrem = stock.set_index("batch_id").quantity_kg.to_dict()
    for (g, c), grp in nv.groupby(["godown_id", "commodity"]):
        tot = grp.qty_kg.sum()
        for r in stock[(stock.godown_id == g) & (stock.commodity == c)].sample(frac=1, random_state=1).itertuples():
            take = min(tot, nrem[r.batch_id])
            nrem[r.batch_id] -= take
            tot -= take
    tc = lambda p: float(sum(r.qty_kg * dist.loc[r.godown_id, r.shop_id] * cost_km for r in p.itertuples()))
    k = {"cost_smart": tc(plan), "cost_naive": tc(nv), "issued": float(plan.qty_kg.sum()),
         "urg_smart": float(stock[stock.near_expiry].left_kg.sum()),
         "urg_naive": float(sum(nrem[b] for b in stock[stock.near_expiry].batch_id))}
    return stock, plan, ledger, k


# ---------------------------------------------------------------- page
godowns, _, shops, dist = make_data()
hist, fc, acc, nxt = make_forecast(1)
gname = dict(zip(godowns.godown_id, godowns.name))

st.title("🌾 iPDS stock control room")
st.caption(f"Dummy data. Planning month: {nxt}. Change the sliders and the whole plan is recalculated.")
with st.sidebar:
    st.header("Settings")
    window = st.slider("Near-expiry window (days)", 10, 90, 30)
    buffer = st.slider("Safety buffer on forecast (%)", 0, 20, 5) / 100
    cost_km = st.slider("Transport cost (Rs per kg per km)", 0.005, 0.10, 0.02, 0.005)
    urg_pen = st.slider("Priority for clearing old stock", 1.0, 20.0, 5.0, 0.5,
                        help="Higher means the optimizer pays more transport cost to clear near-expiry stock.")

stock, plan, ledger, k = make_plan(window, buffer, cost_km, urg_pen)
t1, t2, t3, t4, t5 = st.tabs(["Overview", "FIFO queue", "Allocation", "Forecast", "Download"])

with t1:
    c = st.columns(4)
    c[0].metric("To issue next month", f"{k['issued']:,.0f} kg")
    c[1].metric("Transport saving vs nearest-godown", f"{(1 - k['cost_smart'] / k['cost_naive']) * 100:.1f}%")
    c[2].metric("Near-expiry stock left", f"{k['urg_smart']:,.0f} kg", f"naive: {k['urg_naive']:,.0f} kg", delta_color="off")
    c[3].metric("Forecast error", f"{acc['model_err']:.1f}%", f"last-month guess: {acc['naive_err']:.1f}%", delta_color="off")
    cmp_df = pd.DataFrame({"Naive (nearest godown, no FIFO)": [k["cost_naive"], k["urg_naive"]],
                           "Smart (forecast + optimizer + FIFO)": [k["cost_smart"], k["urg_smart"]]},
                          index=["Transport cost (Rs)", "Near-expiry stock left (kg)"]).round()
    st.dataframe(cmp_df, width="stretch")
    st.subheader("Issue first: near-expiry batches")
    al = stock[stock.near_expiry][["batch_id", "godown_id", "commodity", "quantity_kg", "expiry_date", "days_to_expiry", "left_kg"]]
    st.dataframe(al, width="stretch", hide_index=True) if len(al) else st.info("No batches within the window.")

with t2:
    st.caption("Oldest batch on the left. Colour shows shelf life used (green to red). Hatching is what the plan issues.")
    for g in godowns.godown_id:
        fig, axes = plt.subplots(2, 1, figsize=(10, 2.6))
        for ax, cm in zip(axes, COMMODITIES):
            x = 0
            for r in stock[(stock.godown_id == g) & (stock.commodity == cm)].sort_values("received_date").itertuples():
                ax.barh(0, r.quantity_kg, left=x, color=plt.cm.RdYlGn_r(min(r.age_days / r.shelf_life_days, 1)),
                        edgecolor="white")
                ax.barh(0, r.issued_kg, left=x, color="none", hatch="//", edgecolor="black", lw=0)
                ax.text(x + r.quantity_kg / 2, 0, r.batch_id, ha="center", va="center", fontsize=7)
                x += r.quantity_kg
            ax.set_yticks([]); ax.set_ylabel(cm, rotation=0, ha="right"); ax.margins(x=0)
        fig.suptitle(gname[g], x=0.01, ha="left"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

with t3:
    sel = st.selectbox("Godown", godowns.godown_id, format_func=gname.get)
    pos = {"G1": (50, 8), "G2": (50, 56), "G3": (88, 32), "G4": (12, 32)}
    r3 = np.random.default_rng(3)
    sp = {s: (int(r3.integers(8, 92)), int(r3.integers(8, 60))) for s in shops.shop_id}
    rows = plan[plan.godown_id == sel].groupby("shop_id").qty_kg.sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, q in rows.items():
        ax.plot([pos[sel][0], sp[s][0]], [pos[sel][1], sp[s][1]], color="#1f4e5f", alpha=.5, lw=1 + q / 450)
    ax.scatter(*zip(*sp.values()), c=["#1f4e5f" if s in rows.index else "#cccccc" for s in sp], s=40)
    for g, (gx, gy) in pos.items():
        ax.scatter(gx, gy, marker="s", s=160, c="#b7791f" if g == sel else "#5d6b74")
        ax.text(gx + 2, gy, gname[g], fontsize=9)
    ax.axis("off"); st.pyplot(fig); plt.close(fig)
    st.caption("Schematic layout, not real geography.")
    piv = plan[plan.godown_id == sel].pivot_table(index="shop_id", columns="commodity", values="qty_kg", aggfunc="sum").fillna(0)
    piv["Total"] = piv.sum(axis=1)
    st.write(f"**{gname[sel]}** sends {piv.Total.sum():,.0f} kg to {len(piv)} shops")
    st.dataframe(piv, width="stretch")

with t4:
    a, b = st.columns(2)
    shop = a.selectbox("Shop", shops.shop_id)
    com = b.selectbox("Commodity", COMMODITIES)
    h = hist[(hist.shop_id == shop) & (hist.commodity == com)].set_index("month").qty_kg
    f = float(fc[(fc.shop_id == shop) & (fc.commodity == com)].forecast_kg.iloc[0])
    ch = pd.DataFrame({"Actual": h.values}, index=[str(m) for m in h.index])
    ch.loc[nxt, "Forecast"] = f
    ch.loc[str(h.index[-1]), "Forecast"] = h.iloc[-1]
    st.line_chart(ch)
    ent = int(shops.set_index("shop_id").loc[shop, f"entitled_{com}"])
    st.table(pd.DataFrame({"kg": [ent, round(f), round(f * (1 + buffer))]},
                          index=["Full entitlement", "Model forecast", f"Planned (+{buffer * 100:.0f}% buffer)"]))

with t5:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        stock.to_excel(w, sheet_name="Stock_Batches_FIFO", index=False)
        al.to_excel(w, sheet_name="Near_Expiry_Alerts", index=False)
        plan.to_excel(w, sheet_name="Allocation_Plan", index=False)
        ledger.to_excel(w, sheet_name="FIFO_Dispatch_Ledger", index=False)
        fc.to_excel(w, sheet_name="Demand_Forecast", index=False)
    st.download_button("Download plan as Excel", buf.getvalue(), "ipds_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
