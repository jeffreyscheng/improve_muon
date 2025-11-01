--------------------------------------------------------------------------------
Context (what and why)
--------------------------------------------------------------------------------
Symptoms
- On semilog plots, the actual spectral echo vs singular value deviates from the expected sigmoid and exhibits a ~0.45 plateau beyond s≈5e-3.

Root causes
1) Pairing bug: you flattened per-replica singulars [B,Kc] but paired them with per-direction echoes [Kc]. That mismatches most points and smears the curve.
2) Estimator guard bias: denominator guarding in the “spectral reverb / triple–OLS” applies even when denominators are large, flattening the high-s end.

Fixes
- Aggregate singular values per-direction across replicas (median across B) before pairing with the per-direction echo in all builder functions.
- Make denominator guarding conditional (only when |denominator| is small) and slightly reduce guard strength.

Optional improvement
- Build the alignment consensus matrix using all replica pairs (symmetric), instead of anchoring on replica 0.

Sanity checks
- Assert shapes align ([Kc] vs [Kc]).
- Optional monotonicity diagnostics (Spearman ρ warning).

--------------------------------------------------------------------------------
FILE: empirical/research/analysis/run_gradient_analysis.py
SECTION: build_pred_actual_gptlp
ACTION: Replace entire function body with the following (drop-in)
--------------------------------------------------------------------------------
def build_pred_actual_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    IMPORTANT: Pair per-direction singulars (aggregated across replicas) with per-direction echoes.
    Avoid flattening [B,Kc] singulars; take median across B to get [Kc].
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        # s_rep: [B,Kc], echo: [Kc]
        s_rep = props['aligned_replicate_singular_values']  # torch [B,Kc]
        s_dir = torch.median(s_rep, dim=0).values           # torch [Kc]
        sv = s_dir.detach().cpu().numpy()
        actual = props['spectral_echo'].detach().cpu().numpy()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        if sv.size and actual.size:
            n = min(sv.size, actual.size)
            sv = sv[:n]; actual = actual[:n]
            pred = predict_spectral_echo_curve_np(sv, tau2)
            out[key] = np.vstack([np.clip(pred, 1e-8, 1.0),
                                  np.clip(actual, 1e-8, 1.0)])
    return out

--------------------------------------------------------------------------------
FILE: empirical/research/analysis/run_gradient_analysis.py
SECTION: build_spectral_echo_vs_sv_panel
ACTION: Replace entire function body with the following (drop-in)
--------------------------------------------------------------------------------
def build_spectral_echo_vs_sv_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Store per-direction singulars (median across replicas) and matched per-direction echoes.
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        s_rep = props['aligned_replicate_singular_values']                 # [B,Kc]
        s_dir = torch.median(s_rep, dim=0).values.detach().cpu().numpy()  # [Kc]
        echo = props['spectral_echo'].detach().cpu().numpy()               # [Kc]
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(s_dir.size, echo.size)
        if n:
            out[key] = {
                'sv': s_dir[:n],
                'spectral_echo': echo[:n],
                'tau2': tau2,
                'shape': tuple(int(x) for x in props['checkpoint_weights'].shape[-2:]),
            }
            # optional shape guards (debug-only; remove if noisy)
            assert out[key]['sv'].ndim == 1 and out[key]['spectral_echo'].ndim == 1
            assert out[key]['sv'].shape == out[key]['spectral_echo'].shape
    return out

NOTE:
- With this fix, the plotting helpers (create_spectral_echo_vs_sv_semilog_subplot and create_spectral_echo_vs_sv_semilog_normalized_subplot) do NOT need to flatten. If they currently call .flatten() on the stored arrays, remove that.

--------------------------------------------------------------------------------
FILE: empirical/research/analysis/core_math.py
SECTION: solve_for_spectral_echo_using_reverb (parameters)
ACTION: Tweak defaults to reduce high-s flattening
--------------------------------------------------------------------------------
# Change function signature defaults:
# from: noise_quantile: float = 0.10, tau_mult: float = 4.0
#   to: noise_quantile: float = 0.05, tau_mult: float = 2.0

def solve_for_spectral_echo_using_reverb(
    left_bases_U: torch.Tensor,
    right_bases_V: torch.Tensor,
    noise_quantile: float = 0.05,  # lower-tail quantile to set per-direction noise floor
    tau_mult: float = 2.0,         # gentler guard multiplier to reduce high-s flattening
    weight_power: float = 2.0,
) -> torch.Tensor:

--------------------------------------------------------------------------------
FILE: empirical/research/analysis/core_math.py
SECTION: solve_for_spectral_echo_using_reverb (denominator guarding)
ACTION: Replace denom_safe computation with conditional guard
--------------------------------------------------------------------------------
    denom = spectral_reverb_Z.unsqueeze(0)                               # (1,a,b,Kc)
    sgn = torch.where(denom >= 0, 1.0, -1.0)                             # (1,a,b,Kc)
    # Conditional guarding: only adjust when |denom| is small; otherwise leave unchanged.
    guard = sgn * tau_dir.view(1, 1, 1, Kc)                              # (1,a,b,Kc)
    need_guard = (denom.abs() < tau_dir.view(1, 1, 1, Kc))
    denom_safe = torch.where(need_guard, denom + guard, denom)           # (1,a,b,Kc)

--------------------------------------------------------------------------------
FILE: empirical/research/analysis/core_math.py
SECTION: compute_reverb_echo_and_alignment (OPTIONAL symmetric all-pairs consensus)
ACTION: Replace the “ref=0” consensus block with symmetric all-pairs averaging
--------------------------------------------------------------------------------
# BEFORE (anchored to ref=0), you had code that built M_avg by comparing each replica to U_ref,V_ref.

# AFTER (symmetric all-pairs consensus):
        # ---- build symmetric all-pairs consensus M_avg (no special reference) ----
        Ms = []
        for a in range(B):
            for b in range(a + 1, B):
                MU = (U_b[a].transpose(0, 1) @ U_b[b]).abs()   # [Kc,Kc]
                MV = (V_b[a].transpose(0, 1) @ V_b[b]).abs()   # [Kc,Kc]
                Ms.append((MU * MV).cpu().numpy())
        if not Ms:
            # degenerate B=1: identity alignment
            assign_idx = torch.arange(Kc, device=U_b.device, dtype=torch.long).view(1, Kc).repeat(1, 1)
            spectral_echo = torch.ones(Kc, device=U_b.device, dtype=U_b.dtype)
            return spectral_echo, assign_idx
        M_avg = np.mean(np.stack(Ms, axis=0), axis=0)

# Keep the LAP and permutation code as-is. For sign correction, establish a consensus orientation using the first replica post-permutation:
        U_cons = U_b[0][:, perm]  # [H,Kc]
        V_cons = V_b[0][:, perm]  # [W,Kc]
        for b in range(B):
            idx = assign_idx[b]  # [Kc], equals perm
            u_overlap = torch.sum(U_b[b][:, idx] * U_cons, dim=0)
            v_overlap = torch.sum(V_b[b][:, idx] * V_cons, dim=0)
            sgn = torch.sign(u_overlap * v_overlap)
            sgn[sgn == 0] = 1.0
            U_b[b].index_copy_(1, idx, U_b[b][:, idx] * sgn)
            V_b[b].index_copy_(1, idx, V_b[b][:, idx] * sgn)

# If you prefer the original ref=0 approach, you can skip this optional change; the main fixes are pairing + guard.

--------------------------------------------------------------------------------
Sanity checks (optional but recommended)
--------------------------------------------------------------------------------
After building each panel, keep these for a while:
- Assert 1D shape equality:
    assert sv.shape == echo.shape and sv.ndim == 1
- (Optional) Warn if Spearman rho(sv, echo) < 0.2 for a layer (health check, not a theorem).

--------------------------------------------------------------------------------
Expected outcomes
--------------------------------------------------------------------------------
- The ~0.45 shelf disappears (it was caused by mispairing).
- Curves match E[echo(s)] ≈ (1 + τ² / s²)^(-1); varying κ shifts horizontally in log(s).
- High-s flattening reduces thanks to the conditional guard and softer defaults. If still a bit low, try tau_mult=1.5 or noise_quantile=0.02.

--------------------------------------------------------------------------------
Full replacement functions (for IDEs that prefer copy-in bodies)
--------------------------------------------------------------------------------
# build_pred_actual_gptlp
def build_pred_actual_gptlp(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    IMPORTANT: Pair per-direction singulars (aggregated across replicas) with per-direction echoes.
    Avoid flattening [B,Kc] singulars; take median across B to get [Kc].
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        s_rep = props['aligned_replicate_singular_values']  # [B,Kc]
        s_dir = torch.median(s_rep, dim=0).values           # [Kc]
        sv = s_dir.detach().cpu().numpy()
        actual = props['spectral_echo'].detach().cpu().numpy()
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        if sv.size and actual.size:
            n = min(sv.size, actual.size)
            sv = sv[:n]; actual = actual[:n]
            pred = predict_spectral_echo_curve_np(sv, tau2)
            out[key] = np.vstack([np.clip(pred, 1e-8, 1.0),
                                  np.clip(actual, 1e-8, 1.0)])
    return out

# build_spectral_echo_vs_sv_panel
def build_spectral_echo_vs_sv_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Store per-direction singulars (median across replicas) and matched per-direction echoes.
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        s_rep = props['aligned_replicate_singular_values']                 # [B,Kc]
        s_dir = torch.median(s_rep, dim=0).values.detach().cpu().numpy()  # [Kc]
        echo = props['spectral_echo'].detach().cpu().numpy()               # [Kc]
        tau2 = float(props.get('empirical_phase_constant_tau2', np.nan))
        n = min(s_dir.size, echo.size)
        if n:
            out[key] = {
                'sv': s_dir[:n],
                'spectral_echo': echo[:n],
                'tau2': tau2,
                'shape': tuple(int(x) for x in props['checkpoint_weights'].shape[-2:]),
            }
            assert out[key]['sv'].ndim == 1 and out[key]['spectral_echo'].ndim == 1
            assert out[key]['sv'].shape == out[key]['spectral_echo'].shape
    return out

# denom guard snippet (paste inside solve_for_spectral_echo_using_reverb)
# defaults at top: noise_quantile=0.05, tau_mult=2.0
denom = spectral_reverb_Z.unsqueeze(0)                               # (1,a,b,Kc)
sgn = torch.where(denom >= 0, 1.0, -1.0)                             # (1,a,b,Kc)
guard = sgn * tau_dir.view(1, 1, 1, Kc)                              # (1,a,b,Kc)
need_guard = (denom.abs() < tau_dir.view(1, 1, 1, Kc))
denom_safe = torch.where(need_guard, denom + guard, denom)           # (1,a,b,Kc)

# optional symmetric consensus (replace the ref=0 block)
Ms = []
for a in range(B):
    for b in range(a + 1, B):
        MU = (U_b[a].transpose(0, 1) @ U_b[b]).abs()   # [Kc,Kc]
        MV = (V_b[a].transpose(0, 1) @ V_b[b]).abs()   # [Kc,Kc]
        Ms.append((MU * MV).cpu().numpy())
if not Ms:
    assign_idx = torch.arange(Kc, device=U_b.device, dtype=torch.long).view(1, Kc).repeat(1, 1)
    spectral_echo = torch.ones(Kc, device=U_b.device, dtype=U_b.dtype)
    return spectral_echo, assign_idx
M_avg = np.mean(np.stack(Ms, axis=0), axis=0)
# ... LAP to obtain perm -> assign_idx ...
U_cons = U_b[0][:, perm]
V_cons = V_b[0][:, perm]
for b in range(B):
    idx = assign_idx[b]
    u_overlap = torch.sum(U_b[b][:, idx] * U_cons, dim=0)
    v_overlap = torch.sum(V_b[b][:, idx] * V_cons, dim=0)
    sgn = torch.sign(u_overlap * v_overlap)
    sgn[sgn == 0] = 1.0
    U_b[b].index_copy_(1, idx, U_b[b][:, idx] * sgn)
    V_b[b].index_copy_(1, idx, V_b[b][:, idx] * sgn)

--------------------------------------------------------------------------------
End of patch
--------------------------------------------------------------------------------
