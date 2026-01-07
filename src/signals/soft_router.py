import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def soft_router(C_ab, C_perp, C_R, P,
                tau_ab=0.7, tau_perp=0.5, tau_delta=0.12,
                k=10.0):
    """Return soft weights (comp, alt, noise, overfit).
    - comp: AB consistent & (optionally residual stable)
    - alt: capacity-limited: AB inconsistent but high-rank helps (delta high)
    - overfit: AB consistent but residual inconsistent
    - noise: neither view consistent OR not persistent
    """
    p_ab = sigmoid(k*(C_ab - tau_ab))
    p_perp = sigmoid(k*(C_perp - tau_perp))
    delta = C_R - C_ab
    p_delta = sigmoid(k*(delta - tau_delta))

    # raw scores
    w_comp = p_ab * p_perp * P
    w_overfit = p_ab * (1.0 - p_perp) * P
    w_alt = (1.0 - p_ab) * (1.0 - p_perp) * p_delta * P
    w_noise = (1.0 - p_ab) * (1.0 - p_perp) * (1.0 - p_delta) * (1.0 - P)

    s = w_comp + w_overfit + w_alt + w_noise + 1e-12
    return {
        "comp": w_comp/s,
        "overfit": w_overfit/s,
        "alt": w_alt/s,
        "noise": w_noise/s,
        "delta": delta,
        "p_ab": p_ab,
        "p_perp": p_perp,
        "p_delta": p_delta,
    }
