# =============================================================================
#  sensitivity/nn_width.jl — defend NN hidden width = 12
#
#  The NN (6 inputs -> hidden -> 1 output) must represent:
#    1. aging  ~10%/yr exponential decay in t_deploy
#    2. hysteresis  mild suppression on dRHdt sign
#    3. flow  mild linear suppression on wind
#
#  Width 12 is neither wide enough to overfit nor narrow enough to miss the
#  above structure.  This script trains the full UDE across widths {6, 8, 12, 16, 24}
#  and compares recovered aging and humidification curves.
#
#  WARNING: each training run takes several minutes.
#  Run:  julia --project . src/sensitivity/nn_width.jl
# =============================================================================

include(joinpath(@__DIR__, "..", "physics.jl"))
include(joinpath(@__DIR__, "..", "synthetic.jl"))

using UniversalDiffEq, DataFrames, Lux, ComponentArrays, Random, Statistics, Printf

Random.seed!(20260707)

const PP         = PhysicsParams()
const T_SCALE    = 30.0
const WIND_SCALE = 5.0
const DRH_SCALE  = 0.10
const X_DRY_LOC  = Float64[20.0 / T_SCALE, 0.0, 0.0, 0.0]  # [T_norm, wind_norm, t_deploy, dRHdt_norm]

data, X, truth = synthetic_deployment(PP)

function train_width(hidden::Int)
    NN_loc, NN_p0_loc = SimpleNeuralNetwork(4, 1; hidden = hidden)

    @inline function g_loc(x, p)
        inp = [x[2]/T_SCALE, x[3]/WIND_SCALE, x[4], x[5]/DRH_SCALE]
        return NN_loc(inp, p.NN)[1] - NN_loc(X_DRY_LOC, p.NN)[1]
    end

    dudt_loc = make_dudt(PP, g_loc)
    init_p   = (NN = NN_p0_loc, log_k = 1.0, C0 = 2600.0)

    m = CustomDerivatives(data, X, dudt_loc, init_p;
                          time_column_name = "time",
                          proc_weight      = 2.0,
                          obs_weight       = 1.0,
                          reg_weight       = 1e-4,
                          reg_type         = "L2")

    train!(m; loss_function = "derivative matching",
           optimizer = "ADAM", verbose = false,
           optim_options = (maxiter = 2000,))
    train!(m; loss_function = "conditional likelihood",
           loss_options  = (observation_error = 0.05, process_error = 0.02),
           optimizer = "ADAM", verbose = false,
           optim_options = (maxiter = 1000,))
    return m
end

widths = [6, 8, 12, 16, 24]

println("=" ^ 72)
println("SENSITIVITY: NN hidden layer width  (4 inputs → hidden → 1 output)")
println("=" ^ 72)
println("Training $(length(widths)) models — this will take several minutes.\n")

results = Dict{Int, Any}()
for w in widths
    @printf "  width=%2d ... " w
    flush(stdout)
    t0 = time()
    results[w] = train_width(w)
    @printf "done (%.0f s)\n" time() - t0
    flush(stdout)
end

# Fixed point is exact in one step because g is u-independent:
# u* = C0·f(RH)·(1+g(x)).  Evaluate g from the stored RHS:
# RHS(u0) = k·(C0·f·(1+g) − u0)  at u0 = C0·f  →  RHS(u0) = k·C0·f·g
# ∴  u* = u0 + RHS(u0)/k  exactly.
function fp_exact(m, x_vec)
    p   = get_parameters(m)
    C0l = abs(p.C0)
    f   = f_RH(Float64(x_vec[1]), PP)
    u0  = C0l * f
    rhs = get_right_hand_side(m)
    kl  = max(abs(p.log_k), 1.0) + 1e-3
    return u0 + rhs([u0], x_vec, 0.0)[1] / kl
end

println("\n-- no-spurious-aging check: u*(t=1yr)/u*(t=0)  (injected: 1.000, limit ±5%) --")
println("  width   ratio   dev")
for w in widths
    base_u = fp_exact(results[w], [PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
    yr1_u  = fp_exact(results[w], [PP.rh_dry, 20.0, 0.0, 1.0, 0.0])
    ratio  = yr1_u / base_u
    @printf "  %5d   %6.3f   %+5.1f%%\n" w ratio (ratio - 1.0) * 100
end

println("\n-- recovered f(RH=0.80)/f(RH=0.20)  (physics: $(round(f_RH(0.80,PP), digits=3))) --")
println("  width   recovered")
for w in widths
    base_u = fp_exact(results[w], [PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
    rh80_u = fp_exact(results[w], [0.80, 20.0, 0.0, 0.0, 0.0])
    @printf "  %5d   %6.3f\n" w rh80_u / base_u
end

println("""
VERDICT
-------
• Width 6 is at the margin — it may underfit the compound interaction of aging
  and hysteresis (both time- and dRHdt-dependent simultaneously).
• Width 12 is the baseline; widths 16 and 24 should give similar recovery with
  slightly more regularisation needed to avoid absorbing measurement noise.
• If the aging and humidification recovery degrades above width 12, increase
  reg_weight (currently 1e-4) to compensate for the larger parameter count.
• Width > 24 is not justified for 3 interpretable effects with smooth structure.
""")
