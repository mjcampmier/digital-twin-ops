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
const CH1_SCALE  = 2600.0
const T_SCALE    = 30.0
const WIND_SCALE = 5.0
const DRH_SCALE  = 0.10

data, X, truth = synthetic_deployment(PP)

function train_width(hidden::Int)
    NN_loc, NN_p0_loc = SimpleNeuralNetwork(6, 1; hidden = hidden)

    @inline function res_loc(u, x, p)
        inp = [u[1]/CH1_SCALE, x[1], x[2]/T_SCALE, x[3]/WIND_SCALE, x[4], x[5]/DRH_SCALE]
        return NN_loc(inp, p.NN)[1] * CH1_SCALE
    end

    dudt_loc = make_dudt(PP, res_loc)
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
    train!(m; loss_function = "derivative matching",
           optimizer = "BFGS", verbose = false,
           optim_options = (maxiter = 300,))
    return m
end

widths = [6, 8, 12, 16, 24]

println("=" ^ 72)
println("SENSITIVITY: NN hidden layer width  (6 inputs → hidden → 1 output)")
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

println("\n-- recovered aging at t_deploy=1 yr  (injected: exp(-0.105) = $(round(exp(-0.105), digits=3))) --")
println("  width   u*(t=1yr)/C0")
for w in widths
    m    = results[w]
    pars = get_parameters(m)
    C0h  = abs(pars.C0)
    RHS  = get_right_hand_side(m)
    xa   = [PP.rh_dry, 20.0, 0.0, 1.0, 0.0]
    g(u1) = RHS([u1], xa, 0.0)[1]
    lo, hi = 0.3 * C0h, 2.0 * C0h
    glo = g(lo)
    for _ in 1:60
        mid = 0.5*(lo+hi); gm = g(mid)
        (sign(gm)==sign(glo)) ? (lo=mid; glo=gm) : (hi=mid)
    end
    fp = 0.5*(lo+hi)
    @printf "  %5d   %6.3f\n" w fp / C0h
end

println("\n-- recovered f(RH=0.80)/f(RH=0.20)  (physics: $(round(f_RH(0.80,PP), digits=3))) --")
println("  width   recovered")
for w in widths
    m    = results[w]
    pars = get_parameters(m)
    C0h  = abs(pars.C0)
    RHS  = get_right_hand_side(m)
    function fp_at(x_vec)
        lo, hi = 0.3*C0h, 2.0*C0h
        g(u1) = RHS([u1], x_vec, 0.0)[1]
        glo = g(lo)
        for _ in 1:60
            mid=0.5*(lo+hi); gm=g(mid)
            (sign(gm)==sign(glo)) ? (lo=mid; glo=gm) : (hi=mid)
        end
        return 0.5*(lo+hi)
    end
    base_u = fp_at([PP.rh_dry, 20.0, 0.0, 0.0, 0.0])
    rh80_u = fp_at([0.80, 20.0, 0.0, 0.0, 0.0])
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
